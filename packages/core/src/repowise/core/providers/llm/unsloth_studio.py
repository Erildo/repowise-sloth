"""Unsloth Studio provider for repowise.

Unsloth Studio (https://unsloth.ai) runs fine-tuned models locally with an
OpenAI-compatible inference server. This provider talks to the local Studio
endpoint via the openai SDK with a custom base_url, following the same
pattern as OllamaProvider.

No API key required for local deployments — Unsloth Studio binds to localhost
by default and is intended for:
    - Inference on custom fine-tuned models without leaving the machine
    - Air-gapped or compliance-sensitive codebases
    - Iterating on a model trained in the same Studio session

Usage:
    provider = UnslothStudioProvider(
        model="gemma-4-E2B-it-GGUF",
        base_url="http://localhost:8000",
    )
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import APIStatusError as _OpenAIAPIStatusError
from openai import AsyncOpenAI
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from repowise.core.providers.llm.base import (
    BaseProvider,
    ChatStreamEvent,
    ChatToolCall,
    GeneratedResponse,
    ProviderError,
    ensure_reasoning_supported,
)
from repowise.core.rate_limiter import RateLimiter
from repowise.core.reasoning import ReasoningMode

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_MIN_WAIT = 1.0
_MAX_WAIT = 8.0

_DEFAULT_BASE_URL = "http://localhost:8000"


def _normalize_base_url(url: str) -> str:
    """Ensure base_url has http:// protocol and ends with /v1."""
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if not url.endswith("/v1"):
        url += "/v1"
    return url


class UnslothStudioProvider(BaseProvider):
    """Unsloth Studio provider for local, fine-tuned-model inference.

    Args:
        model:        Model identifier as served by the Studio endpoint
                      (e.g., 'gemma-4-E2B-it-GGUF').
        api_key:      Optional bearer token. Studio defaults to no auth; pass
                      a value (or set UNSLOTH_STUDIO_API_KEY) only if you've
                      enabled token auth on the server.
        base_url:     Studio server URL. Defaults to http://localhost:8000.
                      The /v1 suffix is appended automatically if missing.
        rate_limiter: Optional RateLimiter for resource-constrained machines.
    """

    def __init__(
        self,
        model: str = "gemma-4-E2B-it-GGUF",
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("UNSLOTH_STUDIO_API_KEY") or "unsloth"
        resolved_base_url = (
            base_url or os.environ.get("UNSLOTH_STUDIO_BASE_URL") or _DEFAULT_BASE_URL
        )
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=_normalize_base_url(resolved_base_url),
        )
        self._model = model
        self._rate_limiter = rate_limiter

    @property
    def provider_name(self) -> str:
        return "unsloth_studio"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        request_id: str | None = None,
        reasoning: ReasoningMode = "auto",
        cache_hints: tuple = (),
    ) -> GeneratedResponse:
        ensure_reasoning_supported("unsloth_studio", self._model, reasoning)
        if self._rate_limiter:
            await self._rate_limiter.acquire(estimated_tokens=max_tokens)

        log.debug(
            "unsloth_studio.generate.start",
            model=self._model,
            max_tokens=max_tokens,
            request_id=request_id,
        )

        try:
            return await self._generate_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=request_id,
            )
        except RetryError as exc:
            raise ProviderError(
                "unsloth_studio",
                f"All {_MAX_RETRIES} retries exhausted: {exc}",
            ) from exc

    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential_jitter(initial=_MIN_WAIT, max=_MAX_WAIT),
        reraise=True,
    )
    async def _generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        request_id: str | None,
    ) -> GeneratedResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except _OpenAIAPIStatusError as exc:
            raise ProviderError(
                "unsloth_studio", str(exc), status_code=exc.status_code
            ) from exc

        usage = response.usage
        result = GeneratedResponse(
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_tokens=0,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        )
        log.debug(
            "unsloth_studio.generate.done",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            request_id=request_id,
        )
        return result

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        request_id: str | None = None,
        tool_executor: Any | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream chat via Unsloth Studio's OpenAI-compatible endpoint."""
        import json as _json

        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except _OpenAIAPIStatusError as exc:
            raise ProviderError(
                "unsloth_studio", str(exc), status_code=exc.status_code
            ) from exc

        tool_calls_acc: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta
                finish = choice.finish_reason

                if delta and delta.content:
                    yield ChatStreamEvent(type="text_delta", text=delta.content)

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        acc = tool_calls_acc[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                acc["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc["arguments"] += tc_delta.function.arguments

                if finish:
                    for idx in sorted(tool_calls_acc.keys()):
                        acc = tool_calls_acc[idx]
                        try:
                            args = _json.loads(acc["arguments"]) if acc["arguments"] else {}
                        except Exception:
                            args = {}
                        yield ChatStreamEvent(
                            type="tool_start",
                            tool_call=ChatToolCall(
                                id=acc["id"], name=acc["name"], arguments=args
                            ),
                        )
                    tool_calls_acc.clear()
                    stop_reason = "tool_use" if finish == "tool_calls" else "end_turn"
                    yield ChatStreamEvent(type="stop", stop_reason=stop_reason)
        except _OpenAIAPIStatusError as exc:
            raise ProviderError(
                "unsloth_studio", str(exc), status_code=exc.status_code
            ) from exc
