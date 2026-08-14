"""Unified LLM access.

A thin, dependency-light abstraction over chat-completion backends:

* ``OpenAICompatBackend`` (default) — speaks the OpenAI Chat Completions API, so
  it covers OpenAI, DeepSeek, Ollama, OpenRouter, vLLM, Groq, … via ``LLM_API_BASE``.
* ``LiteLLMBackend`` (optional, needs the ``[multi]`` extra) — native routing to
  Anthropic / Gemini / Bedrock / Cohere for users who need it.

The module-level :func:`chat` / :func:`chat_json` helpers cache a singleton
backend built from :mod:`litreview.config`.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from litreview.config import settings


@runtime_checkable
class LLMBackend(Protocol):
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_json: bool = False,
    ) -> str: ...


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return raw


class OpenAICompatBackend:
    """OpenAI-compatible Chat Completions backend."""

    def __init__(self) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": settings.llm_api_key}
        if settings.llm_api_base:
            kwargs["base_url"] = settings.llm_api_base
        self._client = OpenAI(**kwargs)
        self._model = settings.llm_model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_json: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens_default,
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class LiteLLMBackend:
    """Optional multi-provider backend (requires the ``[multi]`` extra)."""

    def __init__(self) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "LLM_PROVIDER=litellm requires the [multi] extra: "
                "pip install 'litreview[multi]'"
            ) from e
        self._api_key = settings.llm_api_key
        self._api_base = settings.llm_api_base
        self._model = settings.llm_model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_json: bool = False,
    ) -> str:  # pragma: no cover - requires optional dep
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens_default,
        }
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_api_base:
            kwargs["api_base"] = settings.llm_api_base
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content or ""


_backend: LLMBackend | None = None


def get_backend() -> LLMBackend:
    """Return (and cache) the LLM backend chosen by ``LLM_PROVIDER``."""
    global _backend
    if _backend is not None:
        return _backend
    provider = settings.llm_provider.lower()
    if provider == "litellm":
        _backend = LiteLLMBackend()
    elif provider == "openai":
        _backend = OpenAICompatBackend()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}")
    return _backend


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    response_json: bool = False,
) -> str:
    return get_backend().chat(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_json=response_json,
    )


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> dict:
    """Call the LLM and parse a JSON response.

    Resilient to reasoning models whose "thinking" tokens can squeeze the output
    budget to empty: on parse failure it retries once with a larger budget.
    """
    max_tokens = max_tokens or settings.llm_max_tokens_default

    def _call(budget: int) -> str:
        return chat(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=budget,
            response_json=True,
        )

    raw = _strip_fences(_call(max_tokens))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        bigger = max(max_tokens * 2, 8192)
        raw = _strip_fences(_call(bigger))
        return json.loads(raw)
