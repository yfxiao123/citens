"""Unified LLM access.

A thin, dependency-light abstraction over chat-completion backends:

* ``OpenAICompatBackend`` (default) — speaks the OpenAI Chat Completions API, so
  it covers OpenAI, DeepSeek, Ollama, OpenRouter, vLLM, Groq, … via ``LLM_API_BASE``.
* ``LiteLLMBackend`` (optional, needs the ``[multi]`` extra) — native routing to
  Anthropic / Gemini / Bedrock / Cohere for users who need it.

Performance/cost layout:
* **Two model tiers.** Cheap structured stages (planner/filter/extract) use
  ``LLM_MODEL``; the intelligence-heavy stages (writer/synth/verifier) pass
  ``strong=True`` and use ``LLM_MODEL_STRONG`` (falls back to ``LLM_MODEL``).
* **Response cache.** Successful completions are cached on disk keyed by
  (model, prompts, params) — re-runs of the same topic skip repeat calls.
* :func:`run_concurrent` — a small thread-pool map used to parallelize
  per-paper / per-batch LLM calls (``LLM_CONCURRENCY``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol, runtime_checkable

from citens import cache
from citens.config import settings


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
    """OpenAI-compatible Chat Completions backend (one instance per model)."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": settings.llm_api_key}
        if settings.llm_api_base:
            kwargs["base_url"] = settings.llm_api_base
        self._client = OpenAI(**kwargs)
        self._model = model or settings.llm_model

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
        if getattr(resp, "usage", None):
            record_usage(
                self._model,
                resp.usage.prompt_tokens or 0,
                resp.usage.completion_tokens or 0,
            )
        return resp.choices[0].message.content or ""


class LiteLLMBackend:
    """Optional multi-provider backend (requires the ``[multi]`` extra)."""

    def __init__(self, model: str | None = None) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "LLM_PROVIDER=litellm requires the [multi] extra: "
                "pip install 'citens[multi]'"
            ) from e
        self._api_key = settings.llm_api_key
        self._api_base = settings.llm_api_base
        self._model = model or settings.llm_model

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
        if getattr(resp, "usage", None):
            record_usage(
                self._model,
                getattr(resp.usage, "prompt_tokens", 0) or 0,
                getattr(resp.usage, "completion_tokens", 0) or 0,
            )
        return resp.choices[0].message.content or ""


_backends: dict[str, LLMBackend] = {}

# --- usage telemetry ---------------------------------------------------------
# Backends record every completion's token usage here; RunLog attributes
# records to pipeline stages by timestamp (see runlog.token_usage_by_stage).
import threading  # noqa: E402

_usage_lock = threading.Lock()
_usage_records: list[dict] = []


def record_usage(model: str, prompt: int, completion: int) -> None:
    with _usage_lock:
        _usage_records.append(
            {"ts": time.time(), "model": model, "prompt": prompt, "completion": completion}
        )


def usage_records() -> list[dict]:
    """Snapshot of recorded usage events (thread-safe copy)."""
    with _usage_lock:
        return list(_usage_records)


def get_backend(model: str | None = None) -> LLMBackend:
    """Return (and cache) a backend for `model` (default: LLM_MODEL)."""
    model = model or settings.llm_model
    if model in _backends:
        return _backends[model]
    provider = settings.llm_provider.lower()
    if provider == "litellm":
        backend: LLMBackend = LiteLLMBackend(model)
    elif provider == "openai":
        backend = OpenAICompatBackend(model)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}")
    _backends[model] = backend
    return backend


def strong_model() -> str:
    """The model used for intelligence-heavy stages (writer/synth/verifier)."""
    return settings.llm_model_strong or settings.llm_model


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    response_json: bool = False,
    strong: bool = False,
) -> str:
    model = strong_model() if strong else settings.llm_model
    budget = max_tokens or settings.llm_max_tokens_default
    cache_key = {
        "model": model,
        "system": system_prompt,
        "user": user_prompt,
        "temperature": temperature,
        "max_tokens": budget,
        "json": response_json,
    }
    cached = cache.get("llm", cache_key)
    if cached is not None:
        return cached
    text = get_backend(model).chat(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=budget,
        response_json=response_json,
    )
    if text:  # never cache empty outputs (reasoning-model failure mode)
        cache.put("llm", cache_key, text)
    return text


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    strong: bool = False,
) -> dict:
    """Call the LLM and parse a JSON response.

    Resilient to reasoning models whose "thinking" tokens can squeeze the output
    budget to empty: on parse failure it retries once with a larger budget.
    """
    max_tokens = max_tokens or settings.llm_max_tokens_default

    def _call(budget: int, nudge: bool = False) -> str:
        prompt = user_prompt
        if nudge:
            # some backends answer with prose despite response_format; an
            # explicit instruction usually recovers a parseable reply
            prompt += "\n\nRespond with ONLY the JSON object. No prose, no code fences."
        return chat(
            system_prompt,
            prompt,
            temperature=temperature,
            max_tokens=budget,
            response_json=True,
            strong=strong,
        )

    raw = _strip_fences(_call(max_tokens))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # retry 1: same budget, explicit JSON-only nudge (prose answers are the
    # most common non-truncation parse failure)
    try:
        return json.loads(_strip_fences(_call(max_tokens, nudge=True)))
    except json.JSONDecodeError:
        pass
    # retry 2: bigger budget (reasoning models whose thinking squeezes the
    # output to empty/truncated JSON)
    bigger = max(max_tokens * 2, 8192)
    return json.loads(_strip_fences(_call(bigger)))


def run_concurrent(
    fn: Callable[[int, Any], Any],
    items: list[Any],
    *,
    on_done: Callable[[int, Any, Any], None] | None = None,
    max_workers: int | None = None,
) -> list[Any]:
    """Map ``fn(i, item)`` over ``items`` with a thread pool (LLM-bound I/O).

    ``on_done(i, item, result)`` fires in the SUBMITTING thread as each job
    completes (safe for progress events). Returns results in input order;
    ``fn`` should absorb its own exceptions. Falls back to a sequential loop
    when concurrency is 1 or there is a single item.
    """
    n = len(items)
    if n == 0:
        return []
    workers = max_workers if max_workers is not None else settings.llm_concurrency
    if n == 1 or workers <= 1:
        out = []
        for i, item in enumerate(items):
            r = fn(i, item)
            if on_done:
                on_done(i, item, r)
            out.append(r)
        return out

    results: list[Any] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            results[i] = r
            if on_done:
                on_done(i, items[i], r)
    return results
