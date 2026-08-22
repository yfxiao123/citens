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
        thinking: bool | str = True,
    ) -> str: ...


def build_completion_kwargs(
    model: str,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    response_json: bool,
    thinking: bool | str = True,
) -> dict[str, Any]:
    """Chat-completions payload shared by the OpenAI-compatible backends.

    ``thinking=False`` injects ``reasoning_effort: "none"`` — on hybrid
    reasoning models (deepseek-v4-flash etc.) thinking and the visible body
    share one completion budget, so a long deliberation can starve the body
    to empty. Killing the thinking for mechanical calls (or as a writer's
    last-resort attempt) eliminates the empty-body failure mode at the
    source. Backends that ignore the field are unaffected.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}
    if thinking is False or thinking == "none":
        kwargs["extra_body"] = {"reasoning_effort": "none"}
    elif isinstance(thinking, str):
        # named effort ("low"/"medium"/...) — hybrid models budget a short
        # deliberation instead of the default full one
        kwargs["extra_body"] = {"reasoning_effort": thinking}
    return kwargs


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return raw


def _retryable(exc: BaseException) -> bool:
    """Transient transport/server failures worth retrying with backoff."""
    # openai.RateLimitStatusError etc. — match by name, not import, so this
    # also works for backends that wrap errors differently (Ollama, vLLM, …)
    name = type(exc).__name__
    if name in {
        "APITimeoutError", "APIConnectionError", "APIStatusError",
        "RateLimitError", "RateLimitStatusError", "InternalServerError",
        "InternalServerErrorResponse", "ServiceUnavailableError",
        "ServiceUnavailableResponse", "ReadTimeout", "ConnectError",
        "RemoteProtocolError",
    }:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    return isinstance(exc, (ConnectionError, TimeoutError))


def _chat_with_retry(create: Callable[[], Any], model: str) -> str:
    """Call ``create()`` with exponential backoff on transient failures.

    Max 3 attempts (1.5s → 4.5s) — enough to ride out a 429 burst or a
    dropped connection without stalling a stage for minutes. Non-transient
    errors (auth, bad request) surface immediately.
    """
    delays = (1.5, 4.5)
    for attempt in range(len(delays) + 1):
        try:
            resp = create()
        except Exception as exc:  # noqa: BLE001 - classified below
            if attempt < len(delays) and _retryable(exc):
                time.sleep(delays[attempt])
                continue
            raise
        if getattr(resp, "usage", None):
            record_usage(
                model,
                resp.usage.prompt_tokens or 0,
                resp.usage.completion_tokens or 0,
            )
        msg = resp.choices[0].message
        # reasoning models (deepseek-reasoner, GLM hybrid, ...) return their
        # thinking separately from the body; stash it for the trace hook
        _tls.last_reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        return msg.content or ""
    raise RuntimeError("unreachable")  # pragma: no cover


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
        thinking: bool | str = True,
    ) -> str:
        kwargs = build_completion_kwargs(
            self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens or settings.llm_max_tokens_default,
            response_json=response_json,
            thinking=thinking,
        )
        return _chat_with_retry(
            lambda: self._client.chat.completions.create(**kwargs), self._model
        )


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
        thinking: bool | str = True,  # accepted for protocol parity; unused
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
        return _chat_with_retry(lambda: litellm.completion(**kwargs), self._model)


_backends: dict[str, LLMBackend] = {}

# --- usage telemetry ---------------------------------------------------------
# Backends record every completion's token usage here. Each record is tagged
# with the current run scope (contextvar) so concurrent runs in one process
# attribute cleanly; RunLog reads its own run's records. Records with no
# scope fall back to timestamp-window attribution.
import contextvars  # noqa: E402
import threading  # noqa: E402

_usage_lock = threading.Lock()
_usage_records: list[dict] = []
_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "citens_run_id", default=None
)

# --- agent transcript hook ----------------------------------------------------
# When set, every chat() call reports itself here (start / end / cached) so the
# run's event bus can stream a live "what is the agent doing" transcript. The
# hook must be cheap and never raise — tracing must not affect the run.
_tls = threading.local()  # last_reasoning, written by _chat_with_retry
_trace_hook: Callable[[dict], None] | None = None
_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "citens_stage", default=""
)


def set_trace(hook: Callable[[dict], None] | None) -> None:
    """Install/clear the transcript hook (records: phase, call_id, model,
    purpose, thinking, chars_in/out, ms, reasoning excerpt, stage)."""
    global _trace_hook
    _trace_hook = hook


def set_trace_stage(stage: str) -> None:
    """Tag subsequent trace records with the pipeline step they belong to."""
    _current_stage.set(stage)


def _trace_purpose(system_prompt: str) -> str:
    """Short label for the transcript: the system prompt's first line."""
    first = system_prompt.strip().splitlines()[0] if system_prompt.strip() else ""
    return first[:90]


def _fire_trace(record: dict) -> None:
    if _trace_hook is not None:
        import contextlib

        with contextlib.suppress(Exception):  # tracing must never break a run
            _trace_hook(record)


def current_run_id() -> str | None:
    """The run scope active on this thread (trace hooks filter by it)."""
    return _current_run.get()


class run_scope:
    """Tag LLM usage records with a run id for the duration of the block.

    The contextvar propagates into worker threads spawned inside the scope
    (they copy the context), so per-paper extract/verify calls land in the
    right run even when several runs share the process.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> run_scope:
        self._token = _current_run.set(self.run_id)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current_run.reset(self._token)


def record_usage(model: str, prompt: int, completion: int) -> None:
    with _usage_lock:
        _usage_records.append(
            {
                "ts": time.time(),
                "model": model,
                "prompt": prompt,
                "completion": completion,
                "run": _current_run.get(),
            }
        )


def usage_records(run_id: str | None = None) -> list[dict]:
    """Snapshot of recorded usage events (thread-safe copy).

    With ``run_id``: only records tagged to that run (plus untagged records,
    for callers that still rely on timestamp fallback).
    """
    with _usage_lock:
        records = list(_usage_records)
    if run_id is None:
        return records
    return [r for r in records if r.get("run") in (None, run_id)]


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


def reset_backends() -> None:
    """Drop cached backend clients (each holds the key/base_url of its
    creation time). The settings UI calls this after saving new LLM
    credentials so the next call uses them without a restart."""
    with _usage_lock:
        _backends.clear()


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    response_json: bool = False,
    strong: bool = False,
    thinking: bool | str = True,
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
        "thinking": thinking,
    }
    import uuid

    call_id = uuid.uuid4().hex[:8]
    purpose = _trace_purpose(system_prompt)
    stage = _current_stage.get()
    _fire_trace({
        "phase": "start", "call_id": call_id, "model": model, "purpose": purpose,
        "thinking": thinking is not False, "chars_in": len(user_prompt),
        "stage": stage,
    })
    t0 = time.monotonic()
    cached = cache.get("llm", cache_key)
    if cached is not None:
        _fire_trace({
            "phase": "cached", "call_id": call_id, "model": model,
            "purpose": purpose, "chars_out": len(cached), "ms": 0,
            "stage": stage,
        })
        return cached
    text = get_backend(model).chat(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=budget,
        response_json=response_json,
        thinking=thinking,
    )
    reasoning = getattr(_tls, "last_reasoning", "") or ""
    _fire_trace({
        "phase": "end", "call_id": call_id, "model": model, "purpose": purpose,
        "chars_out": len(text), "ms": round((time.monotonic() - t0) * 1000),
        "reasoning": reasoning[:800], "stage": stage,
    })
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
    thinking: bool | str = True,
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
            thinking=thinking,
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
    # worker threads don't inherit the context; re-set the run tag and the
    # transcript stage in each worker's own context so attribution follows
    # pool jobs (usage records AND trace records)
    parent_run = _current_run.get()
    parent_stage = _current_stage.get()

    def wrapped(i: int, item: Any) -> Any:
        if parent_run is None and not parent_stage:
            return fn(i, item)
        run_token = _current_run.set(parent_run) if parent_run is not None else None
        stage_token = _current_stage.set(parent_stage) if parent_stage else None
        try:
            return fn(i, item)
        finally:
            if run_token is not None:
                _current_run.reset(run_token)
            if stage_token is not None:
                _current_stage.reset(stage_token)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(wrapped, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            results[i] = r
            if on_done:
                on_done(i, items[i], r)
    return results
