"""FastAPI app: runs + result retrieval + a minimal web UI.

Endpoints:
    POST /run    {topic, max_papers?, ...}  -> text/event-stream of pipeline
                events (RunStarted/StepStarted/StepProgress/StepCompleted/
                LLMTrace/RunCompleted/RunFailed), then a final `result` event.
                For API consumers that can hold a streaming connection.
    POST /run/start  -> {run_id} immediately; the pipeline runs in a thread
                and every event lands in an in-memory log.
    GET  /run/events/{run_id}?after=seq -> incremental events for polling.
                The web UI uses this pair instead of SSE: plain request/
                response survives proxies/AV that buffer text/event-stream
                (observed: fetch-streaming delivered 0 frames for minutes
                while the run progressed server-side), and `after=0` replays
                the whole transcript after a page refresh.
    GET  /runs                           -> recent run directories
    GET  /result/{run_id}                -> review.md + artifacts of a run
    GET  /health                         -> liveness + config sanity

The API extra (``pip install 'citens[api]'``) provides FastAPI/uvicorn/
sse-starlette; without it the CLI works unchanged.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from citens import __version__
from citens.api.envstore import env_path, read_env_value, update_env_file
from citens.config import settings
from citens.events import Event, EventBus, RunCompleted, RunFailed
from citens.orchestration import RunOptions, run_pipeline_async

app = FastAPI(title="citens", version=__version__, docs_url="/docs")

# CORS: lock down to explicit origins; "*" only via config when you mean it.
_allowed = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if _allowed:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Optional bearer auth: set API_TOKEN when exposing the server beyond
# localhost — /run spends LLM credits, so leaving it open on a public
# interface hands your API key to the internet.
_bearer = HTTPBearer(auto_error=False)


def _require_token(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not settings.api_token:
        return  # auth disabled (local dev default)
    if creds is None or creds.credentials != settings.api_token:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


class RunRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="research topic")
    max_papers: int | None = None
    max_results: int | None = None
    sources: list[str] | None = None
    mode: str | None = None  # quick_scan/deep_review/interactive (auto-detect if None)
    fetch_fulltext: bool = True
    enrich_abstracts: bool = True
    allow_supplement: bool = True
    use_cache: bool = True
    agentic: bool = False  # agentic retrieval harness (Phase 1)
    filters: dict = Field(default_factory=dict)  # answers from /clarify


@app.get("/clarify", dependencies=[Depends(_require_token)])
def clarify(topic: str) -> dict:
    """Pre-run clarifying questions for a topic (UI calls this before /run)."""
    from citens.agents.clarify import generate_clarifying_questions

    return {"topic": topic, "questions": generate_clarifying_questions(topic)}


def _event_to_dict(event: Event) -> dict:
    # the class name AFTER the spread: model_dump() carries a lowercase
    # Literal `type` field ("run_started") that would otherwise clobber the
    # PascalCase tag the UI's handleEvent dispatches on — the exact bug that
    # made every arriving event invisible to the console for releases 1.2.x
    d = event.model_dump()
    d["type"] = event.__class__.__name__
    return d


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "llm_model": settings.llm_model,
        "llm_provider": settings.llm_provider,
        "search_sources": settings.search_sources,
        "sjr_data": Path(settings.sjr_csv_path).is_file(),
        "papers_dir": Path(settings.papers_dir).is_dir(),
        # the console uses this to auto-open the settings page on first run
        "llm_configured": bool(settings.llm_api_key),
        # where this launch's data (runs/.env/cache) lives — the header chip
        # answers "where did my records go" at a glance
        "workdir": str(Path.cwd()),
    }


# --- settings UI (the desktop app's config manager) ---------------------------
#
# The .env next to the exe is the single source of truth; the API reads it,
# masks secrets on the way out, applies saved values to the live settings
# object, and resets the LLM backend cache so new keys take effect on the
# next call without a restart.

_SETTINGS_FIELDS: list[tuple[str, str, str, bool, str]] = [
    # (env key, group, label, secret, hint)
    ("LLM_API_BASE", "llm", "API Base URL", False,
     "任何 OpenAI 兼容服务：DeepSeek / OpenRouter / vLLM / Groq / Ollama …"),
    ("LLM_API_KEY", "llm", "API Key", True, "服务商控制台获取"),
    ("LLM_MODEL", "llm", "模型（日常阶段）", False,
     "planner / 筛选 / 抽取 用；如 deepseek-chat"),
    ("LLM_MODEL_STRONG", "llm", "强模型（写作/核验，可选）", False,
     "留空 = 与日常模型相同"),
    ("SEMANTIC_SCHOLAR_API_KEY", "sources", "Semantic Scholar API Key", True,
     "免费申请；避免 S2 限流掉源"),
    ("OPENALEX_EMAIL", "sources", "OpenAlex 邮箱", False, "礼貌池，更快更稳"),
    ("CROSSREF_EMAIL", "sources", "Crossref 邮箱", False, "礼貌池"),
    ("CORE_API_KEY", "sources", "CORE API Key", True,
     "免费申请；显著提高全文命中率"),
    ("HTTP_PROXY", "access", "HTTP(S) 代理（可选）", False,
     "有校园代理/VPN 时填，可取付费论文"),
    ("EZPROXY_PREFIX", "access", "EZproxy 前缀（可选）", False,
     "如 https://lib.univ.edu.cn/login?url="),
    ("CITELENS_WORKDIR", "app", "工作目录（重启生效）", False,
     "数据（runs/文献库/缓存）存放位置；留空 = exe 旁边"),
]


def _mask(v: str, secret: bool) -> str:
    if not v:
        return ""
    if not secret or len(v) <= 8:
        return v
    return f"{v[:5]}…{v[-4:]}"


def _field_value(env_key: str) -> str:
    raw = read_env_value(env_path(), env_key)
    if raw is not None:
        return raw
    # fall back to the live settings object (env vars set outside .env)
    return str(getattr(settings, env_key.lower(), "") or "")


@app.get("/settings", dependencies=[Depends(_require_token)])
def get_settings() -> dict:
    fields = []
    for env_key, group, label, secret, hint in _SETTINGS_FIELDS:
        v = _field_value(env_key)
        fields.append({
            "key": env_key, "group": group, "label": label,
            "secret": secret, "hint": hint,
            "current": _mask(v, secret), "set": bool(v),
        })
    return {
        "fields": fields,
        "workdir": str(Path.cwd()),
        "env_file": str(env_path()),
    }


class SettingsUpdate(BaseModel):
    updates: dict[str, str] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


@app.post("/settings", dependencies=[Depends(_require_token)])
def save_settings(req: SettingsUpdate) -> dict:
    known = {f[0] for f in _SETTINGS_FIELDS}
    unknown = set(req.updates) - known
    if unknown:
        raise HTTPException(400, f"unknown settings keys: {sorted(unknown)}")

    update_env_file(env_path(), req.updates)

    # apply to the live settings object (same instance every module holds) so
    # changes take effect immediately; the workdir needs a restart by nature
    from citens import llm

    llm_touched = False
    for key, val in req.updates.items():
        if key == "CITELENS_WORKDIR":
            continue
        setattr(settings, key.lower(), val.strip())
        if key.startswith("LLM_"):
            llm_touched = True
    if llm_touched:
        llm.reset_backends()  # cached clients hold the old key/base

    applied = [k for k, v in req.updates.items() if v.strip()]
    return {
        "applied": applied,
        "needs_restart": "CITELENS_WORKDIR" in req.updates,
    }


@app.post("/settings/test", dependencies=[Depends(_require_token)])
def test_llm_connection(req: SettingsUpdate) -> dict:
    """One tiny completion against the given (or current) LLM config."""
    import time

    from citens.llm import build_completion_kwargs

    base = (req.updates.get("LLM_API_BASE") or settings.llm_api_base).strip()
    key = req.updates.get("LLM_API_KEY") or settings.llm_api_key
    model = (req.updates.get("LLM_MODEL") or settings.llm_model).strip()
    if not key:
        return {"ok": False, "error": "缺少 API Key / missing API key"}
    try:
        from openai import OpenAI

        client = (
            OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
        )
        kwargs = build_completion_kwargs(
            model,
            system_prompt="You are a connectivity test.",
            user_prompt="Reply with the single word: OK",
            temperature=0.0,
            max_tokens=512,
            response_json=False,
            thinking=False,
        )
        t0 = time.monotonic()
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        return {
            "ok": True,
            "latency_ms": round((time.monotonic() - t0) * 1000),
            "model": resp.model,
            "reply": text[:40],
        }
    except Exception as e:  # noqa: BLE001 - surface the provider's message
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


@app.post("/run", dependencies=[Depends(_require_token)])
async def run(req: RunRequest):
    """Stream pipeline events over SSE; final `result` event carries artifacts.

    The pipeline runs in a worker thread with its OWN event loop: agent LLM
    calls are synchronous HTTP and would otherwise block the server loop.
    Events cross back via a thread-safe queue handoff.
    """

    async def gen():
        from citens.models import RunMode
        
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        
        # Parse mode if provided
        run_mode = None
        if req.mode:
            try:
                run_mode = RunMode(req.mode)
            except ValueError:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "type": "ValidationError",
                        "message": f"无效的 mode: {req.mode}，可选值: quick_scan/deep_review/interactive"
                    }),
                }
                return
        
        options = RunOptions(
            max_papers=req.max_papers,
            max_results=req.max_results,
            sources=req.sources,
            mode=run_mode,
            fetch_fulltext=req.fetch_fulltext,
            enrich_abstracts=req.enrich_abstracts,
            allow_supplement=req.allow_supplement,
            use_cache=req.use_cache,
            agentic_retrieval=req.agentic,
            filters=req.filters,
        )

        def publish(event: Event) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_in_thread() -> None:
            asyncio.run(run_pipeline_async(req.topic, options, _QueueBus(publish)))

        pipeline_task = loop.run_in_executor(None, run_in_thread)

        event = None
        while True:
            if pipeline_task.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": ""}
                continue
            yield {"event": "message", "data": json.dumps(_event_to_dict(event), ensure_ascii=False)}
            if isinstance(event, RunCompleted | RunFailed):
                break

        exc = pipeline_task.exception() if pipeline_task.done() else None
        if exc is not None and not isinstance(event, RunFailed):
            yield {
                "event": "error",
                "data": json.dumps({"type": "PipelineError", "message": str(exc)}),
            }

    return EventSourceResponse(gen())


class _QueueBus(EventBus):
    """An EventBus that forwards each event to a callback (used by /run SSE)."""

    def __init__(self, publish) -> None:
        super().__init__()
        self._publish = publish

    def emit(self, event: Event) -> None:
        self._publish(event)


# --- polling event log (the UI's transport; survives stream-buffering) --------


class _RunBuffer:
    """In-memory append-only event log for one UI-started run.

    Each event gets a monotonically increasing seq; ``after=0`` replays the
    whole transcript (page-refresh recovery), ``after=last`` is the live
    polling cursor. Bounded by TTL + count GC, not by run length.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self.done = False
        self.error: str | None = None
        self.created = time.time()

    def append(self, event: Event) -> None:
        with self._lock:
            self._events.append({"seq": len(self._events), "event": _event_to_dict(event)})
            if isinstance(event, RunCompleted | RunFailed):
                self.done = True
                if isinstance(event, RunFailed):
                    self.error = event.message

    def tail(self, after: int) -> dict:
        with self._lock:
            return {
                "events": self._events[max(after, 0):],
                "done": self.done,
                "error": self.error,
                "server_time": time.time(),
            }


_RUN_BUFFERS: dict[str, _RunBuffer] = {}
_RUN_BUFFERS_LOCK = threading.Lock()
_RUN_BUFFER_TTL_S = 30 * 60
_RUN_BUFFER_MAX = 8


def _gc_run_buffers_locked() -> None:
    """Drop finished-and-stale logs; keep at most the newest _RUN_BUFFER_MAX."""
    now = time.time()
    stale = [
        k
        for k, b in _RUN_BUFFERS.items()
        if b.done and now - b.created > _RUN_BUFFER_TTL_S
    ]
    for k in stale:
        del _RUN_BUFFERS[k]
    while len(_RUN_BUFFERS) > _RUN_BUFFER_MAX:
        oldest = min(_RUN_BUFFERS, key=lambda k: _RUN_BUFFERS[k].created)
        del _RUN_BUFFERS[oldest]


def _resolve_mode(mode: str | None):
    """Parse the request's mode string; None means auto-detect."""
    from citens.models import RunMode

    if not mode:
        return None
    try:
        return RunMode(mode)
    except ValueError:
        raise HTTPException(
            400, f"无效的 mode: {mode}，可选值: quick_scan/deep_review/interactive"
        ) from None


@app.post("/run/start", dependencies=[Depends(_require_token)])
async def run_start(req: RunRequest) -> dict:
    """Start a run in the background; return its event-log id immediately."""
    options = RunOptions(
        max_papers=req.max_papers,
        max_results=req.max_results,
        sources=req.sources,
        mode=_resolve_mode(req.mode),
        fetch_fulltext=req.fetch_fulltext,
        enrich_abstracts=req.enrich_abstracts,
        allow_supplement=req.allow_supplement,
        use_cache=req.use_cache,
        agentic_retrieval=req.agentic,
        filters=req.filters,
    )
    key = uuid.uuid4().hex[:12]
    buf = _RunBuffer()
    with _RUN_BUFFERS_LOCK:
        _gc_run_buffers_locked()
        _RUN_BUFFERS[key] = buf
        while len(_RUN_BUFFERS) > _RUN_BUFFER_MAX:
            oldest = min(_RUN_BUFFERS, key=lambda k: _RUN_BUFFERS[k].created)
            del _RUN_BUFFERS[oldest]

    def run_in_thread() -> None:
        try:
            asyncio.run(run_pipeline_async(req.topic, options, _QueueBus(buf.append)))
        except Exception as e:  # noqa: BLE001 - the log must always close
            buf.append(RunFailed(message=str(e), step="pipeline"))
        except BaseException as e:  # noqa: BLE001 - SystemExit must not hang the UI
            # the v1.3.3 desktop exe died HERE silently (thread gone, no
            # terminal event) and the UI polled forever — any thread death
            # must land as a visible RunFailed
            buf.append(RunFailed(
                message=f"{type(e).__name__}: {e}", step="pipeline"
            ))
            raise

    threading.Thread(target=run_in_thread, daemon=True, name=f"run-{key}").start()
    return {"run_id": key}


@app.get("/run/events/{key}", dependencies=[Depends(_require_token)])
def run_events(key: str, after: int = 0) -> dict:
    """Events with seq >= after for a /run/start id (the UI's polling feed)."""
    with _RUN_BUFFERS_LOCK:
        buf = _RUN_BUFFERS.get(key)
    if buf is None:
        raise HTTPException(404, "unknown or expired run id")
    return buf.tail(after)


@app.post("/shutdown", dependencies=[Depends(_require_token)])
def shutdown() -> dict:
    """Stop the app (the windowed exe's ⏻ button — there is no console
    window to close anymore). The response is flushed before the exit."""
    import os
    import threading

    threading.Timer(0.4, lambda: os._exit(0)).start()
    return {"ok": True, "bye": True}


@app.get("/runs", dependencies=[Depends(_require_token)])
def list_runs() -> dict:
    """Completed runs, NEWEST FIRST (the history panel's contract).

    Sorted by the run's actual time: the -YYYYMMDD_HHMMSS suffix of the run
    dir, falling back to mtime. Reverse-sorting the dir NAMES puts 中文-topic
    runs in Unicode order, not time order (订单簿 always beat 生成式).
    """
    import datetime as _dt
    import re as _re

    ts_re = _re.compile(r"-(\d{8}_\d{6})$")
    root = Path(settings.output_dir)
    entries: list[tuple[_dt.datetime, dict]] = []
    if root.is_dir():
        for d in root.iterdir():
            if not (d.is_dir() and (d / "review.md").exists()):
                continue
            meta_file = d / "meta.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            ts: _dt.datetime | None = None
            m = ts_re.search(d.name)
            if m:
                try:
                    ts = _dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                except ValueError:
                    ts = None
            if ts is None:
                try:
                    ts = _dt.datetime.fromtimestamp(d.stat().st_mtime)
                except OSError:
                    ts = _dt.datetime.min
            entries.append(
                (
                    ts,
                    {
                        "run_id": d.name,
                        "topic": meta.get("topic", ""),
                        "citation_precision": meta.get("citation_precision"),
                        "time": ts.strftime("%m-%d %H:%M"),
                    },
                )
            )
    entries.sort(key=lambda e: e[0], reverse=True)
    return {"runs": [e[1] for e in entries[:50]]}


@app.get("/result/{run_id}", dependencies=[Depends(_require_token)])
def result(run_id: str) -> dict:
    d = Path(settings.output_dir) / run_id
    if not d.is_dir():
        raise HTTPException(404, f"unknown run: {run_id}")
    review = d / "review.md"
    payload: dict = {"run_id": run_id, "review_md": review.read_text(encoding="utf-8") if review.exists() else ""}
    for key, name in (
        ("verification", "verification.json"),
        ("grounding", "grounding.json"),
        ("meta", "meta.json"),
        ("fetch_list", "fetch_list.md"),
        ("references_bib", "references.bib"),
        ("provenance", "provenance.json"),
    ):
        f = d / name
        if f.exists():
            payload[key] = (
                json.loads(f.read_text(encoding="utf-8")) if name.endswith(".json") else f.read_text(encoding="utf-8")
            )
    return payload


# files the UI may link to directly (the rest stay API-shaped payloads)
_SERVABLE_ARTIFACTS = {
    "review_browser.html": "text/html",
    "references.ris": "application/octet-stream",
    "references.bib": "text/plain",
    "review.md": "text/markdown",
    "fetch_list.md": "text/markdown",
}


@app.get("/artifact/{run_id}/{filename}", dependencies=[Depends(_require_token)])
def artifact(run_id: str, filename: str) -> FileResponse:
    """Serve one whitelisted artifact file of a run (the audit browser, the
    RIS export, ...). Whitelist + resolved-path containment: run ids and
    filenames both come from the URL, so traversal must be impossible."""
    ctype = _SERVABLE_ARTIFACTS.get(filename)
    if ctype is None:
        raise HTTPException(404, f"not a servable artifact: {filename}")
    root = Path(settings.output_dir).resolve()
    d = (root / run_id).resolve()
    if not d.is_relative_to(root) or not d.is_dir():
        raise HTTPException(404, f"unknown run: {run_id}")
    f = d / filename
    if not f.is_file():
        raise HTTPException(404, f"run has no {filename}")
    return FileResponse(f, media_type=ctype)


_static = Path(__file__).parent / "static"
if (_static / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
