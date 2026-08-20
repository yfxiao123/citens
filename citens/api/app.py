"""FastAPI app: SSE-streamed runs + result retrieval + a minimal web UI.

Endpoints:
    POST /run    {topic, max_papers?, ...}  -> text/event-stream of pipeline
                events (RunStarted/StepStarted/StepProgress/StepCompleted/
                RunCompleted/RunFailed), then a final `result` event.
    GET  /runs                           -> recent run directories
    GET  /result/{run_id}                -> review.md + artifacts of a run
    GET  /health                         -> liveness + config sanity

The API extra (``pip install 'citens[api]'``) provides FastAPI/uvicorn/
sse-starlette; without it the CLI works unchanged.
"""

from __future__ import annotations

import asyncio
import json
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
    filters: dict = Field(default_factory=dict)  # answers from /clarify


@app.get("/clarify", dependencies=[Depends(_require_token)])
def clarify(topic: str) -> dict:
    """Pre-run clarifying questions for a topic (UI calls this before /run)."""
    from citens.agents.clarify import generate_clarifying_questions

    return {"topic": topic, "questions": generate_clarifying_questions(topic)}


def _event_to_dict(event: Event) -> dict:
    d = event.model_dump()
    return {"type": event.__class__.__name__, **d}


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


@app.get("/runs", dependencies=[Depends(_require_token)])
def list_runs() -> dict:
    root = Path(settings.output_dir)
    runs = []
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True)[:50]:
            if d.is_dir() and (d / "review.md").exists():
                meta_file = d / "meta.json"
                meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
                runs.append(
                    {
                        "run_id": d.name,
                        "topic": meta.get("topic", ""),
                        "citation_precision": meta.get("citation_precision"),
                        "time": d.name,
                    }
                )
    return {"runs": runs}


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
