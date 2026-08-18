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
    }


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
