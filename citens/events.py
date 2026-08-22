"""Structured pipeline events.

The orchestrator emits events as it runs. Subscribers (CLI Rich renderer,
FastAPI SSE stream) translate them into progress UI / network messages without
coupling to pipeline internals.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "run_started",
    "step_started",
    "step_progress",
    "step_completed",
    "llm_trace",
    "run_completed",
    "run_failed",
]


class _Base(BaseModel):
    type: EventType
    ts: float = Field(default_factory=time.time)


class RunStarted(_Base):
    type: Literal["run_started"] = "run_started"
    topic: str = ""


class StepStarted(_Base):
    type: Literal["step_started"] = "step_started"
    step: str = ""
    title: str = ""


class StepProgress(_Base):
    type: Literal["step_progress"] = "step_progress"
    step: str = ""
    message: str = ""
    current: int | None = None
    total: int | None = None
    # render as an indented detail line under the step (retrieved content:
    # queries, per-source counts, filter verdicts, ...) instead of a status line
    detail: bool = False


class LLMTrace(_Base):
    """One model call in the agent transcript.

    phase "start" fires before the request (the UI shows it as the current
    activity); "end"/"cached" fire with the result. ``reasoning`` carries the
    reasoning-model's own thinking excerpt when the backend returns one.
    """

    type: Literal["llm_trace"] = "llm_trace"
    phase: Literal["start", "end", "cached"] = "start"
    call_id: str = ""
    model: str = ""
    purpose: str = ""
    thinking: bool = True
    reasoning: str = ""
    chars_in: int = 0
    chars_out: int = 0
    ms: int = 0
    stage: str = ""


class StepCompleted(_Base):
    type: Literal["step_completed"] = "step_completed"
    step: str = ""
    message: str = ""


class RunCompleted(_Base):
    type: Literal["run_completed"] = "run_completed"
    run_dir: str = ""
    review_path: str = ""
    summary: dict = Field(default_factory=dict)


class RunFailed(_Base):
    type: Literal["run_failed"] = "run_failed"
    message: str = ""
    step: str = ""


Event = (
    RunStarted
    | StepStarted
    | StepProgress
    | StepCompleted
    | LLMTrace
    | RunCompleted
    | RunFailed
)


class EventBus:
    """A tiny in-process pub/sub. Handlers receive each emitted event."""

    def __init__(self) -> None:
        self._handlers: list = []

    def subscribe(self, handler) -> None:
        self._handlers.append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers:
            handler(event)
