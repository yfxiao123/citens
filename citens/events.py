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
    RunStarted | StepStarted | StepProgress | StepCompleted | RunCompleted | RunFailed
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
