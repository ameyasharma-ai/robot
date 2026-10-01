"""Observability models — Traces, Spans, Structured Events, and Hardware Metrics."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SpanStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    WAITING = "WAITING"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class ObservabilityEvent:
    """Standard structured runtime event for telemetry and real-time observability."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    component: str = ""
    operation: str = ""
    status: str = "COMPLETED"
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Span:
    """Represents a discrete operation within an execution trace."""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    component: str
    operation: str
    status: SpanStatus = SpanStatus.RUNNING
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: SpanStatus = SpanStatus.COMPLETED, metadata: dict[str, Any] | None = None) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status
        if metadata:
            self.metadata.update(metadata)

    def cancel(self, metadata: dict[str, Any] | None = None) -> None:
        self.finish(status=SpanStatus.CANCELLED, metadata=metadata)

    def fail(self, error: str, metadata: dict[str, Any] | None = None) -> None:
        meta = {"error": error}
        if metadata:
            meta.update(metadata)
        self.finish(status=SpanStatus.ERROR, metadata=meta)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Trace:
    """Represents an entire end-to-end user interaction or workflow."""

    trace_id: str
    name: str = "voice_turn"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    status: SpanStatus = SpanStatus.RUNNING
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_span(
        self,
        component: str,
        operation: str,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            component=component,
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def finish(self, status: SpanStatus = SpanStatus.COMPLETED) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status

    def cancel(self) -> None:
        for span in self.spans:
            if span.status == SpanStatus.RUNNING:
                span.cancel()
        self.finish(status=SpanStatus.CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status.value,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }
