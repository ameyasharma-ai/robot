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
    start_monotonic: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: SpanStatus = SpanStatus.COMPLETED, metadata: dict[str, Any] | None = None) -> None:
        """Complete the span with status and record monotonic duration."""
        self.end_time = time.time()
        self.duration_ms = (time.perf_counter() - self.start_monotonic) * 1000.0
        self.status = status
        if metadata:
            self.metadata.update(metadata)

    def cancel(self, metadata: dict[str, Any] | None = None) -> None:
        """Mark span as cancelled."""
        self.finish(status=SpanStatus.CANCELLED, metadata=metadata)

    def fail(self, error: str, metadata: dict[str, Any] | None = None) -> None:
        """Mark span as failed with error details."""
        meta = {"error": error}
        if metadata:
            meta.update(metadata)
        self.finish(status=SpanStatus.ERROR, metadata=meta)

    def to_dict(self) -> dict[str, Any]:
        """Convert span to serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        d["duration_ms"] = round(self.duration_ms, 2)
        return d


@dataclass
class Trace:
    """Represents an entire end-to-end user interaction or workflow."""

    trace_id: str
    name: str = "voice_turn"
    category: str = "conversational"
    start_time: float = field(default_factory=time.time)
    start_monotonic: float = field(default_factory=time.perf_counter)
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
        """Create and register a child span on this trace."""
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            component=component,
            operation=operation,
            start_time=time.time(),
            start_monotonic=time.perf_counter(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def finish(self, status: SpanStatus = SpanStatus.COMPLETED) -> None:
        """Complete the trace and record total monotonic duration."""
        self.end_time = time.time()
        self.duration_ms = (time.perf_counter() - self.start_monotonic) * 1000.0
        self.status = status

    def cancel(self) -> None:
        """Cancel all active spans and mark trace as cancelled."""
        for span in self.spans:
            if span.status == SpanStatus.RUNNING:
                span.cancel()
        self.finish(status=SpanStatus.CANCELLED)

    def get_stage_durations(self) -> dict[str, float]:
        """Extract duration breakdown across all distinct stages."""
        durations: dict[str, float] = {
            "routing_ms": 0.0,
            "llm_ttft_ms": 0.0,
            "llm_generation_ms": 0.0,
            "tool_execution_ms": 0.0,
            "post_tool_ttft_ms": 0.0,
            "post_tool_generation_ms": 0.0,
            "tts_synthesis_ms": 0.0,
            "playback_ms": 0.0,
            "total_e2e_ms": round(self.duration_ms, 2),
        }
        for s in self.spans:
            comp = s.component.upper()
            op = s.operation.upper()
            dur = s.duration_ms
            if comp in ("ROUTING", "TOOL_ROUTER"):
                durations["routing_ms"] += dur
            elif comp == "LLM" and op == "GENERATE":
                durations["llm_generation_ms"] += dur
                if "ttft_ms" in s.metadata:
                    durations["llm_ttft_ms"] = float(s.metadata["ttft_ms"])
            elif comp == "LLM_POST_TOOL":
                durations["post_tool_generation_ms"] += dur
                if "ttft_ms" in s.metadata:
                    durations["post_tool_ttft_ms"] = float(s.metadata["ttft_ms"])
            elif comp in ("TOOL", "TOOLS"):
                durations["tool_execution_ms"] += dur
            elif comp == "TTS":
                durations["tts_synthesis_ms"] += dur
            elif comp in ("PLAYBACK", "AUDIO"):
                durations["playback_ms"] += dur
        return durations

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to serializable dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "category": self.category,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status.value,
            "stages": self.get_stage_durations(),
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }

