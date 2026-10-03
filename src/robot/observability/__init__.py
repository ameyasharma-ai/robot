"""Real-time observability hub — integrates trace tracking, event emission, metrics, and WebSocket streaming."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import deque
from typing import Any

from robot.core.events import EventBus
from robot.observability.models import ObservabilityEvent, Span, SpanStatus, Trace
from robot.observability.resources import SystemMetricsCollector
from robot.telemetry import get_logger

logger = get_logger("observability")

# SENSITIVE DATA REDACTION
REDACTION_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|password|secret|authorization|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})"), r"\1: [REDACTED]"),
    (re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}"), "Bearer [REDACTED]"),
]


def redact_sensitive(data: Any) -> Any:
    """Recursively redact secrets and credentials from event payloads."""
    if isinstance(data, str):
        cleaned = data
        for pattern, replacement in REDACTION_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)
        return cleaned
    elif isinstance(data, dict):
        return {k: redact_sensitive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_sensitive(item) for item in data]
    return data


class ObservabilityHub:
    """Coordinates traces, span hierarchy, event dispatch, and latency calculations."""

    def __init__(self, event_bus: EventBus, max_history: int = 250, redact: bool = True) -> None:
        self.events = event_bus
        self.max_history = max_history
        self.redact = redact
        self._traces: deque[Trace] = deque(maxlen=max_history)
        self._active_traces: dict[str, Trace] = {}
        self._active_spans: dict[str, Span] = {}
        self._current_trace_id: str | None = None
        self._resources = SystemMetricsCollector()

        # Pipeline Node States for Live UI
        self.pipeline_state: dict[str, str] = {
            "MIC": "IDLE",
            "VAD": "IDLE",
            "STT": "IDLE",
            "CONTEXT": "IDLE",
            "LLM": "IDLE",
            "TOOLS": "IDLE",
            "TTS": "IDLE",
            "AVATAR": "IDLE",
            "IMAGE": "IDLE",
        }

    @property
    def current_trace_id(self) -> str | None:
        return self._current_trace_id

    def set_pipeline_node(self, node: str, state: str) -> None:
        """Update live status of a pipeline component (IDLE, RUNNING, COMPLETED, CANCELLED, ERROR)."""
        node_upper = node.upper()
        if node_upper in self.pipeline_state:
            self.pipeline_state[node_upper] = state
            try:
                loop = asyncio.get_running_loop()
                if loop and loop.is_running():
                    asyncio.create_task(
                        self.events.emit(
                            "pipeline_updated",
                            pipeline=dict(self.pipeline_state),
                            active_node=node_upper,
                            node_status=state,
                        )
                    )
            except RuntimeError:
                pass

    def start_trace(self, name: str = "voice_turn", trace_id: str | None = None, metadata: dict[str, Any] | None = None) -> Trace:
        tid = trace_id or str(uuid.uuid4())[:8]
        trace = Trace(trace_id=tid, name=name, start_time=time.time(), metadata=metadata or {})
        self._active_traces[tid] = trace
        self._traces.append(trace)
        self._current_trace_id = tid
        return trace

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._active_traces.get(trace_id)

    def start_span(
        self,
        component: str,
        operation: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        tid = trace_id or self._current_trace_id or str(uuid.uuid4())[:8]
        trace = self._active_traces.get(tid)
        if not trace:
            trace = self.start_trace(trace_id=tid)

        span = trace.create_span(
            component=component,
            operation=operation,
            parent_span_id=parent_span_id,
            metadata=metadata or {},
        )
        self._active_spans[span.span_id] = span
        self.set_pipeline_node(component.upper(), "RUNNING")

        # Emit structured start event
        self.emit_event(
            event_type=f"{component.lower()}.{operation.lower()}.started",
            trace_id=tid,
            span_id=span.span_id,
            parent_span_id=parent_span_id,
            component=component,
            operation=operation,
            status="RUNNING",
            metadata=metadata or {},
        )
        return span

    def finish_span(
        self,
        span: Span | str,
        status: SpanStatus = SpanStatus.COMPLETED,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        span_obj = self._active_spans.pop(span, None) if isinstance(span, str) else span
        if not span_obj:
            return

        span_obj.finish(status=status, metadata=metadata)
        self.set_pipeline_node(span_obj.component.upper(), status.value)

        # Emit structured complete/error/cancel event
        self.emit_event(
            event_type=f"{span_obj.component.lower()}.{span_obj.operation.lower()}.{status.value.lower()}",
            trace_id=span_obj.trace_id,
            span_id=span_obj.span_id,
            parent_span_id=span_obj.parent_span_id,
            component=span_obj.component,
            operation=span_obj.operation,
            status=status.value,
            duration_ms=span_obj.duration_ms,
            metadata=span_obj.metadata,
        )

    def finish_trace(self, trace_id: str | None = None, status: SpanStatus = SpanStatus.COMPLETED) -> Trace | None:
        tid = trace_id or self._current_trace_id
        if not tid:
            return None
        trace = self._active_traces.pop(tid, None)
        if trace:
            trace.finish(status=status)
            if self._current_trace_id == tid:
                self._current_trace_id = None
        return trace

    def cancel_active_traces(self) -> None:
        """Cancel all in-flight spans and traces on barge-in interruption."""
        for tid, trace in list(self._active_traces.items()):
            trace.cancel()
            self.emit_event(
                event_type="trace.cancelled",
                trace_id=tid,
                component="CORE",
                operation="BARGE_IN",
                status="CANCELLED",
            )
        self._active_spans.clear()
        self._active_traces.clear()
        self._current_trace_id = None
        for k in self.pipeline_state:
            self.pipeline_state[k] = "IDLE"
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                asyncio.create_task(
                    self.events.emit(
                        "pipeline_updated",
                        pipeline=dict(self.pipeline_state),
                    )
                )
        except RuntimeError:
            pass

    def emit_event(
        self,
        event_type: str,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        component: str = "",
        operation: str = "",
        status: str = "COMPLETED",
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ObservabilityEvent:
        """Format, redact, and dispatch a structured event over the internal bus."""
        meta = metadata or {}
        if self.redact:
            meta = redact_sensitive(meta)

        evt = ObservabilityEvent(
            event_type=event_type,
            trace_id=trace_id or self._current_trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            component=component,
            operation=operation,
            status=status,
            duration_ms=round(duration_ms, 2),
            metadata=meta,
        )

        # Dispatch via async event bus
        try:
            asyncio.create_task(self.events.emit("observability.event", **evt.to_dict()))
        except RuntimeError:
            pass
        return evt

    def get_recent_traces(self, limit: int = 15) -> list[dict[str, Any]]:
        traces = list(self._traces)[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def get_hardware_metrics(self) -> dict[str, Any]:
        return self._resources.get_metrics()
