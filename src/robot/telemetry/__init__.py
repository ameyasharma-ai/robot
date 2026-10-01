"""Structured event logging and telemetry for Robot Core.

Tags: [USER] [STT] [LLM] [MEMORY] [TOOL] [TTS] [SYSTEM] [ERROR]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import structlog


class EventTag(str, Enum):
    USER = "USER"
    STT = "STT"
    LLM = "LLM"
    MEMORY = "MEMORY"
    TOOL = "TOOL"
    TTS = "TTS"
    FACE = "FACE"
    RGB = "RGB"
    VISION = "VISION"
    SYSTEM = "SYSTEM"
    ERROR = "ERROR"


@dataclass
class LatencyMetric:
    """Accumulates latency samples for a named metric and computes statistics."""

    name: str
    samples: list[float] = field(default_factory=list)
    target_ms: float | None = None

    def record(self, duration_ms: float) -> None:
        self.samples.append(duration_ms)

    @property
    def last(self) -> float:
        return self.samples[-1] if self.samples else 0.0

    @property
    def avg(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def count(self) -> int:
        return len(self.samples)

    def percentile(self, p: float) -> float:
        """Calculate the p-th percentile (0 <= p <= 100)."""
        if not self.samples:
            return 0.0
        sorted_s = sorted(self.samples)
        k = (len(sorted_s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_s) - 1)
        d0 = sorted_s[f] * (c - k)
        d1 = sorted_s[c] * (k - f)
        return round(d0 + d1, 1)

    @property
    def p50(self) -> float:
        return self.percentile(50.0)

    @property
    def p95(self) -> float:
        return self.percentile(95.0)

    @property
    def p99(self) -> float:
        return self.percentile(99.0)

    @property
    def status(self) -> str:
        if self.target_ms is None or not self.samples:
            return "N/A"
        if self.p95 <= self.target_ms:
            return "PASS"
        elif self.p50 <= self.target_ms:
            return "WARNING"
        return "FAIL"

    def reset(self) -> None:
        self.samples.clear()


class Telemetry:
    """Central telemetry and metrics collector with targets and percentile computation."""

    DEFAULT_TARGETS: dict[str, float] = {
        "VAD_LATENCY": 30.0,
        "STT_PARTIAL_LATENCY": 500.0,
        "STT_FINAL_LATENCY": 300.0,
        "LLM_TTFT": 400.0,
        "LLM_GENERATION_LATENCY": 1200.0,
        "TOOL_EXECUTION_LATENCY": 50.0,
        "TTS_FIRST_AUDIO_LATENCY": 150.0,
        "AVATAR_REACTION_LATENCY": 50.0,
        "END_OF_SPEECH_TO_FIRST_AUDIO": 900.0,
    }

    def __init__(self, targets: dict[str, float] | None = None) -> None:
        self._metrics: dict[str, LatencyMetric] = {}
        self._logger = structlog.get_logger("telemetry")
        self._targets = dict(self.DEFAULT_TARGETS)
        if targets:
            self._targets.update(targets)

    def set_target(self, name: str, target_ms: float) -> None:
        self._targets[name] = target_ms
        if name in self._metrics:
            self._metrics[name].target_ms = target_ms

    def _ensure_metric(self, name: str) -> LatencyMetric:
        if name not in self._metrics:
            target = self._targets.get(name)
            self._metrics[name] = LatencyMetric(name=name, target_ms=target)
        return self._metrics[name]

    def record(self, name: str, duration_ms: float) -> None:
        metric = self._ensure_metric(name)
        metric.record(duration_ms)
        self._logger.info("metric", name=name, duration_ms=round(duration_ms, 1))

    def get_metric(self, name: str) -> LatencyMetric | None:
        return self._metrics.get(name)

    def snapshot(self) -> dict[str, dict]:
        return {
            name: {
                "last": round(m.last, 1),
                "avg": round(m.avg, 1),
                "p50": m.p50,
                "p95": m.p95,
                "p99": m.p99,
                "target": m.target_ms,
                "status": m.status,
                "count": m.count,
            }
            for name, m in self._metrics.items()
        }


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, telemetry: Telemetry, metric_name: str) -> None:
        self._telemetry = telemetry
        self._name = metric_name
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._telemetry.record(self._name, elapsed_ms)


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog with processors for console output."""
    import logging

    log_level = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)
