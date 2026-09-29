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
    """Accumulates latency samples for a named metric."""

    name: str
    samples: list[float] = field(default_factory=list)

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

    def reset(self) -> None:
        self.samples.clear()


class Telemetry:
    """Central telemetry and metrics collector.

    Tracks all whitepaper-required latency metrics:
    STT_FINAL_LATENCY, LLM_TTFT, TTS_FIRST_AUDIO_LATENCY,
    END_OF_SPEECH_TO_FIRST_AUDIO, BARGE_IN_STOP_LATENCY, etc.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, LatencyMetric] = {}
        self._logger = structlog.get_logger("telemetry")

    def _ensure_metric(self, name: str) -> LatencyMetric:
        if name not in self._metrics:
            self._metrics[name] = LatencyMetric(name=name)
        return self._metrics[name]

    def record(self, name: str, duration_ms: float) -> None:
        metric = self._ensure_metric(name)
        metric.record(duration_ms)
        self._logger.info("metric", name=name, duration_ms=round(duration_ms, 1))

    def get_metric(self, name: str) -> LatencyMetric | None:
        return self._metrics.get(name)

    def snapshot(self) -> dict[str, dict]:
        return {
            name: {"last": m.last, "avg": round(m.avg, 1), "count": m.count}
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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)
