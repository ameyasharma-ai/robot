"""Service health monitoring and graceful degradation."""

from __future__ import annotations

import asyncio

from robot.core.state import StateManager
from robot.telemetry import get_logger

logger = get_logger("services.health")


class HealthMonitor:
    """Periodically checks service health and updates state.

    If a noncritical subsystem fails, Robot Core starts in degraded mode
    rather than preventing all operation (whitepaper §31).
    """

    def __init__(self, state: StateManager, check_interval_s: float = 30.0) -> None:
        self._state = state
        self._interval = check_interval_s
        self._checks: dict[str, object] = {}  # name -> provider with health_check()
        self._task: asyncio.Task | None = None

    def register(self, name: str, provider: object) -> None:
        """Register a provider that has an async health_check() method."""
        self._checks[name] = provider
        logger.info("health_check_registered", service=name)

    async def check_all(self) -> dict[str, bool]:
        """Run all health checks and update state."""
        results = {}
        for name, provider in self._checks.items():
            try:
                healthy = await provider.health_check()
                await self._state.update_service_health(name, healthy)
                results[name] = healthy
            except Exception as e:
                await self._state.update_service_health(name, False, error=str(e))
                results[name] = False
                logger.error("health_check_failed", service=name, error=str(e))
        return results

    async def start(self) -> None:
        """Start periodic health monitoring."""
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("health_monitor_started", interval_s=self._interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        while True:
            await self.check_all()
            await asyncio.sleep(self._interval)
