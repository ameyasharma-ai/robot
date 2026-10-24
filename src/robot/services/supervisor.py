"""Centralized Process Supervisor & Health Watchdog.

Features:
- Monitors background AI engine daemons (llama-server on RTX 3050, whisper-server on Vulkan)
- Real-time CPU RSS and GPU VRAM leak detection via psutil and nvidia-smi
- Proactive memory compaction and threshold alerts (<2.5GB CPU, <3.8GB VRAM)
- Centralized signal trap (SIGINT, SIGTERM) for clean zero-orphan process termination
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Optional

import psutil

from robot.telemetry import get_logger

logger = get_logger("services.supervisor")

#: A full GC is stop-the-world; while RSS stays above the ceiling, collecting
#: every watchdog tick (15s) would repeatedly stall the audio/LLM event loop.
_GC_COOLDOWN_S = 300.0


class ProcessSupervisor:
    """Centralized process supervisor and health watchdog."""

    def __init__(
        self,
        max_cpu_rss_mb: float = 2500.0,
        max_gpu_vram_mb: float = 3800.0,
        health_check_interval_s: float = 15.0,
    ) -> None:
        """Initialize process supervisor.

        Args:
            max_cpu_rss_mb: Memory warning threshold in megabytes for host CPU.
            max_gpu_vram_mb: Memory warning threshold in megabytes for RTX 3050 VRAM.
            health_check_interval_s: Polling interval for background health monitoring.
        """
        self.max_cpu_rss_mb = max_cpu_rss_mb
        self.max_gpu_vram_mb = max_gpu_vram_mb
        self.health_interval = health_check_interval_s

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_callbacks: list[Callable[[], Any]] = []
        self._registered_processes: dict[str, int] = {}
        self._signal_handled: bool = False
        # Set instead of calling sys.exit() from a signal handler, so a running
        # asyncio loop can observe the flag and perform its own cleanup
        # (await wait_for_shutdown() / poll shutdown_requested).
        self._shutdown_event = asyncio.Event()
        self._last_gc_ts: float = 0.0

    @property
    def shutdown_event(self) -> asyncio.Event:
        """Event set once a termination signal has been received."""
        return self._shutdown_event

    @property
    def shutdown_requested(self) -> bool:
        """True when a SIGINT/SIGTERM has been handled and shutdown is requested."""
        return self._shutdown_event.is_set()

    async def wait_for_shutdown(self) -> None:
        """Awaitable shutdown gate for asyncio consumers (instead of hard process exit)."""
        await self._shutdown_event.wait()

    def register_process(self, name: str, pid: int) -> None:
        """Register a child daemon process ID for supervision.

        Args:
            name: Human-readable daemon identifier (e.g. 'llama-server', 'whisper-server').
            pid: Operating system Process ID.
        """
        self._registered_processes[name] = pid
        logger.info("process_registered_with_supervisor", name=name, pid=pid)

    def on_shutdown(self, callback: Callable[[], Any]) -> None:
        """Register a clean teardown hook to execute on termination signals.

        Args:
            callback: Callable hook executed prior to process exit.
        """
        self._shutdown_callbacks.append(callback)

    def start(self) -> None:
        """Install signal traps and start the background health watchdog."""
        if self._running:
            return

        self._running = True
        self._install_signal_handlers()

        self._thread = threading.Thread(
            target=self._watchdog_worker,
            daemon=True,
            name="SupervisorWatchdogWorker",
        )
        self._thread.start()
        logger.info("process_supervisor_started", interval_s=self.health_interval)

    def stop(self) -> None:
        """Stop supervisor thread and perform clean teardown."""
        if not self._running:
            return
        self._running = False
        self._execute_shutdown_hooks()
        logger.info("process_supervisor_stopped")

    def _install_signal_handlers(self) -> None:
        """Install POSIX signal handlers for SIGINT and SIGTERM."""
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, AttributeError):
            # Not in main thread or on non-POSIX system
            pass

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle incoming termination signal.

        Does NOT call sys.exit(): exiting from a signal handler skips asyncio
        cleanup (pending tasks, transports, embedded API server). Instead the
        supervisor stops its watchdog, runs teardown hooks, and sets the
        shutdown event so the running asyncio loop can shut down gracefully.
        """
        if self._signal_handled:
            return
        self._signal_handled = True
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info("termination_signal_received", signal=sig_name)
        self.stop()
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._shutdown_event.set)
        except RuntimeError:
            # No running loop in this thread/context: set directly (safe pre-loop).
            self._shutdown_event.set()

    def _execute_shutdown_hooks(self) -> None:
        """Execute registered teardown hooks and terminate child processes."""
        for hook in reversed(self._shutdown_callbacks):
            try:
                hook()
            except Exception as e:
                logger.warning("shutdown_hook_failed", error=str(e))

        # Terminate any registered child process trees
        for name, pid in self._registered_processes.items():
            try:
                p = psutil.Process(pid)
                p.terminate()
                p.wait(timeout=1.5)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

    def _watchdog_worker(self) -> None:
        """Continuously monitor process memory, zombie states, and daemon health."""
        while self._running:
            self._check_memory_usage()
            self._check_process_liveness()
            time.sleep(self.health_interval)

    def _check_memory_usage(self) -> None:
        """Inspect host CPU RSS and GPU VRAM usage against safety ceilings."""
        try:
            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)

            if rss_mb > self.max_cpu_rss_mb:
                logger.warning(
                    "high_cpu_memory_usage_detected",
                    current_mb=round(rss_mb, 1),
                    threshold_mb=self.max_cpu_rss_mb,
                )
                now = time.time()
                if now - self._last_gc_ts >= _GC_COOLDOWN_S:
                    self._last_gc_ts = now
                    import gc
                    gc.collect()
        except Exception as e:
            logger.debug("memory_check_failed", error=str(e))

        # Inspect NVIDIA GPU VRAM if available
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0:
                vram_mb = float(res.stdout.strip().split("\n")[0])
                if vram_mb > self.max_gpu_vram_mb:
                    logger.warning(
                        "high_gpu_vram_usage_detected",
                        current_mb=vram_mb,
                        threshold_mb=self.max_gpu_vram_mb,
                    )
        except Exception:
            pass

    def _check_process_liveness(self) -> None:
        """Ensure registered daemons are alive and not zombie processes."""
        for name, pid in list(self._registered_processes.items()):
            try:
                p = psutil.Process(pid)
                if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                    logger.warning("supervised_process_died_or_zombie", name=name, pid=pid)
            except psutil.NoSuchProcess:
                logger.warning("supervised_process_missing", name=name, pid=pid)
