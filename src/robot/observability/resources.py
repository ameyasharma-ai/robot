"""System resource monitoring for CPU, RAM, discrete NVIDIA GPU, and integrated AMD GPU."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("observability.resources")

#: Samples cost a psutil read plus a subprocess probe; dashboard polling would
#: otherwise stack one per request.
METRICS_TTL_S = 2.0


class SystemMetricsCollector:
    """Collect hardware metrics with a short cache and thread-safe refresh."""

    def __init__(self) -> None:
        self._last_metrics: dict[str, Any] = {}
        self._sampled_at: float = 0.0
        self._sample_lock = threading.Lock()

    def get_metrics(self, ttl_s: float = METRICS_TTL_S) -> dict[str, Any]:
        """Return a hardware sample, refreshing only when the cache is stale.

        The collection itself blocks on a subprocess probe, so async callers
        must run this through ``asyncio.to_thread``.

        Args:
            ttl_s: How long a previous sample stays valid.

        Returns:
            CPU, RAM and GPU metrics from the freshest sample.
        """
        with self._sample_lock:
            now = time.monotonic()
            if self._last_metrics and now - self._sampled_at < ttl_s:
                return dict(self._last_metrics)
            metrics = self._collect()
            self._last_metrics = metrics
            self._sampled_at = now
            return dict(metrics)

    def _collect(self) -> dict[str, Any]:
        """Collect current system metrics directly from OS and GPU interfaces."""
        metrics: dict[str, Any] = {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_percent": 0.0,
            "nvidia": None,
            "amd": None,
        }

        # 1. CPU & System RAM (via psutil)
        try:
            import psutil

            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            metrics["ram_used_gb"] = round((mem.total - mem.available) / (1024**3), 2)
            metrics["ram_total_gb"] = round(mem.total / (1024**3), 2)
            metrics["ram_percent"] = mem.percent
        except Exception as e:
            logger.debug("psutil_metrics_error", error=str(e))

        # 2. NVIDIA GPU (RTX 3050 via nvidia-smi)
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    metrics["nvidia"] = {
                        "device": "NVIDIA GeForce RTX 3050 Mobile",
                        "utilization_percent": float(parts[0]),
                        "vram_used_mb": float(parts[1]),
                        "vram_total_mb": float(parts[2]),
                        "temperature_c": float(parts[3]),
                    }
        except Exception:
            metrics["nvidia"] = None

        # 3. AMD Radeon 680M (Integrated GPU via sysfs)
        try:
            amd_busy = None
            # Check DRM sysfs device paths across all detected cards
            candidates = [
                "/sys/class/drm/card2/device/gpu_busy_percent",
                "/sys/class/drm/card1/device/gpu_busy_percent",
                "/sys/class/drm/card0/device/gpu_busy_percent",
                "/sys/class/drm/renderD129/device/gpu_busy_percent",
                "/sys/class/drm/renderD128/device/gpu_busy_percent",
            ]
            for path in candidates:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            val = f.read().strip()
                            amd_busy = float(val)
                            break
                    except Exception:
                        pass

            if amd_busy is not None:
                metrics["amd"] = {
                    "device": "AMD Radeon 680M (Vulkan / WebGL2)",
                    "utilization_percent": amd_busy,
                }
        except Exception:
            metrics["amd"] = None

        return metrics
