"""Llama.cpp advanced server settings and offloading configuration.

Provides comprehensive parameter management matching LM Studio and llama-server CLI flags,
including GPU layer offloading, FlashAttention, context and batch sizing, KV cache quantization,
CPU threading, memory mapping, RoPE scaling, and arbitrary custom CLI arguments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from robot.telemetry import get_logger

logger = get_logger("llama_settings")

SETTINGS_FILE = Path("data/llama_server_settings.json")


@dataclass
class LlamaServerSettings:
    """Configuration settings for the local llama-server process matching LM Studio controls."""

    # ─── GPU Offloading & Hardware ───────────────────────────────────────────
    n_gpu_layers: int = 99  # -ngl / --n-gpu-layers: number of layers offloaded to GPU
    main_gpu: int = 0  # --main-gpu: primary GPU device index
    split_mode: str = "none"  # --split-mode: 'none', 'layer', or 'row'
    tensor_split: str = ""  # -ts / --tensor-split: e.g. '0.5,0.5'
    flash_attn: bool = True  # -fa / --flash-attn: FlashAttention acceleration
    no_kv_offload: bool = False  # --no-kv-offload: keep KV cache on CPU RAM

    # ─── Context & Batch Sizing ──────────────────────────────────────────────
    ctx_size: int = 16384  # -c / --ctx-size: context window token capacity
    batch_size: int = 4096  # -b / --batch-size: logical prompt processing batch size
    ubatch_size: int = 4096  # -ub / --ubatch-size: physical micro-batch size
    parallel: int = 1  # -np / --parallel: concurrent execution slots
    cont_batching: bool = True  # -cb / --cont-batching: continuous batching

    # ─── KV Cache Quantization ───────────────────────────────────────────────
    cache_type_k: str = "q4_0"  # -ctk / --cache-type-k: 'f16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0'
    cache_type_v: str = "q4_0"  # -ctv / --cache-type-v: 'f16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0'

    # ─── CPU & Threading ─────────────────────────────────────────────────────
    threads: int = 8  # -t / --threads: token generation threads
    threads_batch: int = 8  # -tb / --threads-batch: prompt evaluation threads
    cpu_strict: bool = False  # --cpu-strict: strict thread pinning

    # ─── Memory & Optimization ───────────────────────────────────────────────
    no_mmap: bool = False  # --no-mmap: do not memory-map model file
    mlock: bool = False  # --mlock: lock model in RAM to prevent disk swapping
    numa: str = "none"  # --numa: 'none', 'distribute', 'isolate', 'numactl'

    # ─── RoPE & Context Scaling ──────────────────────────────────────────────
    rope_scaling: str = "none"  # --rope-scaling: 'none', 'linear', 'yarn'
    rope_scale: float | None = None  # --rope-scale: context extension scaling factor
    rope_freq_base: float | None = None  # --rope-freq-base: base frequency
    rope_freq_scale: float | None = None  # --rope-freq-scale: frequency scale

    # ─── Multimodal Projector ────────────────────────────────────────────────
    mmproj_path: str = ""  # --mmproj: path to CLIP/vision projector GGUF
    mmproj_offload: bool = True  # --mmproj-offload: offload vision projector to GPU

    # ─── Sampling & Reasoning Defaults ───────────────────────────────────────
    temp: float = 0.7  # --temp: temperature
    top_p: float = 0.9  # --top-p: nucleus sampling
    top_k: int = 40  # --top-k: top-k sampling
    min_p: float = 0.05  # --min-p: min-p sampling
    repeat_penalty: float = 1.1  # --repeat-penalty: repetition penalty
    reasoning_budget: int = -1  # --reasoning-budget: token budget for reasoning (-1 = unlimited)

    # ─── Custom CLI Arguments ────────────────────────────────────────────────
    custom_cli_args: str = ""  # Arbitrary flags passed directly to llama-server

    def to_dict(self) -> dict[str, Any]:
        """Convert settings instance to dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LlamaServerSettings:
        """Create settings instance from dictionary with type coercion.

        Args:
            data: Raw dictionary containing setting overrides.

        Returns:
            LlamaServerSettings: Coerced and validated settings instance.
        """
        valid_keys = set(cls.__annotations__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        # Type coercions
        if "n_gpu_layers" in filtered:
            filtered["n_gpu_layers"] = int(filtered["n_gpu_layers"])
        if "ctx_size" in filtered:
            filtered["ctx_size"] = int(filtered["ctx_size"])
        if "batch_size" in filtered:
            filtered["batch_size"] = int(filtered["batch_size"])
        if "ubatch_size" in filtered:
            filtered["ubatch_size"] = int(filtered["ubatch_size"])
        if "threads" in filtered:
            filtered["threads"] = int(filtered["threads"])
        if "threads_batch" in filtered:
            filtered["threads_batch"] = int(filtered["threads_batch"])
        if "temp" in filtered:
            filtered["temp"] = float(filtered["temp"])
        if "top_p" in filtered:
            filtered["top_p"] = float(filtered["top_p"])
        if "top_k" in filtered:
            filtered["top_k"] = int(filtered["top_k"])
        if "min_p" in filtered:
            filtered["min_p"] = float(filtered["min_p"])
        if "repeat_penalty" in filtered:
            filtered["repeat_penalty"] = float(filtered["repeat_penalty"])
        if "flash_attn" in filtered:
            filtered["flash_attn"] = bool(filtered["flash_attn"])
        if "no_mmap" in filtered:
            filtered["no_mmap"] = bool(filtered["no_mmap"])
        if "mlock" in filtered:
            filtered["mlock"] = bool(filtered["mlock"])

        return cls(**filtered)

    def save(self, filepath: Path | None = None) -> None:
        """Persist settings to JSON file.

        Args:
            filepath: Optional destination path. Defaults to data/llama_server_settings.json.
        """
        p = filepath or SETTINGS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("llama_settings_saved", path=str(p))
        except Exception as e:
            logger.error("llama_settings_save_failed", error=str(e))

    @classmethod
    def load(cls, filepath: Path | None = None) -> LlamaServerSettings:
        """Load settings from JSON file or fall back to defaults.

        Args:
            filepath: Optional source path. Defaults to data/llama_server_settings.json.

        Returns:
            LlamaServerSettings: Loaded or default settings instance.
        """
        p = filepath or SETTINGS_FILE
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                logger.warning("llama_settings_load_fallback", error=str(e))
        return cls()

    def build_cli_args(
        self,
        bin_path: str,
        model_path: str,
        port: int = 8080,
        host: str = "127.0.0.1",
    ) -> list[str]:
        """Construct full command line argument list for llama-server.

        Args:
            bin_path: Path to llama-server binary.
            model_path: Path to GGUF model file.
            port: Network port to bind.
            host: Host interface to bind.

        Returns:
            list[str]: Complete CLI argument list.
        """
        import shlex

        cmd: list[str] = [
            bin_path,
            "-m", model_path,
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-t", str(self.threads),
            "-tb", str(self.threads_batch),
            "-np", str(self.parallel),
            "--host", host,
            "--port", str(port),
        ]

        if self.flash_attn:
            cmd.extend(["-fa", "on"])
        else:
            cmd.extend(["-fa", "off"])

        if self.no_kv_offload:
            cmd.append("--no-kv-offload")

        if self.cache_type_k:
            cmd.extend(["-ctk", self.cache_type_k])
        if self.cache_type_v:
            cmd.extend(["-ctv", self.cache_type_v])

        if self.main_gpu != 0:
            cmd.extend(["--main-gpu", str(self.main_gpu)])

        if self.split_mode and self.split_mode != "none":
            cmd.extend(["--split-mode", self.split_mode])

        if self.tensor_split:
            cmd.extend(["-ts", self.tensor_split])

        if self.no_mmap:
            cmd.append("--no-mmap")

        if self.mlock:
            cmd.append("--mlock")

        if self.numa and self.numa != "none":
            cmd.extend(["--numa", self.numa])

        if self.cpu_strict:
            cmd.extend(["--cpu-strict", "1"])

        if self.cont_batching:
            cmd.append("-cb")

        # RoPE
        if self.rope_scaling and self.rope_scaling != "none":
            cmd.extend(["--rope-scaling", self.rope_scaling])
        if self.rope_scale is not None:
            cmd.extend(["--rope-scale", str(self.rope_scale)])
        if self.rope_freq_base is not None:
            cmd.extend(["--rope-freq-base", str(self.rope_freq_base)])
        if self.rope_freq_scale is not None:
            cmd.extend(["--rope-freq-scale", str(self.rope_freq_scale)])

        # Multimodal Vision Projector
        if self.mmproj_path and Path(self.mmproj_path).exists():
            cmd.extend(["--mmproj", str(self.mmproj_path)])
            if self.mmproj_offload:
                cmd.append("--mmproj-offload")

        # Sampling Defaults
        cmd.extend([
            "--temp", str(self.temp),
            "--top-p", str(self.top_p),
            "--top-k", str(self.top_k),
            "--min-p", str(self.min_p),
            "--repeat-penalty", str(self.repeat_penalty),
        ])

        if self.reasoning_budget != -1:
            cmd.extend(["--reasoning-budget", str(self.reasoning_budget)])

        # Arbitrary custom user arguments
        if self.custom_cli_args.strip():
            try:
                extra = shlex.split(self.custom_cli_args.strip())
                cmd.extend(extra)
            except Exception as e:
                logger.warning("failed_parsing_custom_cli_args", error=str(e))

        return cmd
