"""Llama.cpp advanced server settings and offloading configuration.

Provides comprehensive parameter management matching LM Studio and llama-server CLI flags,
including GPU layer offloading, FlashAttention, context and batch sizing, KV cache quantization,
CPU threading, memory mapping, RoPE scaling, and arbitrary custom CLI arguments.

All user-supplied values are coerced with explicit typing and hard bounds so a malformed
settings payload can neither crash the server with an out-of-range value (e.g. a context
size that exhausts the KV cache) nor smuggle a listener-affecting flag through
``custom_cli_args``.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from robot.telemetry import get_logger

logger = get_logger("llama_settings")

#: Legacy relative location of the settings file (kept for import compatibility).
SETTINGS_FILE = Path("data/llama_server_settings.json")

#: Serializes read-modify-write cycles on the shared settings file.
_SETTINGS_FILE_LOCK = threading.Lock()

#: Upper/lower bounds for numeric settings. Values outside are rejected rather than clamped
#: so an operator notices the mistake instead of silently running an unsafe configuration.
INT_BOUNDS: dict[str, tuple[int, int]] = {
    "n_gpu_layers": (0, 999),
    "main_gpu": (0, 64),
    "ctx_size": (512, 131072),
    "batch_size": (1, 4096),
    "ubatch_size": (1, 4096),
    "parallel": (1, 64),
    "threads": (1, 256),
    "threads_batch": (1, 256),
    "top_k": (0, 1000),
    "reasoning_budget": (-1, 1_000_000),
}

FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "temp": (0.0, 5.0),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
    "repeat_penalty": (0.0, 10.0),
    "rope_scale": (0.0, 1000.0),
    "rope_freq_base": (0.0, 1e9),
    "rope_freq_scale": (0.0, 1000.0),
}

BOOL_FIELDS: tuple[str, ...] = (
    "flash_attn",
    "no_kv_offload",
    "cont_batching",
    "cpu_strict",
    "no_mmap",
    "mlock",
    "mmproj_offload",
)

#: CLI flags a custom argument list may never override: they control the listening
#: interface, authentication, port, or model files and could expose the unauthenticated
#: inference server on a public interface.
RESERVED_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--host",
        "--api-key",
        "--port",
        "--model",
        "-m",
        "--mmproj",
    }
)

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on", "enabled"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off", "disabled", ""})


def _coerce_bool(field_name: str, value: Any) -> bool:
    """Parse a boolean setting without Python's truthiness trap.

    ``bool("false")`` evaluates to ``True``; JSON/YAML clients frequently send the
    string form, so only an explicit allow-list of spellings is accepted.

    Args:
        field_name: Setting name (used in the error message).
        value: Raw value from JSON, YAML, or an API payload.

    Returns:
        bool: Parsed boolean value.

    Raises:
        ValueError: If the value is not a recognised boolean spelling.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"Setting '{field_name}' must be a boolean (true/false).")


def _coerce_int(field_name: str, value: Any) -> int:
    """Coerce and bound-check an integer setting.

    Args:
        field_name: Setting name.
        value: Raw value.

    Returns:
        int: Validated integer.

    Raises:
        ValueError: If the value is non-numeric, boolean, or out of bounds.
    """
    if isinstance(value, bool):
        raise ValueError(f"Setting '{field_name}' must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Setting '{field_name}' must be an integer.") from None
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"Setting '{field_name}' must be an integer.")
    low, high = INT_BOUNDS[field_name]
    if not low <= parsed <= high:
        raise ValueError(f"Setting '{field_name}' must be between {low} and {high}.")
    return parsed


def _coerce_float(field_name: str, value: Any) -> float:
    """Coerce and bound-check a floating point setting.

    Args:
        field_name: Setting name.
        value: Raw value.

    Returns:
        float: Validated value.

    Raises:
        ValueError: If the value is non-numeric or out of bounds.
    """
    if isinstance(value, bool):
        raise ValueError(f"Setting '{field_name}' must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Setting '{field_name}' must be a number.") from None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"Setting '{field_name}' must be a finite number.")
    low, high = FLOAT_BOUNDS[field_name]
    if not low <= parsed <= high:
        raise ValueError(f"Setting '{field_name}' must be between {low} and {high}.")
    return parsed


def validate_custom_cli_args(raw: str) -> list[str]:
    """Split and validate arbitrary custom llama-server arguments.

    Args:
        raw: Shell-like argument string supplied by the user.

    Returns:
        list[str]: Parsed argument tokens.

    Raises:
        ValueError: If quoting is invalid or a reserved flag is present.
    """
    if not raw or not str(raw).strip():
        return []
    if "\x00" in str(raw):
        raise ValueError("Custom CLI arguments may not contain NUL bytes.")
    try:
        tokens = shlex.split(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"Could not parse custom CLI arguments: {exc}") from None
    for token in tokens:
        flag = token.split("=", 1)[0].strip()
        if flag in RESERVED_CLI_FLAGS:
            raise ValueError(f"Custom CLI arguments may not override the reserved flag '{flag}'.")
    return tokens


def _default_settings_path() -> Path:
    """Return the settings file anchored to the project root instead of the CWD.

    Returns:
        Path: Absolute settings file path.
    """
    try:
        from robot.config import get_project_root

        return get_project_root() / SETTINGS_FILE
    except Exception:  # pragma: no cover - config import is always available in production
        return Path(SETTINGS_FILE).resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a temp file, fsync, and atomic replace.

    Args:
        path: Destination file.
        payload: JSON-serialisable mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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
    batch_size: int = 2048  # -b / --batch-size: logical prompt processing batch size
    ubatch_size: int = 1024  # -ub / --ubatch-size: physical micro-batch size (1024 prevents RTX 3050 CUDA OOM in cudaGraphInstantiate)
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
        """Create settings instance from dictionary with typed coercion and bounds checks.

        Args:
            data: Raw dictionary containing setting overrides.

        Returns:
            LlamaServerSettings: Coerced and validated settings instance.

        Raises:
            ValueError: If a supplied value cannot be coerced or violates its bounds.
        """
        if not isinstance(data, dict):
            raise ValueError("Settings payload must be an object.")

        valid_keys = set(cls.__annotations__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        # An explicit JSON null means "no override": drop the key so the
        # dataclass default applies instead of persisting None into the command.
        for name in list(INT_BOUNDS) + list(FLOAT_BOUNDS):
            if filtered.get(name) is None:
                filtered.pop(name, None)
            else:
                filtered[name] = _coerce_int(name, filtered[name]) if name in INT_BOUNDS else _coerce_float(name, filtered[name])
        for name in BOOL_FIELDS:
            if name in filtered:
                filtered[name] = _coerce_bool(name, filtered[name])

        if "custom_cli_args" in filtered:
            raw = filtered["custom_cli_args"]
            if raw is None:
                filtered["custom_cli_args"] = ""
            elif not isinstance(raw, str):
                raise ValueError("Setting 'custom_cli_args' must be a string.")
            else:
                # shlex.join (not " ".join) so a quoted value containing spaces
                # survives the round-trip through validate/build.
                filtered["custom_cli_args"] = shlex.join(validate_custom_cli_args(raw))

        return cls(**filtered)

    def save(self, filepath: Path | None = None) -> None:
        """Persist settings atomically to the JSON settings file.

        The write is guarded by a module lock and goes through a temp-file + fsync +
        os.replace cycle so a crash never leaves a truncated settings file behind.

        Args:
            filepath: Optional destination path. Defaults to <project>/data/llama_server_settings.json.
        """
        p = Path(filepath) if filepath else _default_settings_path()
        try:
            with _SETTINGS_FILE_LOCK:
                _atomic_write_json(p, self.to_dict())
            logger.info("llama_settings_saved", path=str(p))
        except Exception as e:
            logger.error("llama_settings_save_failed", error=str(e))

    @classmethod
    def load(cls, filepath: Path | None = None) -> LlamaServerSettings:
        """Load settings from JSON file or fall back to defaults.

        Args:
            filepath: Optional source path. Defaults to <project>/data/llama_server_settings.json.

        Returns:
            LlamaServerSettings: Loaded or default settings instance.
        """
        p = Path(filepath) if filepath else _default_settings_path()
        if p.exists():
            try:
                with _SETTINGS_FILE_LOCK:
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

        Raises:
            ValueError: If the custom argument string contains a reserved flag.
        """
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

        # Arbitrary custom user arguments (reserved flags rejected).
        extra = validate_custom_cli_args(self.custom_cli_args)
        cmd.extend(extra)

        return cmd

