"""File handles for the logs of externally spawned daemon processes.

llama-server and whisper-server write to their stdout/stderr for the whole
lifetime of a session. Appending that to a single file forever eventually fills
the disk, and the failure mode (server starts, then the filesystem is full) is
invisible until it is expensive. This module keeps one generation of history and
drops the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

from robot.telemetry import get_logger

logger = get_logger("daemon_log")

#: Rotate a daemon log once it grows past this many bytes.
MAX_DAEMON_LOG_BYTES = 5_000_000
#: Size kept for the previous generation, so the last crash is still diagnosable.
TRIMMED_LOG_BYTES = 1_000_000


def open_daemon_log(path: Path, max_bytes: int = MAX_DAEMON_LOG_BYTES) -> IO[Any]:
    """Open a daemon log for appending, rotating an oversized file first.

    Args:
        path: Log file path; its parent directory is created if missing.
        max_bytes: Rotation threshold for the previous session's log.

    Returns:
        An open text file handle positioned for appending.

    Raises:
        OSError: If the directory or file cannot be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if size > max_bytes:
        backup = path.with_suffix(path.suffix + ".1")
        try:
            keep = path.read_bytes()[-TRIMMED_LOG_BYTES:]
            backup.write_bytes(keep)
            path.write_bytes(b"")
            logger.info("daemon_log_rotated", path=str(path), previous_bytes=size)
        except OSError as exc:
            # A log that cannot rotate still must not block the daemon from starting.
            logger.warning("daemon_log_rotation_failed", path=str(path), error=str(exc))

    return open(path, "a", encoding="utf-8")
