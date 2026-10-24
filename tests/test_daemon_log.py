"""Tests for bounded daemon log rotation."""

from robot.daemon_log import MAX_DAEMON_LOG_BYTES, open_daemon_log


def test_open_daemon_log_rotates_an_oversized_file(tmp_path):
    path = tmp_path / "llama-server.log"
    path.write_bytes(b"x" * (MAX_DAEMON_LOG_BYTES + 1024))

    handle = open_daemon_log(path)
    try:
        handle.write("new session\n")
    finally:
        handle.close()

    assert path.stat().st_size < MAX_DAEMON_LOG_BYTES
    backup = tmp_path / "llama-server.log.1"
    assert backup.exists()
    assert backup.stat().st_size <= 1_000_000
    assert "new session" in path.read_text()


def test_open_daemon_log_keeps_a_small_file(tmp_path):
    path = tmp_path / "whisper-server.log"
    path.write_text("previous run\n")

    handle = open_daemon_log(path)
    handle.close()

    assert (tmp_path / "whisper-server.log.1").exists() is False
    assert path.read_text() == "previous run\n"


def test_open_daemon_log_creates_missing_directory(tmp_path):
    path = tmp_path / "nested" / "deep" / "server.log"

    handle = open_daemon_log(path)
    handle.close()

    assert path.exists()
