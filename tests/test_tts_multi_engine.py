"""Unit and integration test suite for multi-engine Text-to-Speech (Kokoro & Piper).

Verifies engine initialization, voice catalog resolution, dynamic prosody,
hot-swapping, state persistence, and REST API endpoints.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot.api import create_app
from robot.tts.kokoro import KokoroTTS
from robot.tts.manager import TTSManager
from robot.tts.prosody import ProsodyModulation


def test_kokoro_initialization_and_voices_catalog() -> None:
    """Verify KokoroTTS initializes properly and exposes comprehensive voice catalog."""
    model_path = Path("models/tts/kokoro/kokoro-v1.0.onnx")
    voices_path = Path("models/tts/kokoro/voices-v1.0.bin")

    if not model_path.exists() or not voices_path.exists():
        pytest.skip("Kokoro model files not found locally.")

    kokoro = KokoroTTS(str(model_path), str(voices_path), default_voice="af_heart")
    assert kokoro.SAMPLE_RATE == 24000
    assert kokoro.active_voice == "af_heart"

    voices = kokoro.list_voices()
    assert len(voices) >= 50

    voice_ids = {v["id"] for v in voices}
    assert "af_heart" in voice_ids
    assert "af_bella" in voice_ids
    assert "am_adam" in voice_ids
    assert "bf_emma" in voice_ids
    assert "bm_george" in voice_ids

    # Verify voice metadata structure
    heart_meta = next(v for v in voices if v["id"] == "af_heart")
    assert heart_meta["gender"] == "female"
    assert heart_meta["accent"] == "American"
    assert heart_meta["lang"] == "en-us"
    assert heart_meta["is_active"] is True


def test_kokoro_voice_resolution_and_fallback() -> None:
    """Verify voice identifier resolution, prefix matching, and graceful fallbacks."""
    model_path = Path("models/tts/kokoro/kokoro-v1.0.onnx")
    voices_path = Path("models/tts/kokoro/voices-v1.0.bin")

    if not model_path.exists() or not voices_path.exists():
        pytest.skip("Kokoro model files not found locally.")

    kokoro = KokoroTTS(str(model_path), str(voices_path), default_voice="af_heart")

    # Direct match
    assert kokoro._resolve_voice("am_adam") == "am_adam"

    # Suffix match
    assert kokoro._resolve_voice("bella") == "af_bella"

    # Legacy Piper model mapping to Kokoro counterpart
    assert kokoro._resolve_voice("en-us-lessac-medium.onnx") == "am_adam"
    assert kokoro._resolve_voice("en-us-amy-medium.onnx") == "af_heart"


@pytest.mark.asyncio
async def test_kokoro_synthesis_and_prosody() -> None:
    """Verify speech synthesis and prosody modulation producing 24kHz float32 audio."""
    model_path = Path("models/tts/kokoro/kokoro-v1.0.onnx")
    voices_path = Path("models/tts/kokoro/voices-v1.0.bin")

    if not model_path.exists() or not voices_path.exists():
        pytest.skip("Kokoro model files not found locally.")

    kokoro = KokoroTTS(str(model_path), str(voices_path), default_voice="af_heart")

    # Empty text returns empty audio
    empty_audio, sr = await kokoro.synthesize("")
    assert len(empty_audio) == 0
    assert sr == 24000

    # Short phrase synthesis
    audio, sr = await kokoro.synthesize("Hello from Kokoro neural TTS test.", speed=1.0)
    assert len(audio) > 1000
    assert sr == 24000
    assert audio.dtype == np.float32

    # Synthesize with prosody
    audio2, sr2, mod = await kokoro.synthesize_with_prosody("I am so excited to see you today!", personality_name="luna")
    assert len(audio2) > 1000
    assert sr2 == 24000
    assert isinstance(mod, ProsodyModulation)


@pytest.mark.asyncio
async def test_piper_multi_voice_support() -> None:
    """Verify PiperTTS loads multiple installed voice checkpoints."""
    from robot.tts import PiperTTS
    tts_dir = Path("models/tts")
    lessac_path = tts_dir / "en-us-lessac-medium.onnx"
    ryan_path = tts_dir / "en-us-ryan-medium.onnx"

    if not lessac_path.exists():
        pytest.skip("Piper lessac voice not found.")

    piper = PiperTTS(str(lessac_path))
    assert await piper.health_check()

    # Switch to Ryan voice if installed
    if ryan_path.exists():
        piper.set_active_voice(str(ryan_path))
        audio, sr = piper.warm_phrase("Testing Piper Ryan voice synthesis.")
        assert len(audio) > 0
        assert sr in (16000, 22050)


def test_tts_manager_switching_and_persistence(tmp_path: Path) -> None:
    """Verify TTSManager hot-swapping between Kokoro and Piper and persisting state."""
    active_engine_file = tmp_path / "active_tts_engine.txt"
    active_voice_file = tmp_path / "active_tts_voice.txt"

    with patch("robot.tts.manager.ACTIVE_ENGINE_FILE", active_engine_file), \
         patch("robot.tts.manager.ACTIVE_VOICE_FILE", active_voice_file), \
         patch("robot.tts.manager.DATA_DIR", tmp_path):

        mgr = TTSManager({"tts": {"provider": "kokoro"}})
        assert mgr.get_active_engine_name() == "kokoro"

        # List engines
        engines = mgr.list_engines()
        assert len(engines) == 2
        engine_ids = [e["id"] for e in engines]
        assert "kokoro" in engine_ids
        assert "piper" in engine_ids

        # Switch to Piper
        res = mgr.switch_engine("piper", voice="en-us-lessac-medium")
        assert res["status"] == "switched"
        assert res["engine"] == "piper"
        assert mgr.get_active_engine_name() == "piper"
        assert active_engine_file.read_text().strip() == "piper"

        # Switch back to Kokoro
        res2 = mgr.switch_engine("kokoro", voice="af_heart")
        assert res2["status"] == "switched"
        assert res2["engine"] == "kokoro"
        assert mgr.get_active_engine_name() == "kokoro"
        assert active_engine_file.read_text().strip() == "kokoro"

    # State lives only in the runtime files; the hand-edited YAML is untouched.
    assert not (tmp_path / "default.yaml").exists()


def test_tts_switch_never_rewrites_system_config(tmp_path: Path) -> None:
    """Verify engine switching leaves config/default.yaml byte-for-byte unchanged.

    A load/dump cycle silently destroys the comments and layout of the main
    hand-maintained config file, so the manager must persist to ``data/`` only.
    """
    config_file = tmp_path / "default.yaml"
    original = "# Speaking voice — pick kokoro or piper\ntts:\n  provider: kokoro  # keep me\n"
    config_file.write_text(original)

    with patch("robot.tts.manager.DATA_DIR", tmp_path / "data"), \
         patch("robot.tts.manager.ACTIVE_ENGINE_FILE", tmp_path / "data" / "active_tts_engine.txt"), \
         patch("robot.tts.manager.ACTIVE_VOICE_FILE", tmp_path / "data" / "active_tts_voice.txt"):
        mgr = TTSManager({"tts": {"provider": "kokoro"}})
        mgr.switch_engine("piper", voice="en-us-lessac-medium")

    assert config_file.read_text() == original
    assert (tmp_path / "data" / "active_tts_engine.txt").read_text().strip() == "piper"


@pytest.mark.asyncio
async def test_tts_api_endpoints() -> None:
    """Verify GET /api/tts/status, GET /api/tts/voices, POST /api/tts/switch, and POST /api/tts/test."""
    mock_engine = MagicMock()
    mock_engine.state.state.active_personality = "neo"
    mock_engine.events.emit = AsyncMock()
    mock_engine.config = {"tts": {"provider": "kokoro"}}

    mgr = TTSManager({"tts": {"provider": "kokoro"}})
    mgr.switch_engine("kokoro", voice="af_heart")
    mock_engine.tts = mgr

    app = create_app(mock_engine)
    client = TestClient(app)

    # 1. GET /api/tts/status
    res = client.get("/api/tts/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["active_engine"] == "kokoro"
    assert len(data["engines"]) >= 2

    # 2. GET /api/tts/voices
    res_voices = client.get("/api/tts/voices")
    assert res_voices.status_code == 200
    voices_data = res_voices.json()
    assert voices_data["status"] == "ok"
    assert len(voices_data["voices"]) >= 50

    # 3. POST /api/tts/switch
    res_switch = client.post("/api/tts/switch", json={"engine": "piper", "voice": "en-us-lessac-medium"})
    assert res_switch.status_code == 200
    switch_data = res_switch.json()
    assert switch_data["status"] == "ok"
    assert switch_data["engine"] == "piper"
    mock_engine.events.emit.assert_called()

    # 4. POST /api/tts/test
    res_test = client.post("/api/tts/test", json={"engine": "piper", "voice": "en-us-lessac-medium", "text": "Testing preview"})
    assert res_test.status_code == 200
    assert res_test.headers["content-type"] == "audio/wav"
    assert len(res_test.content) > 100

    # Verify generated WAV header
    with wave.open(io.BytesIO(res_test.content), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() in (16000, 22050, 24000)
        assert wf.getnframes() > 100


def test_piper_resolve_model_path_alien_voice_fallback() -> None:
    """Verify PiperTTS._resolve_model_path falls back cleanly to default model on alien voices."""
    from robot.tts import PiperTTS
    with patch.object(PiperTTS, "_load_voice", return_value=MagicMock()):
        piper = PiperTTS("models/tts/en-us-lessac-medium.onnx")
        piper._default_model = "models/tts/en-us-lessac-medium.onnx"

        # Calling with Kokoro voice 'af_heart' must fall back to default model
        resolved = piper._resolve_model_path("af_heart")
        assert resolved == "models/tts/en-us-lessac-medium.onnx"


@pytest.mark.asyncio
async def test_tts_manager_cross_engine_routing() -> None:
    """Verify TTSManager.synthesize auto-routes Kokoro voices to Kokoro even when Piper is active."""
    mgr = TTSManager({"tts": {"provider": "piper"}})
    mock_piper = MagicMock()
    mock_piper.synthesize = AsyncMock(return_value=(np.zeros(160, dtype=np.float32), 22050))
    mock_kokoro = MagicMock()
    mock_kokoro.synthesize = AsyncMock(return_value=(np.zeros(240, dtype=np.float32), 24000))

    mgr._engines["piper"] = mock_piper
    mgr._engines["kokoro"] = mock_kokoro
    mgr._active_engine_name = "piper"

    # Synthesize with Kokoro voice af_heart
    samples, sr = await mgr.synthesize("Hello world", voice_model="af_heart")
    assert sr == 24000
    mock_kokoro.synthesize.assert_awaited_once()
    mock_piper.synthesize.assert_not_called()
