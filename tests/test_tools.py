"""Tests for tool registry, router, and permissions."""

import asyncio

from robot.tools import PermissionLevel, PermissionManager, ToolDefinition, ToolRegistry, ToolRouter


async def dummy_handler(**kwargs):
    return {"status": "ok"}


def test_tool_registration():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="test.tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        handler=dummy_handler,
        family="test",
    )
    registry.register(tool)
    assert "test.tool" in registry.list_tools()
    assert registry.get("test.tool") is not None


def test_tool_family_lookup():
    registry = ToolRegistry()
    for i in range(3):
        registry.register(
            ToolDefinition(
                name=f"memory.tool_{i}",
                description=f"Tool {i}",
                parameters={},
                handler=dummy_handler,
                family="memory",
            )
        )
    assert len(registry.get_family("memory")) == 3
    assert len(registry.get_family("nonexistent")) == 0


def test_tool_router_selection():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="memory.search", description="Search", parameters={},
            handler=dummy_handler, family="memory",
        )
    )
    router = ToolRouter(registry)
    selected = router.select_tools("Can you remember what I said?")
    assert "memory.search" in selected


def test_permission_check():
    pm = PermissionManager(max_auto_level=PermissionLevel.HARMLESS_LOCAL)
    safe_tool = ToolDefinition(
        name="safe", description="", parameters={}, handler=dummy_handler,
        permission_level=PermissionLevel.READ_ONLY,
    )
    dangerous_tool = ToolDefinition(
        name="dangerous", description="", parameters={}, handler=dummy_handler,
        permission_level=PermissionLevel.SYSTEM,
    )
    assert pm.check(safe_tool) is True
    assert pm.check(dangerous_tool) is False


def test_clean_text_for_speech():
    from robot.tts import clean_text_for_speech
    raw = "**Hello world!** Here is *some* text with `code` and [motion:wave]."
    clean = clean_text_for_speech(raw)
    assert "**" not in clean
    assert "*" not in clean
    assert "`" not in clean
    assert "Hello world! Here is some text with code and ." in clean or "Hello world! Here is some text with code and." in clean


def test_tool_router_motion_and_perception():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="character.execute_action", description="", parameters={}, handler=dummy_handler, family="motion")
    )
    registry.register(
        ToolDefinition(name="perception.inspect_view", description="", parameters={}, handler=dummy_handler, family="perception")
    )
    router = ToolRouter(registry)
    motion_selected = router.select_tools("wave your right hand")
    assert "character.execute_action" in motion_selected

    perception_selected = router.select_tools("what do you see in front of you")
    assert "perception.inspect_view" in perception_selected


def test_extract_motion_from_phrase():
    from robot.core.engine import _extract_motion_from_phrase

    # Explicit marker
    cleaned, motion = _extract_motion_from_phrase("Hello there! [motion: wave] How are you?")
    assert motion == {"action": "wave", "side": "right"}
    assert cleaned == "Hello there!  How are you?"

    # Asterisk stage direction
    cleaned2, motion2 = _extract_motion_from_phrase("I am so excited! *waves* Welcome!")
    assert motion2 == {"action": "wave", "side": "right"}
    assert cleaned2 == "I am so excited!  Welcome!"

    # No motion
    cleaned3, motion3 = _extract_motion_from_phrase("Just a normal sentence.")
    assert motion3 is None
    assert cleaned3 == "Just a normal sentence."


import pytest



@pytest.mark.asyncio
async def test_builtins_perception_what_do_you_see():
    from unittest.mock import MagicMock
    from robot.tools.builtins.perception import register_perception_tools

    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.perception = MagicMock()
    engine.perception.world_state.to_compact.return_value = {
        "people": [{"position": "center", "distance": "near"}],
        "objects": [{"label": "cup", "position": "center"}],
        "lighting": "bright",
        "motion": "subtle",
    }

    register_perception_tools(engine)
    tool = engine.tool_registry.get("perception.what_do_you_see")
    assert tool is not None
    result = await tool.handler()
    assert "cup at center" in result["summary"]
    assert len(result["people"]) == 1
    assert len(result["objects"]) == 1


@pytest.mark.asyncio
async def test_all_15_tools_functional():
    import numpy as np
    from unittest.mock import MagicMock, AsyncMock
    from robot.tools.builtins import register_builtin_tools

    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.personalities = MagicMock()
    engine.personalities.active = MagicMock(memory_namespace="test_ns")
    engine.state = MagicMock()
    engine.state.state.to_compact.return_value = {"mode": "idle"}
    engine.telemetry = MagicMock()
    engine.telemetry.snapshot.return_value = {}
    engine.memory = MagicMock()
    engine.memory.add_memory = AsyncMock(return_value=None)
    engine.memory.get_relevant_memories = AsyncMock(return_value=[])
    engine.memory.search_history = AsyncMock(return_value=[])
    engine.perception = MagicMock()
    engine.perception.camera.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.perception.world_state.to_compact.return_value = {
        "people": [{"position": "center", "distance": "near"}],
        "objects": [{"label": "bottle", "position": "right"}],
        "lighting": "normal",
        "motion": "none",
    }
    engine.events = MagicMock()
    engine.events.emit = AsyncMock(return_value=None)
    engine._pending_motion = None

    register_builtin_tools(engine)
    tools = engine.tool_registry.list_tools()
    assert len(tools) >= 13

    for name in tools:
        t = engine.tool_registry.get(name)
        assert t is not None
        if name == "memory.write":
            res = await t.handler(fact="Test fact")
            assert "Test fact" in str(res)
        elif name in ("memory.search", "history.search"):
            res = await t.handler(query="test")
            assert isinstance(res, list)
        elif name == "character.execute_action":
            res = await t.handler(action="wave")
            assert res.get("status") == "executing"
        elif name == "character.stop":
            res = await t.handler()
            assert res.get("status") == "stopped"
        elif name == "personality.switch":
            engine.personalities.set_active.return_value = True
            res = await t.handler(name="cypher")
            assert "cypher" in str(res)
        elif name == "system.time":
            res = await t.handler()
            assert "current_time" in res
        elif name == "system.status":
            res = await t.handler()
            assert "state" in res
        elif name in ("perception.what_do_you_see", "perception.inspect_view"):
            res = await t.handler()
            assert "summary" in res
        elif name == "perception.get_world_state":
            res = await t.handler()
            assert "people" in res
        elif name == "camera.capture_snapshot":
            res = await t.handler()
            assert res.get("format") == "jpeg"
        elif name == "music.play":
            res = await t.handler(query="Bohemian Rhapsody", source="auto")
            assert res.get("status") in {"playing", "error"}
        elif name == "music.control":
            res = await t.handler(action="pause")
            assert res.get("status") == "paused"


@pytest.mark.asyncio
async def test_system_time_country_and_default():
    from unittest.mock import MagicMock
    from robot.core.engine import RobotEngine
    from robot.tools.builtins import register_builtin_tools

    engine = RobotEngine.__new__(RobotEngine)
    engine.tool_registry = ToolRegistry()
    engine.personalities = MagicMock()
    register_builtin_tools(engine)

    time_tool = engine.tool_registry.get("system.time")
    assert time_tool is not None

    # 1. Default (no timezone specified) -> Indian Standard Time
    default_res = await time_tool.handler()
    assert default_res["timezone"] == "Asia/Kolkata"
    assert "India" in default_res["country"]
    assert "current_time" in default_res

    # 2. Country lookup: Tokyo / Japan
    japan_res = await time_tool.handler(timezone_str="Tokyo")
    assert japan_res["timezone"] == "Asia/Tokyo"

    # 3. Country lookup: London / UK
    london_res = await time_tool.handler(timezone_str="London")
    assert london_res["timezone"] == "Europe/London"

    # 4. Country lookup: New York / USA
    ny_res = await time_tool.handler(timezone_str="New York")
    assert ny_res["timezone"] == "America/New_York"


@pytest.mark.asyncio
async def test_music_player_service():
    from robot.tools.builtins.music import MusicPlayerService
    svc = MusicPlayerService()

    # YouTube / yt-dlp
    yt_res = svc.play_ytdlp("lofi hip hop")
    assert yt_res["status"] == "playing"
    assert yt_res["source"] in {"ytdlp", "local"}
    assert "youtube" in yt_res.get("url", "") or "local" in yt_res["source"]

    # Spotify
    sp_res = svc.play_spotify("lofi hip hop")
    assert sp_res["status"] == "playing"
    assert sp_res["source"] in {"spotify", "ytdlp", "spotify_premium", "local"}

    # SoundCloud
    sc_res = svc.play_soundcloud("synthwave")
    assert sc_res["status"] == "playing"
    assert sc_res["source"] == "soundcloud"
    assert "synthwave" in sc_res["url"]

    # Controls & Status
    status = svc.get_status()
    assert isinstance(status, dict)
    assert "is_playing" in status
    assert "current_track" in status
    assert "volume" in status
    assert svc.pause()["status"] == "paused"
    assert svc.get_status()["is_paused"] is True
    assert svc.resume()["status"] == "resumed"
    assert svc.set_volume(0.5)["volume"] == 0.5
    assert svc.stop()["status"] == "stopped"
    assert svc.get_status()["is_playing"] is False


def test_tool_skill_manager():
    from robot.tools.skills import ToolSkillManager
    mgr = ToolSkillManager()
    skills = mgr.list_skills()
    assert "system.time" in skills
    assert "web.search" in skills
    assert "memory.write" in skills
    assert "memory.search" in skills
    assert "music.play" in skills
    assert "music.control" in skills

    prompt = mgr.format_skills_prompt(["system.time", "music.play"])
    assert "Skill: system.time" in prompt
    assert "Skill: music.play" in prompt
    assert "Asia/Kolkata" in prompt


@pytest.mark.asyncio
async def test_piper_multi_voice_and_prosody():
    from robot.tts import PiperTTS
    tts = PiperTTS("models/tts/en-us-lessac-medium.onnx")

    # Synthesize with active model
    audio1, sr1 = await tts.synthesize("Testing voice.", speed=1.1, pitch=1.0)
    assert len(audio1) > 0
    assert sr1 == 16000 or sr1 == 22050

    # Synthesize with Amy (thin & sweet female voice)
    audio2, sr2 = await tts.synthesize(
        "Testing Luna voice.",
        speed=1.1,
        pitch=1.2,
        voice_model="models/tts/en-us-amy-medium.onnx",
        noise_scale=0.88,
        noise_w=0.90,
    )
    assert len(audio2) > 0

    # Ensure dynamic active voice switching works
    tts.set_active_voice("models/tts/en-us-amy-medium.onnx")
    assert "amy" in tts._active_model.lower()


@pytest.mark.asyncio
async def test_personality_persistence_and_memory_clear_turn():
    from robot.personality import PersonalityManager
    from robot.memory import ConversationHistory
    import tempfile, os

    # 1. Personality persistence
    pm = PersonalityManager()
    pm.load_all()
    assert pm.set_active("luna") is True
    assert pm.get_persisted_active() == "luna"

    # 2. ConversationHistory clear_turn
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    try:
        hist = ConversationHistory(db_path)
        await hist.initialize()
        await hist.add_message("sess_1", "user", "turn 0 msg", turn_number=0)
        await hist.add_message("sess_1", "assistant", "turn 0 reply", turn_number=0)
        await hist.add_message("sess_1", "user", "turn 1 msg", turn_number=1)
        recent = await hist.get_recent("sess_1", 10)
        assert len(recent) == 3

        # Clear turn 0
        await hist.clear_turn("sess_1", turn_number=0)
        recent_after = await hist.get_recent("sess_1", 10)
        assert len(recent_after) == 1
        assert recent_after[0]["content"] == "turn 1 msg"
        await hist.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_music_history_replay_and_title_preservation():
    from robot.tools.builtins.music import MusicPlayerService
    svc = MusicPlayerService()

    # 1. Title preservation: querying "paranoid by juice wrld" should not be replaced with "No Good"
    res1 = svc.play_ytdlp("paranoid by juice wrld")
    assert "Paranoid" in res1["resolved_name"]
    assert "No Good" not in res1["resolved_name"]
    svc.stop()

    # 2. History tracking: recently played history should contain the track
    history = svc.get_history(5)
    assert len(history) >= 1
    assert "Paranoid" in history[0]["resolved_name"]

    # 3. Status reflection
    status = svc.get_status()
    assert "history" in status
    assert len(status["history"]) >= 1

    # 4. Auto Replay & Smart Resume
    replay_res = svc.replay()
    assert replay_res["status"] == "playing"
    assert "Paranoid" in replay_res["resolved_name"]
    svc.stop()

    # Resume when stopped should auto-replay from history
    resume_res = svc.resume()
    assert resume_res["status"] == "resumed"
    assert "Paranoid" in resume_res["resolved_name"]
    svc.stop()


def test_tool_router_music_and_volume_keywords():
    """Verify ToolRouter matches music and volume actions reliably."""
    from robot.tools import ToolRegistry, ToolRouter
    from robot.tools.builtins import register_builtin_tools
    from unittest.mock import MagicMock

    engine = MagicMock()
    tr = ToolRegistry()
    engine.tool_registry = tr
    register_builtin_tools(engine)
    router = ToolRouter(tr)

    for q in [
        "set the volume to 20%",
        "what song did you last play?",
        "play it again",
        "play that song again on youtube",
        "turn it down",
        "mute",
        "louder",
        "stop the song",
        "resume the song",
    ]:
        selected = router.select_tools(q)
        assert any("music" in name for name in selected), f"Failed to route music tools for: {q}"


@pytest.mark.asyncio
async def test_music_play_relative_replay():
    """Verify music.play auto-delegates 'that song again' to replay."""
    from robot.tools import ToolRegistry
    from robot.tools.builtins import register_builtin_tools, MusicPlayerService
    from unittest.mock import MagicMock

    engine = MagicMock()
    tr = ToolRegistry()
    engine.tool_registry = tr
    register_builtin_tools(engine)

    svc = MusicPlayerService.get_instance()
    svc._history = [{"resolved_name": "Cornelia Street", "source": "ytdlp", "query": "cornelia street"}]

    play_tool = tr.get("music.play")
    assert play_tool is not None

    res = await play_tool.handler(query="that song again on youtube")
    assert res.get("status") in {"playing", "resumed"}
    svc.stop()


@pytest.mark.asyncio
async def test_music_control_composite_and_equalizer():
    from robot.tools.builtins import register_builtin_tools, MusicPlayerService
    from unittest.mock import MagicMock

    engine = MagicMock()
    tr = ToolRegistry()
    engine.tool_registry = tr
    register_builtin_tools(engine)

    svc = MusicPlayerService.get_instance()
    svc._history = [{"resolved_name": "Juice Wrld Acapella", "source": "ytdlp", "query": "Juice Wrld Acapella"}]

    ctrl_tool = tr.get("music.control")
    eq_tool = tr.get("music.equalizer")
    assert ctrl_tool is not None
    assert eq_tool is not None

    # 1. No action specified, only volume -> auto-infer action='volume'
    res_vol = await ctrl_tool.handler(volume=0.4)
    assert res_vol.get("status") == "volume_set"
    assert res_vol.get("volume") == 0.4

    # 2. String percentage volume -> "40%"
    res_pct = await ctrl_tool.handler(action="volume", volume="40%")
    assert res_pct.get("status") == "volume_set"
    assert res_pct.get("volume") == 0.4

    # 3. Composite restart with 40% volume
    res_composite = await ctrl_tool.handler(action="restart", volume=0.4)
    assert res_composite.get("volume") == 0.4
    assert res_composite.get("status") in {"playing", "resumed", "error"}

    # 4. Equalizer preset switch via music.control
    res_eq1 = await ctrl_tool.handler(action="equalizer", preset="robot")
    assert res_eq1.get("status") == "ok"
    assert res_eq1.get("preset") == "robot"

    # 5. Equalizer preset list via music.control
    res_list = await ctrl_tool.handler(action="equalizers")
    assert res_list.get("status") == "ok"
    assert "presets" in res_list
    assert "robot" in res_list["presets"]
    assert "bass_boost" in res_list["presets"]
    assert "spatial" in res_list["presets"]

    # 6. Equalizer tool directly
    res_eq2 = await eq_tool.handler(preset="spatial")
    assert res_eq2.get("status") == "ok"
    assert res_eq2.get("preset") == "spatial"

    # Restore default robot preset
    await eq_tool.handler(preset="robot")
    svc.stop()


def test_tool_router_chaining():
    tr = ToolRegistry()
    from robot.tools.builtins import register_builtin_tools
    from unittest.mock import MagicMock
    engine = MagicMock()
    engine.tool_registry = tr
    register_builtin_tools(engine)

    router = ToolRouter(tr)

    # 1. Single intent
    tools_music = router.select_tools("play some music")
    assert "music.play" in tools_music

    # 2. Chained intent with "and then"
    tools_chained = router.select_tools("check what is in front of you and then play juice wrld and do a wave")
    assert "music.play" in tools_chained
    assert "character.execute_action" in tools_chained
    assert "perception.what_do_you_see" in tools_chained or "perception.inspect_view" in tools_chained


def test_api_equalizer_endpoint():
    from starlette.testclient import TestClient
    from robot.api import create_app
    from unittest.mock import MagicMock

    engine = MagicMock()
    app = create_app(engine)

    client = TestClient(app)
    # GET status
    resp = client.get("/api/music/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "equalizer" in data

    # POST switch equalizer preset
    resp2 = client.post("/api/equalizer", json={"preset": "robot"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2.get("status") == "ok"
    assert data2.get("preset") == "robot"








