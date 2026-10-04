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
            res = await t.handler(query="Bohemian Rhapsody", source="spotify")
            assert res.get("status") == "playing"
            assert "spotify" in res.get("url", "")
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

    # Controls
    assert svc.pause()["status"] == "paused"
    assert svc.resume()["status"] == "resumed"
    assert svc.set_volume(0.5)["volume"] == 0.5
    assert svc.stop()["status"] == "stopped"


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




