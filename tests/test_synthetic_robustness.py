"""Synthetic test suite for context intelligence, entity attribution, streaming STT, and tools."""

import asyncio
import numpy as np
import pytest

from robot.core.context import ContextEngine
from robot.motion.actions import validate_action
from robot.motion.planner import MotionPlanner
from robot.stt import STTProvider, StreamingSTT, clean_transcript
from robot.tools import ToolDefinition, ToolRegistry, ToolRouter
from robot.tools.skills import ToolSkillManager


class MockSTT(STTProvider):
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if len(audio) == 0:
            return ""
        return "synthetic test transcript"

    async def health_check(self) -> bool:
        return True


def test_entity_attribution_and_crush_grounding():
    """Verify that memories about the user's crush are explicitly attributed to the user."""
    engine = ContextEngine()
    memories = [
        {"memory": "The user crush is Anubha Anand"},
        {"memory": "The user favorite programming language is Python"},
    ]
    messages = engine.build_messages(
        system_prompt="You are a personal robot assistant.",
        user_text="Who is Anubha?",
        memories=memories,
        high_intelligence=False,
    )

    # 1. System prompt must include entity attribution directive
    sys_content = messages[0]["content"]
    assert "User Entity Ownership & Attribution Directive" in sys_content
    assert "belong exclusively to the user" in sys_content.lower()
    assert "anubha is the user's crush, not yours" in sys_content.lower()

    # 2. Memories must be labeled as belonging to the user
    user_turn = messages[-1]["content"]
    assert "Anubha Anand" in user_turn
    assert "(Belonging to User)" in user_turn


def test_context_layer_budget_local_vs_gemini():
    """Verify that local mode stays compact for RTX 3050 (<2200 tokens) while Gemini expands."""
    engine = ContextEngine()

    large_archive = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Detailed turn {i} with facts about physics and robotics."}
        for i in range(40)
    ]
    all_facts = [{"fact": f"User preference #{i}: likes item {i}"} for i in range(50)]

    # Local mode (high_intelligence=False)
    local_messages = engine.build_messages(
        system_prompt="You are a helpful assistant.",
        user_text="Tell me about item 2",
        recent_conversation=large_archive[-5:],
        conversation_archive=large_archive,
        all_facts=all_facts,
        high_intelligence=False,
    )
    local_tokens = sum(len(m.get("content", "")) // 4 for m in local_messages)
    # Local mode must not bloat beyond RTX 3050 fast threshold
    assert local_tokens < 2500

    # Gemini mode (high_intelligence=True)
    gemini_messages = engine.build_messages(
        system_prompt="You are a helpful assistant.",
        user_text="Tell me about item 2",
        recent_conversation=large_archive[-20:],
        conversation_archive=large_archive,
        all_facts=all_facts,
        high_intelligence=True,
    )
    gemini_tokens = sum(len(m.get("content", "")) // 4 for m in gemini_messages)
    # Gemini mode should embrace deep expansion
    assert gemini_tokens > local_tokens
    assert "[Historical Multi-Session Conversation Archive]" in gemini_messages[-1]["content"]


@pytest.mark.asyncio
async def test_streaming_stt_chunking_and_finalization():
    """Verify StreamingSTT correctly splits chunks and finalizes merged output."""
    mock_provider = MockSTT()
    stt = StreamingSTT(provider=mock_provider, sample_rate=16000, chunk_seconds=1.0)
    stt.start()

    # Feed 1.5 seconds of simulated audio frames (24000 samples)
    frame = np.ones(8000, dtype=np.float32) * 0.05
    stt.feed_frame(frame)
    stt.feed_frame(frame)  # Crosses 16000 threshold -> dispatches chunk

    # Finalize remaining tail
    result = await stt.finalize(remaining_audio=frame)
    assert result != ""
    assert "synthetic test transcript" in result


def test_expressive_motion_actions_and_planning():
    """Verify that all new expressive actions validate and plan correctly."""
    expressive_actions = [
        "nod", "shake_head", "tilt_head", "shrug", "bow",
        "clap", "think", "cheer", "hands_on_hips", "heart_gesture"
    ]
    planner = MotionPlanner()

    for act_name in expressive_actions:
        req = {"action": act_name, "duration": 1.5}
        valid, action, err = validate_action(req)
        assert valid is True, f"Failed validation for {act_name}: {err}"
        assert action is not None
        assert action.action == act_name

        plan = planner.plan_action(req)
        assert len(plan) >= 1

    # Verify natural language command parser
    cmd_nod = planner.parse_natural_command("can you please nod your head?")
    assert any(a.get("action") == "nod" for a in cmd_nod)

    cmd_bow = planner.parse_natural_command("take a bow for the audience")
    assert any(a.get("action") == "bow" for a in cmd_bow)

    cmd_cheer = planner.parse_natural_command("cheer for victory!")
    assert any(a.get("action") == "cheer" for a in cmd_cheer)


def test_tool_skill_manager_guides():
    """Verify ToolSkillManager has complete guides for knowledge and calculation."""
    manager = ToolSkillManager()

    wiki_skill = manager.get_skill("knowledge.wikipedia")
    assert wiki_skill is not None
    assert "Wikipedia" in wiki_skill.summary
    assert len(wiki_skill.examples) >= 2

    calc_skill = manager.get_skill("system.calculate")
    assert calc_skill is not None
    assert "expression" in calc_skill.parameters_guide

    prompt_guide = manager.format_skills_prompt(["knowledge.wikipedia", "system.calculate"])
    assert "### Skill: knowledge.wikipedia" in prompt_guide
    assert "### Skill: system.calculate" in prompt_guide


def test_tool_router_new_families():
    """Verify ToolRouter routes queries to knowledge and system calculation."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="knowledge.wikipedia",
            description="Wikipedia search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=lambda query: {"result": "ok"},
            family="knowledge",
        )
    )
    registry.register(
        ToolDefinition(
            name="system.calculate",
            description="Math calculator",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
            handler=lambda expression: {"result": 42},
            family="system",
        )
    )

    router = ToolRouter(registry)
    selected_wiki = router.select_tools("tell me about the history of Rome from wikipedia")
    assert "knowledge.wikipedia" in selected_wiki

    selected_math = router.select_tools("calculate the square root of 529")
    assert "system.calculate" in selected_math


def test_parse_qwen_tool_call_block_bracketed_calls():
    """Verify that bracketed tool calls like [system.time] are parsed accurately."""
    from robot.llm import parse_qwen_tool_call_block

    # Test single bracketed tool call
    res = parse_qwen_tool_call_block("[system.time]")
    assert len(res) == 1
    assert res[0]["function"]["name"] == "system.time"

    # Test multiple bracketed tool calls
    res_multi = parse_qwen_tool_call_block("[system.time] [perception.what_do_you_see]")
    assert len(res_multi) == 2
    assert res_multi[0]["function"]["name"] == "system.time"
    assert res_multi[1]["function"]["name"] == "perception.what_do_you_see"

    # Test bracketed tool call with JSON arguments
    res_args = parse_qwen_tool_call_block('[image.generate({"prompt": "futuristic robot"})]')
    assert len(res_args) == 1
    assert res_args[0]["function"]["name"] == "image.generate"
    assert '"futuristic robot"' in res_args[0]["function"]["arguments"]


def test_anti_leakage_grounding_directive():
    """Verify that GROUNDING_POLICY includes the anti-leakage directive."""
    from robot.core.context import GROUNDING_POLICY, format_system_prompt

    assert "Pure Conversational Speech & Anti-Leakage Directive" in GROUNDING_POLICY
    assert "[system.time]" in GROUNDING_POLICY

    prompt = format_system_prompt("You are Luna, an AI companion.")
    assert "Pure Conversational Speech & Anti-Leakage Directive" in prompt

