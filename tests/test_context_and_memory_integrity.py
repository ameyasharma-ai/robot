"""Unit tests for Phase 5E: Memory and Context Integrity."""

import pytest
from robot.core.context import ContextEngine, format_system_prompt
from robot.memory import MemoryManager


@pytest.fixture
def context_engine():
    return ContextEngine({
        "system_personality": 600,
        "tool_schemas": 1500,
        "retrieved_memories": 400,
        "recent_conversation": 1200,
        "world_state": 300,
        "robot_state": 200,
        "generation_headroom": 1500,
    })


@pytest.fixture
async def memory_mgr(tmp_path):
    db_path = str(tmp_path / "test_integrity.db")
    config = {
        "mem0": {"enabled": False},
        "history": {"database_path": db_path},
    }
    mgr = MemoryManager(config, db_path)
    await mgr.initialize()
    return mgr


def test_bounded_context_growth(context_engine):
    """Verify that simulating 100 turns never overflows the token budget."""
    long_conversation = []
    for i in range(100):
        long_conversation.append({"role": "user", "content": f"Turn {i}: What is the status of subsystem {i}?"})
        long_conversation.append({"role": "assistant", "content": f"Subsystem {i} is operating normally with 100% efficiency."})

    system_prompt = format_system_prompt("You are a helpful robot.", high_intelligence=False)
    messages = context_engine.build_messages(
        system_prompt=system_prompt,
        user_text="What is the overall summary?",
        recent_conversation=long_conversation,
    )

    # Context must have system prompt, trimmed conversation, and current turn
    assert len(messages) >= 3
    assert messages[0]["role"] == "system"
    # First message in history must be a 'user' turn boundary
    assert messages[1]["role"] == "user"
    assert messages[-1]["role"] == "user"

    # Total tokens must not exceed budget
    total_tokens = sum(len(m["content"]) // 4 for m in messages if isinstance(m.get("content"), str))
    assert total_tokens < 3500


def test_system_prompt_prefix_invariance(context_engine):
    """Verify system prompt is 100% byte-for-byte invariant across distinct conversation turns."""
    p_neo = format_system_prompt("You are Cypher, an ultra-precise tactical AI.")
    msgs_1 = context_engine.build_messages(system_prompt=p_neo, user_text="Hello")
    msgs_2 = context_engine.build_messages(system_prompt=p_neo, user_text="What's the weather?")
    msgs_3 = context_engine.build_messages(system_prompt=p_neo, user_text="Calculate 5 * 5")

    # Invariant Slot 0 Prefix Guarantee
    assert msgs_1[0]["content"] == msgs_2[0]["content"]
    assert msgs_2[0]["content"] == msgs_3[0]["content"]
    assert len(msgs_1[0]["content"]) > 500


@pytest.mark.asyncio
async def test_duplicate_turn_prevention(memory_mgr):
    """Verify that identical turns stored in rapid succession are suppressed."""
    sess_id = "test_sess_dedup"

    # Turn 1
    await memory_mgr.store_turn(sess_id, "Hello robot", "Hello human!", personality="cypher", turn_number=1)
    # Turn 1 duplicate (e.g. from retry or retransmission)
    await memory_mgr.store_turn(sess_id, "Hello robot", "Hello human!", personality="cypher", turn_number=1)

    history = await memory_mgr.get_recent_conversation(sess_id, limit=10)
    # Exactly 2 messages (1 user, 1 assistant), NOT 4
    assert len(history) == 2
    assert history[0]["content"] == "Hello robot"
    assert history[1]["content"] == "Hello human!"


@pytest.mark.asyncio
async def test_persistent_memory_write_and_retrieval(memory_mgr):
    """Verify durable biographical facts are stored and isolated by namespace."""
    user_id = "cypher_workspace"

    await memory_mgr.history.add_fact("User is a senior software engineer working on robotics", user_id=user_id)
    await memory_mgr.history.add_fact("User's favorite programming language is Python", user_id=user_id)

    facts = await memory_mgr.get_all_facts(user_id=user_id)
    assert len(facts) >= 2
    fact_texts = [f["fact"] for f in facts]
    assert any("robotics" in ft for ft in fact_texts)
    assert any("Python" in ft for ft in fact_texts)

    # Verify namespace isolation: querying another workspace returns empty
    other_facts = await memory_mgr.get_all_facts(user_id="other_workspace")
    assert len(other_facts) == 0
