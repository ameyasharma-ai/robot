"""Tests for context engine."""

from robot.core.context import ContextEngine, _estimate_tokens


def test_estimate_tokens():
    assert _estimate_tokens("Hello world") > 0
    assert _estimate_tokens("") == 0


def test_context_basic_build():
    engine = ContextEngine()
    messages = engine.build_messages(
        system_prompt="You are a helpful assistant.",
        user_text="What time is it?",
    )
    assert len(messages) >= 2  # system + user
    assert messages[0]["role"] == "system"
    assert "What time is it?" in messages[-1]["content"]
    assert "Indian Standard Time" in messages[-1]["content"]


def test_context_with_memories():
    engine = ContextEngine()
    memories = [
        {"memory": "User prefers dark mode"},
        {"memory": "User's name is Ameya"},
    ]
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="Hello",
        memories=memories,
    )
    # Memories are grounded either in system prompt or dynamically in the turn
    full_content = " ".join(m.get("content", "") for m in messages)
    assert "Ameya" in full_content or "dark mode" in full_content


def test_context_with_conversation():
    engine = ContextEngine()
    recent = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="Tell me more",
        recent_conversation=recent,
    )
    assert len(messages) >= 4  # system + 2 history + user


def test_context_alias_keyword_arguments():
    """Verify build_messages accepts legacy / alias keyword arguments without throwing TypeError."""
    engine = ContextEngine()
    messages = engine.build_messages(
        personality_prompt="You are Luna.",
        user_input="Hello from chat interface",
        history=[{"role": "user", "content": "previous question"}],
        retrieved_memories=[{"memory": "favorite color is cyan"}],
    )
    assert len(messages) >= 3
    assert messages[0]["role"] == "system"
    assert "Luna" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "Hello from chat interface" in messages[-1]["content"]
    assert "cyan" in messages[-1]["content"]


def test_context_action_commands_do_not_inject_memories():
    """Verify action commands like volume or replay never get polluted by memories."""
    engine = ContextEngine()
    memories = [{"memory": "User likes jazz"}, {"memory": "User crush is Anubha"}]
    for action_cmd in ["set the volume to 20%", "pause the music", "replay", "what time is it?"]:
        messages = engine.build_messages(
            system_prompt="You are Luna.",
            user_text=action_cmd,
            memories=memories,
        )
        user_turn = messages[-1]["content"]
        assert "User crush" not in user_turn
        assert "User likes jazz" not in user_turn
        assert action_cmd in user_turn


def test_context_greetings_zero_memories():
    """Verify casual greetings never inject irrelevant personal facts or memories."""
    engine = ContextEngine()
    poison_memories = [
        {"memory": "Ameya's crush is Anubha Anand"},
        {"memory": "The primary optical collecting area of our benchmark space telescope is exactly 33.18 square meters"},
        {"memory": "I have a severe tree-nut allergy"},
    ]
    for greeting in ["Hi", "Hello!", "hey robot", "How are you?", "Good morning"]:
        messages = engine.build_messages(
            system_prompt="You are Neo.",
            user_text=greeting,
            memories=poison_memories,
        )
        user_turn = messages[-1]["content"]
        assert "Anubha" not in user_turn
        assert "33.18" not in user_turn
        assert "tree-nut" not in user_turn
        assert greeting in user_turn


def test_context_recalled_memories_on_specific_queries():
    """Verify query-specific memory injection operates accurately for targeted questions."""
    engine = ContextEngine()
    memories = [
        {"memory": "Ameya's crush is Anubha Anand"},
        {"memory": "User favorite food is spicy chicken ramen"},
    ]
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="Who is my crush?",
        memories=memories,
    )
    user_turn = messages[-1]["content"]
    assert "Anubha Anand" in user_turn
    assert "ramen" not in user_turn or "Anubha" in user_turn


def test_assistant_history_thought_and_bracket_stripping():
    """Verify historical assistant turns strip thought tags and leaked tool brackets."""
    engine = ContextEngine()
    history = [
        {"role": "user", "content": "Play some jazz"},
        {
            "role": "assistant",
            "content": "<thought>I should play jazz music</thought>[music.play(query='jazz')] Playing jazz now for you Ameya!",
        },
    ]
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="What is playing?",
        recent_conversation=history,
    )
    # Check the historical assistant turn
    hist_assistant = messages[2]
    assert hist_assistant["role"] == "assistant"
    assert "<thought>" not in hist_assistant["content"]
    assert "[music.play" not in hist_assistant["content"]
    assert "Playing jazz now for you Ameya!" in hist_assistant["content"]


def test_dynamic_rolling_context_budget_trimmer_and_compaction():
    """Verify that exceeding conversation budget prunes older turns and synthesizes a compact digest."""
    engine = ContextEngine({"recent_conversation": 80})

    history = [
        {"role": "user", "content": "What is the capital of France and what is the best bakery there?"},
        {"role": "assistant", "content": "The capital is Paris, and Du Pain et des Idées is widely celebrated for bakeries."},
        {"role": "user", "content": "What about Italy and its best pasta places in Rome?"},
        {"role": "assistant", "content": "The capital is Rome, and Roscioli serves phenomenal carbonara."},
        {"role": "user", "content": "Tell me about Tokyo and ramen spots in Shinjuku."},
        {"role": "assistant", "content": "Tokyo features Fuunji for exceptional tsukemen and Ichiran for tonkotsu."},
    ]

    messages = engine.build_messages(
        system_prompt="You are Cypher.",
        user_text="Which of these cities should I visit first?",
        recent_conversation=history,
    )

    # Context should retain only the most recent turns within 80 tokens
    assert len(messages) >= 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[-1]["role"] == "user"

    # Compacted summary of earlier turns must be injected into the user turn grounding
    user_turn = messages[-1]["content"]
    assert "Earlier Conversation Summary (Compacted for Rolling Budget)" in user_turn
    assert "France" in user_turn or "Paris" in user_turn

    # Diagnostics verification
    diag = engine.get_last_diagnostics()
    assert "rolling_budget" in diag
    rb = diag["rolling_budget"]
    assert rb["budget_ceiling"] == 80
    assert rb["is_compacted"] is True
    assert rb["dropped_turns"] > 0
    assert rb["compacted_summary_tokens"] > 0


def test_explicit_empty_profile_never_leaks_default_operator_profile():
    """An isolated persona passes '' to mean no profile; it must not fall back to Ameya's."""
    engine = ContextEngine()
    messages = engine.build_messages(
        system_prompt="You are Ruby.",
        user_text="Who am I?",
        user_profile="",
    )
    assert "USER CORE IDENTITY PROFILE" not in messages[-1]["content"]
    assert "User Name: Ameya" not in messages[-1]["content"]


def test_robot_state_is_injected_into_grounding():
    """Robot state is an explicit build_messages input and must not be silently dropped."""
    engine = ContextEngine()
    messages = engine.build_messages(
        system_prompt="You are Neo.",
        user_text="Give me a status report",
        robot_state={"mode": "IDLE", "personality": "neo", "turn": 3},
    )
    user_turn = messages[-1]["content"]
    assert "[Robot Internal State]" in user_turn
    assert "mode: IDLE" in user_turn
    assert "turn: 3" in user_turn


def test_context_never_scans_music_library(monkeypatch):
    """Context assembly must use the playback snapshot, never the disk-scanning status."""
    from robot.tools.builtins.music import MusicPlayerService

    service = MusicPlayerService.get_instance()
    scans: list[int] = []
    monkeypatch.setattr(service, "list_local_tracks", lambda *a, **k: scans.append(1) or [])

    ContextEngine().build_messages(system_prompt="You are Neo.", user_text="Hello")
    assert scans == []


def test_summarize_conversation_turns_helper():
    """Verify summarize_conversation_turns produces concise digest with user and assistant highlights."""
    from robot.core.context import summarize_conversation_turns

    dropped = [
        {"role": "user", "content": "Can you open the Spotify app?"},
        {"role": "assistant", "content": "Opening Spotify for you now Ameya."},
        {"role": "user", "content": "Now play some lo-fi beats."},
        {"role": "assistant", "content": "Playing chill lo-fi beats playlist."},
    ]
    summary = summarize_conversation_turns(dropped, max_tokens=100)
    assert "User previously inquired about:" in summary
    assert "Spotify" in summary
    assert "Assistant previously responded:" in summary
    assert "lo-fi" in summary or "Spotify" in summary




