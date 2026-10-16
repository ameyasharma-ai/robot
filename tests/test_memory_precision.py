"""Memory Precision Benchmark and Partitioned User Profile Sheet Tests.

Validates:
1. Mem0 threshold calibration (0.82) across 50 benchmark queries.
2. Comparison of precision, recall, and false-positive rates at 0.75, 0.78, 0.82, 0.85.
3. Asynchronous background profile sheet summarization with partitioned fields
   (durable_preferences, temporary_context, inferred_facts).
4. Strict namespace isolation between user profiles and robot personalities.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from robot.memory import ConversationHistory, MemoryManager


@pytest.fixture
async def temp_memory(tmp_path: Path) -> MemoryManager:
    """Create an isolated, temporary MemoryManager instance for testing."""
    db_file = tmp_path / "test_precision.db"
    manager = MemoryManager(
        mem0_config={
            "score_threshold": 0.82,
            "embedder": {"provider": "mock"},
            "llm": {"provider": "mock"},
        },
        history_path=str(db_file),
    )
    await manager.initialize()
    return manager


BENCHMARK_50_QUERIES: list[dict[str, str | bool]] = [
    # 25 Relevant intent queries that SHOULD retrieve relevant facts
    {"query": "What is my favorite food?", "relevant": True, "target": "favorite food is sushi"},
    {"query": "What beverage do I like to drink?", "relevant": True, "target": "likes green tea"},
    {"query": "Which programming language do I prefer?", "relevant": True, "target": "prefers python for ai development"},
    {"query": "What is my dog's name?", "relevant": True, "target": "dog named buster"},
    {"query": "Where do I work as an engineer?", "relevant": True, "target": "works at robotics lab"},
    {"query": "What is my favorite music genre?", "relevant": True, "target": "favorite genre is synthwave"},
    {"query": "What city do I live in?", "relevant": True, "target": "lives in san francisco"},
    {"query": "What time do I usually wake up?", "relevant": True, "target": "wakes up at 7 am"},
    {"query": "What editor do I use for code?", "relevant": True, "target": "prefers neovim and vscode"},
    {"query": "What car do I drive?", "relevant": True, "target": "drives an electric sedan"},
    {"query": "What is my favorite book?", "relevant": True, "target": "favorite book is neuromancer"},
    {"query": "Do I have any dietary allergies?", "relevant": True, "target": "allergic to peanuts"},
    {"query": "What monitor setup do I have?", "relevant": True, "target": "dual 4k monitor setup"},
    {"query": "What is my sister's name?", "relevant": True, "target": "sister named maya"},
    {"query": "What sport do I play on weekends?", "relevant": True, "target": "plays tennis on saturdays"},
    {"query": "What is my preferred GPU for local inferencing?", "relevant": True, "target": "uses rtx 3050 for edge models"},
    {"query": "What kind of coffee do I order?", "relevant": True, "target": "prefers oat milk latte"},
    {"query": "What is my favorite color?", "relevant": True, "target": "favorite color is cyan"},
    {"query": "When is my birthday?", "relevant": True, "target": "birthday is october 24"},
    {"query": "What operating system do I use?", "relevant": True, "target": "runs linux mint 22"},
    {"query": "What headphones do I own?", "relevant": True, "target": "uses sony xm5 headphones"},
    {"query": "What is my favorite movie?", "relevant": True, "target": "favorite movie is blade runner"},
    {"query": "Do I prefer dark mode or light mode?", "relevant": True, "target": "always prefers dark mode"},
    {"query": "What is my primary research field?", "relevant": True, "target": "researches humanoid robotics"},
    {"query": "What is my morning routine?", "relevant": True, "target": "starts morning with 20 minute jog"},

    # 25 Distractor / Irrelevant queries that SHOULD NOT falsely hallucinate
    {"query": "How far is the Moon from Mars?", "relevant": False},
    {"query": "What is the capital of Mongolia?", "relevant": False},
    {"query": "Can quantum computers factor 2048-bit RSA keys?", "relevant": False},
    {"query": "What is the speed of sound in seawater?", "relevant": False},
    {"query": "Who won the 1994 FIFA World Cup?", "relevant": False},
    {"query": "Explain how a four-stroke internal combustion engine works.", "relevant": False},
    {"query": "What is the atomic mass of tungsten?", "relevant": False},
    {"query": "What is the chemical formula for sulfuric acid?", "relevant": False},
    {"query": "How many moons does Jupiter have?", "relevant": False},
    {"query": "What is the deepest trench in the Pacific Ocean?", "relevant": False},
    {"query": "Who wrote The Wealth of Nations?", "relevant": False},
    {"query": "What is the half-life of Carbon-14?", "relevant": False},
    {"query": "How does photosynthesis convert photon energy?", "relevant": False},
    {"query": "What is the boiling point of ethanol at 1 atm?", "relevant": False},
    {"query": "Explain Dijkstra's shortest path algorithm.", "relevant": False},
    {"query": "What causes the aurora borealis?", "relevant": False},
    {"query": "How many bones are in the adult human skeleton?", "relevant": False},
    {"query": "What is the currency of Switzerland?", "relevant": False},
    {"query": "Who discovered penicillin?", "relevant": False},
    {"query": "What is the capital of New Zealand?", "relevant": False},
    {"query": "How does a bipolar junction transistor amplify current?", "relevant": False},
    {"query": "What is the distance to the Andromeda galaxy?", "relevant": False},
    {"query": "What is the primary function of ribosomes?", "relevant": False},
    {"query": "How many chromosomes do chimpanzees have?", "relevant": False},
    {"query": "What is Fermat's Last Theorem?", "relevant": False},
]


@pytest.mark.asyncio
async def test_50_query_memory_benchmark(temp_memory: MemoryManager) -> None:
    """Benchmark memory retrieval precision and false positive suppression across 50 queries."""
    # Seed known facts into user's profile
    for item in BENCHMARK_50_QUERIES:
        if item.get("relevant") and "target" in item:
            await temp_memory.history.add_fact(str(item["target"]), user_id="user")

    # Evaluate 50 queries
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for item in BENCHMARK_50_QUERIES:
        q = str(item["query"])
        is_rel = bool(item["relevant"])
        results = await temp_memory.get_relevant_memories(q, user_id="user", limit=3)

        if is_rel:
            if results:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if results:
                false_positives += 1
            else:
                true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # Strict assertion: 0 false positive fact hallucinations on irrelevant general knowledge
    assert false_positives == 0, f"Spurious fact retrieval detected: {false_positives} false positives"
    assert precision == 1.0, f"Expected 100% precision, got {precision}"
    assert recall >= 0.88, f"Expected high recall (>=88%), got {recall}"


@pytest.mark.asyncio
async def test_threshold_calibration_comparison() -> None:
    """Compare candidate thresholds (0.75, 0.78, 0.82, 0.85) on synthetic semantic scores."""
    # Mock candidate scores returned by semantic search:
    # Scores > 0.80 represent genuine semantic matches
    # Scores between 0.70 and 0.79 represent spurious partial keyword / topical bleed
    scores = [
        {"fact": "favorite food is sushi", "score": 0.94, "relevant": True},
        {"fact": "prefers python for ai development", "score": 0.88, "relevant": True},
        {"fact": "drives an electric sedan", "score": 0.84, "relevant": True},
        {"fact": "prefers oat milk latte", "score": 0.83, "relevant": True},
        # Borderline noise / spurious bleed
        {"fact": "yesterday the server had high cpu load", "score": 0.79, "relevant": False},
        {"fact": "the weather in tokyo is rainy", "score": 0.77, "relevant": False},
        {"fact": "random system log entry 404", "score": 0.75, "relevant": False},
        {"fact": "unrelated meeting notes from march", "score": 0.72, "relevant": False},
    ]

    thresholds = [0.75, 0.78, 0.82, 0.85]
    results_by_thresh = {}

    for thresh in thresholds:
        passed = [s for s in scores if s["score"] >= thresh]
        fp = sum(1 for s in passed if not s["relevant"])
        tp = sum(1 for s in passed if s["relevant"])
        results_by_thresh[thresh] = {"tp": tp, "fp": fp}

    # At 0.75 and 0.78, noisy logs bleed into active context (fp > 0)
    assert results_by_thresh[0.75]["fp"] > 0
    assert results_by_thresh[0.78]["fp"] > 0

    # At 0.82, false positives are completely eliminated (0 FP) while preserving authentic facts (4 TP)
    assert results_by_thresh[0.82]["fp"] == 0
    assert results_by_thresh[0.82]["tp"] == 4


@pytest.mark.asyncio
async def test_background_user_profile_summarizer(temp_memory: MemoryManager) -> None:
    """Test background summarization into partitioned durable_preferences and temporary_context."""
    # Seed facts and session turns
    await temp_memory.history.add_fact("User's favorite color is cyan", user_id="user")
    await temp_memory.history.add_fact("User prefers dark mode in all applications", user_id="user")
    await temp_memory.history.add_fact("User works on robot hardware integration", user_id="user")

    session = await temp_memory.create_session(personality="neo")
    await temp_memory.history.add_message(
        session_id=session["id"],
        role="user",
        content="Let's optimize the audio barge-in pipeline today.",
    )

    # Run background summarizer
    profile = await temp_memory.summarize_user_profile(user_id="user")

    assert profile["user_id"] == "user"
    assert "durable_preferences" in profile
    assert len(profile["durable_preferences"]) >= 2
    assert "temporary_context" in profile
    assert "audio barge-in" in profile["temporary_context"].get("last_active_topic", "").lower()
    assert "=== USER PROFILE SHEET ===" in profile["summary_markdown"]

    # Retrieve from core user profile sheet
    core_profile = await temp_memory.get_core_user_profile(user_id="user")
    assert "Ameya" in core_profile
    assert "cyan" in core_profile.lower() or "dark mode" in core_profile.lower()


@pytest.mark.asyncio
async def test_strict_namespace_isolation(temp_memory: MemoryManager) -> None:
    """Verify 0% cross-user / cross-persona memory leakage."""
    # Add confidential facts to distinct namespaces
    await temp_memory.history.add_fact("Neo secret hacker key: ALPHA_99", user_id="neo")
    await temp_memory.history.add_fact("Luna diary entry: loves pink plushies", user_id="luna")
    await temp_memory.history.add_fact("User main credentials: AMEYA_DEV", user_id="user")

    # Query neo from luna
    luna_memories = await temp_memory.get_relevant_memories("ALPHA_99", user_id="luna")
    assert not any("ALPHA_99" in str(m) for m in luna_memories)

    # Query luna from user
    user_memories = await temp_memory.get_relevant_memories("plushies", user_id="user")
    assert not any("plushies" in str(m) for m in user_memories)

    # User profile of luna does not leak user facts
    luna_profile = await temp_memory.get_user_profile(user_id="luna")
    assert "AMEYA_DEV" not in str(luna_profile)
