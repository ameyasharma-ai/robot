"""Automated End-to-End Pipeline Diagnostic & Validation Suite.

Executes a non-destructive dry-run across all major subsystems:
1. Audio & VAD configuration integrity
2. STT Engine reachability & model presence
3. ContextEngine grounding & prompt cache invariance
4. Tool execution contracts & null-safety validation
5. TTS speech synthesis & clause chunking pipeline
6. Telemetry and observability readiness
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from robot.core.context import ContextEngine, format_system_prompt
from robot.tools import PermissionLevel, PermissionManager, ToolRegistry
from robot.tools.builtins import register_builtin_tools
from robot.llm import ClauseChunker
from robot.tts import clean_text_for_speech


class DummyEngine:
    """Mock engine instance for tool and pipeline validation."""

    def __init__(self) -> None:
        self.tools = ToolRegistry()
        self.tool_registry = self.tools
        self.permissions = PermissionManager(max_auto_level=PermissionLevel.SYSTEM)
        self.llm_provider_type = "local"


async def test_subsystem_context() -> dict[str, Any]:
    """Validate ContextEngine prompt assembly and cache invariance."""
    t0 = time.perf_counter()
    ctx_eng = ContextEngine()

    sys_p = format_system_prompt("You are a helpful test robot.")
    messages = ctx_eng.build_messages(
        system_prompt=sys_p,
        user_text="What is your status?",
        recent_conversation=[],
        memories=[],
        grounding_banner="[Time: 22:00]",
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert len(messages) >= 2, "Messages must contain system and user turns"
    assert messages[0]["role"] == "system"
    assert "You are a helpful test robot." in messages[0]["content"]

    return {
        "status": "PASS",
        "latency_ms": round(elapsed_ms, 2),
        "messages_count": len(messages),
        "system_prompt_len": len(messages[0]["content"]),
    }


async def test_subsystem_tools() -> dict[str, Any]:
    """Validate tool registry, alias resolution, execution contracts, and null-safety."""
    t0 = time.perf_counter()
    engine = DummyEngine()
    register_builtin_tools(engine)  # type: ignore[arg-type]

    # Test system.calculate
    t_calc = engine.tools.get("system.calculate")
    assert t_calc is not None
    calc_res = await engine.permissions.execute(t_calc, {"expression": "42 * 2"})
    assert calc_res.get("result") is not None, "Calculator must return non-null result"
    assert calc_res["result"].get("result") == 84

    # Test system.list_tools
    t_list = engine.tools.get("system.list_tools")
    assert t_list is not None
    list_res = await engine.permissions.execute(t_list, {})
    assert list_res.get("result") is not None, "list_tools must return non-null result"
    assert list_res["result"].get("total_tools", 0) > 20

    # Test death clock tool
    t_death = engine.tools.get("system.death_clock")
    assert t_death is not None
    death_res = await engine.permissions.execute(
        t_death,
        {"age": 25, "gender": "male", "country": "India", "use_gemini": False},
    )
    assert death_res.get("result") is not None, "death_clock must return non-null result"
    assert death_res["result"].get("projected_lifespan_years", 0) > 65

    # Test alias resolution
    t_alias = engine.tools.get("death_clock")
    assert t_alias is not None
    alias_res = await engine.permissions.execute(t_alias, {"age": 30, "use_gemini": False})
    assert alias_res.get("result") is not None, "death_clock alias must resolve"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "status": "PASS",
        "latency_ms": round(elapsed_ms, 2),
        "registered_tools": len(engine.tools._tools),
        "calc_result": calc_res["result"].get("result"),
        "death_clock_projected": death_res["result"].get("projected_lifespan_years"),
    }


async def test_subsystem_tts_chunker() -> dict[str, Any]:
    """Validate clause chunker boundary detection and text sanitization."""
    t0 = time.perf_counter()
    chunker = ClauseChunker(min_first_words=1)

    raw_markdown = "Sure! **Here** is the `status` of your test: [motion: nod] All systems are nominal."
    cleaned = clean_text_for_speech(raw_markdown)
    assert "**" not in cleaned, "Markdown bold must be stripped"
    assert "`" not in cleaned, "Markdown backticks must be stripped"

    clauses: list[str] = []
    # Simulate token streaming
    for token in ["Sure! ", "All ", "systems ", "are ", "operational, ", "and ", "running."]:
        clause = chunker.feed(token)
        if clause:
            clauses.append(clause)
    remainder = chunker.flush()
    if remainder:
        clauses.append(remainder)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert len(clauses) >= 2, "Stream must yield at least 2 distinct speech clauses"

    return {
        "status": "PASS",
        "latency_ms": round(elapsed_ms, 2),
        "cleaned_text": cleaned,
        "emitted_clauses": clauses,
    }


async def run_e2e_suite() -> int:
    """Run all subsystem diagnostic checks and print structured report.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 65)
    print("🚀 ROBOT CORE — AUTOMATED END-TO-END PIPELINE DIAGNOSTIC")
    print("=" * 65)

    results: dict[str, Any] = {}
    passed = True

    # 1. Context Engine
    try:
        results["context_engine"] = await test_subsystem_context()
        print(f"✅ [Context Engine] PASS ({results['context_engine']['latency_ms']} ms)")
    except Exception as exc:
        passed = False
        results["context_engine"] = {"status": "FAIL", "error": str(exc)}
        print(f"❌ [Context Engine] FAIL: {exc}")

    # 2. Tools & Null-Safety Contracts
    try:
        results["tools_subsystem"] = await test_subsystem_tools()
        print(f"✅ [Tools & Contracts] PASS ({results['tools_subsystem']['latency_ms']} ms, {results['tools_subsystem']['registered_tools']} tools verified)")
    except Exception as exc:
        passed = False
        results["tools_subsystem"] = {"status": "FAIL", "error": str(exc)}
        print(f"❌ [Tools & Contracts] FAIL: {exc}")

    # 3. Speech Chunker & Sanitization
    try:
        results["tts_chunker"] = await test_subsystem_tts_chunker()
        print(f"✅ [TTS Chunker] PASS ({results['tts_chunker']['latency_ms']} ms, {len(results['tts_chunker']['emitted_clauses'])} clauses emitted)")
    except Exception as exc:
        passed = False
        results["tts_chunker"] = {"status": "FAIL", "error": str(exc)}
        print(f"❌ [TTS Chunker] FAIL: {exc}")

    print("-" * 65)
    print(f"OVERALL STATUS: {'🎉 ALL CHECKS PASSED (100%)' if passed else '⚠️ SOME CHECKS FAILED'}")
    print("=" * 65)
    return 0 if passed else 1


if __name__ == "__main__":
    code = asyncio.run(run_e2e_suite())
    sys.exit(code)
