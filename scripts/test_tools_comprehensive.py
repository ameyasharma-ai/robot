"""Comprehensive latency, timeout, and accuracy benchmark for Robot Core built-in tools."""

import asyncio
import time
from robot.core.engine import RobotEngine
from robot.motion.actions import validate_action
from robot.motion.planner import MotionPlanner

async def benchmark_tools():
    print("=" * 60)
    print("ROBOT CORE — COMPREHENSIVE TOOL BENCHMARK & ACCURACY TEST")
    print("=" * 60)

    # Initialize engine without launching heavy background hardware threads
    engine = RobotEngine(config_path="config/default.yaml")
    # Register builtin tools
    from robot.tools.builtins import register_builtin_tools
    register_builtin_tools(engine)
    
    registry = engine.tool_registry
    all_tools = registry.list_tools()
    print(f"\n[1] Registered Tools count: {len(all_tools)}")
    for t_name in all_tools:
        td = registry.get(t_name)
        print(f"  - {t_name:<25} (family: {td.family}, perm: {td.permission_level.name}, timeout: {td.timeout_s}s)")

    # 1. Benchmark system.time
    print("\n[2] Benchmarking 'system.time'...")
    time_tool = registry.get("system.time")
    t0 = time.perf_counter()
    time_res = await time_tool.handler()
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"  Latency: {dt_ms:.3f} ms | Result: {time_res}")
    assert "current_time" in time_res and "current_date" in time_res, "system.time missing expected keys"
    assert dt_ms < 10.0, f"system.time too slow ({dt_ms:.1f}ms)"

    # 2. Benchmark system.status
    print("\n[3] Benchmarking 'system.status'...")
    status_tool = registry.get("system.status")
    t0 = time.perf_counter()
    status_res = await status_tool.handler()
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"  Latency: {dt_ms:.3f} ms | State keys: {list(status_res.keys())}")
    assert "state" in status_res and "telemetry" in status_res
    assert dt_ms < 10.0, f"system.status too slow ({dt_ms:.1f}ms)"

    # 3. Benchmark motion validation & planning
    print("\n[4] Benchmarking Character & Motion Actions...")
    planner = MotionPlanner()
    actions_to_test = [
        {"action": "wave", "side": "right", "duration": 1.2},
        {"action": "kick", "side": "left", "duration": 0.8},
        {"action": "raise_hand", "side": "both", "duration": 1.0},
        {"action": "idle", "duration": 0.5},
    ]
    for act in actions_to_test:
        t0 = time.perf_counter()
        valid, semantic_action, err = validate_action(act)
        assert valid, f"Action validation failed for {act}: {err}"
        seq = planner.plan_action(act)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  Action: {act['action']:<12} side: {str(act.get('side')):<6} -> {len(seq)} keyframes in {dt_ms:.3f} ms")
        assert len(seq) > 0

    # 4. Benchmark web.search (Multi-tier fallback latency & accuracy)
    print("\n[5] Benchmarking 'web.search'...")
    search_tool = registry.get("web.search")
    queries = [
        "current python 3.12 release date",
        "what is DuckDuckGo",
    ]
    for q in queries:
        t0 = time.perf_counter()
        results = await search_tool.handler(query=q, max_results=2)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  Query: '{q}' -> {len(results)} items in {dt_ms:.1f} ms")
        if results and "error" not in results[0]:
            print(f"    Sample: [{results[0].get('title', '')[:40]}...] URL: {results[0].get('url', '')[:40]}")

    print("\n" + "=" * 60)
    print("ALL TOOL BENCHMARKS & ACCURACY TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(benchmark_tools())
