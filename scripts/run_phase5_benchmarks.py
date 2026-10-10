"""Empirical Benchmark & Latency Distribution Evaluator for Phase 5.

Evaluates P50, P95, and P99 latency across all 5 operational categories:
1. Fast-Path Local Utility (<50ms execution, <350ms E2E)
2. Conversational Banter (KV-cache accelerated, <1s TTFT)
3. Tool-Assisted Local Actions (hardware status, calculators)
4. Network-Dependent Operations (weather, news, quotes, capped at 3.0s)
5. Sequential Multi-Step Planner–Executor (structured plans with step boundaries)
"""

import asyncio
import numpy as np
import time

from robot.core.context import ContextEngine, format_system_prompt
from robot.observability import ObservabilityHub
from robot.tools import (
    PermissionLevel,
    PermissionManager,
    SequentialPlannerExecutor,
    ToolDefinition,
    ToolRegistry,
)
from robot.tools.builtins import register_builtin_tools
from robot.tools.fast_path import FastPathRouter


class DummyEngine:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.permissions = PermissionManager(max_auto_level=PermissionLevel.HARMLESS_LOCAL)
        register_builtin_tools(self)
        self.fast_path = FastPathRouter(self.tool_registry, self.permissions)


async def run_benchmarks():
    print("=" * 70)
    print("🚀 RUNNING PHASE 5 BENCHMARK & LATENCY DISTRIBUTION EVALUATION")
    print("=" * 70)

    engine = DummyEngine()
    planner = SequentialPlannerExecutor(engine.tool_registry, engine.permissions)

    # 1. Fast-Path Benchmark
    fast_path_queries = [
        "What is the time?",
        "what time is it right now?",
        "current time",
        "battery status",
        "system status",
        "hardware status",
        "what is the display brightness?",
        "set display brightness to 70%",
    ]
    fast_path_latencies = []
    for _ in range(5):
        for q in fast_path_queries:
            t0 = time.perf_counter()
            intent = engine.fast_path.classify(q)
            if intent:
                outcome, phrase = await engine.fast_path.execute_intent(intent)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                fast_path_latencies.append(elapsed_ms)

    # 2. Conversational Context & KV-Cache Invariance Benchmark
    ctx_engine = ContextEngine()
    sys_prompt = format_system_prompt("You are Cypher, an ultra-precise tactical AI.")
    conversational_latencies = []
    for i in range(25):
        t0 = time.perf_counter()
        msgs = ctx_engine.build_messages(
            system_prompt=sys_prompt,
            user_text=f"Hello robot, status check iteration {i}",
            recent_conversation=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Greetings."}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        conversational_latencies.append(elapsed_ms)

    # 3. Tool-Assisted Local Actions (Calculators, Hardware)
    tool_assisted_latencies = []
    calc_tool = engine.tool_registry.get("system.calculate")
    hw_tool = engine.tool_registry.get("system.hardware")
    for i in range(20):
        t0 = time.perf_counter()
        if i % 2 == 0 and calc_tool:
            await engine.permissions.execute(calc_tool, {"expression": f"{i} * 15 + 4"})
        elif hw_tool:
            await engine.permissions.execute(hw_tool, {})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tool_assisted_latencies.append(elapsed_ms)

    # 4. Network-Dependent Tools (Weather, News, Finance with 3.0s cap)
    network_latencies = []
    weather_tool = engine.tool_registry.get("weather.forecast")
    news_tool = engine.tool_registry.get("news.headlines")
    finance_tool = engine.tool_registry.get("finance.quote")
    network_tasks = [
        (weather_tool, {"location": "Pune"}),
        (weather_tool, {"location": "Tokyo"}),
        (news_tool, {"category": "tech"}),
        (finance_tool, {"symbol": "bitcoin"}),
        (finance_tool, {"symbol": "apple"}),
    ]
    for tool, args in network_tasks * 3:
        if tool:
            t0 = time.perf_counter()
            res = await engine.permissions.execute(tool, args)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            network_latencies.append(elapsed_ms)

    # 5. Multi-Step Planner Execution
    multi_step_latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        plan = planner.create_plan(
            goal="Multi-step system health and calculation",
            raw_steps=[
                {"step_id": "s1", "tool_name": "system.calculate", "arguments": {"expression": "25*4"}},
                {"step_id": "s2", "tool_name": "system.hardware", "arguments": {}, "depends_on": ["s1"]},
            ],
        )
        await planner.execute_plan(plan)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        multi_step_latencies.append(elapsed_ms)

    # Print Distribution Summary
    def calc_stats(latencies):
        arr = np.array(latencies)
        return {
            "count": len(arr),
            "p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
            "min": np.min(arr),
            "max": np.max(arr),
        }

    results = {
        "fast_path": calc_stats(fast_path_latencies),
        "conversational_context": calc_stats(conversational_latencies),
        "tool_assisted_local": calc_stats(tool_assisted_latencies),
        "network_dependent": calc_stats(network_latencies),
        "multi_step_planner": calc_stats(multi_step_latencies),
    }

    print("\n" + "=" * 70)
    print(f"{'CATEGORY':<25} | {'COUNT':<6} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9}")
    print("-" * 70)
    for cat, s in results.items():
        print(f"{cat:<25} | {s['count']:<6} | {s['p50']:<9.2f} | {s['p95']:<9.2f} | {s['p99']:<9.2f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
