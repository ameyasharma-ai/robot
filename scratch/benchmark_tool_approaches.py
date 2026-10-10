"""Benchmark Approach A vs Approach B for Tool Exposure and Prompt Caching.

Approach A: Invariant set containing all 59 tool schemas across all turns.
Approach B: Invariant core toolset (18 tools) with selectively exposed specialized tools.
"""

import asyncio
import json
import time
import httpx

from robot.core.engine import RobotEngine
from robot.tools.builtins import register_builtin_tools

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Core tools family (high frequency embodied tools)
CORE_FAMILIES = {"system", "music", "scheduler", "perception", "os", "web", "memory"}

TEST_QUERIES = [
    ("Set a timer for 10 seconds", "timer.set"),
    ("What time is it right now?", "system.time"),
    ("Check the git status of the repository", "git.workspace_commit"),
    ("Scan nearby wifi access points", "network.wifi_scan"),
    ("Set screen brightness to 30 percent", "os.display_brightness"),
    ("Tell me a quick joke about robots", None),
]


async def send_chat_completion(client: httpx.AsyncClient, messages: list[dict], tools: list[dict] | None):
    payload = {
        "model": "local-qwen",
        "messages": messages,
        "tools": tools,
        "stream": True,
        "max_tokens": 150,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    first_chunk_time = None
    gen_tokens = 0
    full_text = ""
    tool_calls = []

    async with client.stream("POST", SERVER_URL, json=payload, timeout=30.0) as resp:
        if resp.status_code != 200:
            err = await resp.aread()
            raise RuntimeError(f"Server error {resp.status_code}: {err.decode()}")

        async for line in resp.aiter_lines():
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "tool_calls" in delta and delta["tool_calls"]:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        tool_calls.extend(delta["tool_calls"])
                    c = delta.get("content", "")
                    if c:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        gen_tokens += 1
                        full_text += c

    total_time = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms": ttft or total_time,
        "total_ms": total_time,
        "gen_tokens": gen_tokens,
        "text": full_text,
        "tool_calls": tool_calls,
    }


async def get_server_slot_info(client: httpx.AsyncClient):
    try:
        r = await client.get("http://127.0.0.1:8080/slots")
        if r.status_code == 200:
            slots = r.json()
            if slots and isinstance(slots, list):
                s0 = slots[0]
                return {
                    "n_past": s0.get("n_past", 0),
                    "n_prompt_tokens": s0.get("n_prompt_tokens", 0),
                }
    except Exception:
        pass
    return {}


async def run_benchmark():
    engine = RobotEngine()
    register_builtin_tools(engine)
    from robot.tools import ToolRouter
    engine.tool_router = ToolRouter(engine.tool_registry)
    all_schemas = engine.tool_registry.get_schemas()
    all_tool_names = engine.tool_registry.list_tools()
    print(f"Loaded {len(all_schemas)} total tool schemas.")

    # Core schemas
    core_tool_defs = [
        t for t in engine.tool_registry.get_all()
        if t.family in CORE_FAMILIES or t.name in {"os.display_brightness", "system.time", "timer.set"}
    ]
    core_schemas = [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in core_tool_defs
    ]
    print(f"Loaded {len(core_schemas)} core tool schemas.")

    system_prompt = (
        "You are Lisa, an intelligent embodied companion robot.\n"
        "Grounding Directives: Always invoke tools when requested by the user."
    )

    async with httpx.AsyncClient() as client:
        # === TEST APPROACH A: Invariant All 59 Tools ===
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARK: APPROACH A (Invariant All 59 Tools)")
        print("=" * 60)

        # Clear server slots first
        history_a: list[dict] = [{"role": "system", "content": system_prompt}]
        results_a = []

        for idx, (query, expected_tool) in enumerate(TEST_QUERIES):
            turn_msgs = list(history_a) + [{"role": "user", "content": query}]
            res = await send_chat_completion(client, turn_msgs, all_schemas)
            slot_info = await get_server_slot_info(client)

            # Record turn
            assistant_content = res["text"] or ""
            if res["tool_calls"]:
                call_names = [tc.get("function", {}).get("name", "") for tc in res["tool_calls"]]
                called_str = f"Tools: {call_names}"
            else:
                called_str = "No tools"

            results_a.append({
                "turn": idx + 1,
                "query": query,
                "expected": expected_tool,
                "ttft_ms": round(res["ttft_ms"], 1),
                "total_ms": round(res["total_ms"], 1),
                "gen_tokens": res["gen_tokens"],
                "called": called_str,
                "slot_info": slot_info,
            })
            print(f"Turn {idx+1}: '{query[:30]}...' -> TTFT: {res['ttft_ms']:.1f}ms | Total: {res['total_ms']:.1f}ms | {called_str}")
            history_a.append({"role": "user", "content": query})
            history_a.append({"role": "assistant", "content": assistant_content or "Action executed."})

        # === TEST APPROACH B: Invariant Core (35) + Selective Specialized ===
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARK: APPROACH B (Invariant Core + Selective Specialized)")
        print("=" * 60)

        # Reset conversation history for clean test
        history_b: list[dict] = [{"role": "system", "content": system_prompt}]
        results_b = []

        for idx, (query, expected_tool) in enumerate(TEST_QUERIES):
            # Select tools: if query requires a specialized tool not in core, add it
            selected_names = engine.tool_router.select_tools(query)
            active_tools = list(core_schemas)
            # Add specialized tools if selected and not in core
            core_names = {t["function"]["name"] for t in core_schemas}
            added_specialized = []
            for nm in selected_names:
                if nm not in core_names and nm in engine.tool_registry._tools:
                    t_def = engine.tool_registry.get(nm)
                    if t_def:
                        active_tools.append({
                            "type": "function",
                            "function": {"name": t_def.name, "description": t_def.description, "parameters": t_def.parameters},
                        })
                        added_specialized.append(nm)

            turn_msgs = list(history_b) + [{"role": "user", "content": query}]
            res = await send_chat_completion(client, turn_msgs, active_tools)
            slot_info = await get_server_slot_info(client)

            assistant_content = res["text"] or ""
            if res["tool_calls"]:
                call_names = [tc.get("function", {}).get("name", "") for tc in res["tool_calls"]]
                called_str = f"Tools: {call_names}"
            else:
                called_str = "No tools"

            results_b.append({
                "turn": idx + 1,
                "query": query,
                "expected": expected_tool,
                "added_specialized": added_specialized,
                "tool_count": len(active_tools),
                "ttft_ms": round(res["ttft_ms"], 1),
                "total_ms": round(res["total_ms"], 1),
                "gen_tokens": res["gen_tokens"],
                "called": called_str,
                "slot_info": slot_info,
            })
            print(f"Turn {idx+1}: '{query[:30]}...' -> TTFT: {res['ttft_ms']:.1f}ms | Tools in req: {len(active_tools)} (Added: {added_specialized}) | {called_str}")
            history_b.append({"role": "user", "content": query})
            history_b.append({"role": "assistant", "content": assistant_content or "Action executed."})

        # Print comparison summary table
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY: APPROACH A vs APPROACH B")
        print("=" * 70)
        print(f"{'Turn':<5} | {'Query':<25} | {'App A TTFT':<12} | {'App B TTFT':<12} | {'App B Diff':<12}")
        print("-" * 70)
        for ra, rb in zip(results_a, results_b):
            diff = rb["ttft_ms"] - ra["ttft_ms"]
            diff_str = f"{diff:+.1f}ms"
            print(f"{ra['turn']:<5} | {ra['query'][:25]:<25} | {ra['ttft_ms']:<10.1f}ms | {rb['ttft_ms']:<10.1f}ms | {diff_str:<12}")

        with open("scratch/tool_benchmark_results.json", "w") as f:
            json.dump({"approach_a": results_a, "approach_b": results_b}, f, indent=2)
        print("\nDetailed results saved to scratch/tool_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
