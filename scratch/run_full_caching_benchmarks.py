"""Comprehensive End-to-End Benchmark & Verification Suite for Prompt Caching.

Covers:
1. Repeated requests with identical prefix (cold vs warm).
2. Consecutive conversation turns (Turn 1 -> Turn 2 -> Turn 3).
3. Tool calls and tool result feedback loop (Pass 1 -> Pass 2).
4. Tool-set shifts and stability under invariant canonical tools.
5. Dynamic memory and user profile updates.
6. New conversations and persona changes.
7. Context-window rolling truncation near limit.
8. Concurrent requests (slot isolation / queuing).
9. Live system telemetry and resource footprint (RAM / VRAM).
"""

import asyncio
import json
import time
import httpx
import statistics

from robot.core.engine import RobotEngine
from robot.tools.builtins import register_builtin_tools
from robot.core.context import ContextEngine
from robot.llm import LlamaServerLLM

SERVER_BASE = "http://127.0.0.1:8080"


async def get_slot_info(client: httpx.AsyncClient):
    try:
        r = await client.get(f"{SERVER_BASE}/slots", timeout=3.0)
        if r.status_code == 200:
            slots = r.json()
            if slots and isinstance(slots, list):
                s = slots[0]
                return {
                    "n_ctx": s.get("n_ctx", 16384),
                    "n_prompt_tokens": s.get("n_prompt_tokens", 0),
                    "is_processing": s.get("is_processing", False),
                }
    except Exception:
        pass
    return {}


async def send_request(client: httpx.AsyncClient, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 40):
    payload = {
        "model": "local-qwen",
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    gen_tokens = 0
    full_text = ""
    tool_calls = []

    async with client.stream("POST", f"{SERVER_BASE}/v1/chat/completions", json=payload, timeout=30.0) as resp:
        if resp.status_code != 200:
            err = await resp.aread()
            raise RuntimeError(f"Server error {resp.status_code}: {err.decode()}")

        async for line in resp.aiter_lines():
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                data = json.loads(line[6:])
                choices = data.get("choices", [])
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

    t_end = time.perf_counter()
    total_ms = (t_end - t0) * 1000
    ttft_val = ttft or total_ms
    gen_time_s = max(0.001, (total_ms - ttft_val) / 1000.0)
    tok_per_sec = round(gen_tokens / gen_time_s, 1)

    return {
        "ttft_ms": round(ttft_val, 1),
        "total_ms": round(total_ms, 1),
        "gen_tokens": gen_tokens,
        "tok_per_sec": tok_per_sec,
        "text": full_text.strip(),
        "tool_calls": tool_calls,
    }


async def main():
    print("=" * 70)
    print("ROBOT V2 LOCAL LLM PROMPT CACHING VERIFICATION & BENCHMARK SUITE")
    print("=" * 70)

    engine = RobotEngine()
    register_builtin_tools(engine)
    tools = engine.tool_registry.get_schemas()
    llm = LlamaServerLLM(canonical_tools=tools)
    print(f"Loaded {len(tools)} canonical tools into LlamaServerLLM.")

    async with httpx.AsyncClient() as client:
        # ---------------------------------------------------------
        # Scenario 1: Identical Prefix (Cold vs Warm Repeat)
        # ---------------------------------------------------------
        print("\n--- Scenario 1: Identical Prefix (Cold vs Warm Repeated Requests) ---")
        prompt_1 = engine.context.build_messages(
            system_prompt="You are Lisa, an embodied companion.",
            user_text="Give me a 1-sentence thought about the future.",
        )
        # Run 1: Cold start for this prompt
        r_cold = await send_request(client, prompt_1, tools)
        slot_cold = await get_slot_info(client)
        print(f"  Run 1 (Cold): TTFT={r_cold['ttft_ms']}ms | Total={r_cold['total_ms']}ms | GenSpeed={r_cold['tok_per_sec']} tok/s | SlotTokens={slot_cold.get('n_prompt_tokens')}")

        # Repeated warm runs
        warm_ttfts = []
        for i in range(3):
            r_warm = await send_request(client, prompt_1, tools)
            warm_ttfts.append(r_warm["ttft_ms"])
            print(f"  Run {i+2} (Warm): TTFT={r_warm['ttft_ms']}ms | Total={r_warm['total_ms']}ms | GenSpeed={r_warm['tok_per_sec']} tok/s")
        print(f"  Median Warm TTFT: {statistics.median(warm_ttfts):.1f}ms")

        # ---------------------------------------------------------
        # Scenario 2: Consecutive Turns in Same Conversation
        # ---------------------------------------------------------
        print("\n--- Scenario 2: Consecutive Multi-Turn Conversation ---")
        conv_turns = [
            "Hi Lisa, how are you today?",
            "What can you do for me?",
            "What is the current time?",
            "Remember that I love coding in Python.",
            "What was that thing I just told you I loved?",
        ]
        history = []
        turn_metrics = []
        for idx, query in enumerate(conv_turns):
            msgs = engine.context.build_messages(
                system_prompt="You are Lisa, an intelligent embodied companion.",
                user_text=query,
                recent_conversation=history,
            )
            res = await send_request(client, msgs, tools)
            slot = await get_slot_info(client)
            turn_metrics.append(res)
            print(f"  Turn {idx+1}: '{query[:28]}...' -> TTFT: {res['ttft_ms']}ms | Total: {res['total_ms']}ms | SlotTokens: {slot.get('n_prompt_tokens')}")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": res["text"] or "Understood."})

        # ---------------------------------------------------------
        # Scenario 3: Request Invoking a Tool & 2-Pass Feedback
        # ---------------------------------------------------------
        print("\n--- Scenario 3: Tool Invocation & Second-Pass Feedback Loop ---")
        tool_query = "Set a timer for 10 seconds"
        msgs_t1 = engine.context.build_messages(
            system_prompt="You are Lisa, an intelligent embodied companion.",
            user_text=tool_query,
        )
        res_pass1 = await send_request(client, msgs_t1, tools)
        tc_names = [tc.get("function", {}).get("name", "") for tc in res_pass1["tool_calls"]]
        print(f"  Pass 1 (Tool Call): TTFT={res_pass1['ttft_ms']}ms | Emitted Tools: {tc_names}")

        # Pass 2: Feed tool output back with properly structured tool call and tool response
        formatted_tool_calls = [
            {"id": "call_123", "type": "function", "function": {"name": "timer.set", "arguments": json.dumps({"duration_seconds": 10})}}
        ]
        msgs_pass2 = list(msgs_t1) + [
            {"role": "assistant", "content": None, "tool_calls": formatted_tool_calls},
            {"role": "tool", "tool_call_id": "call_123", "content": json.dumps({"status": "success", "duration_seconds": 10, "message": "Timer set"})},
        ]
        res_pass2 = await send_request(client, msgs_pass2, tools)
        print(f"  Pass 2 (Synthesis): TTFT={res_pass2['ttft_ms']}ms | Total={res_pass2['total_ms']}ms | Response='{res_pass2['text']}'")

        # ---------------------------------------------------------
        # Scenario 4: Tool Schema Invariance Across Tool Types
        # ---------------------------------------------------------
        print("\n--- Scenario 4: Switching Actions Under Canonical Tools ---")
        action_queries = [
            "Set screen brightness to 40 percent",
            "What is the system battery status?",
            "Scan nearby wifi access points",
            "Play some lo-fi beats",
        ]
        for aq in action_queries:
            msgs = engine.context.build_messages(
                system_prompt="You are Lisa.",
                user_text=aq,
                recent_conversation=history[-4:],
            )
            r_act = await send_request(client, msgs, tools)
            called = [tc.get("function", {}).get("name", "") for tc in r_act["tool_calls"]]
            print(f"  Query: '{aq[:30]}...' -> TTFT: {r_act['ttft_ms']}ms | Emitted: {called}")

        # ---------------------------------------------------------
        # Scenario 5: Dynamic Memory Updates
        # ---------------------------------------------------------
        print("\n--- Scenario 5: Conversation With Memory Updates ---")
        mems = [{"memory": "User enjoys lo-fi and electronic music"}, {"memory": "User is building Robot V2"}]
        msgs_mem = engine.context.build_messages(
            system_prompt="You are Lisa.",
            user_text="What music do I like?",
            memories=mems,
        )
        r_mem = await send_request(client, msgs_mem, tools)
        print(f"  Memory Query TTFT: {r_mem['ttft_ms']}ms | Response: '{r_mem['text'][:60]}...'")

        # ---------------------------------------------------------
        # Scenario 6: New Conversation & Persona Switch
        # ---------------------------------------------------------
        print("\n--- Scenario 6: Persona Switching (Lisa -> Cypher) ---")
        msgs_cypher = engine.context.build_messages(
            system_prompt="You are Cypher, an analytical cybernetic AI.",
            user_text="Initiate system diagnostics.",
        )
        r_cypher = await send_request(client, msgs_cypher, tools)
        slot_cypher = await get_slot_info(client)
        print(f"  Cypher Turn 1 (Persona Divergence): TTFT={r_cypher['ttft_ms']}ms | SlotTokens={slot_cypher.get('n_prompt_tokens')}")

        r_cypher_2 = await send_request(client, msgs_cypher, tools)
        print(f"  Cypher Turn 2 (Cached Cypher): TTFT={r_cypher_2['ttft_ms']}ms")

        # ---------------------------------------------------------
        # Scenario 7: Context Window Truncation Near Limit
        # ---------------------------------------------------------
        print("\n--- Scenario 7: Context Window Rolling Truncation ---")
        long_history = []
        for i in range(25):
            long_history.append({"role": "user", "content": f"Turn {i}: Remember fact number {i} about our robotics project."})
            long_history.append({"role": "assistant", "content": f"Acknowledged fact {i} recorded."})
        msgs_long = engine.context.build_messages(
            system_prompt="You are Lisa.",
            user_text="Summarize all that we discussed.",
            recent_conversation=long_history,
        )
        r_long = await send_request(client, msgs_long, tools)
        slot_long = await get_slot_info(client)
        print(f"  Deep History (25 turns truncated): TTFT={r_long['ttft_ms']}ms | Messages={len(msgs_long)} | SlotTokens={slot_long.get('n_prompt_tokens')}")

        # ---------------------------------------------------------
        # Scenario 8: Concurrent Requests
        # ---------------------------------------------------------
        print("\n--- Scenario 8: Concurrent Requests (Slot Exclusivity) ---")
        t_c0 = time.perf_counter()
        req_a = send_request(client, msgs_t1, tools)
        req_b = send_request(client, msgs_mem, tools)
        res_a, res_b = await asyncio.gather(req_a, req_b)
        dur_conc = (time.perf_counter() - t_c0) * 1000
        print(f"  Concurrent pair duration: {dur_conc:.1f}ms | Req A TTFT={res_a['ttft_ms']}ms | Req B TTFT={res_b['ttft_ms']}ms")

        # ---------------------------------------------------------
        # System Resource Footprint
        # ---------------------------------------------------------
        import subprocess
        print("\n--- Hardware & Resource Utilization ---")
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            vram_used, vram_total, gpu_util = [x.strip() for x in smi.stdout.strip().split(",")]
            print(f"  GPU VRAM: {vram_used} MB / {vram_total} MB ({int(vram_used)/int(vram_total)*100:.1f}%)")
            print(f"  GPU Utilization: {gpu_util}%")
        except Exception as e:
            print(f"  GPU query error: {e}")

        # Save results to json
        out_summary = {
            "scenario_1_cold_ttft": r_cold["ttft_ms"],
            "scenario_1_warm_median_ttft": statistics.median(warm_ttfts),
            "scenario_2_consecutive_ttfts": [t["ttft_ms"] for t in turn_metrics],
            "scenario_3_pass1_ttft": res_pass1["ttft_ms"],
            "scenario_3_pass2_ttft": res_pass2["ttft_ms"],
            "scenario_6_persona_switch_ttft": r_cypher["ttft_ms"],
            "scenario_6_persona_cached_ttft": r_cypher_2["ttft_ms"],
            "scenario_7_truncated_ttft": r_long["ttft_ms"],
        }
        with open("scratch/final_caching_benchmarks.json", "w") as f:
            json.dump(out_summary, f, indent=2)
        print("\nAll benchmark scenarios completed and saved to scratch/final_caching_benchmarks.json")


if __name__ == "__main__":
    asyncio.run(main())
