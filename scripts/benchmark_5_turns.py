#!/usr/bin/env python3
"""Automated 5-turn benchmark to measure TTFT and prompt eval latency collapse."""

import json
import re
import time
import urllib.request
from pathlib import Path

BASE_API = "http://127.0.0.1:8888"
LOG_PATH = Path("logs/llama-server.log")


def get_latest_llama_log_timing() -> dict:
    """Parse the most recent launch_slot / print_timing block in llama-server.log."""
    if not LOG_PATH.exists():
        return {}
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
    result = {}
    for line in reversed(lines):
        if "prompt eval time =" in line and "prompt_eval_ms" not in result:
            m = re.search(r"prompt eval time =\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
            if m:
                result["prompt_eval_ms"] = float(m.group(1))
                result["prompt_eval_tokens"] = int(m.group(2))
        if "selected slot by LCP similarity" in line and "f_sim_best" not in result:
            m = re.search(r"f_sim_best =\s*([\d\.]+)", line)
            if m:
                result["f_sim_best"] = float(m.group(1))
        if "prompt_eval_ms" in result and "f_sim_best" in result:
            break
    return result


def send_chat_stream(text: str) -> dict:
    """Send text message to /chat/stream and measure TTFT and total response time."""
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{BASE_API}/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_ms = None
    response_tokens = []
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    ev = json.loads(data_str)
                    if ev.get("event") == "token":
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        response_tokens.append(ev.get("token", ""))
                except Exception:
                    pass
    
    total_ms = (time.perf_counter() - t0) * 1000
    if ttft_ms is None:
        ttft_ms = total_ms
        
    full_text = "".join(response_tokens)
    timing = get_latest_llama_log_timing()
    
    return {
        "text": text,
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "response": full_text[:80].strip(),
        "prompt_eval_ms": timing.get("prompt_eval_ms", 0.0),
        "prompt_eval_tokens": timing.get("prompt_eval_tokens", 0),
        "f_sim_best": timing.get("f_sim_best", 0.0),
    }


def main():
    test_queries = [
        "Hi",
        "What's the weather today?",
        "Play some upbeat music",
        "What time is it right now?",
        "Who are you and what can you do?",
    ]
    
    print("=" * 80)
    print("🚀 EXECUTING 5 AUTOMATED RUNTIME LATENCY TESTS")
    print("=" * 80)
    
    results = []
    for i, q in enumerate(test_queries, 1):
        print(f"\n[Test {i}/5] Sending: '{q}'...")
        res = send_chat_stream(q)
        results.append(res)
        print(f"  • TTFT: {res['ttft_ms']} ms")
        print(f"  • Prompt Eval: {res['prompt_eval_ms']} ms ({res['prompt_eval_tokens']} tokens)")
        print(f"  • KV-Cache Match (f_sim_best): {res['f_sim_best']}")
        print(f"  • Total Turn Duration: {res['total_ms']} ms")
        print(f"  • Response snippet: {res['response']}...")
        time.sleep(1.0)
        
    print("\n" + "=" * 80)
    print("📊 LATENCY COLLAPSE REPORT — SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Turn':<6} | {'User Query':<32} | {'TTFT (ms)':<10} | {'Prompt Eval':<12} | {'Eval Tokens':<12} | {'KV Match':<8}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        print(f"Turn {i:<1} | {r['text']:<32} | {r['ttft_ms']:<10.1f} | {r['prompt_eval_ms']:<10.1f}ms | {r['prompt_eval_tokens']:<12} | {r['f_sim_best']:<8}")
    print("=" * 80)


if __name__ == "__main__":
    main()
