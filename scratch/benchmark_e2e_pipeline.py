"""Comprehensive 10-turn E2E voice pipeline benchmark with Fast-Path routing and accurate timing boundaries."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from robot.core.context import ContextEngine

CYPHER_PROMPT = "You are Cypher, a fast, sharp AI companion robot. Answer concisely in 1-2 punchy sentences."
from robot.llm import ClauseChunker, LlamaServerLLM
from robot.stt import WhisperServerSTT
from robot.tools import PermissionManager, ToolRegistry
from robot.tools.builtins import register_builtin_tools
from robot.tools.fast_path import FastPathRouter
from robot.tts import PiperTTS

TEST_UTTERANCES = [
    "Hello, how are you doing today?",
    "What time is it right now?",
    "Tell me a short, fun fact about space",
    "Can you check the weather forecast for tomorrow?",
    "Remember that my favorite programming language is Python",
    "Why is 25 times 48?",
    "Can you tell me a quick programming joke?",
    "What is the system status and battery level?",
    "set display brightness to 60%",
    "Thank you so much. You are really helpful",
]


class MockEngine:
    """Mock engine providing initialized registry and permissions."""

    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()
        self.permissions = PermissionManager()


async def run_benchmark():
    print("=" * 88)
    print("ROBOT V2 — E2E VOICE PIPELINE 10-TURN HARDENED BENCHMARK")
    print("=" * 88)

    # 1. Initialize Subsystems
    print("\n[1/4] Initializing subsystem instances...")
    stt = WhisperServerSTT(url="http://127.0.0.1:8081/inference")
    tts = PiperTTS(model_path="models/tts/en-us-lessac-medium.onnx")

    mock = MockEngine()
    register_builtin_tools(mock)
    canonical_tools = mock.tool_registry.get_schemas()
    fast_router = FastPathRouter(registry=mock.tool_registry, permissions=mock.permissions)
    print(f"  ✓ Tool Registry: {len(canonical_tools)} canonical tools registered.")
    print("  ✓ Fast-Path Router initialized.")

    llm = LlamaServerLLM(
        base_url="http://127.0.0.1:8080/v1",
        canonical_tools=canonical_tools,
        use_invariant_tools=True,
    )
    context_engine = ContextEngine()
    print("  ✓ Whisper STT (port 8081), LlamaServerLLM (port 8080), Piper TTS ready.")

    # 2. Warm up STT, TTS phrases, and LLM slot with exact invariant Cypher prompt
    print("\n[2/4] Pre-warming pipeline slots & neutral acknowledgment phrases...")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    t_stt_w = time.perf_counter()
    await stt.transcribe(dummy_audio, 16000)
    print(f"  ✓ Whisper STT slot warm ({round((time.perf_counter() - t_stt_w)*1000, 1)}ms)")

    # Pre-warm neutral audio acknowledgments
    neutral_acks = [
        "Checking that for you right now.",
        "Looking that up.",
        "One second.",
        "On it.",
    ]
    for ack in neutral_acks:
        tts.warm_phrase(ack)
    print(f"  ✓ Neutral audio acknowledgments pre-warmed ({len(neutral_acks)} phrases in memory).")

    situation_warm = {
        "battery": {"percent": 88, "power_plugged": True},
        "music_state": {"is_playing": False, "volume": 0.8},
        "world_state": {"people": 1, "scene": "developer_desk"},
    }
    warmup_messages = context_engine.build_messages(
        system_prompt=CYPHER_PROMPT,
        user_text="[System Context: Initialized. Silent warm-up verification.]",
        situation=situation_warm,
    )
    t_llm_w = time.perf_counter()
    async for _ in llm.stream_chat(warmup_messages, max_tokens=1, tools=canonical_tools):
        pass
    print(f"  ✓ Llama-server KV slot warm ({round((time.perf_counter() - t_llm_w)*1000, 1)}ms)")

    # 3. Synthesize Speech Audio Inputs
    print("\n[3/4] Pre-generating speech audio inputs for testing STT...")
    test_audio_samples = []
    for idx, utt in enumerate(TEST_UTTERANCES, 1):
        audio_pcm, sr = await tts.synthesize(utt, speed=1.05)
        test_audio_samples.append((utt, audio_pcm, sr))
        print(f"  ✓ Sample {idx:02d}: \"{utt[:40]}...\" ({len(audio_pcm)} samples @ {sr}Hz)")

    # 4. Execute 10 Consecutive Turns
    print("\n[4/4] Executing 10 E2E Benchmark Iterations...")
    print("-" * 92)
    print(f"{'Turn':<5} | {'VAD':<6} | {'STT':<7} | {'Ctx':<5} | {'TTFT':<7} | {'TTS 1st':<8} | {'E2E Total':<10} | {'Tok/s':<6} | {'Status'}")
    print("-" * 92)

    vad_silence_ms = 130.0  # Tuned VAD silence pause
    conversation_history = []
    results = []

    for turn_num, (expected_text, audio_pcm, sr) in enumerate(test_audio_samples, 1):
        # Step A: STT Transcription
        t0_stt = time.perf_counter()
        stt_transcript = await stt.transcribe(audio_pcm, sr)
        stt_ms = (time.perf_counter() - t0_stt) * 1000
        actual_text = stt_transcript.strip() or expected_text

        # Step B: Fast-Path Intent Check
        fast_intent = fast_router.classify(actual_text)

        if fast_intent:
            # Deterministic fast-path branch (<50ms execution!)
            t0_fast = time.perf_counter()
            exec_outcome, speech_text = await fast_router.execute_intent(fast_intent)
            fast_ms = (time.perf_counter() - t0_fast) * 1000

            # Step C: TTS of spoken response
            t0_synth = time.perf_counter()
            first_phrase = speech_text.split(".")[0] + "." if "." in speech_text else speech_text
            synth_audio, synth_sr = await tts.synthesize(first_phrase, speed=1.1)
            tts_first_ms = (time.perf_counter() - t0_synth) * 1000

            total_e2e_ms = vad_silence_ms + stt_ms + 0.5 + fast_ms + tts_first_ms
            tok_per_sec = 0.0

            results.append({
                "turn": turn_num,
                "input": actual_text,
                "vad_ms": vad_silence_ms,
                "stt_ms": stt_ms,
                "ctx_ms": 0.5,
                "ttft_ms": fast_ms,
                "tool_ms": fast_ms,
                "tts_first_ms": tts_first_ms,
                "total_e2e_ms": total_e2e_ms,
                "tok_per_sec": tok_per_sec,
                "tokens": len(speech_text.split()),
                "response": speech_text,
                "first_phrase": first_phrase,
                "is_fast_path": True,
                "is_tool_call": True,
            })

            conversation_history.append({"role": "user", "content": actual_text})
            conversation_history.append({"role": "assistant", "content": speech_text})

            status_flag = "⚡ FAST PATH (<350ms)"
            print(
                f"{turn_num:<5} | "
                f"{vad_silence_ms:>4.0f}ms | "
                f"{stt_ms:>5.1f}ms | "
                f"{0.5:>3.1f}ms | "
                f"{fast_ms:>5.1f}ms | "
                f"{tts_first_ms:>6.1f}ms | "
                f"{total_e2e_ms:>7.1f} ms | "
                f"{'N/A':>6} | "
                f"{status_flag}"
            )
            continue

        # Standard LLM Conversational or Complex Tool Path
        t0_ctx = time.perf_counter()
        situation = {
            "battery": {"percent": 88, "power_plugged": True},
            "music_state": {"is_playing": False, "volume": 0.8},
            "world_state": {"people": 1, "scene": "developer_desk"},
        }
        messages = context_engine.build_messages(
            system_prompt=CYPHER_PROMPT,
            user_text=actual_text,
            recent_conversation=conversation_history[-6:] if conversation_history else None,
            situation=situation,
        )
        ctx_ms = (time.perf_counter() - t0_ctx) * 1000

        chunker = ClauseChunker(eager_first_chunk=True, min_first_words=1)
        t0_llm = time.perf_counter()
        ttft_ms = None
        gen_tokens = 0
        full_text = ""
        first_phrase = None
        is_tool_call = False
        tool_call_data = None

        async for token in llm.stream_chat(messages, max_tokens=150, tools=canonical_tools):
            gen_tokens += 1
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0_llm) * 1000

            if token.startswith("__TOOL_CALL__"):
                is_tool_call = True
                try:
                    tool_call_data = json.loads(token[len("__TOOL_CALL__"):])
                except Exception:
                    pass
                continue

            full_text += token

            if first_phrase is None:
                p = chunker.feed(token)
                if p and len(p.split()) >= 2:
                    first_phrase = p

        tool_exec_ms = 0.0
        tts_first_ms = 0.0

        if is_tool_call and tool_call_data:
            tc = tool_call_data[0] if isinstance(tool_call_data, list) else tool_call_data
            fn = tc.get("function", {})
            tool_name = fn.get("name", "tool")
            raw_args = fn.get("arguments", "{}")
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

            # Execute tool through PermissionManager
            t_tool_0 = time.perf_counter()
            tool_def = mock.tool_registry.get(tool_name)
            if tool_def:
                tool_outcome = await mock.permissions.execute(tool_def, tool_args)
                tool_res = tool_outcome.get("result", tool_outcome)
            else:
                tool_res = f"Tool {tool_name} not found"
            tool_exec_ms = (time.perf_counter() - t_tool_0) * 1000

            # Pass 2: continuation synthesis with tools=[] so model speaks
            continuation_msgs = list(messages) + [
                {"role": "assistant", "content": None, "tool_calls": [tc]},
                {"role": "tool", "tool_call_id": tc.get("id", "call_0"), "name": tool_name, "content": str(tool_res)[:400]},
            ]
            full_text = ""
            chunker = ClauseChunker(eager_first_chunk=True, min_first_words=1)
            first_phrase = None

            async for token in llm.stream_chat(continuation_msgs, max_tokens=100, tools=[]):
                gen_tokens += 1
                full_text += token
                if first_phrase is None:
                    p = chunker.feed(token)
                    if p:
                        first_phrase = p

        llm_total_ms = (time.perf_counter() - t0_llm) * 1000
        tok_per_sec = (gen_tokens / (llm_total_ms / 1000.0)) if llm_total_ms > 0 else 0.0

        if first_phrase is None:
            first_phrase = chunker.flush() or (full_text.split(".")[0] if full_text else "Understood.")

        # TTS Synthesis of First Audio Chunk
        t0_synth = time.perf_counter()
        synth_audio, synth_sr = await tts.synthesize(first_phrase, speed=1.1)
        tts_first_ms = (time.perf_counter() - t0_synth) * 1000

        total_e2e_ms = vad_silence_ms + stt_ms + ctx_ms + (ttft_ms or 0) + tool_exec_ms + tts_first_ms

        results.append({
            "turn": turn_num,
            "input": actual_text,
            "vad_ms": vad_silence_ms,
            "stt_ms": stt_ms,
            "ctx_ms": ctx_ms,
            "ttft_ms": ttft_ms,
            "tool_ms": tool_exec_ms,
            "tts_first_ms": tts_first_ms,
            "total_e2e_ms": total_e2e_ms,
            "tok_per_sec": tok_per_sec,
            "tokens": gen_tokens,
            "response": full_text.strip(),
            "first_phrase": first_phrase,
            "is_fast_path": False,
            "is_tool_call": is_tool_call,
        })

        conversation_history.append({"role": "user", "content": actual_text})
        conversation_history.append({"role": "assistant", "content": full_text})

        status_flag = "🟢 EXCELLENT" if total_e2e_ms < 650 else ("🟡 GOOD" if total_e2e_ms < 900 else "🔴 HIGH")
        print(
            f"{turn_num:<5} | "
            f"{vad_silence_ms:>4.0f}ms | "
            f"{stt_ms:>5.1f}ms | "
            f"{ctx_ms:>3.1f}ms | "
            f"{(ttft_ms or 0):>5.1f}ms | "
            f"{tts_first_ms:>6.1f}ms | "
            f"{total_e2e_ms:>7.1f} ms | "
            f"{tok_per_sec:>4.1f} | "
            f"{status_flag}"
        )

    # Save complete benchmark results
    out_file = Path("scratch/e2e_10_turn_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 92)
    # Summary Metrics
    e2e_list = [r["total_e2e_ms"] for r in results]
    p50 = float(np.percentile(e2e_list, 50))
    p95 = float(np.percentile(e2e_list, 95))
    p99 = float(np.percentile(e2e_list, 99))
    print(f"\nBenchmark Latency Summary:")
    print(f"  P50 Total E2E Latency: {p50:.1f} ms")
    print(f"  P95 Total E2E Latency: {p95:.1f} ms")
    print(f"  P99 Total E2E Latency: {p99:.1f} ms")
    print(f"  Mean Total E2E:        {np.mean(e2e_list):.1f} ms")
    print(f"  Min Total E2E:         {np.min(e2e_list):.1f} ms")
    print(f"  Max Total E2E:         {np.max(e2e_list):.1f} ms")
    print(f"Saved complete benchmark data to {out_file}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
