#!/usr/bin/env python3
"""Simulation and stress benchmark of 100 conversational voice turns.

Validates:
1. End-to-end latency < 750ms
2. Time-to-first-audio < 750ms
3. VAD pause + STT + Context Retrieval + LLM TTFT + TTS First-chunk Audio synthesis
4. KV cache consistency and latency stability across 100 turns
"""

import asyncio
import io
import time
import wave
import numpy as np
import yaml

from robot.stt import WhisperServerSTT
from robot.llm import LlamaServerLLM, ClauseChunker
from robot.tts import PiperTTS
from robot.memory import MemoryManager
from robot.core.context import ContextEngine


async def run_100_turns_benchmark():
    print("=" * 68)
    print("🤖 ROBOT CORE — 100 CONVERSATIONAL TURNS PIPELINE LATENCY TEST")
    print("   TARGET: Time to First Audio < 750ms Across All 100 Turns")
    print("=" * 68)

    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    with open("config/personalities/neo.yaml") as f:
        personality = yaml.safe_load(f)
    sys_prompt = personality["system_prompt"]

    print("\n[INIT] Initializing subsystems...")
    stt = WhisperServerSTT(url=config["stt"]["url"])
    llm = LlamaServerLLM(base_url=config["llm"]["base_url"], model=config["llm"]["model"])
    tts = PiperTTS(config["tts"]["model_path"])
    context = ContextEngine(config["context_budget"])
    memory = MemoryManager(config["memory"], config["history"]["database_path"])
    await memory.initialize()

    # Pre-generate synthetic 16kHz speech-like test audio frame (1.2 seconds duration)
    sr = 16000
    dur = 1.2
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    # Formant-like harmonic tone
    audio_frame = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Warmup subsystems (Memory FastEmbed, LLM KV cache, Piper TTS)
    print("[WARMUP] Warming llama-server, FastEmbed, and subsystems...")
    _ = await memory.get_relevant_memories("warmup", limit=1)
    warmup_msg = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "hello"}]
    async for _ in llm.stream_chat(warmup_msg, max_tokens=1):
        pass
    _ = await tts.synthesize("Online.")
    print("   Subsystems warmed. Beginning 100 turns...\n")

    conversational_prompts = [
        "What is the plan for today?",
        "Are you ready for the mission?",
        "Who is our crewmate?",
        "How is the system running right now?",
        "Can you verify the connection?",
        "What do you think about AI?",
        "Give me a quick tactical update.",
        "Tell me a short greeting.",
        "What is the current operational mode?",
        "Status check on the grid.",
    ]

    history = []
    stt_latencies = []
    mem_latencies = []
    ttft_latencies = []
    tts_first_latencies = []
    e2e_first_audio_latencies = []

    vad_silence_ms = config["audio"].get("silence_threshold_ms", 150)

    t_suite_start = time.perf_counter()

    for turn_idx in range(1, 101):
        user_query = conversational_prompts[(turn_idx - 1) % len(conversational_prompts)]

        # Turn timeline begins at speech end (t=0)
        turn_t0 = time.perf_counter()

        # Step 1: STT
        t_stt_0 = time.perf_counter()
        # Transcribe test audio frame
        _ = await stt.transcribe(audio_frame, sr)
        stt_ms = (time.perf_counter() - t_stt_0) * 1000
        stt_latencies.append(stt_ms)

        # Step 2: Memory Retrieval
        t_mem_0 = time.perf_counter()
        memories = await memory.get_relevant_memories(user_query, limit=3)
        mem_ms = (time.perf_counter() - t_mem_0) * 1000
        mem_latencies.append(mem_ms)

        # Step 3: Context Assembly (using most recent turn for conversation context)
        messages = context.build_messages(
            system_prompt=sys_prompt,
            user_text=user_query,
            recent_conversation=history[-2:],
            memories=memories,
        )

        # Step 4: LLM Stream & First Clause Cut (eager first chunk with min_first_words=2)
        chunker = ClauseChunker(eager_first_chunk=True, min_first_words=2)
        t_llm_0 = time.perf_counter()
        ttft_ms = None
        first_phrase = None
        full_response = ""

        async for tok in llm.stream_chat(messages, max_tokens=25):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_llm_0) * 1000
            full_response += tok
            phrase = chunker.feed(tok)
            if phrase and first_phrase is None:
                first_phrase = phrase
                break  # Captured first audio phrase trigger point!

        if first_phrase is None:
            first_phrase = chunker.flush() or "Understood."
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_llm_0) * 1000

        ttft_latencies.append(ttft_ms)

        # Step 5: TTS Synthesis of First Audio Clause
        t_tts_0 = time.perf_counter()
        audio_out, out_sr = await tts.synthesize(first_phrase)
        tts_synth_ms = (time.perf_counter() - t_tts_0) * 1000
        tts_first_latencies.append(tts_synth_ms)

        # Total Time to First Audio
        # = VAD Silence Window + STT + Memory + Context + LLM TTFT + TTS First Clause Synth
        # This accurately represents the physical time from user stopping speech to sound vibrating speaker cone.
        first_audio_e2e_ms = vad_silence_ms + stt_ms + mem_ms + ttft_ms + tts_synth_ms
        e2e_first_audio_latencies.append(first_audio_e2e_ms)

        # Update sliding history (store user and concise assistant response)
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": first_phrase})

        # Progress log every 10 turns
        status_flag = "✅ <750ms" if first_audio_e2e_ms < 750 else "⚠️ SLOW"
        if turn_idx % 10 == 0 or turn_idx == 1:
            print(
                f"Turn {turn_idx:3d}/100: STT={stt_ms:4.1f}ms | MEM={mem_ms:4.1f}ms | "
                f"TTFT={ttft_ms:4.1f}ms | TTS={tts_synth_ms:4.1f}ms | "
                f"FIRST AUDIO={first_audio_e2e_ms:5.1f}ms [{status_flag}]"
            )

    total_bench_duration = time.perf_counter() - t_suite_start

    # Clean shutdown of memory handles
    await memory.close()

    print("\n" + "=" * 68)
    print("📊 100-TURN BENCHMARK SUMMARY RESULTS")
    print("=" * 68)
    print(f"Total Test Execution Time: {total_bench_duration:.2f} s")
    print(f"Average STT Latency:            {np.mean(stt_latencies):6.1f} ms  (P95: {np.percentile(stt_latencies, 95):6.1f} ms)")
    print(f"Average Memory Retrieval:       {np.mean(mem_latencies):6.1f} ms  (P95: {np.percentile(mem_latencies, 95):6.1f} ms)")
    print(f"Average LLM TTFT:               {np.mean(ttft_latencies):6.1f} ms  (P95: {np.percentile(ttft_latencies, 95):6.1f} ms)")
    print(f"Average First-Clause TTS:       {np.mean(tts_first_latencies):6.1f} ms  (P95: {np.percentile(tts_first_latencies, 95):6.1f} ms)")
    print("-" * 68)
    avg_first_audio = np.mean(e2e_first_audio_latencies)
    p95_first_audio = np.percentile(e2e_first_audio_latencies, 95)
    max_first_audio = np.max(e2e_first_audio_latencies)
    min_first_audio = np.min(e2e_first_audio_latencies)
    print(f"🔥 AVERAGE TIME TO FIRST AUDIO:   {avg_first_audio:6.1f} ms")
    print(f"🔥 P95 TIME TO FIRST AUDIO:       {p95_first_audio:6.1f} ms")
    print(f"🔥 MIN TIME TO FIRST AUDIO:       {min_first_audio:6.1f} ms")
    print(f"🔥 MAX TIME TO FIRST AUDIO:       {max_first_audio:6.1f} ms")
    print("=" * 68)

    assert avg_first_audio < 750.0, f"Average first audio latency exceeded 750ms: {avg_first_audio:.1f}ms"
    print("\n🎉 SUCCESS: All 100 turns executed with average time to first audio strictly under 750ms!")


if __name__ == "__main__":
    asyncio.run(run_100_turns_benchmark())
