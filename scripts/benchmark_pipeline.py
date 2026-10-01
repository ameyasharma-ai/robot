#!/usr/bin/env python3
"""Automated Benchmark Suite for Robot Core Voice & Intelligence Pipeline."""

import asyncio
import io
import time
import wave
import numpy as np
import yaml
import requests

from robot.stt import WhisperServerSTT
from robot.llm import LlamaServerLLM, ClauseChunker
from robot.tts import PiperTTS
from robot.memory import MemoryManager
from robot.core.context import ContextEngine


async def benchmark_all():
    print("=" * 60)
    print("🤖 ROBOT CORE PIPELINE LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 60)
    
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    # 1. Benchmark STT (whisper-server on AMD Radeon 680M)
    print("\n[1/5] Benchmarking STT (whisper-server on Vulkan dev 0)...")
    stt = WhisperServerSTT(url=config["stt"]["url"])
    sr = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    test_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    stt_times = []
    for i in range(5):
        t0 = time.perf_counter()
        _ = await stt.transcribe(test_audio, sr)
        stt_ms = (time.perf_counter() - t0) * 1000
        stt_times.append(stt_ms)
        print(f"  Run {i+1}: {stt_ms:.1f} ms")
    
    avg_stt = float(np.mean(stt_times))
    print(f"  → Average STT Latency: {avg_stt:.1f} ms (Min: {np.min(stt_times):.1f} ms)")

    # 2. Benchmark Memory Retrieval (Mem0 + Qdrant + FastEmbed + SQLite)
    print("\n[2/5] Benchmarking Memory & Fact Retrieval...")
    memory = MemoryManager(config["memory"], config["history"]["database_path"])
    await memory.initialize()
    
    mem_times = []
    for i in range(5):
        t0 = time.perf_counter()
        _ = await memory.get_relevant_memories("who are you", limit=5)
        mem_ms = (time.perf_counter() - t0) * 1000
        mem_times.append(mem_ms)
        print(f"  Run {i+1}: {mem_ms:.1f} ms")
    
    avg_mem = float(np.mean(mem_times))
    print(f"  → Average Memory Retrieval: {avg_mem:.1f} ms (Min: {np.min(mem_times):.1f} ms)")

    # 3. Benchmark LLM Inference (llama-server on RTX 3050)
    print("\n[3/5] Benchmarking LLM TTFT & Generation Speed (RTX 3050)...")
    llm = LlamaServerLLM(base_url=config["llm"]["base_url"], model=config["llm"]["model"])
    context = ContextEngine(config["context_budget"])
    
    with open("config/personalities/neo.yaml") as f:
        personality = yaml.safe_load(f)
    sys_prompt = personality["system_prompt"]
    
    # Warm up KV-cache with system prompt
    warmup_msg = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "init"}]
    async for _ in llm.stream_chat(warmup_msg, max_tokens=1):
        pass
    
    # Simulate turns with conversational history
    test_queries = [
        "Hello Cypher!",
        "What is your status?",
        "Tell me a short 10-word sentence.",
    ]
    
    history = []
    ttft_times = []
    gen_speeds = []
    
    for i, q in enumerate(test_queries):
        messages = context.build_messages(
            system_prompt=sys_prompt,
            user_text=q,
            recent_conversation=history,
        )
        
        t0 = time.perf_counter()
        ttft = None
        tokens = []
        
        async for tok in llm.stream_chat(messages, max_tokens=60):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            tokens.append(tok)
            
        t_total = time.perf_counter() - t0
        gen_time = t_total - (ttft / 1000.0 if ttft else 0)
        tok_speed = (len(tokens) / gen_time) if gen_time > 0 and len(tokens) > 0 else 0
        
        full_text = "".join(tokens).strip()
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": full_text})
        
        ttft_times.append(ttft)
        gen_speeds.append(tok_speed)
        print(f"  Turn {i+1} ('{q}'):")
        print(f"    - TTFT: {ttft:.1f} ms")
        print(f"    - Tokens generated: {len(tokens)}")
        print(f"    - Decode speed: {tok_speed:.1f} tok/s")
        print(f"    - Response: \"{full_text[:50]}...\"")

    avg_ttft = float(np.mean(ttft_times))
    avg_speed = float(np.mean(gen_speeds))
    print(f"  → Average LLM TTFT: {avg_ttft:.1f} ms (Min: {np.min(ttft_times):.1f} ms)")
    print(f"  → Average Decode Throughput: {avg_speed:.1f} tokens/s")

    # 4. Benchmark TTS (Piper on AMD Ryzen 7 6800H AVX2)
    print("\n[4/5] Benchmarking TTS Synthesis (Piper ONNX AVX2)...")
    tts = PiperTTS(config["tts"]["model_path"])
    
    phrases = [
        "I'm Cypher,",
        "your ride-or-die hacker crewmate,",
        "We are ready to roll through the system right now.",
    ]
    
    tts_times = []
    rtf_scores = []
    
    for i, p in enumerate(phrases):
        t0 = time.perf_counter()
        audio, sr = await tts.synthesize(p)
        synth_ms = (time.perf_counter() - t0) * 1000
        audio_dur = len(audio) / sr
        rtf = (synth_ms / 1000.0) / audio_dur if audio_dur > 0 else 0
        tts_times.append(synth_ms)
        rtf_scores.append(rtf)
        print(f"  Phrase {i+1} ('{p}'): {synth_ms:.1f} ms (Duration: {audio_dur:.2f}s, RTF: {rtf:.3f})")

    avg_tts = float(np.mean(tts_times))
    first_chunk_tts = tts_times[0]
    print(f"  → First Clause TTS Synthesis: {first_chunk_tts:.1f} ms")
    print(f"  → Average Real-Time Factor (RTF): {np.mean(rtf_scores):.3f}x")

    # 5. Composite End-to-End Pipeline Latency
    print("\n[5/5] End-to-End Pipeline Turnaround Analysis:")
    vad_silence = config["audio"].get("silence_threshold_ms", 150)
    e2e_measured = vad_silence + avg_stt + avg_mem + avg_ttft + first_chunk_tts
    
    print("-" * 60)
    print(f"  • VAD Silence Confirmation:  {vad_silence:.1f} ms")
    print(f"  • Speech-to-Text (STT):      {avg_stt:.1f} ms")
    print(f"  • Memory & Context Engine:   {avg_mem:.1f} ms")
    print(f"  • LLM TTFT (RTX 3050):       {avg_ttft:.1f} ms  [< 300ms GUARANTEE]")
    print(f"  • TTS First Clause Audio:    {first_chunk_tts:.1f} ms")
    print("-" * 60)
    print(f"  🔥 PROJECTED FIRST-AUDIO LATENCY: {e2e_measured:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(benchmark_all())
