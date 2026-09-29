"""Diagnostic test: Record audio exactly through the engine's capture and VAD pipeline,
save the raw and processed WAV files to disk, play back to verify human intelligibility,
and test transcription via whisper-server with full diagnostics.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from robot.audio.capture import AudioCapture
from robot.audio.vad import VoiceActivityDetector
from robot.stt import WhisperServerSTT, clean_transcript


async def run_diagnostics():
    out_dir = Path("/home/ameya/robot/data/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    capture = AudioCapture(sample_rate=sample_rate, frame_size=512, pre_roll_ms=600)
    vad = VoiceActivityDetector(
        sample_rate=sample_rate,
        normal_threshold=0.40,
        barge_in_threshold=0.60,
        echo_guard_threshold=0.60,
        energy_threshold=0.002,
        barge_in_energy=0.006,
    )
    stt = WhisperServerSTT(url="http://127.0.0.1:8081/inference")

    loop = asyncio.get_running_loop()
    queue = await capture.start(loop)

    print("\n" + "=" * 60)
    print("🎙️  DIAGNOSTIC RECORDING: SPEAK A SENTENCE INTO YOUR MIC NOW")
    print("=" * 60)

    audio_buffer: list[float] = []
    is_speaking = False
    silence_frames = 0
    silence_threshold = int(sample_rate * 0.65 / 512)

    while True:
        frame = await queue.get()
        clean_chunk, rms = capture.process_frame(frame)
        is_voice, prob = vad.detect(clean_chunk, rms, is_bot_speaking=False, in_echo_guard=False)

        if is_voice:
            if not is_speaking:
                print("🟢 Speech detected! Recording...")
                is_speaking = True
                audio_buffer = list(capture.get_pre_roll())

            audio_buffer.extend(frame.flatten().tolist())
            silence_frames = 0
        else:
            if is_speaking:
                audio_buffer.extend(frame.flatten().tolist())
                silence_frames += 1
                if silence_frames >= silence_threshold:
                    print("🛑 End of speech detected. Processing captured audio...")
                    break

    await capture.stop()

    audio_np = np.array(audio_buffer, dtype=np.float32)
    duration = len(audio_np) / sample_rate
    raw_peak = float(np.max(np.abs(audio_np)))
    raw_rms = float(np.sqrt(np.mean(audio_np**2)))

    # Save raw audio as delivered by capture
    raw_path = out_dir / "captured_raw.wav"
    with wave.open(str(raw_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int16_raw = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(int16_raw.tobytes())

    # Apply STT preprocessing
    processed_pcm = audio_np - np.mean(audio_np)
    peak = float(np.max(np.abs(processed_pcm)))
    if peak > 1e-4:
        gain = min(0.9 / peak, 8.0)
        processed_pcm = processed_pcm * gain
    else:
        gain = 1.0

    proc_path = out_dir / "captured_processed_stt.wav"
    with wave.open(str(proc_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int16_proc = (processed_pcm * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(int16_proc.tobytes())

    print("\n📊 Audio Capture Diagnostics:")
    print(f" • Duration:           {duration:.2f} seconds")
    print(f" • Raw Peak Energy:    {raw_peak:.4f}")
    print(f" • Raw RMS Energy:     {raw_rms:.4f}")
    print(f" • Preprocessing Gain: {gain:.2f}x")
    print(f" • Saved Raw:          {raw_path}")
    print(f" • Saved Processed:    {proc_path}")

    print("\n🔊 Playing back the EXACT audio sent to Whisper STT...")
    sd.play(processed_pcm.reshape(-1, 1), samplerate=sample_rate)
    sd.wait()

    print("\n⏳ Sending to whisper-server...")
    t0 = time.perf_counter()
    transcription = await stt.transcribe(audio_np, sample_rate)
    t_stt = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 60)
    print(f"⏱️  Transcription Latency: {t_stt:.1f}ms")
    print(f"📝 Whisper Transcription:  \"{transcription}\"")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
