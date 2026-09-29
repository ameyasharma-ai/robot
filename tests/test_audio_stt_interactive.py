"""Interactive STT verification test script.

Records audio from microphone, plays it back through the speakers,
sends it to whisper-server for transcription, and prints the result.
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
import sounddevice as sd

from robot.stt import WhisperServerSTT


def record_audio(duration: float = 4.0, sample_rate: int = 16000) -> np.ndarray:
    print(f"\n🎙️  Recording {duration:.1f} seconds of speech... Speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten()


def playback_audio(audio: np.ndarray, sample_rate: int = 16000) -> None:
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    print(f"📊 Audio Signal Stats -> Peak: {peak:.4f}, RMS: {rms:.4f}")
    if peak > 1e-4:
        audio = audio * min(0.9 / peak, 8.0)
    print("🔊 Playing back normalized audio...")
    sd.play(audio.reshape(-1, 1), samplerate=sample_rate)
    sd.wait()
    print("✅ Playback finished.")


async def main() -> None:
    stt = WhisperServerSTT(url="http://127.0.0.1:8081/inference")
    is_healthy = await stt.health_check()
    if not is_healthy:
        print("❌ whisper-server not detected on http://127.0.0.1:8081.")
        print("Please ensure whisper-server is running.")
        return

    sample_rate = 16000
    audio = record_audio(duration=4.0, sample_rate=sample_rate)
    playback_audio(audio, sample_rate=sample_rate)

    print("\n⏳ Sending audio to whisper-server for transcription...")
    start_t = time.perf_counter()
    transcript = await stt.transcribe(audio, sample_rate=sample_rate)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    print(f"\n{'='*50}")
    print(f"⏱️  Transcription Latency: {elapsed_ms:.1f} ms")
    print(f"📝 Transcribed Text: \"{transcript}\"")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
