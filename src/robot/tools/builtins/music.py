"""Music playback tool supporting Spotify, SoundCloud, and local audio files."""

from __future__ import annotations

import asyncio
import os
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

import yaml

from robot.telemetry import get_logger

logger = get_logger("tools.music")


class MusicPlayerService:
    """Singleton-like music player service handling local playback and streaming links."""

    _instance: MusicPlayerService | None = None

    def __init__(self, config_path: str = "config/tools/tools.yaml") -> None:
        self.config_path = Path(config_path)
        self._load_config()
        self._current_process: subprocess.Popen | None = None
        self._is_paused = False
        self._current_track = "None"
        self._current_source = "None"
        self._volume = self.config.get("default_volume", 0.8)

    def _load_config(self) -> None:
        self.config = {
            "default_source": "spotify",
            "local_music_dir": "data/music",
            "default_volume": 0.8,
        }
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    if "music" in data:
                        self.config.update(data["music"])
            except Exception as e:
                logger.warning("music_config_load_failed", error=str(e))

    @classmethod
    def get_instance(cls) -> MusicPlayerService:
        if cls._instance is None:
            cls._instance = MusicPlayerService()
        return cls._instance

    def play_local(self, query: str) -> dict[str, Any]:
        """Search local music directories and play matching audio file."""
        music_dir = Path(self.config.get("local_music_dir", "data/music"))
        music_dir.mkdir(parents=True, exist_ok=True)

        # Candidate directories to scan
        search_dirs = [music_dir, Path.home() / "Music"]
        valid_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

        found_files = []
        for s_dir in search_dirs:
            if s_dir.exists():
                for f in s_dir.rglob("*"):
                    if f.suffix.lower() in valid_exts:
                        found_files.append(f)

        if not found_files:
            return {
                "status": "error",
                "source": "local",
                "message": f"No local audio files found in {music_dir} or ~/Music. Place .mp3 or .wav files there.",
            }

        # Fuzzy match query or pick first if generic
        clean_q = query.strip().lower()
        matched = None
        if clean_q and clean_q not in {"music", "song", "something", "any"}:
            for f in found_files:
                if clean_q in f.stem.lower():
                    matched = f
                    break

        if not matched:
            matched = found_files[0]

        # Stop any existing playback
        self.stop()

        # Play in background using paplay (for wav) or mpv/ffplay or sounddevice
        try:
            if matched.suffix.lower() == ".wav" and os.path.exists("/usr/bin/paplay"):
                cmd = ["paplay", str(matched)]
                self._current_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            elif os.path.exists("/usr/bin/mpv"):
                cmd = ["mpv", "--no-video", f"--volume={int(self._volume * 100)}", str(matched)]
                self._current_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                # Use Python torchaudio + sounddevice in thread
                import sounddevice as sd
                import torchaudio
                waveform, sr = torchaudio.load(str(matched))
                audio_np = waveform.numpy().T
                sd.play(audio_np * self._volume, sr)
        except Exception as e:
            logger.error("local_music_playback_failed", error=str(e), file=str(matched))
            return {"status": "error", "message": f"Failed to play {matched.name}: {e}"}

        self._current_track = matched.stem
        self._current_source = "local"
        self._is_paused = False
        return {
            "status": "playing",
            "source": "local",
            "track": matched.name,
            "message": f"Now playing local audio track: '{matched.stem}'.",
        }

    def play_spotify(self, query: str) -> dict[str, Any]:
        """Launch Spotify web player or app with query."""
        clean_q = query.strip()
        encoded = urllib.parse.quote(clean_q)
        url = f"https://open.spotify.com/search/{encoded}"
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning("spotify_browser_launch_failed", error=str(e))

        self._current_track = clean_q
        self._current_source = "spotify"
        return {
            "status": "playing",
            "source": "spotify",
            "query": clean_q,
            "url": url,
            "message": f"Playing '{clean_q}' on Spotify.",
        }

    def play_soundcloud(self, query: str) -> dict[str, Any]:
        """Launch SoundCloud search/stream with query."""
        clean_q = query.strip()
        encoded = urllib.parse.quote(clean_q)
        url = f"https://soundcloud.com/search?q={encoded}"
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning("soundcloud_browser_launch_failed", error=str(e))

        self._current_track = clean_q
        self._current_source = "soundcloud"
        return {
            "status": "playing",
            "source": "soundcloud",
            "query": clean_q,
            "url": url,
            "message": f"Playing '{clean_q}' on SoundCloud.",
        }

    def stop(self) -> dict[str, Any]:
        """Stop local playback."""
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
            self._current_process = None

        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        self._is_paused = False
        track = self._current_track
        self._current_track = "None"
        return {"status": "stopped", "track": track, "message": "Music playback stopped."}

    def pause(self) -> dict[str, Any]:
        """Pause playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._is_paused = True
        return {"status": "paused", "track": self._current_track, "message": "Music playback paused."}

    def resume(self) -> dict[str, Any]:
        """Resume playback."""
        self._is_paused = False
        return {"status": "resumed", "track": self._current_track, "message": "Resumed music playback."}

    def set_volume(self, volume: float) -> dict[str, Any]:
        """Adjust playback volume between 0.0 and 1.0."""
        self._volume = max(0.0, min(1.0, float(volume)))
        return {
            "status": "volume_set",
            "volume": self._volume,
            "message": f"Volume set to {int(self._volume * 100)}%.",
        }
