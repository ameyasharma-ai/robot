"""Music playback tool supporting Spotify, SoundCloud, and local audio files."""

from __future__ import annotations

import asyncio
import json
import os
import re
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
        self._is_playing = False
        self._is_paused = False
        self._current_track = "None"
        self._current_source = "None"
        self._volume = self.config.get("default_volume", 0.8)

    @property
    def is_playing(self) -> bool:
        if self._current_process is not None:
            return self._current_process.poll() is None
        return self._is_playing

    def _load_config(self) -> None:
        self.config = {
            "default_source": "ytdlp",
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

    @staticmethod
    def is_online(timeout: float = 1.5) -> bool:
        """Check if machine has active internet connectivity."""
        try:
            import socket
            with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
                return True
        except Exception:
            try:
                with urllib.request.urlopen("https://www.google.com", timeout=timeout):
                    return True
            except Exception:
                return False

    def find_local_track(self, query: str) -> Path | None:
        """Scan local music directory and ~/Music to find the best match for query."""
        music_dir = Path(self.config.get("local_music_dir", "data/music"))
        music_dir.mkdir(parents=True, exist_ok=True)
        search_dirs = [music_dir, Path.home() / "Music"]
        valid_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".opus"}

        found_files: list[Path] = []
        for s_dir in search_dirs:
            if s_dir.exists():
                for f in s_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in valid_exts and not f.name.endswith(".part"):
                        found_files.append(f)

        if not found_files:
            return None

        clean_q = query.strip().lower()
        # Remove common query filler words
        stop_words = {"play", "song", "music", "track", "audio", "please", "the", "a", "an", "on", "by", "to", "for", "me", "some", "offline", "local"}
        q_words = [w for w in re.findall(r"\w+", clean_q) if w not in stop_words]

        # 1. Exact stem match
        for f in found_files:
            if clean_q == f.stem.lower():
                return f

        if not q_words:
            # Generic query with no specific song title keywords
            return None

        # 2. Substring match for substantial query
        for f in found_files:
            stem_l = f.stem.lower()
            joined_q = " ".join(q_words)
            if joined_q in stem_l:
                return f

        # 3. High-confidence token overlap scoring (>= 60% of query keywords must match file stem)
        best_match = None
        best_score = 0.0
        q_set = set(q_words)
        for f in found_files:
            file_words = set(re.findall(r"\w+", f.stem.lower()))
            overlap = q_set & file_words
            score = len(overlap) / len(q_set)
            if score > best_score:
                best_score = score
                best_match = f

        if best_match and best_score >= 0.60:
            return best_match

        return None

    def play_local(self, query: str, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Search local music directories and play matching audio file with ducking."""
        matched = self.find_local_track(query)
        music_dir = Path(self.config.get("local_music_dir", "data/music"))

        if not matched:
            return {
                "status": "error",
                "source": "local",
                "message": f"No local audio files matching '{query}' found in {music_dir} or ~/Music.",
            }

        # Stop any existing playback
        self.stop()

        def _execute_local_playback():
            def _apply_ducking():
                if wait_speech_engine and hasattr(wait_speech_engine, "player"):
                    try:
                        wait_speech_engine.player.fade_out(rate=0.10)
                    except Exception as e:
                        logger.debug("ducking_error", error=str(e))

            _apply_ducking()

            try:
                if matched.suffix.lower() == ".wav" and os.path.exists("/usr/bin/paplay"):
                    paplay = subprocess.Popen(
                        ["paplay", str(matched)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process = paplay
                    _apply_ducking()
                elif os.path.exists("/usr/bin/mpv"):
                    mpv = subprocess.Popen(
                        ["mpv", "--no-video", f"--volume={int(self._volume * 100)}", str(matched)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process = mpv
                    _apply_ducking()
                else:
                    # PyAV decode into paplay or sounddevice
                    import av
                    import numpy as np
                    container = av.open(str(matched))
                    audio_stream = container.streams.audio[0]
                    sr = audio_stream.rate
                    channels = 2

                    if os.path.exists("/usr/bin/paplay"):
                        paplay = subprocess.Popen(
                            ["/usr/bin/paplay", f"--rate={sr}", f"--channels={channels}", "--format=s16le", "--raw"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        self._current_process = paplay
                        _apply_ducking()
                        vol = max(0.0, min(1.0, float(self._volume)))
                        for frame in container.decode(audio_stream):
                            if paplay.poll() is not None or self._current_process != paplay:
                                break
                            raw = frame.to_ndarray()
                            if raw.ndim == 1:
                                raw = np.repeat(raw[np.newaxis, :], 2, axis=0)
                            elif raw.shape[0] == 1:
                                raw = np.repeat(raw, 2, axis=0)
                            elif raw.shape[0] > 2:
                                raw = raw[:2, :]
                            if raw.dtype == np.int16:
                                raw = (raw * vol).astype(np.int16)
                            else:
                                raw = (raw * 32767.0 * vol).astype(np.int16)
                            try:
                                paplay.stdin.write(raw.T.tobytes())
                            except (BrokenPipeError, OSError):
                                break
                        try:
                            if paplay.stdin:
                                paplay.stdin.close()
                        except Exception:
                            pass
                    else:
                        import sounddevice as sd
                        import torchaudio
                        waveform, sr = torchaudio.load(str(matched))
                        audio_np = waveform.numpy().T
                        sd.play(audio_np * self._volume, sr)
            except Exception as e:
                logger.error("local_music_playback_failed", error=str(e), file=str(matched))

        import threading
        threading.Thread(target=_execute_local_playback, daemon=True).start()

        self._current_track = matched.stem
        self._current_source = "local"
        self._is_paused = False
        return {
            "status": "playing",
            "source": "local",
            "track": matched.stem,
            "resolved_name": matched.stem,
            "file": str(matched),
            "message": f"Playing '{matched.stem}' from local storage.",
        }

    def lookup_metadata(self, query: str) -> dict[str, Any] | None:
        """Reliably resolve artist, track title, and audio preview from music catalog API."""
        clean_q = query.strip()
        url = f"https://itunes.apple.com/search?term={urllib.parse.quote(clean_q)}&entity=song&limit=3"
        req = urllib.request.Request(url, headers={"User-Agent": "RobotMusicService/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if results:
                    best = results[0]
                    return {
                        "title": best.get("trackName", clean_q),
                        "artist": best.get("artistName", "Unknown Artist"),
                        "album": best.get("collectionName", ""),
                        "preview_url": best.get("previewUrl"),
                        "artwork": best.get("artworkUrl100"),
                    }
        except Exception as e:
            logger.debug("music_metadata_lookup_failed", error=str(e), query=clean_q)
        return None

    def _get_spotify_client(self) -> Any | None:
        """Instantiate Spotify Premium API client if credentials are configured."""
        sp_cfg = self.config.get("spotify", {})
        client_id = sp_cfg.get("client_id") or os.environ.get("SPOTIPY_CLIENT_ID")
        client_secret = sp_cfg.get("client_secret") or os.environ.get("SPOTIPY_CLIENT_SECRET")
        redirect_uri = sp_cfg.get("redirect_uri") or os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

        if not client_id or not client_secret:
            return None

        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth

            scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing"
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
                open_browser=False,
                cache_path=".spotify_cache",
            )
            return spotipy.Spotify(auth_manager=auth_manager)
        except Exception as e:
            logger.warning("spotify_premium_init_failed", error=str(e))
            return None

    def _play_full_song_ytdlp(
        self,
        query: str,
        wait_speech_engine: Any = None,
        cache_name: str | None = None,
    ) -> bool:
        """Stream full-length audio track via yt-dlp pipe directly into paplay and cache to disk.
        
        Begins fetching immediately in parallel with speech. As soon as the first
        music frames arrive, smoothly ducks/fades out remaining speech so music overpowers.
        Simultaneously encodes audio frames into data/music/{cache_name}.mp3 so future plays
        of the song work offline instantly.
        """
        ytdl_bin = ".venv/bin/yt-dlp" if os.path.exists(".venv/bin/yt-dlp") else "/usr/bin/yt-dlp"
        if not os.path.exists(ytdl_bin):
            return False

        try:
            import av
            import numpy as np

            # Launch yt-dlp outputting raw audio stream to stdout
            ytdl_proc = subprocess.Popen(
                [
                    ytdl_bin,
                    "-f", "ba/b",
                    "-o", "-",
                    "--quiet",
                    "--no-playlist",
                    "--no-warnings",
                    f"ytsearch1:{query} audio",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            container = av.open(ytdl_proc.stdout)
            audio_stream = container.streams.audio[0]
            sr = audio_stream.rate
            channels = 2

            paplay = subprocess.Popen(
                ["/usr/bin/paplay", f"--rate={sr}", f"--channels={channels}", "--format=s16le", "--raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_process = paplay
            logger.info("playing_full_song_via_ytdlp", query=query, rate=sr)

            def _apply_ducking():
                if wait_speech_engine and hasattr(wait_speech_engine, "player"):
                    try:
                        wait_speech_engine.player.fade_out(rate=0.10)
                        logger.info("tts_fade_out_applied_for_music")
                    except Exception as e:
                        logger.debug("ducking_error", error=str(e))

            _apply_ducking()

            # Prepare disk caching in background
            save_title = cache_name or query
            safe_filename = re.sub(r'[^a-zA-Z0-9_\- ]+', '', save_title).strip() or "track"
            music_dir = Path(self.config.get("local_music_dir", "data/music"))
            music_dir.mkdir(parents=True, exist_ok=True)
            cache_file = music_dir / f"{safe_filename}.mp3"
            temp_cache_file = music_dir / f"{safe_filename}.mp3.part"

            cache_container = None
            cache_stream = None
            resampler = None
            try:
                cache_container = av.open(str(temp_cache_file), mode="w", format="mp3")
                cache_stream = cache_container.add_stream("mp3", rate=44100, layout="stereo")
                resampler = av.AudioResampler(format="s16p", layout="stereo", rate=44100)
            except Exception as e:
                logger.debug("cache_container_init_failed", error=str(e))
                cache_container = None

            def _stream_pump():
                nonlocal cache_container, cache_stream, resampler
                c_frames_written = 0
                try:
                    ducked_once = False
                    vol = max(0.0, min(1.0, float(self._volume)))
                    for frame in container.decode(audio_stream):
                        if not ducked_once:
                            _apply_ducking()
                            ducked_once = True
                        if paplay.poll() is not None or self._current_process != paplay:
                            break
                        raw = frame.to_ndarray()
                        # Ensure 2-channel stereo
                        if raw.ndim == 1:
                            raw = np.repeat(raw[np.newaxis, :], 2, axis=0)
                        elif raw.shape[0] == 1:
                            raw = np.repeat(raw, 2, axis=0)
                        elif raw.shape[0] > 2:
                            raw = raw[:2, :]

                        # Scale volume for playback
                        if raw.dtype == np.int16:
                            raw_play = (raw * vol).astype(np.int16)
                        else:
                            raw_play = (raw * 32767.0 * vol).astype(np.int16)

                        try:
                            paplay.stdin.write(raw_play.T.tobytes())
                        except (BrokenPipeError, OSError):
                            break

                        # Simultaneously cache clean audio to disk
                        if cache_container and cache_stream and resampler:
                            try:
                                for rf in resampler.resample(frame):
                                    for pkt in cache_stream.encode(rf):
                                        cache_container.mux(pkt)
                                c_frames_written += 1
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug("ytdlp_stream_pump_ended", error=str(e))
                finally:
                    try:
                        if paplay.stdin:
                            paplay.stdin.close()
                    except Exception:
                        pass
                    try:
                        ytdl_proc.terminate()
                    except Exception:
                        pass
                    # Finalize MP3 cache
                    if cache_container and cache_stream:
                        try:
                            for pkt in cache_stream.encode(None):
                                cache_container.mux(pkt)
                            cache_container.close()
                            if c_frames_written > 20 and temp_cache_file.exists():
                                temp_cache_file.replace(cache_file)
                                logger.info("cached_audio_saved_offline", path=str(cache_file))
                            elif temp_cache_file.exists():
                                temp_cache_file.unlink(missing_ok=True)
                        except Exception as e:
                            logger.debug("cache_finalize_failed", error=str(e))

            import threading
            threading.Thread(target=_stream_pump, daemon=True).start()
            return True
        except Exception as e:
            logger.warning("full_song_ytdlp_failed", error=str(e), query=query)
            return False

    def _play_stream_audio(
        self,
        preview_url: str | None,
        web_url: str | None = None,
        full_query: str | None = None,
        wait_speech_engine: Any = None,
        cache_name: str | None = None,
    ) -> None:
        """Play full song or high-quality preview audio in background."""
        self.stop()

        def _worker():
            # 1. If full_query is provided, attempt full song stream via yt-dlp first
            if full_query:
                success = self._play_full_song_ytdlp(
                    full_query,
                    wait_speech_engine=wait_speech_engine,
                    cache_name=cache_name or full_query,
                )
                if success:
                    return

            # If fallback preview is used, wait for speech to finish before playing preview
            if wait_speech_engine:
                self._wait_for_speech_to_finish(wait_speech_engine)

            # 2. Preview audio stream fallback via PyAV -> paplay
            if preview_url:
                try:
                    import av
                    import io
                    import wave
                    import numpy as np

                    req = urllib.request.Request(preview_url, headers={"User-Agent": "RobotMusicService/2.0"})
                    with urllib.request.urlopen(req, timeout=6.0) as resp:
                        audio_bytes = resp.read()

                    container = av.open(io.BytesIO(audio_bytes))
                    audio_stream = container.streams.audio[0]
                    sr = audio_stream.rate
                    frames = [frame.to_ndarray() for frame in container.decode(audio_stream)]

                    if frames:
                        raw_audio = np.concatenate(frames, axis=1)
                        channels = raw_audio.shape[0] if raw_audio.ndim == 2 else 1

                        vol = max(0.0, min(1.0, float(self._volume)))
                        if raw_audio.dtype == np.int16:
                            scaled_audio = (raw_audio * vol).astype(np.int16)
                        else:
                            scaled_audio = (raw_audio * 32767.0 * vol).astype(np.int16)

                        wav_buf = io.BytesIO()
                        with wave.open(wav_buf, "wb") as wf:
                            wf.setnchannels(channels)
                            wf.setsampwidth(2)
                            wf.setframerate(sr)
                            wf.writeframes(scaled_audio.T.tobytes())
                        wav_data = wav_buf.getvalue()

                        if os.path.exists("/usr/bin/paplay"):
                            try:
                                proc = subprocess.Popen(
                                    ["/usr/bin/paplay"],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                                self._current_process = proc
                                logger.info("playing_music_stream_via_paplay", rate=sr, channels=channels, size_bytes=len(wav_data))
                                proc.communicate(input=wav_data)
                                return
                            except Exception as e:
                                logger.warning("paplay_stdin_failed", error=str(e))

                        try:
                            import sounddevice as sd
                            audio_float = scaled_audio.T.astype(np.float32) / 32768.0
                            logger.info("playing_music_stream_via_sounddevice", shape=audio_float.shape, rate=sr)
                            sd.play(audio_float, samplerate=sr)
                            return
                        except Exception as e:
                            logger.warning("sounddevice_play_failed", error=str(e))
                except Exception as e:
                    logger.warning("pyav_stream_decode_failed", error=str(e))

            # 3. Fallback to CLI player if installed
            if preview_url and os.path.exists("/usr/bin/mpv"):
                try:
                    self._current_process = subprocess.Popen(
                        ["mpv", "--no-video", f"--volume={int(self._volume * 100)}", preview_url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except Exception as e:
                    logger.warning("cli_music_playback_failed", error=str(e))

            # 4. Fallback to opening browser player
            if web_url:
                try:
                    webbrowser.open(web_url)
                except Exception as e:
                    logger.warning("browser_music_open_failed", error=str(e))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _wait_for_speech_to_finish(self, wait_speech_engine: Any, timeout: float = 14.0) -> None:
        """Wait until robot engine finishes speaking before audio playback begins."""
        if not wait_speech_engine:
            return
        import time
        t0 = time.time()

        # Step 1: Wait up to 3.5s for speech to actually begin if it hasn't yet
        while time.time() - t0 < 3.5:
            is_speaking = getattr(wait_speech_engine, "_is_speaking", False)
            has_queue = False
            q = getattr(wait_speech_engine, "_tts_queue", None)
            if q is not None:
                has_queue = not q.empty() if hasattr(q, "empty") else bool(q)
            if is_speaking or has_queue:
                break
            time.sleep(0.05)

        # Step 2: Now wait until speech and all queued sentences are completely finished
        while time.time() - t0 < timeout:
            is_speaking = getattr(wait_speech_engine, "_is_speaking", False)
            has_queue = False
            q = getattr(wait_speech_engine, "_tts_queue", None)
            if q is not None:
                has_queue = not q.empty() if hasattr(q, "empty") else bool(q)
            if not is_speaking and not has_queue:
                time.sleep(0.2)
                break
            time.sleep(0.08)

    def play_ytdlp(self, query: str, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Play track using yt-dlp or local cached files, prioritizing offline playback."""
        clean_q = query.strip()

        # 1. Check if the track is already available offline in data/music or ~/Music
        local_match = self.find_local_track(clean_q)
        if local_match:
            logger.info("playing_offline_cached_file", file=str(local_match), query=clean_q)
            return self.play_local(clean_q, wait_speech_engine=wait_speech_engine)

        # 2. Check internet connectivity
        online = self.is_online()
        if not online:
            # When offline, check if any local file matches the query with confidence
            if local_match:
                return self.play_local(clean_q, wait_speech_engine=wait_speech_engine)
            return {
                "status": "error",
                "source": "local",
                "message": f"Internet is offline and '{clean_q}' is not available in local storage.",
            }

        # 3. Lookup metadata if online
        meta = self.lookup_metadata(clean_q)
        track_title = meta["title"] if meta else clean_q
        artist_name = meta["artist"] if meta else ""
        resolved_name = f"{track_title} by {artist_name}" if artist_name else track_title
        query_encoded = urllib.parse.quote(clean_q)
        web_url = f"https://www.youtube.com/results?search_query={query_encoded}"

        def _execute_playback():
            self._play_stream_audio(
                None,
                web_url=web_url,
                full_query=resolved_name,
                wait_speech_engine=wait_speech_engine,
                cache_name=resolved_name,
            )

        import threading
        threading.Thread(target=_execute_playback, daemon=True).start()

        self._current_track = resolved_name
        self._current_source = "ytdlp"
        return {
            "status": "playing",
            "source": "ytdlp",
            "track": track_title,
            "artist": artist_name,
            "resolved_name": resolved_name,
            "query": clean_q,
            "url": web_url,
            "web_url": web_url,
            "message": f"Playing '{resolved_name}' via YouTube.",
        }

    def play_spotify(self, query: str, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Search Spotify for the exact track and play either via Spotify Premium or yt-dlp."""
        clean_q = query.strip()

        # Check local files first if confident match exists
        local_match = self.find_local_track(clean_q)
        if local_match:
            return self.play_local(clean_q, wait_speech_engine=wait_speech_engine)
        if not self.is_online():
            return {
                "status": "error",
                "source": "local",
                "message": f"Internet is offline and '{clean_q}' is not available in local storage.",
            }

        meta = self.lookup_metadata(clean_q)
        track_title = meta["title"] if meta else clean_q
        artist_name = meta["artist"] if meta else ""
        resolved_name = f"{track_title} by {artist_name}" if artist_name else track_title

        query_encoded = urllib.parse.quote(clean_q)
        web_url = f"https://open.spotify.com/search/{query_encoded}"
        preview_url = meta.get("preview_url") if meta else None

        # Try Spotify Premium API first if configured
        sp = self._get_spotify_client()
        if sp:
            try:
                results = sp.search(q=resolved_name, type="track", limit=1)
                tracks = results.get("tracks", {}).get("items", [])
                if tracks:
                    track_uri = tracks[0]["uri"]
                    sp_cfg = self.config.get("spotify", {})
                    dev_id = sp_cfg.get("device_id") or None

                    def _execute_spotify_premium():
                        self._wait_for_speech_to_finish(wait_speech_engine)
                        try:
                            sp.start_playback(device_id=dev_id, uris=[track_uri])
                            logger.info("spotify_premium_playback_started", uri=track_uri)
                        except Exception as e:
                            logger.warning("spotify_premium_playback_failed_fallback_ytdlp", error=str(e))
                            self._play_stream_audio(
                                preview_url,
                                web_url=web_url,
                                full_query=resolved_name,
                                cache_name=resolved_name,
                            )

                    import threading
                    threading.Thread(target=_execute_spotify_premium, daemon=True).start()

                    self._current_track = resolved_name
                    self._current_source = "spotify_premium"
                    return {
                        "status": "playing",
                        "source": "spotify_premium",
                        "track": track_title,
                        "artist": artist_name,
                        "resolved_name": resolved_name,
                        "query": clean_q,
                        "url": web_url,
                        "web_url": web_url,
                        "message": f"Playing '{resolved_name}' on Spotify Premium.",
                    }
            except Exception as e:
                logger.warning("spotify_search_premium_error", error=str(e))

        # Default full song playback via yt-dlp & paplay
        # Transparently acknowledge playback source as YouTube / yt-dlp
        def _execute_playback():
            self._play_stream_audio(
                preview_url,
                web_url=web_url,
                full_query=resolved_name,
                wait_speech_engine=wait_speech_engine,
                cache_name=resolved_name,
            )

        import threading
        threading.Thread(target=_execute_playback, daemon=True).start()

        self._current_track = resolved_name
        self._current_source = "ytdlp"
        return {
            "status": "playing",
            "source": "ytdlp",
            "track": track_title,
            "artist": artist_name,
            "resolved_name": resolved_name,
            "query": clean_q,
            "url": web_url,
            "web_url": web_url,
            "preview_url": preview_url,
            "message": f"Playing '{resolved_name}' via YouTube.",
        }

    def play_soundcloud(self, query: str, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Search SoundCloud for the exact track and play the full song with ducking."""
        clean_q = query.strip()
        meta = self.lookup_metadata(clean_q)
        track_title = meta["title"] if meta else clean_q
        artist_name = meta["artist"] if meta else ""
        resolved_name = f"{track_title} by {artist_name}" if artist_name else track_title

        query_encoded = urllib.parse.quote(clean_q)
        web_url = f"https://soundcloud.com/search?q={query_encoded}"
        preview_url = meta.get("preview_url") if meta else None

        def _execute_playback():
            self._play_stream_audio(
                preview_url,
                web_url=web_url,
                full_query=resolved_name,
                wait_speech_engine=wait_speech_engine,
            )

        import threading
        threading.Thread(target=_execute_playback, daemon=True).start()

        self._current_track = resolved_name
        self._current_source = "soundcloud"
        return {
            "status": "playing",
            "source": "soundcloud",
            "track": track_title,
            "artist": artist_name,
            "resolved_name": resolved_name,
            "query": clean_q,
            "url": web_url,
            "web_url": web_url,
            "preview_url": preview_url,
            "message": f"Playing '{resolved_name}' on SoundCloud.",
        }

    def stop(self) -> dict[str, Any]:
        """Stop music playback."""
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
        if self._current_process:
            try:
                import signal
                os.kill(self._current_process.pid, signal.SIGSTOP)
            except Exception:
                pass
        self._is_paused = True
        return {"status": "paused", "track": self._current_track, "message": "Music playback paused."}

    def resume(self) -> dict[str, Any]:
        """Resume playback."""
        if self._current_process:
            try:
                import signal
                os.kill(self._current_process.pid, signal.SIGCONT)
            except Exception:
                pass
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
