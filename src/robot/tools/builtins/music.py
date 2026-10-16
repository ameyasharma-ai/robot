"""Music playback tool supporting Spotify, SoundCloud, and local audio files."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import struct
import subprocess
import threading
import time
import urllib.parse
import wave
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.music")


def ensure_easyeffects_running() -> bool:
    """Ensure the EasyEffects audio processing service is running in background.
    
    If not already running, launches it in service mode so PipeWire audio filters
    and equalizer profiles work reliably without manual user intervention.
    """
    try:
        check = subprocess.run(["pgrep", "-f", "easyeffects"], capture_output=True, text=True, timeout=1.0)
        if check.returncode == 0 and check.stdout.strip():
            return True
        logger.info("starting_easyeffects_service_daemon")
        subprocess.Popen(
            ["easyeffects", "--gapplication-service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(0.4)
        return True
    except Exception as e:
        logger.warning("ensure_easyeffects_running_failed", error=str(e))
        return False


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
        self._history: list[dict[str, Any]] = []
        # Event set when music audio frames actually begin playing (for TTS coordination)
        self._music_started_event = asyncio.Event()
        self._music_active = False
        self._active_preset = "robot"
        # Ensure EasyEffects audio daemon is active in background from startup
        ensure_easyeffects_running()

    @property
    def is_playing(self) -> bool:
        if self._current_process is not None:
            return self._current_process.poll() is None
        return self._is_playing or self._music_active

    def list_local_tracks(self) -> list[str]:
        """Scan local music directory, generated audio directory, and ~/Music and return list of track names."""
        music_dir = Path(self.config.get("local_music_dir", "data/music"))
        gen_dir = Path("data/generated_audio")
        search_dirs = [music_dir, gen_dir, Path.home() / "Music"]
        valid_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".opus"}
        tracks = []
        for s_dir in search_dirs:
            if s_dir.exists():
                for f in s_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in valid_exts and not f.name.endswith(".part"):
                        tracks.append(f.stem)
        return sorted(list(set(tracks)))

    def get_status(self) -> dict[str, Any]:
        """Return the current live music playback state, history, and last played song."""
        active = self.is_playing
        last = self._history[0] if self._history else None
        current = self._current_track if (active or self._is_paused) else (last["resolved_name"] if last else "None")
        source = self._current_source if (active or self._is_paused) else (last["source"] if last else "None")
        local_tracks = self.list_local_tracks()
        return {
            "is_playing": active,
            "is_paused": self._is_paused,
            "current_track": current,
            "source": source,
            "volume": self._volume,
            "equalizer_preset": getattr(self, "_active_preset", "robot"),
            "history": self.get_history(5),
            "last_played": last,
            "local_collection": local_tracks[:30],
            "total_local_songs": len(local_tracks),
        }

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
        """Scan local music directory, generated audio directory, and ~/Music to find the best match for query."""
        music_dir = Path(self.config.get("local_music_dir", "data/music"))
        music_dir.mkdir(parents=True, exist_ok=True)
        gen_dir = Path("data/generated_audio")
        gen_dir.mkdir(parents=True, exist_ok=True)
        search_dirs = [music_dir, gen_dir, Path.home() / "Music"]
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

        if best_match and best_score >= 0.80:
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
            try:
                if matched.suffix.lower() == ".wav" and os.path.exists("/usr/bin/paplay"):
                    paplay = subprocess.Popen(
                        ["paplay", str(matched)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process = paplay
                elif os.path.exists("/usr/bin/mpv"):
                    mpv = subprocess.Popen(
                        ["mpv", "--no-video", f"--volume={int(self._volume * 100)}", str(matched)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process = mpv
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
                            ["/usr/bin/paplay", f"--rate={sr}", f"--channels={channels}", "--format=s16le", "--raw", "--latency-msec=300"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        self._current_process = paplay
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
                            vol = max(0.0, min(1.0, float(self._volume)))
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

        threading.Thread(target=_execute_local_playback, daemon=True).start()

        self._current_track = matched.stem
        self._current_source = "local"
        self._is_paused = False
        self._record_history(matched.stem, "", matched.stem, query, "local")
        return {
            "status": "playing",
            "source": "local",
            "track": matched.stem,
            "resolved_name": matched.stem,
            "file": str(matched),
            "message": f"Playing '{matched.stem}' from local storage.",
        }

    @staticmethod
    def _parse_query_target(clean_q: str) -> tuple[str, str]:
        """Split 'title by artist' or 'artist - title' into (title, artist)."""
        stop_lead = {"play", "song", "music", "track", "listen to", "put on"}
        q = clean_q.strip()
        for sl in stop_lead:
            if q.lower().startswith(sl + " "):
                q = q[len(sl) + 1:].strip()
                break

        for sep in [" by ", " from "]:
            if sep in q.lower():
                parts = re.split(re.escape(sep), q, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    return parts[0].strip(), parts[1].strip()

        if " - " in q:
            parts = q.split(" - ", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return parts[1].strip(), parts[0].strip()

        return q, ""

    def _record_history(
        self,
        track: str,
        artist: str,
        resolved_name: str,
        query: str,
        source: str,
    ) -> None:
        """Record played track into recently played history (max 50 tracks)."""
        import datetime
        entry = {
            "track": track,
            "artist": artist,
            "resolved_name": resolved_name,
            "query": query,
            "source": source,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if not self._history or self._history[0].get("resolved_name") != resolved_name:
            self._history.insert(0, entry)
        self._history = self._history[:50]

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

    def _get_spotify_search_client(self) -> Any | None:
        """Instantiate Spotify client using Developer Client Credentials (zero login needed)."""
        sp_cfg = self.config.get("spotify", {})
        client_id = sp_cfg.get("client_id") or os.environ.get("SPOTIPY_CLIENT_ID")
        client_secret = sp_cfg.get("client_secret") or os.environ.get("SPOTIPY_CLIENT_SECRET")
        if not client_id or not client_secret:
            return None
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            return spotipy.Spotify(auth_manager=auth_manager)
        except Exception as e:
            logger.debug("spotify_search_client_init_failed", error=str(e))
            return None

    def _get_spotify_oauth_client(self) -> Any | None:
        """Instantiate Spotify Premium API client ONLY if an authorized token exists on disk."""
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
            token_info = auth_manager.validate_token(auth_manager.cache_handler.get_cached_token())
            if not token_info:
                # Token missing or expired without refresh capability; NEVER call get_access_token() in headless mode
                return None
            return spotipy.Spotify(auth_manager=auth_manager)
        except Exception as e:
            logger.debug("spotify_oauth_client_init_failed", error=str(e))
            return None

    def _get_spotify_client(self) -> Any | None:
        """Alias for backward compatibility."""
        return self._get_spotify_oauth_client()

    def handle_spotify_callback(self, code: str) -> tuple[bool, str]:
        """Process OAuth authorization code from /callback endpoint."""
        sp_cfg = self.config.get("spotify", {})
        client_id = sp_cfg.get("client_id") or os.environ.get("SPOTIPY_CLIENT_ID")
        client_secret = sp_cfg.get("client_secret") or os.environ.get("SPOTIPY_CLIENT_SECRET")
        redirect_uri = sp_cfg.get("redirect_uri") or os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
        if not client_id or not client_secret:
            return False, "Spotify credentials not configured."
        try:
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
            auth_manager.get_access_token(code, as_dict=False)
            return True, "Authorized"
        except Exception as e:
            logger.error("spotify_callback_failed", error=str(e))
            return False, str(e)

    def lookup_spotify_metadata(self, clean_q: str) -> dict[str, Any] | None:
        """Search Spotify Web API for canonical track metadata using client credentials."""
        sp = self._get_spotify_search_client()
        if not sp:
            return None
        try:
            results = sp.search(q=clean_q, type="track", limit=1)
            tracks = results.get("tracks", {}).get("items", [])
            if tracks:
                t = tracks[0]
                artists = [a.get("name", "") for a in t.get("artists", []) if a.get("name")]
                artist_name = ", ".join(artists) if artists else "Unknown Artist"
                images = t.get("album", {}).get("images", [])
                artwork = images[0].get("url") if images else None
                return {
                    "title": t.get("name", clean_q),
                    "artist": artist_name,
                    "album": t.get("album", {}).get("name", ""),
                    "uri": t.get("uri"),
                    "url": t.get("external_urls", {}).get("spotify", ""),
                    "preview_url": t.get("preview_url"),
                    "artwork": artwork,
                }
        except Exception as e:
            logger.debug("spotify_search_failed", error=str(e), query=clean_q)
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
                ["/usr/bin/paplay", f"--rate={sr}", f"--channels={channels}", "--format=s16le", "--raw", "--latency-msec=300"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_process = paplay
            logger.info("playing_full_song_via_ytdlp", query=query, rate=sr)

            # Signal that music frames are now actively playing
            self._music_active = True
            try:
                self._music_started_event.set()
            except Exception:
                pass

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

            import queue
            cache_q = queue.Queue(maxsize=500) if cache_container else None
            c_frames_written = 0

            def _disk_cache_worker():
                nonlocal c_frames_written
                if not cache_container or not cache_stream or not resampler:
                    return
                while True:
                    try:
                        f = cache_q.get()
                        if f is None:
                            break
                        for rf in resampler.resample(f):
                            for pkt in cache_stream.encode(rf):
                                cache_container.mux(pkt)
                                c_frames_written += 1
                        cache_q.task_done()
                    except Exception:
                        break

            if cache_q:
                threading.Thread(target=_disk_cache_worker, daemon=True, name="MusicDiskCache").start()

            def _stream_pump():
                nonlocal cache_container, cache_stream, resampler, c_frames_written
                try:
                    for frame in container.decode(audio_stream):
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

                        # Read volume dynamically on every frame for instant response
                        vol = max(0.0, min(1.0, float(self._volume)))
                        if raw.dtype == np.int16:
                            raw_play = (raw * vol).astype(np.int16)
                        else:
                            raw_play = (raw * 32767.0 * vol).astype(np.int16)

                        try:
                            paplay.stdin.write(raw_play.T.tobytes())
                        except (BrokenPipeError, OSError):
                            break

                        # Asynchronously send clean audio to background cache worker
                        if cache_q:
                            try:
                                cache_q.put_nowait(frame)
                            except queue.Full:
                                pass
                except Exception as e:
                    logger.debug("ytdlp_stream_pump_ended", error=str(e))
                finally:
                    if cache_q:
                        try:
                            cache_q.put(None)
                        except Exception:
                            pass
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

        # 1. Check internet connectivity first
        online = self.is_online()

        # 2. Check local match with high confidence
        local_match = self.find_local_track(clean_q)

        if not online:
            # Offline: must use local file if any match exists (even relaxed)
            if local_match:
                logger.info("playing_offline_cached_file", file=str(local_match), query=clean_q)
                return self.play_local(clean_q, wait_speech_engine=wait_speech_engine)
            return {
                "status": "error",
                "source": "local",
                "message": f"Internet is offline and '{clean_q}' is not available in local storage.",
            }

        # 3. Online: only use local match if it's a strong exact/substring match
        #    (i.e. the full query appears in the filename, not just keyword overlap)
        if local_match:
            stem_l = local_match.stem.lower()
            stop_words = {"play", "song", "music", "track", "audio", "please", "the", "a", "an", "on", "by", "to", "for", "me", "some", "offline", "local"}
            q_words = [w for w in re.findall(r"\w+", clean_q.lower()) if w not in stop_words]
            joined_q = " ".join(q_words)
            # Only use local file if the full query is a substring of the filename OR exact stem match
            if clean_q.lower() == stem_l or (joined_q and joined_q in stem_l):
                logger.info("playing_offline_cached_file_exact", file=str(local_match), query=clean_q)
                return self.play_local(clean_q, wait_speech_engine=wait_speech_engine)

        # 3. Lookup metadata if online
        target_title, target_artist = self._parse_query_target(clean_q)
        meta = self.lookup_spotify_metadata(clean_q) or self.lookup_metadata(clean_q)
        track_title = target_title.title()
        artist_name = target_artist.title()
        if meta:
            t_words = set(re.findall(r"\w+", target_title.lower())) - {"play", "song", "music", "track", "by", "on", "the", "a", "an", "audio"}
            m_words = set(re.findall(r"\w+", meta.get("title", "").lower()))
            # Only adopt external title if at least one meaningful title keyword matches
            if t_words and (t_words & m_words):
                track_title = meta.get("title", track_title)
                artist_name = meta.get("artist", artist_name)
            elif not target_artist and meta.get("artist") and (t_words and (t_words & m_words)):
                artist_name = meta.get("artist")

        resolved_name = f"{track_title} by {artist_name}" if artist_name else track_title
        query_encoded = urllib.parse.quote(clean_q)
        web_url = f"https://www.youtube.com/results?search_query={query_encoded}"

        # Reset music started event for TTS coordination
        self._music_started_event.clear()
        self._music_active = False

        def _execute_playback():
            # CRITICAL: Always use clean_q as full_query so unreleased/freestyle tracks
            # are queried accurately on YouTube instead of substituted titles!
            self._play_stream_audio(
                None,
                web_url=web_url,
                full_query=clean_q,
                wait_speech_engine=wait_speech_engine,
                cache_name=resolved_name,
            )

        threading.Thread(target=_execute_playback, daemon=True).start()

        self._current_track = resolved_name
        self._current_source = "ytdlp"
        self._is_paused = False
        self._record_history(track_title, artist_name, resolved_name, clean_q, "ytdlp")
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
        """Search Spotify for the exact track and play either via Spotify Premium or full audio stream."""
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

        # 1. Search Spotify for canonical track metadata
        target_title, target_artist = self._parse_query_target(clean_q)
        meta = self.lookup_spotify_metadata(clean_q) or self.lookup_metadata(clean_q)
        track_title = target_title.title()
        artist_name = target_artist.title()
        if meta:
            t_words = set(re.findall(r"\w+", target_title.lower())) - {"play", "song", "music", "track", "by", "on", "the", "a", "an", "audio"}
            m_words = set(re.findall(r"\w+", meta.get("title", "").lower()))
            if t_words and (t_words & m_words):
                track_title = meta.get("title", track_title)
                artist_name = meta.get("artist", artist_name)
            elif not target_artist and meta.get("artist") and (t_words and (t_words & m_words)):
                artist_name = meta.get("artist")

        resolved_name = f"{track_title} by {artist_name}" if artist_name else track_title
        track_url = meta.get("url") if (meta and meta.get("url")) else f"https://open.spotify.com/search/{urllib.parse.quote(clean_q)}"
        track_uri = meta.get("uri") if meta else None
        preview_url = meta.get("preview_url") if meta else None

        # Reset music started event for TTS coordination
        self._music_started_event.clear()
        self._music_active = False

        # 2. Try Spotify Premium API first if configured and authenticated
        sp_oauth = self._get_spotify_oauth_client()
        if sp_oauth and track_uri:
            try:
                sp_cfg = self.config.get("spotify", {})
                dev_id = sp_cfg.get("device_id") or None
                devices = sp_oauth.devices().get("devices", [])
                if devices:
                    target_dev = dev_id or devices[0]["id"]
                    def _execute_spotify_premium():
                        try:
                            sp_oauth.start_playback(device_id=target_dev, uris=[track_uri])
                            logger.info("spotify_premium_playback_started", uri=track_uri, device=target_dev)
                        except Exception as e:
                            logger.warning("spotify_premium_playback_failed_fallback_ytdlp", error=str(e))
                            self._play_stream_audio(
                                preview_url,
                                web_url=track_url,
                                full_query=clean_q,
                                wait_speech_engine=wait_speech_engine,
                                cache_name=resolved_name,
                            )

                    threading.Thread(target=_execute_spotify_premium, daemon=True).start()

                    self._current_track = resolved_name
                    self._current_source = "spotify_premium"
                    self._is_paused = False
                    self._record_history(track_title, artist_name, resolved_name, clean_q, "spotify_premium")
                    return {
                        "status": "playing",
                        "source": "spotify_premium",
                        "track": track_title,
                        "artist": artist_name,
                        "resolved_name": resolved_name,
                        "query": clean_q,
                        "url": track_url,
                        "web_url": track_url,
                        "message": f"Playing '{resolved_name}' on Spotify Premium.",
                    }
            except Exception as e:
                logger.warning("spotify_premium_start_failed_falling_back", error=str(e))

        # 3. Default full song playback via yt-dlp & paplay
        # Transparently stream audio using the clean user query
        def _execute_playback():
            self._play_stream_audio(
                preview_url,
                web_url=track_url,
                full_query=clean_q,
                wait_speech_engine=wait_speech_engine,
                cache_name=resolved_name,
            )

        threading.Thread(target=_execute_playback, daemon=True).start()

        self._current_track = resolved_name
        self._current_source = "spotify"
        self._is_paused = False
        self._record_history(track_title, artist_name, resolved_name, clean_q, "spotify")
        return {
            "status": "playing",
            "source": "spotify",
            "track": track_title,
            "artist": artist_name,
            "resolved_name": resolved_name,
            "query": clean_q,
            "url": track_url,
            "web_url": track_url,
            "message": f"Playing '{resolved_name}' on Spotify.",
        }

    def play_soundcloud(self, query: str, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Search SoundCloud for the exact track and play the full song with ducking."""
        clean_q = query.strip()
        target_title, target_artist = self._parse_query_target(clean_q)
        meta = self.lookup_metadata(clean_q)
        track_title = target_title.title()
        artist_name = target_artist.title()
        if meta:
            t_words = set(re.findall(r"\w+", target_title.lower())) - {"play", "song", "music", "track", "by", "on", "the", "a", "an", "audio"}
            m_words = set(re.findall(r"\w+", meta.get("title", "").lower()))
            if t_words and (t_words & m_words):
                track_title = meta.get("title", track_title)
                artist_name = meta.get("artist", artist_name)

        resolved_name = f"{track_title} by {artist_name}" if artist_name else track_title

        query_encoded = urllib.parse.quote(clean_q)
        web_url = f"https://soundcloud.com/search?q={query_encoded}"
        preview_url = meta.get("preview_url") if meta else None

        def _execute_playback():
            self._play_stream_audio(
                preview_url,
                web_url=web_url,
                full_query=clean_q,
                wait_speech_engine=wait_speech_engine,
            )

        threading.Thread(target=_execute_playback, daemon=True).start()

        self._current_track = resolved_name
        self._current_source = "soundcloud"
        self._is_paused = False
        self._record_history(track_title, artist_name, resolved_name, clean_q, "soundcloud")
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
        self._music_active = False
        self._music_started_event.clear()
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

    def resume(self, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Resume playback, or replay if the process was stopped or finished."""
        if self._current_process and self._current_process.poll() is None:
            try:
                import signal
                os.kill(self._current_process.pid, signal.SIGCONT)
                self._is_paused = False
                return {"status": "resumed", "track": self._current_track, "message": f"Resumed playback of '{self._current_track}'."}
            except Exception:
                pass

        self._is_paused = False
        # If process has terminated or was stopped, automatically replay the last track from history!
        if self._history or (self._current_track and self._current_track != "None"):
            logger.info("resume_falling_back_to_replay")
            res = self.replay(wait_speech_engine=wait_speech_engine)
            res["status"] = "resumed"
            return res

        return {"status": "error", "message": "No active or recent track to resume."}

    def replay(self, query: str | None = None, wait_speech_engine: Any = None) -> dict[str, Any]:
        """Replay the most recent song or a specific track from history."""
        target_entry = None
        if query and query.strip().lower() not in {"", "last", "again", "recent", "replay", "this song", "the song"}:
            clean_q = query.strip().lower()
            for h in self._history:
                if clean_q in h.get("resolved_name", "").lower() or clean_q in h.get("query", "").lower() or clean_q in h.get("track", "").lower():
                    target_entry = h
                    break
            if not target_entry:
                return self.play_ytdlp(query, wait_speech_engine=wait_speech_engine)
        elif self._history:
            target_entry = self._history[0]
        elif self._current_track and self._current_track != "None":
            target_entry = {"query": self._current_track, "resolved_name": self._current_track}

        if not target_entry:
            return {"status": "error", "message": "No recently played songs in history to replay."}

        song_query = target_entry.get("query") or target_entry.get("resolved_name")
        logger.info("replaying_recent_song", song=target_entry.get("resolved_name"))
        res = self.play_ytdlp(song_query, wait_speech_engine=wait_speech_engine)
        res["message"] = f"Replaying '{target_entry.get('resolved_name', song_query)}'."
        return res

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recently played tracks list."""
        return self._history[:limit]

    def set_volume(self, volume: float) -> dict[str, Any]:
        """Adjust playback volume between 0.0 and 1.0, updating dynamic stream gain and system mixer."""
        self._volume = max(0.0, min(1.0, float(volume)))
        try:
            pct = int(self._volume * 100)
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"], capture_output=True, timeout=1.0)
        except Exception as e:
            logger.debug("pactl_volume_sync_failed", error=str(e))
        return {
            "status": "volume_set",
            "volume": self._volume,
            "message": f"Volume set to {int(self._volume * 100)}%.",
        }

    PRESETS: dict[str, dict[str, str]] = {
        "robot": {
            "name": "robot",
            "file": "robot",
            "description": "User's signature robot preset with dynamic warmth, tuned clarity, and rich staging",
        },
        "bass_boost": {
            "name": "Max_Bass_Pristine_Reverb",
            "file": "Max_Bass_Pristine_Reverb",
            "description": "Max bass punch with pristine studio reverb and powerful low-end resonance",
        },
        "spatial": {
            "name": "Apple_Spatial_MaxBass_Vocals",
            "file": "Apple_Spatial_MaxBass_Vocals",
            "description": "Apple spatial 3D audio emulation with elevated sub-bass and crystal-clear vocal projection",
        },
        "sub_bass": {
            "name": "SubBass75_ClearVocals",
            "file": "SubBass75_ClearVocals",
            "description": "Deep 75Hz sub-bass rumble paired with high-definition vocal separation",
        },
        "reverb": {
            "name": "Pristine_Reverb",
            "file": "Pristine_Reverb",
            "description": "Spacious acoustic concert hall reverb with airy harmonics",
        },
        "immersive": {
            "name": "Immersive_Sub75_ClearVocals",
            "file": "Immersive_Sub75_ClearVocals",
            "description": "Wide immersive stage acoustics with deep sub-bass and crisp vocal imaging",
        },
    }

    def list_equalizer_presets(self) -> dict[str, Any]:
        """List all available EasyEffects equalizer presets."""
        presets_info = {}
        for key, info in self.PRESETS.items():
            presets_info[key] = {
                "name": info["name"],
                "description": info["description"],
                "active": getattr(self, "_active_preset", "robot") == key,
            }
        return {
            "status": "ok",
            "current_preset": getattr(self, "_active_preset", "robot"),
            "presets": presets_info,
            "message": f"Available equalizer profiles: {', '.join(self.PRESETS.keys())}. Current active preset: '{getattr(self, '_active_preset', 'robot')}'.",
        }

    def set_equalizer_preset(self, preset_name: str | None = None) -> dict[str, Any]:
        """Load an equalizer profile via EasyEffects CLI.
        
        Supported profiles: 'robot', 'bass_boost', 'spatial', 'sub_bass', 'reverb', 'immersive'.
        """
        if not preset_name or preset_name.strip().lower() in {"", "list", "status", "show"}:
            return self.list_equalizer_presets()

        p_clean = preset_name.strip().lower().replace("-", "_").replace(" ", "_")

        alias_map = {
            "robot": "robot",
            "default": "robot",
            "signature": "robot",
            "bass": "bass_boost",
            "bass_boost": "bass_boost",
            "max_bass": "bass_boost",
            "bassboost": "bass_boost",
            "max_bass_pristine_reverb": "bass_boost",
            "spatial": "spatial",
            "spatial_audio": "spatial",
            "apple_spatial": "spatial",
            "apple": "spatial",
            "3d": "spatial",
            "apple_spatial_maxbass_vocals": "spatial",
            "sub_bass": "sub_bass",
            "subbass": "sub_bass",
            "deep_bass": "sub_bass",
            "subbass75_clearvocals": "sub_bass",
            "reverb": "reverb",
            "concert": "reverb",
            "hall": "reverb",
            "pristine_reverb": "reverb",
            "immersive": "immersive",
            "stage": "immersive",
            "immersive_stage": "immersive",
            "immersive_sub75_clearvocals": "immersive",
        }

        matched_key = alias_map.get(p_clean)
        if not matched_key:
            for k in self.PRESETS:
                if k in p_clean or p_clean in k:
                    matched_key = k
                    break

        if not matched_key or matched_key not in self.PRESETS:
            return {
                "status": "error",
                "message": f"Unknown equalizer preset '{preset_name}'. Available options: {list(self.PRESETS.keys())}.",
                "available_presets": list(self.PRESETS.keys()),
            }

        target_info = self.PRESETS[matched_key]
        target_file = target_info["file"]

        # Guarantee EasyEffects background daemon is running before sending preset load command
        ensure_easyeffects_running()

        try:
            res = subprocess.run(
                ["easyeffects", "-l", target_file],
                capture_output=True,
                text=True,
                timeout=2.5,
            )
            if res.returncode == 0:
                self._active_preset = matched_key
                logger.info("equalizer_preset_loaded", preset=matched_key, file=target_file)
                return {
                    "status": "ok",
                    "preset": matched_key,
                    "profile_name": target_info["name"],
                    "description": target_info["description"],
                    "message": f"Equalizer preset switched to '{matched_key}' ({target_info['description']}).",
                }
            else:
                logger.warning("easyeffects_load_failed", returncode=res.returncode, stderr=res.stderr)
                return {
                    "status": "error",
                    "message": f"EasyEffects returned code {res.returncode}: {res.stderr.strip() or 'Unknown error'}",
                }
        except Exception as e:
            logger.error("easyeffects_exec_exception", error=str(e))
            return {
                "status": "error",
                "message": f"Failed to execute EasyEffects command: {str(e)}",
            }


class MusicGenerator:
    """Generative procedural music synthesizer.

    Synthesizes rich multi-layered WAV audio clips (cyberpunk, synthwave, ambient,
    lofi, cinematic, chiptune, upbeat, calm) across customizable BPM, duration,
    and scales without blocking the main event loop.
    """

    SCALES: dict[str, list[float]] = {
        "cyberpunk": [110.0, 130.81, 146.83, 164.81, 196.0, 220.0, 261.63],
        "synthwave": [130.81, 146.83, 164.81, 174.61, 196.0, 220.0, 246.94],
        "ambient": [146.83, 164.81, 174.61, 196.0, 220.0, 246.94, 261.63],
        "lofi": [130.81, 155.56, 174.61, 196.0, 233.08, 261.63],
        "cinematic": [98.0, 116.54, 130.81, 146.83, 164.81, 196.0],
        "chiptune": [261.63, 293.66, 329.63, 392.0, 440.0, 523.25],
        "upbeat": [130.81, 146.83, 164.81, 196.0, 220.0, 261.63],
        "calm": [110.0, 130.81, 146.83, 164.81, 220.0],
    }

    def __init__(self, output_dir: str = "data/generated_audio") -> None:
        """Initialize the procedural music generator.

        Args:
            output_dir: Directory where generated WAV artifacts will be stored.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(
        self,
        prompt: str = "cyberpunk neon pulse",
        style: str = "synthwave",
        duration_seconds: float = 15.0,
        bpm: int = 120,
    ) -> Path:
        """Synthesize a complete multi-layered audio clip.

        Args:
            prompt: Text prompt or theme description.
            style: Genre or musical mood preset.
            duration_seconds: Length of output audio clip (clamped between 10.0 and 30.0).
            bpm: Tempo in beats per minute (clamped between 60 and 180).

        Returns:
            Path to the generated 16-bit PCM WAV audio file.

        Raises:
            ValueError: If parameters are invalid.
        """
        dur = max(10.0, min(30.0, float(duration_seconds)))
        tempo = max(60, min(180, int(bpm)))
        clean_style = style.lower().strip() if style else "synthwave"
        if clean_style not in self.SCALES:
            clean_style = "synthwave"

        sample_rate = 44100
        total_samples = int(dur * sample_rate)

        scale = self.SCALES[clean_style]
        beat_dur = 60.0 / tempo
        measure_dur = beat_dur * 4.0

        # Layer 1: Bassline (Sawtooth + Sine sub-bass with exponential decay envelope)
        bass_audio = np.zeros(total_samples, dtype=np.float32)
        bass_step = beat_dur
        num_bass_notes = int(np.ceil(dur / bass_step))
        for i in range(num_bass_notes):
            n_start = int(i * bass_step * sample_rate)
            n_end = min(total_samples, int((i + 1) * bass_step * sample_rate))
            if n_start >= total_samples:
                break
            note_len = n_end - n_start
            note_t = np.linspace(0, note_len / sample_rate, note_len, endpoint=False)
            freq = scale[(i % len(scale))] * 0.5
            env = np.exp(-note_t * 2.5)
            sub = np.sin(2 * np.pi * freq * note_t) * 0.6
            harm = np.sin(2 * np.pi * freq * 2.0 * note_t) * 0.25
            bass_audio[n_start:n_end] += (sub + harm) * env * 0.45

        # Layer 2: Arpeggiator / Lead Melody
        arp_audio = np.zeros(total_samples, dtype=np.float32)
        step_dur = beat_dur / 2.0
        num_steps = int(np.ceil(dur / step_dur))
        for step in range(num_steps):
            n_start = int(step * step_dur * sample_rate)
            n_end = min(total_samples, int((step + 1) * step_dur * sample_rate))
            if n_start >= total_samples:
                break
            note_len = n_end - n_start
            note_t = np.linspace(0, note_len / sample_rate, note_len, endpoint=False)
            idx = (step * 2 + (step // 4)) % len(scale)
            freq = scale[idx]
            if clean_style == "chiptune":
                sig = np.sign(np.sin(2 * np.pi * freq * note_t)) * 0.2
            else:
                vib = 1.0 + 0.015 * np.sin(2 * np.pi * 5.0 * note_t)
                sig = np.sin(2 * np.pi * freq * vib * note_t) * 0.35
            env = np.exp(-note_t * 4.0)
            arp_audio[n_start:n_end] += sig * env

        # Layer 3: Lush Chords / Ambient Pad
        pad_audio = np.zeros(total_samples, dtype=np.float32)
        chord_step = measure_dur
        num_chords = int(np.ceil(dur / chord_step))
        for c in range(num_chords):
            n_start = int(c * chord_step * sample_rate)
            n_end = min(total_samples, int((c + 1) * chord_step * sample_rate))
            if n_start >= total_samples:
                break
            note_len = n_end - n_start
            note_t = np.linspace(0, note_len / sample_rate, note_len, endpoint=False)
            root_idx = (c * 2) % len(scale)
            third_idx = (root_idx + 2) % len(scale)
            fifth_idx = (root_idx + 4) % len(scale)
            f1, f2, f3 = scale[root_idx], scale[third_idx], scale[fifth_idx]
            chord = (
                np.sin(2 * np.pi * f1 * note_t)
                + np.sin(2 * np.pi * (f1 * 1.004) * note_t) * 0.5
                + np.sin(2 * np.pi * f2 * note_t) * 0.7
                + np.sin(2 * np.pi * f3 * note_t) * 0.7
            )
            attack_s = min(0.6, (n_end - n_start) / sample_rate / 3)
            attack_samples = int(attack_s * sample_rate)
            pad_env = np.ones(note_len, dtype=np.float32)
            if attack_samples > 0:
                pad_env[:attack_samples] = np.linspace(0, 1, attack_samples)
            decay_samples = int(0.5 * sample_rate)
            if note_len > decay_samples:
                pad_env[-decay_samples:] = np.linspace(1, 0, decay_samples)
            pad_audio[n_start:n_end] += chord * pad_env * 0.22

        # Layer 4: Percussion / Drums (Kick, Snare, Hi-hat)
        drum_audio = np.zeros(total_samples, dtype=np.float32)
        num_beats = int(np.ceil(dur / beat_dur))
        np_rng = np.random.default_rng(seed=abs(hash(prompt)) % (2**31))

        for b in range(num_beats):
            beat_time = b * beat_dur
            n_start = int(beat_time * sample_rate)
            if n_start >= total_samples:
                break

            beat_in_measure = b % 4
            is_kick = (beat_in_measure in (0, 2)) or (clean_style in ("synthwave", "cyberpunk", "upbeat"))
            if is_kick:
                k_len = min(int(0.25 * sample_rate), total_samples - n_start)
                k_t = np.linspace(0, 0.25, k_len, endpoint=False)
                k_freq = 45.0 + 95.0 * np.exp(-k_t * 22.0)
                phase = 2 * np.pi * np.cumsum(k_freq) / sample_rate
                k_env = np.exp(-k_t * 14.0)
                drum_audio[n_start : n_start + k_len] += np.sin(phase) * k_env * 0.65

            if beat_in_measure in (1, 3):
                s_len = min(int(0.20 * sample_rate), total_samples - n_start)
                s_t = np.linspace(0, 0.20, s_len, endpoint=False)
                s_env = np.exp(-s_t * 18.0)
                noise = np_rng.uniform(-1.0, 1.0, s_len).astype(np.float32)
                tone = np.sin(2 * np.pi * 180.0 * s_t) * 0.3
                drum_audio[n_start : n_start + s_len] += (noise * 0.7 + tone) * s_env * 0.35

            offbeat_start = n_start + int(0.5 * beat_dur * sample_rate)
            if offbeat_start < total_samples:
                h_len = min(int(0.06 * sample_rate), total_samples - offbeat_start)
                h_t = np.linspace(0, 0.06, h_len, endpoint=False)
                h_env = np.exp(-h_t * 40.0)
                h_noise = np_rng.uniform(-1.0, 1.0, h_len).astype(np.float32)
                drum_audio[offbeat_start : offbeat_start + h_len] += h_noise * h_env * 0.20

        # Master Stereo Mix
        left = bass_audio + arp_audio * 0.7 + pad_audio * 0.85 + drum_audio * 0.9
        right = bass_audio + arp_audio * 0.85 + pad_audio * 0.7 + drum_audio * 0.9

        # Master Limiter with headroom
        peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-4)
        left = (left / peak) * 0.89
        right = (right / peak) * 0.89

        # Fade in and Fade out
        fade_in_len = int(0.2 * sample_rate)
        fade_out_len = int(0.8 * sample_rate)
        if total_samples > fade_in_len + fade_out_len:
            in_curve = np.linspace(0, 1, fade_in_len, dtype=np.float32)
            out_curve = np.linspace(1, 0, fade_out_len, dtype=np.float32)
            left[:fade_in_len] *= in_curve
            right[:fade_in_len] *= in_curve
            left[-fade_out_len:] *= out_curve
            right[-fade_out_len:] *= out_curve

        # Convert to 16-bit PCM stereo
        left_i16 = (np.clip(left, -1.0, 1.0) * 32767).astype(np.int16)
        right_i16 = (np.clip(right, -1.0, 1.0) * 32767).astype(np.int16)
        stereo_interleaved = np.empty((total_samples * 2,), dtype=np.int16)
        stereo_interleaved[0::2] = left_i16
        stereo_interleaved[1::2] = right_i16

        clean_tag = re.sub(r"[^\w\-]", "_", prompt[:20].lower()).strip("_") or clean_style
        ts = int(time.time())
        filename = f"gen_music_{clean_style}_{clean_tag}_{ts}.wav"
        file_path = self.output_dir / filename

        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(stereo_interleaved.tobytes())

        logger.info(
            "music_synthesized_successfully",
            file=str(file_path),
            duration_s=dur,
            bpm=tempo,
            style=clean_style,
        )
        return file_path


def register_music_generator_tools(engine: RobotEngine) -> None:
    """Register music.generate tool with the Robot Core engine.

    Args:
        engine: RobotEngine instance containing tool_registry.
    """
    generator = MusicGenerator()
    player = MusicPlayerService.get_instance()

    async def generate_music(
        prompt: str = "cyberpunk neon pulse",
        style: str = "synthwave",
        duration_seconds: float = 15.0,
        bpm: int = 120,
        auto_play: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synthesize a rich procedural music clip and optionally play it immediately.

        Args:
            prompt: Text description or theme of music.
            style: Musical style ('cyberpunk', 'synthwave', 'ambient', 'lofi', 'cinematic', 'chiptune', 'upbeat', 'calm').
            duration_seconds: Clip duration in seconds (10 to 30).
            bpm: Tempo in beats per minute (60 to 180).
            auto_play: Whether to immediately queue and start playback of the generated clip.

        Returns:
            Dictionary containing audio file metadata and playback status.
        """
        p = (prompt or kwargs.get("query") or kwargs.get("theme") or kwargs.get("text") or "cyberpunk synth pulse").strip()
        st = (style or kwargs.get("genre") or kwargs.get("mood") or "synthwave").strip().lower()
        d = float(duration_seconds if duration_seconds is not None else kwargs.get("duration", 15.0))
        b = int(bpm if bpm is not None else kwargs.get("tempo", 120))

        # Run synthesis in background thread to never block voice pipeline
        file_path = await asyncio.to_thread(
            generator.synthesize,
            prompt=p,
            style=st,
            duration_seconds=d,
            bpm=b,
        )

        play_result = None
        if auto_play:
            play_result = await asyncio.to_thread(player.play_local, file_path.stem, wait_speech_engine=engine)

        return {
            "status": "ok",
            "file_path": str(file_path),
            "filename": file_path.name,
            "style": st,
            "duration_seconds": d,
            "bpm": b,
            "prompt": p,
            "auto_play": auto_play,
            "playback": play_result,
            "message": f"Generated {d:.1f}s {st} music track saved to {file_path.name}"
            + (" and playback started." if auto_play else "."),
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="music.generate",
            description="Synthesize procedural / generative AI music tracks (10 to 30s) across styles (cyberpunk, synthwave, ambient, lofi, cinematic, chiptune, upbeat, calm). Outputs high-fidelity stereo WAV audio, saves to data/generated_audio/, and optionally starts playback immediately.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Theme, description, or mood of the music (e.g. 'synthwave midnight drive')"},
                    "style": {"type": "string", "description": "Style: 'cyberpunk', 'synthwave', 'ambient', 'lofi', 'cinematic', 'chiptune', 'upbeat', 'calm'", "default": "synthwave"},
                    "duration_seconds": {"type": "number", "description": "Duration in seconds (between 10.0 and 30.0)", "default": 15.0},
                    "bpm": {"type": "integer", "description": "Beats per minute (60 to 180)", "default": 120},
                    "auto_play": {"type": "boolean", "description": "Automatically play the generated track upon completion", "default": True},
                },
                "required": ["prompt"],
            },
            handler=generate_music,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="music",
            timeout_s=30.0,
            is_idempotent=False,
        )
    )

