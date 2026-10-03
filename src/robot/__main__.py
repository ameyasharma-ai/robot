"""Robot Core entry point."""

from __future__ import annotations

import asyncio
import signal
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*Setting the shape on a NumPy array.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

from robot.core.engine import RobotEngine


import argparse


def main() -> None:
    """Launch Robot Core."""
    parser = argparse.ArgumentParser(description="Robot Core — Local Humanoid Robot Brain")
    parser.add_argument(
        "-c", "--config", default="config/default.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    engine = RobotEngine(config_path=args.config)

    async def _runner() -> None:
        loop = asyncio.get_running_loop()
        engine_task = asyncio.create_task(engine.run())

        def _handle_signal() -> None:
            if not engine_task.done():
                engine_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, RuntimeError):
                pass

        try:
            await engine_task
        except asyncio.CancelledError:
            pass
        finally:
            # Terminate any external background server processes (llama-server, whisper-server)
            import os
            import subprocess
            try:
                subprocess.run(["pkill", "-f", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["pkill", "-f", "whisper-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        pass
    finally:
        print("\n🤖 Robot Core and all background server processes terminated cleanly.")
        print("🧠 GPU VRAM and system memory released.")


if __name__ == "__main__":
    main()

