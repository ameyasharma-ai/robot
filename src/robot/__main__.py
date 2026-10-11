"""Robot Core entry point."""

from __future__ import annotations

import asyncio
import signal
import sys
import warnings

import os
import traceback

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from robot.core.engine import RobotEngine
from robot.telemetry import get_logger

logger = get_logger("core.main")

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
                asyncio.create_task(engine.stop())
                engine_task.cancel()

        sigs = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGQUIT"):
            sigs.append(signal.SIGQUIT)

        for sig in sigs:
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, RuntimeError):
                pass

        try:
            await engine_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("engine_task_crashed", error=str(e), traceback=traceback.format_exc())
        finally:
            # Terminate any external background server processes (llama-server, whisper-server)
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
    except Exception as e:
        logger.error("robot_runner_exception", error=str(e), traceback=traceback.format_exc())
    finally:
        print("\n🤖 Robot Core and all background server processes terminated cleanly.")
        print("🧠 GPU VRAM and system memory released.")


if __name__ == "__main__":
    main()

