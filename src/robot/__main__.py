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

        async def _graceful_shutdown() -> None:
            """Shut down engine components gracefully before terminating tasks."""
            await engine.stop()
            for _ in range(20):
                if engine_task.done():
                    return
                await asyncio.sleep(0.1)
            if not engine_task.done():
                engine_task.cancel()

        def _handle_signal() -> None:
            """Handle termination signals and schedule graceful shutdown."""
            if not engine_task.done():
                loop.create_task(_graceful_shutdown())

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
            # A crashed engine can return from run() before its teardown runs.
            # Stop only the servers this process actually started; never reach
            # for every process on the machine that happens to match a name.
            try:
                from robot.llm import LlamaServerManager
                from robot.stt import WhisperServerManager

                await WhisperServerManager.stop()
                await LlamaServerManager.stop()
            except Exception as exc:
                logger.warning("background_server_cleanup_failed", error=str(exc))

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("robot_runner_exception", error=str(e), traceback=traceback.format_exc())
    finally:
        print("\nRobot Core stopped. Background model servers released.")
        print("GPU VRAM and system memory released.")


if __name__ == "__main__":
    main()

