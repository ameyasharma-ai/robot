"""Robot Core entry point."""

from __future__ import annotations

import asyncio
import sys

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

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\n🤖 Robot Core shutting down.")


if __name__ == "__main__":
    main()
