"""Robot Core entry point."""

from __future__ import annotations

import asyncio
import sys

from robot.core.engine import RobotEngine


def main() -> None:
    """Launch Robot Core."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    engine = RobotEngine(config_path=config_path)

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\n🤖 Robot Core shutting down.")


if __name__ == "__main__":
    main()
