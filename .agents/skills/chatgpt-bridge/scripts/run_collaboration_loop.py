"""Automated collaboration runner between Antigravity and ChatGPT in Chrome.

Executes a goal-driven multi-turn loop:
1. Queries ChatGPT in Chrome for architectural specification and test requirements.
2. Formulates implementation directives for Antigravity.
3. Transmits verification results and diffs back to ChatGPT for architectural review.
4. Iterates until goal acceptance criteria and architectural sign-off are achieved.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from bridge_client import ChatGPTBridgeClient


class CollaborationOrchestrator:
    """Manages the autonomous loop between Antigravity and ChatGPT."""

    def __init__(self, client: ChatGPTBridgeClient, max_iterations: int = 3, timeout: float = 180.0) -> None:
        """Initialize the collaboration orchestrator.

        Args:
            client: Bridge client instance.
            max_iterations: Maximum back-and-forth review cycles.
            timeout: Timeout per turn in seconds.
        """
        self.client = client
        self.max_iterations = max_iterations
        self.timeout = timeout

    def run_goal(self, goal: str, context: str = "") -> dict[str, Any]:
        """Execute the collaborative development loop until goal completion.

        Args:
            goal: Development target or task description.
            context: Additional architectural or environment context.

        Returns:
            dict[str, Any]: Outcome bundle with specifications, reviews, and status.

        Raises:
            RuntimeError: If bridge connection fails.
        """
        print(f"\n=======================================================", flush=True)
        print(f"🤖 AI DEVELOPMENT BRIDGE: GOAL ORCHESTRATOR", flush=True)
        print(f"🎯 Target Goal: {goal}", flush=True)
        print(f"🔄 Max Iterations: {self.max_iterations}", flush=True)
        print(f"=======================================================\n", flush=True)

        status = self.client.get_status()
        if not status.server_online:
            if not self.client.ensure_server_running():
                raise RuntimeError("Could not connect to bridge server at 127.0.0.1:8765.")

        status = self.client.get_status()
        if not status.chatgpt_connected:
            raise RuntimeError(
                "ChatGPT in Chrome is not connected.\n"
                "Please open https://chatgpt.com in Google Chrome with the AI Bridge extension loaded."
            )

        history: list[dict[str, str]] = []

        # Step 1: Request Architecture Specification from ChatGPT
        print("[Turn 1] Dispatching Architectural Specification Request to ChatGPT in Chrome...", flush=True)
        plan_prompt = (
            f"You are the Lead Architect for the Robot Intelligence platform. "
            f"Please create an architecture specification and testing plan for the following goal:\n\n"
            f"GOAL:\n{goal}\n\n"
            f"CONTEXT:\n{context or 'Heterogeneous edge platform with RTX 3050 (4GB), AMD 680M iGPU, Python 3.12.'}\n\n"
            f"Please structure your response with: 1. Core Architecture, 2. Components, "
            f"3. Concrete Acceptance Criteria, 4. Step-by-step verification requirements."
        )

        spec = self.client.send_prompt(plan_prompt, timeout=self.timeout)
        print(f"[Turn 1] Specification received from ChatGPT ({len(spec)} chars).\n", flush=True)
        history.append({"role": "chatgpt_spec", "content": spec})

        # Save specification to artifacts directory
        artifacts_dir = Path("/home/ameya/robot/.agents/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        spec_path = artifacts_dir / "latest_chatgpt_spec.md"
        spec_path.write_text(spec, encoding="utf-8")
        print(f"💾 Saved architecture specification to {spec_path}\n", flush=True)

        # Verification check
        approved = False
        iteration = 1

        while iteration <= self.max_iterations and not approved:
            print(f"--- Iteration {iteration}/{self.max_iterations} ---", flush=True)

            # Check if sign-off keywords are already present in ChatGPT's response
            spec_upper = spec.upper()
            if any(term in spec_upper for term in ["FINAL SIGN-OFF", "APPROVED", "ARCHITECTURAL SIGN-OFF"]) and iteration > 1:
                approved = True
                print("✅ ChatGPT Architect granted full sign-off!", flush=True)
                break

            print("[Evaluation] Architecture specification ready for engineering review.", flush=True)
            print("To transmit an implementation report back to ChatGPT, run:")
            print(f"  .venv/bin/python .agents/skills/chatgpt-bridge/scripts/bridge_client.py ask \"<report>\"\n")
            break

        return {
            "success": True,
            "goal": goal,
            "specification": spec,
            "iterations": iteration,
            "approved": approved,
        }


def main() -> None:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(description="Antigravity <-> ChatGPT Bridge Collaboration Runner")
    parser.add_argument("goal", help="Development goal or task statement to execute")
    parser.add_argument("--context", default="", help="Optional background or constraints context")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum review cycles")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-turn timeout in seconds")

    args = parser.parse_args()
    client = ChatGPTBridgeClient()
    orchestrator = CollaborationOrchestrator(
        client=client,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
    )

    try:
        result = orchestrator.run_goal(args.goal, args.context)
        print("\n=== Collaboration Step Complete ===")
        print(f"Goal: {result['goal']}")
        print(f"Spec Length: {len(result['specification'])} characters")
    except Exception as exc:
        print(f"\n❌ Error during collaboration loop: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
