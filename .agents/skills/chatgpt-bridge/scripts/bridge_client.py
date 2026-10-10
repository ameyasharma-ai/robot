"""Client library for interfacing with the AI Development Bridge.

Communicates with the local bridge server running at http://127.0.0.1:8765,
connecting Antigravity IDE with ChatGPT running in Google Chrome.
"""

from dataclasses import dataclass
import subprocess
import sys
import time
from typing import Any

import httpx

DEFAULT_BRIDGE_URL = "http://127.0.0.1:8765"


@dataclass
class BridgeStatus:
    """Connection and operational status of the AI Development Bridge."""

    server_online: bool
    chatgpt_connected: bool
    connected_tabs: int
    pending_prompts: int
    error: str | None = None


class ChatGPTBridgeClient:
    """Client for dispatching prompts and receiving responses from ChatGPT in Chrome."""

    def __init__(self, base_url: str = DEFAULT_BRIDGE_URL) -> None:
        """Initialize the bridge client.

        Args:
            base_url: Base URL of the bridge server (default: http://127.0.0.1:8765).
        """
        self.base_url = base_url.rstrip("/")

    def get_status(self) -> BridgeStatus:
        """Query real-time health of the bridge server and Chrome extension.

        Returns:
            BridgeStatus: Diagnostic status report.
        """
        try:
            resp = httpx.get(f"{self.base_url}/api/live/chatgpt/status", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return BridgeStatus(
                    server_online=True,
                    chatgpt_connected=bool(data.get("connected", False)),
                    connected_tabs=int(data.get("connected_tabs", 0)),
                    pending_prompts=int(data.get("pending_prompts", 0)),
                )
            return BridgeStatus(
                server_online=True,
                chatgpt_connected=False,
                connected_tabs=0,
                pending_prompts=0,
                error=f"Server returned status {resp.status_code}",
            )
        except httpx.ConnectError:
            return BridgeStatus(
                server_online=False,
                chatgpt_connected=False,
                connected_tabs=0,
                pending_prompts=0,
                error="Bridge server is offline (connection refused).",
            )
        except Exception as exc:
            return BridgeStatus(
                server_online=False,
                chatgpt_connected=False,
                connected_tabs=0,
                pending_prompts=0,
                error=str(exc),
            )

    def ensure_server_running(self) -> bool:
        """Verify the bridge server is running; launch as daemon if offline.

        Returns:
            bool: True if server is running and accessible.
        """
        status = self.get_status()
        if status.server_online:
            return True

        print("[Bridge Client] Bridge server is offline. Starting server daemon...", flush=True)
        try:
            app_script = "/home/ameya/ai-dev-bridge/app.py"
            venv_python = "/home/ameya/ai-dev-bridge/.venv/bin/python"
            subprocess.Popen(
                [venv_python, app_script],
                cwd="/home/ameya/ai-dev-bridge",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Wait up to 5 seconds for startup
            for _ in range(10):
                time.sleep(0.5)
                if self.get_status().server_online:
                    print("[Bridge Client] Bridge server started successfully.", flush=True)
                    return True
        except Exception as exc:
            print(f"[Bridge Client] Failed to launch server daemon: {exc}", file=sys.stderr)
        return False

    def send_prompt(self, prompt: str, timeout: float = 180.0) -> str:
        """Transmit a prompt into ChatGPT in Chrome and await its response.

        Args:
            prompt: Text prompt to transmit.
            timeout: Maximum response wait duration in seconds.

        Returns:
            str: Full markdown text response from ChatGPT.

        Raises:
            RuntimeError: If the bridge server or Chrome tab is disconnected.
            TimeoutError: If response generation exceeds timeout.
        """
        if not self.ensure_server_running():
            raise RuntimeError("Cannot reach AI Development Bridge server at http://127.0.0.1:8765.")

        status = self.get_status()
        if not status.chatgpt_connected:
            raise RuntimeError(
                "ChatGPT in Google Chrome is not connected.\n"
                "Ensure Chrome is open to https://chatgpt.com and the extension is loaded from:\n"
                "~/ai-dev-bridge/extensions/chatgpt_bridge"
            )

        try:
            resp = httpx.post(
                f"{self.base_url}/api/live/chatgpt/send",
                json={"prompt": prompt, "timeout_seconds": timeout},
                timeout=timeout + 10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
            elif resp.status_code == 504:
                raise TimeoutError("Timed out waiting for ChatGPT in Chrome to complete generation.")
            else:
                err = resp.json().get("detail", resp.text)
                raise RuntimeError(f"Bridge server error ({resp.status_code}): {err}")
        except httpx.TimeoutException:
            raise TimeoutError(f"HTTP request timed out after {timeout} seconds.")

    def fetch_latest_response(self, timeout: float = 10.0) -> str:
        """Fetch the most recent assistant response from the active ChatGPT tab.

        Args:
            timeout: Maximum query timeout in seconds.

        Returns:
            str: Latest assistant response text.

        Raises:
            RuntimeError: If retrieval fails or tab is disconnected.
        """
        try:
            resp = httpx.get(f"{self.base_url}/api/live/chatgpt/latest", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
            err = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Failed to fetch response: {err}")
        except Exception as exc:
            raise RuntimeError(f"Error communicating with bridge: {exc}")
