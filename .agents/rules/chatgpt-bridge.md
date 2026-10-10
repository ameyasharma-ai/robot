# ChatGPT AI Development Bridge & Collaboration Rule

Whenever you are working on the robot project and the task involves complex architectural planning, new system features, major refactorings, or when the user requests multi-agent collaboration:

1. **Invoke the Bridge Automatically**:
   - Check if the local bridge is available via:
     ```bash
     .venv/bin/python .agents/skills/chatgpt-bridge/scripts/bridge_client.py
     ```
   - Connect to ChatGPT running in Google Chrome to act as the Lead Architect & Reviewer.

2. **Execute the Back-and-Forth Collaboration Cycle**:
   - **Architectural Specification**: Transmit the goal to ChatGPT in Chrome to generate requirements, component boundaries, and acceptance criteria.
   - **Engineering Implementation**: Implement the code in the robot repository according to ChatGPT's specification.
   - **Test & Import Verification**: Ensure 100% of tests pass (`.venv/bin/pytest tests/`) and zero import errors exist (`.venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py`).
   - **Architectural Review**: Transmit the diff and test report back into ChatGPT in Chrome for review.
   - **Iterate**: Continue until ChatGPT provides architectural sign-off and all acceptance criteria are met.

3. **Preserve System Boundaries**:
   - Keep hardware constraints strictly respected (RTX 3050 4GB VRAM reserved for local Qwen3.5-4B; audio buffer latency <800ms).
   - Never scrape Chrome credentials or user profiles.
   - Update `TRACKER.md` with feature status and iteration milestones upon completion.
