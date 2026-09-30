"""Context Engine — assembles the optimal 16K context window per turn.

For each inference turn, Robot Core assembles:
1. Active personality system prompt
2. Core robot/system policy
3. Current task state
4. Relevant Mem0 memories
5. Recent conversation
6. Current user input
7. Relevant tool definitions only
8. Current system/robot state
9. Relevant tool results
10. Compact visual/world state
"""

from __future__ import annotations

from typing import Any

from robot.telemetry import get_logger

logger = get_logger("context")


class ContextEngine:
    """Dynamic context assembly with budget-aware allocation.

    The 16K context window is divided into dynamic (not hard) partitions:
    - System/personality:    1–1.5K tokens
    - Relevant tool schemas: 1.5–2.5K tokens
    - Retrieved memories:    1.5–2.5K tokens
    - Recent conversation:   4–6K tokens
    - World/vision state:    0.5–1K tokens
    - Robot state:           0.25–0.5K tokens
    - User input:            Variable
    - Generation headroom:   3–4K tokens
    """

    def __init__(self, budget: dict[str, int] | None = None) -> None:
        self._budget = budget or {
            "system_personality": 1500,
            "tool_schemas": 2000,
            "retrieved_memories": 2000,
            "recent_conversation": 5000,
            "world_state": 750,
            "robot_state": 375,
            "generation_headroom": 3500,
        }

    def build_messages(
        self,
        system_prompt: str,
        user_text: str,
        recent_conversation: list[dict] | None = None,
        memories: list[dict] | None = None,
        robot_state: dict | None = None,
        world_state: dict | None = None,
        tool_results: list[dict] | None = None,
    ) -> list[dict]:
        """Assemble the complete message list for LLM inference.

        Optimizes context usage by:
        - Injecting only relevant memories (not all)
        - Compacting robot state and physical world state
        - Trimming conversation to budget
        """
        messages = []

        # 1. System prompt (personality + policy)
        system_parts = [system_prompt]

        # 2. Robot state (compact)
        if robot_state:
            state_str = _format_state(robot_state)
            system_parts.append(f"\n[Current State]\n{state_str}")

        # 3. Physical World State (Perception)
        if world_state:
            world_str = _format_world_state(world_state)
            system_parts.append(f"\n[Physical World State]\n{world_str}")

        # 4. Retrieved memories
        if memories:
            mem_str = _format_memories(memories, self._budget["retrieved_memories"])
            if mem_str:
                system_parts.append(f"\n[Relevant Memories]\n{mem_str}")

        messages.append({"role": "system", "content": "\n".join(system_parts)})

        # 4. Recent conversation history
        if recent_conversation:
            trimmed = _trim_conversation(
                recent_conversation, self._budget["recent_conversation"]
            )
            messages.extend(trimmed)

        # 5. Tool results from current turn
        if tool_results:
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "content": str(tr.get("result", tr.get("error", ""))),
                    "tool_call_id": tr.get("id", ""),
                })

        # 6. Current user input (with live camera visual grounding if user asks what is this / what am I holding / what do you see)
        user_content = user_text
        lower_user = user_text.lower()
        visual_triggers = (
            "what is this", "what is that", "what do you see", "who is here",
            "what are these", "describe what you see", "what am i holding",
            "what's this", "what's that", "what i'm holding", "what im holding",
            "look at this", "look at what i have", "can you see this", "what do i have",
            "what object", "identify this", "check this out", "in my hand"
        )
        if any(w in lower_user for w in visual_triggers):
            if world_state:
                detected_items = []
                for o in world_state.get("objects", []):
                    # Filter out redundant generic 'person' when user is asking about an object being held
                    if o.get("label") == "person" and len(world_state.get("objects", [])) > 1:
                        continue
                    detected_items.append(f"{o.get('label')} at {o.get('position')}")
                for p in world_state.get("people", []):
                    detected_items.append(f"person ({p.get('distance')} range, {p.get('position')})")
                
                if detected_items:
                    user_content += f"\n[Visual Stream in Front of Robot]: Live camera actively detects: {', '.join(detected_items)}."
                else:
                    user_content += f"\n[Visual Stream in Front of Robot]: Camera is pointed forward, but no distinct object is currently recognized by YOLOv8. Ask user to hold the object clearly towards the camera."

        messages.append({"role": "user", "content": user_content})

        token_estimate = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        logger.info(
            "context_built",
            messages=len(messages),
            estimated_tokens=token_estimate,
        )

        return messages


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English."""
    return len(text) // 4


def _format_memories(memories: list[dict], budget_tokens: int) -> str:
    """Format retrieved memories within budget."""
    parts = []
    total = 0
    for mem in memories:
        text = mem.get("memory", mem.get("text", str(mem)))
        tokens = _estimate_tokens(text)
        if total + tokens > budget_tokens:
            break
        parts.append(f"- {text}")
        total += tokens
    return "\n".join(parts)


def _format_state(state: dict) -> str:
    """Format robot state compactly."""
    lines = []
    for key, value in state.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _format_world_state(world_state: dict) -> str:
    """Format physical world state compactly for LLM context."""
    lines = []
    # People
    people = world_state.get("people", [])
    people_count = world_state.get("people_count", len(people))
    lines.append(f"  people_present: {people_count}")
    if people:
        for idx, p in enumerate(people):
            lines.append(
                f"    - person_{idx+1}: pos={p.get('position', 'center')} "
                f"distance={p.get('distance', 'medium')} "
                f"attentive={p.get('looking_at_robot', True)}"
            )

    # Physical Objects (from continuous object detection & vision reasoning)
    objects = world_state.get("objects", [])
    objects_count = world_state.get("objects_count", len(objects))
    lines.append(f"  objects_detected: {objects_count}")
    if objects:
        for obj in objects:
            lines.append(
                f"    - {obj.get('label')}: pos={obj.get('position', 'center')} "
                f"conf={obj.get('confidence', 0.8)}"
            )

    lines.append(f"  motion_level: {world_state.get('motion', 'none')}")
    lines.append(f"  room_lighting: {world_state.get('lighting', 'normal')}")
    lines.append(f"  scene: {world_state.get('scene', 'indoor')}")
    return "\n".join(lines)


def _trim_conversation(messages: list[dict], budget_tokens: int) -> list[dict]:
    """Keep as many recent messages as fit within the token budget."""
    result = []
    total = 0
    for msg in reversed(messages):
        tokens = _estimate_tokens(msg.get("content", ""))
        if total + tokens > budget_tokens:
            break
        result.insert(0, msg)
        total += tokens
    return result
