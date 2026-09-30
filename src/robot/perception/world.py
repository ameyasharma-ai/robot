"""World state models and tracking.

Represents continuous physical perception of the environment:
- People present (position, bounding box, attention)
- Visual objects / landmarks
- Motion / activity level
- Lighting conditions
- Compact YAML/dict representation for LLM context injection
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DetectedPerson:
    id: str = "user"
    position: str = "center"  # "left", "center", "right"
    distance_estimate: str = "near"  # "near", "medium", "far"
    confidence: float = 0.0
    looking_at_robot: bool = True
    bbox: list[int] = field(default_factory=list)  # [x, y, w, h]
    norm_x: float = 0.5  # Normalized 0..1 (left to right in frame)
    norm_y: float = 0.5  # Normalized 0..1 (top to bottom in frame)


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: list[int] = field(default_factory=list)  # [x, y, w, h]
    position: str = "center"  # "left", "center", "right"


@dataclass
class WorldState:
    """Continuous physical world state snapshot."""

    people: list[DetectedPerson] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    motion: str = "none"  # "none", "low", "medium", "high"
    motion_score: float = 0.0
    lighting: str = "normal"  # "dark", "dim", "normal", "bright"
    scene: str = "indoor"
    updated_at: float = field(default_factory=time.time)

    def to_compact(self) -> dict[str, Any]:
        """Compact representation for 0.5–1K context budget injection."""
        people_summary = []
        for p in self.people:
            people_summary.append({
                "id": p.id,
                "position": p.position,
                "distance": p.distance_estimate,
                "looking_at_robot": p.looking_at_robot,
                "norm_x": round(p.norm_x, 3),
                "norm_y": round(p.norm_y, 3),
            })

        objects_summary = []
        for obj in self.objects:
            objects_summary.append({
                "label": obj.label,
                "confidence": round(obj.confidence, 2),
                "position": obj.position,
            })

        return {
            "people_count": len(self.people),
            "people": people_summary,
            "objects_count": len(self.objects),
            "objects": objects_summary,
            "motion": self.motion,
            "lighting": self.lighting,
            "scene": self.scene,
            "time_since_update_s": round(time.time() - self.updated_at, 1),
        }

    def to_yaml_str(self) -> str:
        """Format as clean YAML text for system prompt inclusion."""
        compact = self.to_compact()
        lines = []
        lines.append(f"people_count: {compact['people_count']}")
        if compact["people"]:
            lines.append("people:")
            for p in compact["people"]:
                lines.append(f"  - id: {p['id']}")
                lines.append(f"    position: {p['position']}")
                lines.append(f"    distance: {p['distance']}")
                lines.append(f"    looking_at_robot: {str(p['looking_at_robot']).lower()}")
        else:
            lines.append("people: []")

        lines.append(f"objects_count: {compact['objects_count']}")
        if compact["objects"]:
            lines.append("objects:")
            for o in compact["objects"]:
                lines.append(f"  - label: {o['label']}")
                lines.append(f"    position: {o['position']}")
                lines.append(f"    confidence: {o['confidence']}")
        else:
            lines.append("objects: []")

        lines.append(f"motion: {compact['motion']}")
        lines.append(f"lighting: {compact['lighting']}")
        lines.append(f"scene: {compact['scene']}")
        return "\n".join(lines)
