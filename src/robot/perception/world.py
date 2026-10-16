"""World state models and spatial tracking.

Represents continuous physical perception of the environment:
- People present (position, bounding box, attention)
- Visual objects / landmarks
- Tabletop items with 3D spatial schema and temporal smoothing
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
    """Detected person in visual frame."""

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
    """Detected general visual object."""

    label: str
    confidence: float
    bbox: list[int] = field(default_factory=list)  # [x, y, w, h]
    position: str = "center"  # "left", "center", "right"


@dataclass
class TabletopItem:
    """Compact 3D spatial representation of an object on the tabletop or desk."""

    object_name: str
    confidence: float
    bounding_box: list[int] = field(default_factory=list)  # [x, y, w, h] in pixels
    estimated_distance: str = "near"  # "near" (<0.5m), "medium" (0.5-1.2m), "far" (>1.2m)
    estimated_distance_m: float = 0.5  # Estimated radial distance in meters
    position_clock: str = "12 o'clock"  # Relative clock position (e.g., "10 o'clock", "12 o'clock", "2 o'clock")
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert tabletop item to dictionary format."""
        return {
            "object_name": self.object_name,
            "confidence": round(self.confidence, 2),
            "bounding_box": self.bounding_box,
            "estimated_distance": self.estimated_distance,
            "estimated_distance_m": round(self.estimated_distance_m, 2),
            "position_clock": self.position_clock,
            "age_seconds": round(time.time() - self.last_seen, 1),
        }


@dataclass
class WorldState:
    """Continuous physical world state snapshot with spatial tabletop tracking."""

    people: list[DetectedPerson] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    tabletop_items: list[TabletopItem] = field(default_factory=list)
    motion: str = "none"  # "none", "low", "medium", "high"
    motion_score: float = 0.0
    lighting: str = "normal"  # "dark", "dim", "normal", "bright"
    scene: str = "indoor"
    updated_at: float = field(default_factory=time.time)

    def update_tabletop_tracking(
        self, new_items: list[TabletopItem], expiration_s: float = 3.0, alpha: float = 0.7
    ) -> None:
        """Track tabletop items across successive frames with temporal smoothing and expiration.

        Args:
            new_items: Freshly detected tabletop items in the current frame.
            expiration_s: Seconds before an unseen item is expired from memory.
            alpha: Exponential smoothing factor for bounding box and confidence (0.0 to 1.0).
        """
        now = time.time()
        updated_list: list[TabletopItem] = []

        # Retain items with temporal smoothing for matched labels
        for new_item in new_items:
            match = None
            for existing in self.tabletop_items:
                if existing.object_name.lower() == new_item.object_name.lower():
                    # Calculate center distance between bounding boxes
                    if existing.bounding_box and new_item.bounding_box:
                        cx1 = existing.bounding_box[0] + existing.bounding_box[2] / 2
                        cy1 = existing.bounding_box[1] + existing.bounding_box[3] / 2
                        cx2 = new_item.bounding_box[0] + new_item.bounding_box[2] / 2
                        cy2 = new_item.bounding_box[1] + new_item.bounding_box[3] / 2
                        if abs(cx1 - cx2) < 120 and abs(cy1 - cy2) < 120:
                            match = existing
                            break

            if match:
                # Apply exponential smoothing to bounding box and distance
                smoothed_bbox = [
                    int(alpha * n + (1 - alpha) * o)
                    for n, o in zip(new_item.bounding_box, match.bounding_box)
                ]
                smoothed_conf = alpha * new_item.confidence + (1 - alpha) * match.confidence
                smoothed_dist_m = (
                    alpha * new_item.estimated_distance_m + (1 - alpha) * match.estimated_distance_m
                )
                updated_list.append(
                    TabletopItem(
                        object_name=new_item.object_name,
                        confidence=smoothed_conf,
                        bounding_box=smoothed_bbox,
                        estimated_distance=new_item.estimated_distance,
                        estimated_distance_m=smoothed_dist_m,
                        position_clock=new_item.position_clock,
                        last_seen=now,
                    )
                )
            else:
                new_item.last_seen = now
                updated_list.append(new_item)

        # Retain previously seen items that haven't expired yet
        for old_item in self.tabletop_items:
            if not any(u.object_name.lower() == old_item.object_name.lower() for u in updated_list):
                if (now - old_item.last_seen) < expiration_s:
                    updated_list.append(old_item)

        self.tabletop_items = updated_list

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

        tabletop_summary = [item.to_dict() for item in self.tabletop_items]

        return {
            "people_count": len(self.people),
            "people": people_summary,
            "objects_count": len(self.objects),
            "objects": objects_summary,
            "tabletop_items": tabletop_summary,
            "tabletop_count": len(self.tabletop_items),
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

        if compact["tabletop_items"]:
            lines.append("tabletop_items:")
            for t in compact["tabletop_items"]:
                lines.append(f"  - item: {t['object_name']}")
                lines.append(f"    distance: {t['estimated_distance']} (~{t['estimated_distance_m']}m)")
                lines.append(f"    clock_position: {t['position_clock']}")

        lines.append(f"motion: {compact['motion']}")
        lines.append(f"lighting: {compact['lighting']}")
        lines.append(f"scene: {compact['scene']}")
        return "\n".join(lines)
