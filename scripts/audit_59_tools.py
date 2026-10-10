"""Audit all registered robot tools for signatures, schemas, permissions, and risk tiers."""

from __future__ import annotations

import inspect
import json
from typing import Any

from robot.tools import PermissionLevel, ToolDefinition, ToolRegistry
from robot.tools.builtins import register_builtin_tools


class DummyEngine:
    """Mock engine providing a clean ToolRegistry for auditing."""

    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()


def run_audit() -> dict[str, Any]:
    """Execute complete inventory audit across all registered tools."""
    engine = DummyEngine()
    register_builtin_tools(engine)
    registry = engine.tool_registry

    all_tools: list[ToolDefinition] = registry.get_all()
    audit_report: dict[str, Any] = {
        "total_tools": len(all_tools),
        "families": {},
        "permission_counts": {},
        "tools": [],
    }

    for tool in sorted(all_tools, key=lambda t: t.name):
        # Inspect handler signature
        sig = inspect.signature(tool.handler)
        params_info = {}
        for p_name, p in sig.parameters.items():
            params_info[p_name] = {
                "default": str(p.default) if p.default is not inspect.Parameter.empty else "<REQUIRED>",
                "annotation": str(p.annotation) if p.annotation is not inspect.Parameter.empty else "<ANY>",
            }

        # Check schema
        schema_props = tool.parameters.get("properties", {})
        schema_required = tool.parameters.get("required", [])

        t_entry = {
            "name": tool.name,
            "family": tool.family,
            "permission_level": tool.permission_level.name,
            "timeout_s": tool.timeout_s,
            "risk_level": tool.risk_level,
            "signature_parameters": params_info,
            "schema_properties": list(schema_props.keys()),
            "schema_required": schema_required,
            "has_description": bool(tool.description),
        }
        audit_report["tools"].append(t_entry)

        # Counters
        fam = tool.family or "unknown"
        audit_report["families"][fam] = audit_report["families"].get(fam, 0) + 1

        perm = tool.permission_level.name
        audit_report["permission_counts"][perm] = audit_report["permission_counts"].get(perm, 0) + 1

    return audit_report


if __name__ == "__main__":
    report = run_audit()
    print("=" * 70)
    print(f"TOOL AUDIT REPORT: {report['total_tools']} TOOLS REGISTERED")
    print("=" * 70)
    print("Family Breakdown:")
    for fam, cnt in sorted(report["families"].items()):
        print(f"  - {fam:<15}: {cnt:>2} tools")
    print("\nPermission Levels:")
    for perm, cnt in sorted(report["permission_counts"].items()):
        print(f"  - {perm:<15}: {cnt:>2} tools")
    print("=" * 70)

    # Save detailed JSON report
    with open("scratch/tool_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved complete audit report to scratch/tool_audit_report.json")
