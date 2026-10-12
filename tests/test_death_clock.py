"""Unit tests for Actuarial Death Clock tool and epidemiological models."""

from __future__ import annotations

import pytest

from robot.tools.builtins.death_clock import (
    _calculate_actuarial_baseline,
    _compute_lifestyle_delta,
    estimate_death_clock,
    register_death_clock_tools,
)
from robot.tools import ToolRegistry


def test_actuarial_country_baselines() -> None:
    """Test country and gender baseline longevity lookups."""
    assert _calculate_actuarial_baseline("India", "male") == pytest.approx(70.4, 0.1)
    assert _calculate_actuarial_baseline("India", "female") == pytest.approx(73.6, 0.1)
    assert _calculate_actuarial_baseline("Japan", "female") == pytest.approx(87.6, 0.1)
    assert _calculate_actuarial_baseline("UnknownLand", "male") == pytest.approx(71.0, 0.1)


def test_lifestyle_delta_computation() -> None:
    """Test actuarial year deltas across positive and hazardous habits."""
    # Healthy subject
    delta_healthy, impacts_healthy = _compute_lifestyle_delta(
        smoking="never",
        exercise_days_per_week=5,
        diet_quality="healthy",
        sleep_hours=7.5,
        stress_level="low",
        alcohol="none",
        bmi=22.0,
    )
    assert delta_healthy > 5.0
    assert any(item["factor"].startswith("High Physical Activity") for item in impacts_healthy)

    # High hazard subject
    delta_hazard, impacts_hazard = _compute_lifestyle_delta(
        smoking="heavy",
        exercise_days_per_week=0,
        diet_quality="poor",
        sleep_hours=5.0,
        stress_level="extreme",
        alcohol="heavy",
        bmi=33.0,
    )
    assert delta_hazard < -20.0
    assert any(item["factor"] == "Smoking (Heavy)" for item in impacts_hazard)


@pytest.mark.asyncio
async def test_estimate_death_clock_deterministic() -> None:
    """Test full death clock estimation pipeline with deterministic model."""
    res = await estimate_death_clock(
        age=21,
        gender="male",
        country="India",
        smoking="never",
        exercise_days_per_week=4,
        diet_quality="healthy",
        sleep_hours=8.0,
        stress_level="moderate",
        alcohol="none",
        bmi=23.0,
        use_gemini=False,
    )

    assert res["status"] == "success"
    assert res["current_age"] == 21
    assert res["projected_lifespan_years"] > 70.0
    assert res["years_remaining"] > 45.0
    assert res["days_remaining"] > 16000
    assert res["hours_remaining"] > 380000
    assert res["projected_death_year"] > 2070
    assert isinstance(res["actionable_recommendations"], list)
    assert len(res["actionable_recommendations"]) >= 1
    assert res["engine"] == "Deterministic Epidemiological Actuarial Model"


def test_death_clock_tool_registration() -> None:
    """Test tool registration in ToolRegistry."""
    class DummyEngine:
        def __init__(self) -> None:
            self.tools = ToolRegistry()

    engine = DummyEngine()
    register_death_clock_tools(engine)  # type: ignore[arg-type]

    tool = engine.tools.get("system.death_clock")
    assert tool is not None
    assert tool.name == "system.death_clock"
    assert tool.family == "system"

    # Verify aliases
    alias_tool = engine.tools.get("death_clock")
    assert alias_tool is not None
    assert alias_tool.name == "system.death_clock"

    health_tool = engine.tools.get("health.death_clock")
    assert health_tool is not None
    assert health_tool.name == "system.death_clock"
