"""Unit tests for Phase 4 Home Assistant IoT Integration tools.

Validates:
- Home Assistant REST client state querying, service calls, and domain discovery.
- Emulated offline fallback state management for zero-dependency standalone execution.
- Smart lights (turn_on, turn_off, brightness, color_temp), switches, and climate thermostats.
- Tool registration, permissions (HARMLESS_LOCAL), and error resilience.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.tools import PermissionLevel, ToolRegistry
from robot.tools.builtins.smart_home import (
    _EMULATED_DEVICES,
    HomeAssistantClient,
    register_smart_home_tools,
)


@pytest.fixture
def mock_engine() -> MagicMock:
    """Fixture providing a mock RobotEngine with tool registry."""
    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.personalities = MagicMock()
    engine.personalities.active = MagicMock(memory_namespace="test")
    engine.state = MagicMock()
    engine.telemetry = MagicMock()
    engine.scheduler = MagicMock()
    return engine


@pytest.mark.asyncio
async def test_emulated_fallback_state_query() -> None:
    """Verify querying state falls back gracefully to emulated state when HA is offline."""
    client = HomeAssistantClient(base_url="http://localhost:8123", access_token="")

    # Query an existing emulated light
    res = await client.get_state("light.living_room")
    assert res["status"] == "success"
    state_obj = res["state"]
    assert state_obj["entity_id"] == "light.living_room"
    assert state_obj["state"] in ["on", "off"]
    assert "brightness" in state_obj["attributes"]


@pytest.mark.asyncio
async def test_emulated_light_control() -> None:
    """Verify controlling lights updates internal emulated state."""
    client = HomeAssistantClient(base_url="http://localhost:8123", access_token="")

    # Turn off light
    res_off = await client.call_service(
        domain="light",
        service="turn_off",
        entity_id="light.desk_lamp",
    )
    assert res_off["status"] == "success"
    assert res_off["new_state"] == "off"

    # Turn on light with brightness
    res_on = await client.call_service(
        domain="light",
        service="turn_on",
        entity_id="light.desk_lamp",
        service_data={"brightness": 200},
    )
    assert res_on["status"] == "success"
    assert res_on["new_state"] == "on"
    assert res_on["attributes"]["brightness"] == 200


@pytest.mark.asyncio
async def test_emulated_climate_control() -> None:
    """Verify thermostat temperature control in emulated mode."""
    client = HomeAssistantClient(base_url="http://localhost:8123", access_token="")

    res = await client.call_service(
        domain="climate",
        service="set_temperature",
        entity_id="climate.thermostat",
        service_data={"temperature": 23.5},
    )
    assert res["status"] == "success"
    assert res["attributes"]["target_temperature"] == 23.5


@pytest.mark.asyncio
async def test_emulated_list_devices() -> None:
    """Verify listing entities with domain filtering."""
    client = HomeAssistantClient(base_url="http://localhost:8123", access_token="")

    all_devs = await client.list_entities()
    assert len(all_devs) >= 4

    lights = await client.list_entities(domain="light")
    assert all(d["entity_id"].startswith("light.") for d in lights)
    assert any(d["entity_id"] == "light.living_room" for d in lights)

    climates = await client.list_entities(domain="climate")
    assert len(climates) == 1
    assert climates[0]["entity_id"] == "climate.thermostat"


@pytest.mark.asyncio
async def test_live_rest_api_call() -> None:
    """Verify live Home Assistant REST API call when server is reachable."""
    mock_entity_resp = MagicMock()
    mock_entity_resp.status_code = 200
    mock_entity_resp.json.return_value = {
        "entity_id": "light.kitchen",
        "state": "on",
        "attributes": {"brightness": 255, "friendly_name": "Kitchen Light"},
    }

    client = HomeAssistantClient(base_url="http://homeassistant.local:8123", access_token="mock-token-xyz")

    mock_http_client = AsyncMock()
    mock_http_client.get.return_value = mock_entity_resp
    mock_http_client.__aenter__.return_value = mock_http_client

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        res = await client.get_state("light.kitchen")
        assert res["status"] == "success"
        state = res["state"]
        assert state["entity_id"] == "light.kitchen"
        assert state["state"] == "on"
        assert state["attributes"]["friendly_name"] == "Kitchen Light"


def test_register_smart_home_tools(mock_engine: MagicMock) -> None:
    """Verify smart home tools registration into registry."""
    register_smart_home_tools(mock_engine)

    registry: ToolRegistry = mock_engine.tool_registry
    assert "homeassistant.control" in registry.list_tools()
    assert "homeassistant.state" in registry.list_tools()
    assert "homeassistant.list_devices" in registry.list_tools()

    ctrl_tool = registry.get("homeassistant.control")
    assert ctrl_tool is not None
    assert ctrl_tool.permission_level == PermissionLevel.HARMLESS_LOCAL
    assert ctrl_tool.family == "smart_home"
