"""Built-in tools for smart home and IoT automation via Home Assistant REST/WebSocket API.

Provides:
- homeassistant.control: Control smart lights, switches, plugs, and climate thermostats.
- homeassistant.state: Inspect the real-time state and attributes of any IoT entity.
- homeassistant.list_devices: Enumerate smart home entities, lights, switches, and sensors.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.smart_home")

# Local emulated device registry when Home Assistant is offline or in standalone mode
_EMULATED_DEVICES: dict[str, dict[str, Any]] = {
    "light.living_room": {
        "entity_id": "light.living_room",
        "state": "on",
        "attributes": {"friendly_name": "Living Room Light", "brightness": 204, "color_mode": "rgb"},
    },
    "light.desk_lamp": {
        "entity_id": "light.desk_lamp",
        "state": "on",
        "attributes": {"friendly_name": "Studio Desk Lamp", "brightness": 255, "rgb_color": [0, 255, 128]},
    },
    "switch.fan": {
        "entity_id": "switch.fan",
        "state": "off",
        "attributes": {"friendly_name": "Ceiling Fan", "device_class": "fan"},
    },
    "switch.cyber_deck_led": {
        "entity_id": "switch.cyber_deck_led",
        "state": "on",
        "attributes": {"friendly_name": "Cyber-Deck Ambient Lighting"},
    },
    "climate.thermostat": {
        "entity_id": "climate.thermostat",
        "state": "cool",
        "attributes": {
            "friendly_name": "Main AC Thermostat",
            "current_temperature": 23.5,
            "target_temperature": 22.0,
            "hvac_modes": ["off", "cool", "heat", "fan_only"],
        },
    },
}


class HomeAssistantClient:
    """Async client for Home Assistant REST API with graceful local emulation."""

    def __init__(self, base_url: str = "http://homeassistant.local:8123", access_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self._is_connected: bool | None = None

    @property
    def is_configured(self) -> bool:
        """Check if Home Assistant credentials are provided."""
        return bool(self.access_token and self.access_token.strip())

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        service_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a service call against Home Assistant or fallback to emulated state."""
        data = dict(service_data or {})
        data["entity_id"] = entity_id

        if self.is_configured:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            try:
                async with httpx.AsyncClient(headers=headers, timeout=3.0) as client:
                    url = f"{self.base_url}/api/services/{domain}/{service}"
                    resp = await client.post(url, json=data)
                    if resp.status_code in (200, 201):
                        self._is_connected = True
                        return {
                            "status": "success",
                            "message": f"Successfully executed {domain}.{service} on {entity_id}.",
                            "data": resp.json() if resp.text else {},
                        }
            except Exception as e:
                logger.warning("home_assistant_rest_fallback", error=str(e))
                self._is_connected = False

        # Emulated device state updater
        clean_eid = entity_id.strip().lower()
        if clean_eid not in _EMULATED_DEVICES:
            # Auto-register entity
            _EMULATED_DEVICES[clean_eid] = {
                "entity_id": clean_eid,
                "state": "off",
                "attributes": {"friendly_name": clean_eid.replace(".", " ").title()},
            }

        dev = _EMULATED_DEVICES[clean_eid]
        if service in ("turn_on", "on"):
            dev["state"] = "on"
            if "brightness" in data:
                dev["attributes"]["brightness"] = data["brightness"]
            if "temperature" in data:
                dev["attributes"]["target_temperature"] = data["temperature"]
        elif service in ("turn_off", "off"):
            dev["state"] = "off"
        elif service in ("toggle",):
            dev["state"] = "off" if dev["state"] == "on" else "on"
        elif service in ("set_temperature", "temperature"):
            if "temperature" in data:
                dev["attributes"]["target_temperature"] = data["temperature"]

        friendly = dev.get("attributes", {}).get("friendly_name", clean_eid)
        return {
            "status": "success",
            "mode": "live" if self._is_connected else "local_emulated",
            "entity_id": clean_eid,
            "friendly_name": friendly,
            "new_state": dev["state"],
            "attributes": dev.get("attributes", {}),
            "message": f"Set {friendly} ({clean_eid}) to {dev['state']}.",
        }

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        """Fetch current state of an IoT entity."""
        if self.is_configured:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            try:
                async with httpx.AsyncClient(headers=headers, timeout=3.0) as client:
                    url = f"{self.base_url}/api/states/{entity_id}"
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return {"status": "success", "state": resp.json()}
            except Exception:
                pass

        clean_eid = entity_id.strip().lower()
        if clean_eid in _EMULATED_DEVICES:
            return {"status": "success", "mode": "local_emulated", "state": _EMULATED_DEVICES[clean_eid]}
        return {"status": "not_found", "message": f"Entity '{entity_id}' not found."}

    async def list_entities(self, domain: str = "all") -> list[dict[str, Any]]:
        """List all discovered smart home entities."""
        if self.is_configured:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            try:
                async with httpx.AsyncClient(headers=headers, timeout=3.0) as client:
                    url = f"{self.base_url}/api/states"
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        all_states = resp.json()
                        if domain != "all":
                            all_states = [s for s in all_states if s.get("entity_id", "").startswith(f"{domain}.")]
                        return all_states
            except Exception:
                pass

        devices = list(_EMULATED_DEVICES.values())
        if domain != "all":
            devices = [d for d in devices if d.get("entity_id", "").startswith(f"{domain}.")]
        return devices


def register_smart_home_tools(engine: "RobotEngine") -> None:
    """Register Home Assistant and IoT smart home tools with the engine's tool registry.

    Args:
        engine: Reference to the RobotEngine instance.
    """
    ha_client = HomeAssistantClient()

    # 1. homeassistant.control
    async def ha_control(
        entity_id: str = "light.living_room",
        action: str = "turn_on",
        value: int | float | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Control smart lights, switches, fans, or climate thermostats."""
        target_entity = (entity_id or kwargs.get("device") or kwargs.get("target") or "light.living_room").strip().lower()
        target_action = (action or kwargs.get("command") or kwargs.get("act") or "turn_on").strip().lower()

        # Infer domain from entity_id
        domain = target_entity.split(".")[0] if "." in target_entity else "light"
        service = target_action

        service_data: dict[str, Any] = {}

        if target_action in ("on", "enable", "start"):
            service = "turn_on"
        elif target_action in ("off", "disable", "stop"):
            service = "turn_off"
        elif target_action in ("dim", "brightness", "set_brightness"):
            service = "turn_on"
            if value is not None:
                try:
                    b_pct = int(float(value) * 2.55) if float(value) <= 100 else int(float(value))
                    service_data["brightness"] = max(1, min(255, b_pct))
                except Exception:
                    pass
        elif target_action in ("temp", "set_temp", "temperature", "set_temperature"):
            domain = "climate"
            service = "set_temperature"
            if value is not None:
                try:
                    service_data["temperature"] = float(value)
                except Exception:
                    pass

        return await ha_client.call_service(domain, service, target_entity, service_data)

    engine.tool_registry.register(
        ToolDefinition(
            name="homeassistant.control",
            description="Control smart home IoT devices (smart lights, switches, ceiling fans, AC thermostats) via Home Assistant. Spoken triggers: 'Turn on the living room light', 'Turn off desk lamp', 'Set thermostat to 22 degrees', 'Dim the lights to 50%'.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID of the smart device (e.g. 'light.living_room', 'light.desk_lamp', 'switch.fan', 'climate.thermostat')",
                    },
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'turn_on', 'turn_off', 'toggle', 'set_brightness', 'set_temperature'",
                        "default": "turn_on",
                    },
                    "value": {
                        "type": "string",
                        "description": "Optional value (e.g. 50 for 50% brightness, 22 for 22°C temperature)",
                    },
                },
                "required": ["entity_id"],
            },
            handler=ha_control,
            permission_level=PermissionLevel.HARMLESS_LOCAL,
            family="smart_home",
            is_idempotent=False,
            max_retries=0,
        )
    )

    # 2. homeassistant.state
    async def ha_state(entity_id: str = "light.living_room", **kwargs: Any) -> dict[str, Any]:
        """Query state and attributes of a smart home entity."""
        target_entity = (entity_id or kwargs.get("device") or "light.living_room").strip().lower()
        return await ha_client.get_state(target_entity)

    engine.tool_registry.register(
        ToolDefinition(
            name="homeassistant.state",
            description="Check the current real-time state, temperature, or brightness of a smart home device. Spoken triggers: 'Is the living room light on?', 'What is the AC thermostat temperature?'.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID (e.g. 'light.living_room', 'climate.thermostat')",
                    },
                },
                "required": ["entity_id"],
            },
            handler=ha_state,
            permission_level=PermissionLevel.READ_ONLY,
            family="smart_home",
        )
    )

    # 3. homeassistant.list_devices
    async def ha_list_devices(domain: str = "all", **kwargs: Any) -> dict[str, Any]:
        """List all available smart home devices and IoT entities."""
        eff_domain = (domain or kwargs.get("type") or "all").strip().lower()
        devices = await ha_client.list_entities(eff_domain)
        return {
            "status": "success",
            "domain": eff_domain,
            "count": len(devices),
            "devices": devices,
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="homeassistant.list_devices",
            description="List all discovered smart home devices, lights, switches, and thermostats in the home. Spoken triggers: 'What smart devices do I have?', 'List smart lights'.",
            parameters={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filter by domain ('light', 'switch', 'climate', 'all')",
                        "default": "all",
                    },
                },
            },
            handler=ha_list_devices,
            permission_level=PermissionLevel.READ_ONLY,
            family="smart_home",
        )
    )
