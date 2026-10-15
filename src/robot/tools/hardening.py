"""Tool input hardening, strict type coercion, alias resolution, and validation framework.

Architectural Rationale:
1. Zero-Failure Execution: Local LLM tool calls frequently output subtle schema discrepancies
   (e.g., passing strings for integers, alternative parameter aliases like 'query' vs 'track',
   or percentage strings like '60%'). This module normalizes inputs deterministically without
   destabilizing tool semantics or inventing unintended values.
2. Safety & Conflict Detection: Silently ignoring conflicting arguments or blindly coercing
   arbitrary values creates security and state hazards. If a canonical argument and its alias
   conflict, a structured validation error is produced immediately.
3. Universal Compatibility: Supports both synchronous and asynchronous tool handlers,
   preserving function signatures, metadata, and asyncio cancellation semantics.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

from robot.telemetry import get_logger

logger = get_logger("tools.hardening")

F = TypeVar("F", bound=Callable[..., Any])


class ToolOutcomeStatus(str, Enum):
    """Standardized outcome statuses across all tool invocations.

    Standard status values:
        SUCCESS: Tool executed completely and produced output.
        FAILURE: Tool handler failed or raised an exception.
        DENIED: Permission denied or consequential confirmation required.
        TIMEOUT: Tool execution exceeded allotted time cap.
        CANCELLED: Tool execution aborted by user or task supervisor.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    VALIDATION_ERROR = "validation_error"
    UNSUPPORTED = "unsupported"
    PERMISSION_DENIED = "denial"
    EXECUTION_ERROR = "failure"


class ToolValidationError(ValueError):
    """Structured validation error raised when tool arguments fail validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        received_value: Any = None,
        expected_type: str | None = None,
        code: str = "validation_error",
    ) -> None:
        """Initialize the structured validation error.

        Args:
            message: Human-readable error description.
            field: Name of the parameter that failed validation.
            received_value: Actual value received from the caller.
            expected_type: Expected type or constraint description.
            code: Machine-readable error code.
        """
        super().__init__(message)
        self.field: str | None = field
        self.received_value: Any = received_value
        self.expected_type: str | None = expected_type
        self.code: str = code

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a standardized structured outcome dictionary.

        Returns:
            Dictionary detailing the validation failure.
        """
        return {
            "status": ToolOutcomeStatus.VALIDATION_ERROR.value,
            "error": str(self),
            "code": self.code,
            "field": self.field,
            "received_value": repr(self.received_value) if self.received_value is not None else None,
            "expected_type": self.expected_type,
        }


@dataclass
class ParameterConstraint:
    """Validation rules and coercion constraints for a single tool parameter."""

    target_type: type | None = None
    required: bool = False
    default: Any = None
    min_val: float | int | None = None
    max_val: float | int | None = None
    allowed_choices: list[Any] | None = None
    aliases: list[str] = field(default_factory=list)
    regex_pattern: str | None = None


# Canonical approved parameter aliases across the entire 59-tool ecosystem
GLOBAL_APPROVED_ALIASES: dict[str, list[str]] = {
    "fact": ["text", "memory", "content", "note", "information", "info", "data", "message", "value"],
    "query": [
        "q", "search", "search_query", "term", "keyword", "keywords", "topic", "text",
        "song", "track", "title", "name", "prompt", "song_name", "track_name",
        "music", "audio", "video", "url", "target", "item", "subject"
    ],
    "action": ["act", "motion", "command", "type", "operation", "gesture", "movement", "mode"],
    "prompt": ["text", "description", "query", "image_prompt", "caption"],
    "name": ["personality", "character", "persona", "mode", "bot_name", "name_str"],
    "timezone_str": ["timezone", "tz", "country", "city", "location", "place", "region"],
    "source": ["platform", "src", "provider", "music_source"],
    "volume": ["vol", "level", "val"],
    "expression": ["expr", "calc", "formula", "math", "equation", "calculation", "query"],
    "job_id": ["id", "job", "task_id", "jobId"],
    "preset": ["preset_name", "profile", "eq", "name"],
    "duration_seconds": ["duration", "seconds", "sec", "secs", "time", "val", "minutes", "min", "mins"],
    "target_time": ["time", "time_expression", "target", "when", "target_time_expr", "at", "time_str"],
    "time_expression": ["target_time", "time", "when", "time_expr", "target", "at"],
    "message": ["note", "text", "content", "reminder", "task", "msg", "label", "body"],
    "identifier": ["id", "job_id", "timer_id", "alarm_id", "reminder_id", "name", "label", "title"],
    "app_name": ["app", "name", "application", "program", "command", "target", "app_name_str", "query", "input", "arg", "args", "val", "value", "text"],
    "title": ["summary", "header", "topic", "subject", "heading", "name"],
    "text": ["content", "data", "value", "string", "snippet", "body", "code"],
    "location": ["city", "place", "where", "locale", "area", "address", "town", "query"],
    "category": ["topic", "section", "type", "genre", "category_name", "feed"],
    "symbol": ["ticker", "coin", "stock", "crypto", "asset", "company", "token", "query", "target"],
    "entity_id": ["device", "target", "entity", "light", "switch", "lamp", "thermostat", "name", "id"],
    "domain": ["type", "filter", "category", "scope"],
    "level_percent": ["level", "percent", "brightness", "val", "value", "percentage"],
}


def coerce_value(val: Any, target_type: type | None, param_name: str) -> Any:
    """Strictly coerce an argument value into the specified target type.

    Args:
        val: Raw argument value.
        target_type: Expected Python type (int, float, bool, str, list, dict).
        param_name: Parameter name for contextual error reporting.

    Returns:
        Coerced value matching target_type.

    Raises:
        ToolValidationError: If value cannot be strictly and safely coerced.
    """
    if val is None or target_type is None or target_type is Any:
        return val

    # Strict check: Boolean is an int subclass in Python; reject booleans for int/float
    if target_type in (int, float) and isinstance(val, bool):
        raise ToolValidationError(
            f"Boolean value {val} cannot be coerced to {target_type.__name__} for '{param_name}'",
            field=param_name,
            received_value=val,
            expected_type=target_type.__name__,
        )

    # Direct match
    if isinstance(val, target_type):
        return val

    try:
        # 1. Integer Coercion
        if target_type is int:
            if isinstance(val, float):
                return int(round(val))
            if isinstance(val, str):
                cleaned = val.strip().rstrip("%").rstrip("px").rstrip("s")
                return int(cleaned)
            raise ValueError()

        # 2. Float Coercion
        if target_type is float:
            if isinstance(val, (int, str)):
                cleaned = str(val).strip().rstrip("%").rstrip("s")
                return float(cleaned)
            raise ValueError()

        # 3. Boolean Coercion
        if target_type is bool:
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("true", "1", "yes", "y", "on", "enable", "enabled"):
                    return True
                if s in ("false", "0", "no", "n", "off", "disable", "disabled"):
                    return False
            raise ValueError()

        # 4. String Coercion
        if target_type is str:
            return str(val)

        # 5. List / Sequence Coercion
        if target_type in (list, tuple):
            if isinstance(val, str):
                val_s = val.strip()
                if val_s.startswith("[") and val_s.endswith("]"):
                    return json.loads(val_s)
                return [item.strip() for item in val_s.split(",") if item.strip()]
            if isinstance(val, (set, tuple, list)):
                return list(val)
            raise ValueError()

        # 6. Dict Coercion
        if target_type is dict:
            if isinstance(val, str):
                return json.loads(val)
            raise ValueError()

    except Exception as exc:
        raise ToolValidationError(
            f"Cannot coerce parameter '{param_name}' with value {val!r} to expected type {target_type.__name__}",
            field=param_name,
            received_value=val,
            expected_type=target_type.__name__,
        ) from exc

    return val


def harden_tool_arguments(
    raw_args: dict[str, Any] | None,
    constraints: dict[str, ParameterConstraint] | None = None,
    custom_aliases: dict[str, list[str]] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize, resolve aliases, detect conflicts, and validate tool input arguments.

    Args:
        raw_args: Raw arguments provided by caller or LLM tool call.
        constraints: Optional mapping of parameter constraints.
        custom_aliases: Optional parameter-specific alias mapping.
        schema: Optional JSON schema from ToolDefinition.

    Returns:
        Clean, validated, and type-coerced arguments dictionary.

    Raises:
        ToolValidationError: If arguments conflict, fail type coercion, or violate constraints.
    """
    raw = dict(raw_args or {})
    result: dict[str, Any] = {}
    effective_aliases: dict[str, list[str]] = dict(GLOBAL_APPROVED_ALIASES)
    if custom_aliases:
        effective_aliases.update(custom_aliases)

    # 1. Extract schema-based constraints first so target parameter names are known
    effective_constraints: dict[str, ParameterConstraint] = dict(constraints or {})
    if schema:
        props = schema.get("properties", {})
        required_list = schema.get("required", [])
        for p_name, p_spec in props.items():
            if p_name not in effective_constraints:
                p_type_str = p_spec.get("type", "string")
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                c = ParameterConstraint(
                    target_type=type_map.get(p_type_str, str),
                    required=(p_name in required_list),
                    default=p_spec.get("default", None),
                    min_val=p_spec.get("minimum", None),
                    max_val=p_spec.get("maximum", None),
                    allowed_choices=p_spec.get("enum", None),
                )
                effective_constraints[p_name] = c

    known_target_params = set(effective_constraints.keys())

    # 2. Contextual Alias Resolution & Conflict Detection
    # If the tool has defined parameters, prioritize matching aliases to those parameters
    for key, val in list(raw.items()):
        canonical_key = key
        if known_target_params and key in known_target_params:
            canonical_key = key
        elif known_target_params:
            for target_param in known_target_params:
                aliases_for_target = effective_aliases.get(target_param, [])
                if key in aliases_for_target:
                    canonical_key = target_param
                    break
        else:
            # Fallback global lookup
            for canonical, alias_list in effective_aliases.items():
                if key == canonical:
                    canonical_key = canonical
                    break
                if key in alias_list:
                    canonical_key = canonical
                    break

        if canonical_key in result:
            existing_val = result[canonical_key]
            if existing_val != val:
                raise ToolValidationError(
                    f"Conflicting arguments received for parameter '{canonical_key}': "
                    f"canonical '{canonical_key}'={existing_val!r} conflicts with alias '{key}'={val!r}",
                    field=canonical_key,
                    received_value={canonical_key: existing_val, key: val},
                    code="conflicting_arguments",
                )
        else:
            result[canonical_key] = val

    # 3. Type Coercion, Default Application, and Range Validation
    for p_name, constraint in effective_constraints.items():
        if p_name in result:
            val = result[p_name]
            # Coerce type
            coerced = coerce_value(val, constraint.target_type, p_name)

            # Range checks
            if constraint.min_val is not None and isinstance(coerced, (int, float)):
                if coerced < constraint.min_val:
                    raise ToolValidationError(
                        f"Parameter '{p_name}' value {coerced} is below minimum allowed value {constraint.min_val}",
                        field=p_name,
                        received_value=coerced,
                        code="value_below_minimum",
                    )
            if constraint.max_val is not None and isinstance(coerced, (int, float)):
                if coerced > constraint.max_val:
                    raise ToolValidationError(
                        f"Parameter '{p_name}' value {coerced} exceeds maximum allowed value {constraint.max_val}",
                        field=p_name,
                        received_value=coerced,
                        code="value_above_maximum",
                    )

            # Allowed choices (enum)
            if constraint.allowed_choices is not None:
                if coerced not in constraint.allowed_choices:
                    raise ToolValidationError(
                        f"Parameter '{p_name}' value {coerced!r} is not in allowed choices: {constraint.allowed_choices}",
                        field=p_name,
                        received_value=coerced,
                        code="invalid_choice",
                    )

            # Regex pattern
            if constraint.regex_pattern is not None and isinstance(coerced, str):
                if not re.match(constraint.regex_pattern, coerced):
                    raise ToolValidationError(
                        f"Parameter '{p_name}' value {coerced!r} does not match required pattern",
                        field=p_name,
                        received_value=coerced,
                        code="pattern_mismatch",
                    )

            result[p_name] = coerced
        else:
            # Check if required
            if constraint.required and constraint.default is None:
                raise ToolValidationError(
                    f"Missing required parameter '{p_name}'",
                    field=p_name,
                    received_value=None,
                    code="missing_required_parameter",
                )
            if constraint.default is not None:
                result[p_name] = constraint.default

    return result


def harden_tool_inputs(
    constraints: dict[str, ParameterConstraint] | None = None,
    custom_aliases: dict[str, list[str]] | None = None,
    schema: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to harden, validate, and normalize inputs for synchronous or asynchronous tool handlers.

    Args:
        constraints: Optional mapping of parameter constraints.
        custom_aliases: Optional approved alias mapping.
        schema: Optional JSON schema dictionary.

    Returns:
        Decorated tool handler function.
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    hardened_kwargs = harden_tool_arguments(
                        kwargs,
                        constraints=constraints,
                        custom_aliases=custom_aliases,
                        schema=schema,
                    )
                    return await func(*args, **hardened_kwargs)
                except ToolValidationError as err:
                    logger.warning("tool_input_validation_failed", error=str(err), tool=func.__name__)
                    return err.to_dict()

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    hardened_kwargs = harden_tool_arguments(
                        kwargs,
                        constraints=constraints,
                        custom_aliases=custom_aliases,
                        schema=schema,
                    )
                    return func(*args, **hardened_kwargs)
                except ToolValidationError as err:
                    logger.warning("tool_input_validation_failed", error=str(err), tool=func.__name__)
                    return err.to_dict()

            return sync_wrapper  # type: ignore[return-value]

    return decorator
