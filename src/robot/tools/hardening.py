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
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

from robot.telemetry import get_logger

logger = get_logger("tools.hardening")

F = TypeVar("F", bound=Callable[..., Any])

# ─── Global safety bounds ────────────────────────────────────────────────────
# Tool arguments and tool outputs both cross a trust boundary (the LLM).
# These ceilings keep a single malformed tool call from exhausting memory or
# bloating the model context.  Per-schema ``maxLength`` / ``maxItems`` values
# may lower the limits but never raise them above these hard ceilings.
MAX_STRING_LENGTH: int = 64 * 1024
MAX_ARRAY_ITEMS: int = 100
MAX_NESTING_DEPTH: int = 6
MAX_STRING_COERCION_LENGTH: int = 256
TRUNCATION_MARKER: str = "...[truncated]"

# Handler-reported failure statuses.  A handler may return a dict instead of
# raising; the framework must never treat these as successful (a timed-out or
# denied tool is not a completed tool).
FAILURE_STATUS_VALUES: frozenset[str] = frozenset(
    {
        "error",
        "failed",
        "failure",
        "timeout",
        "timed_out",
        "denied",
        "denial",
        "not_sent",
        "cancelled",
        "canceled",
        "validation_error",
        "unsupported",
    }
)

_FAILURE_STATUS_NORMALIZATION: dict[str, str] = {
    "error": "failure",
    "failed": "failure",
    "failure": "failure",
    "validation_error": "failure",
    "unsupported": "failure",
    "not_sent": "failure",
    "timeout": "timeout",
    "timed_out": "timeout",
    "denied": "denial",
    "denial": "denial",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}

# Control characters are stripped before tool text is inserted into a model
# prompt or a log line.  Tab/newline/carriage-return are preserved.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Obvious credential shapes.  This is defence-in-depth for prompt/log hygiene,
# not a replacement for not putting secrets in tool arguments at all.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"), "AIza[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|token|password|passwd|secret|authorization)"
            r"\b(\s*[=:]\s*)([^\s,;\"']+)"
        ),
        r"\1\2[REDACTED]",
    ),
)

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|token|password|passwd|secret|"
    r"authorization|credential|cookie)"
)

#: One trailing unit suffix that local models attach to numeric arguments.
_NUMERIC_SUFFIX_RE = re.compile(
    r"\s*(?:%|px|em|rem|pt|"
    r"ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|hours?|days?|"
    r"kb|mb|gb|hz|khz|mhz)$",
    re.IGNORECASE,
)


#: Ceiling for a handler-reported error message before it reaches the model,
#: the event bus, or a log line. Handler errors routinely embed scraped page
#: text or exception reprs, which must not be echoed verbatim.
MAX_CLASSIFIED_ERROR_CHARS = 500


def _sanitize_reported_text(text: str) -> str:
    """Redact, strip control characters from, and bound handler-reported text.

    Args:
        text: Raw text produced by a tool handler.

    Returns:
        str: Text safe to place in a model turn, event payload, or log line.
    """
    cleaned = strip_control_characters(redact_secrets(text))
    return bound_text(cleaned, MAX_CLASSIFIED_ERROR_CHARS)


def classify_tool_result(result: Any) -> tuple[bool, str | None, str | None]:
    """Classify a handler return value into success or failure.

    Handlers in this codebase may report failure either by raising, by
    returning ``{"status": "error"|"timeout"|"denied"|"not_sent"|"cancelled"}``
    or by returning a non-empty ``error`` field.  A returned dictionary must
    not be blindly wrapped as a successful tool outcome.

    Args:
        result: Raw handler return value.

    Returns:
        Tuple of ``(is_failure, normalized_status, error_message)``.  When the
        result is successful, ``is_failure`` is False and the status/error are
        None (or the original non-failure status is ignored).
    """
    if not isinstance(result, dict):
        return False, None, None

    raw_status = result.get("status")
    status_text = str(raw_status).strip().lower() if isinstance(raw_status, str) else ""
    is_failure_status = status_text in FAILURE_STATUS_VALUES

    raw_error = result.get("error")
    has_error = raw_error is not None and str(raw_error).strip() != ""

    if not is_failure_status and not has_error:
        return False, None, None

    normalized = _FAILURE_STATUS_NORMALIZATION.get(status_text, "failure")
    error_message: str | None = None
    if has_error:
        error_message = _sanitize_reported_text(str(raw_error))
    elif status_text:
        error_message = f"Tool reported status '{_sanitize_reported_text(status_text)}'"
    else:
        error_message = "Tool reported an error"
    return True, normalized, error_message


def strip_control_characters(text: str) -> str:
    """Remove non-printable control characters from untrusted text.

    Args:
        text: Input string.

    Returns:
        Text with C0/C1 control bytes removed (newlines/tabs preserved).
    """
    if not text:
        return text
    return _CONTROL_CHAR_RE.sub("", text)


def redact_secrets(text: str) -> str:
    """Redact obvious bearer tokens, API keys, AWS keys, and password fields.

    Args:
        text: Potentially sensitive text.

    Returns:
        Text with credential-like substrings replaced by ``[REDACTED]``.
    """
    if not text:
        return text
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def bound_text(text: str, max_chars: int) -> str:
    """Truncate text to ``max_chars`` characters, appending a visible marker.

    Args:
        text: Input text.
        max_chars: Maximum allowed length.  Non-positive values produce only
            the truncation marker.

    Returns:
        Bounded text containing ``[truncated]`` when clipping occurred.
    """
    if not isinstance(text, str):
        text = str(text)
    if max_chars <= 0:
        return TRUNCATION_MARKER
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_chars]
    return text[: max_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def bound_payload(
    value: Any,
    *,
    max_string_chars: int = MAX_STRING_LENGTH,
    max_items: int = MAX_ARRAY_ITEMS,
    max_depth: int = MAX_NESTING_DEPTH,
) -> Any:
    """Recursively bound, sanitize, and redact a tool result payload.

    Strings are control-character-stripped, secret-redacted, and truncated.
    Containers are capped by item count and recursion depth so a hostile or
    broken tool cannot exhaust memory or trigger a recursion error when the
    payload is serialized into the model context.

    Args:
        value: Arbitrary JSON-like payload.
        max_string_chars: Per-string character ceiling.
        max_items: Per-container item ceiling.
        max_depth: Maximum nesting depth (top-level value is depth 0).

    Returns:
        A sanitized copy of the payload (or a truncation marker at excessive
        depth).
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        cleaned = strip_control_characters(redact_secrets(value))
        return bound_text(cleaned, max_string_chars)

    if isinstance(value, (bytes, bytearray)):
        decoded = bytes(value).decode("utf-8", errors="replace")
        return bound_text(strip_control_characters(redact_secrets(decoded)), max_string_chars)

    if isinstance(value, dict):
        if max_depth == 0:
            return TRUNCATION_MARKER
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated_items"] = len(value) - max_items
                break
            out[str(key)] = bound_payload(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        if max_depth == 0:
            return TRUNCATION_MARKER
        items = list(value)
        out_list = [
            bound_payload(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            out_list.append(TRUNCATION_MARKER)
        return out_list

    return bound_text(strip_control_characters(redact_secrets(str(value))), max_string_chars)


def redact_tool_arguments(
    arguments: Any,
    *,
    max_string_chars: int = 200,
    max_items: int = MAX_ARRAY_ITEMS,
    max_depth: int = MAX_NESTING_DEPTH,
) -> Any:
    """Redact sensitive keys and bound long strings before logging arguments.

    Args:
        arguments: Raw tool-call arguments.
        max_string_chars: Maximum retained length for non-secret strings.
        max_items: Per-container item ceiling.
        max_depth: Maximum nesting depth.

    Returns:
        A log-safe copy of the arguments.
    """
    if arguments is None or isinstance(arguments, (int, float, bool)):
        return arguments
    if isinstance(arguments, str):
        return bound_text(strip_control_characters(redact_secrets(arguments)), max_string_chars)
    if max_depth <= 0:
        return TRUNCATION_MARKER
    if isinstance(arguments, dict):
        out: dict[str, Any] = {}
        for idx, (key, value) in enumerate(arguments.items()):
            if idx >= max_items:
                out["_truncated_items"] = len(arguments) - max_items
                break
            key_str = str(key)
            if _SENSITIVE_KEY_RE.search(key_str):
                out[key_str] = "[REDACTED]"
            else:
                out[key_str] = redact_tool_arguments(
                    value,
                    max_string_chars=max_string_chars,
                    max_items=max_items,
                    max_depth=max_depth - 1,
                )
        return out
    if isinstance(arguments, (list, tuple, set, frozenset)):
        items = list(arguments)
        out_list = [
            redact_tool_arguments(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            out_list.append(TRUNCATION_MARKER)
        return out_list
    return bound_text(
        strip_control_characters(redact_secrets(str(arguments))), max_string_chars
    )


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
            "error": _sanitize_reported_text(str(self)),
            "code": self.code,
            "field": self.field,
            "received_value": (
                _sanitize_reported_text(repr(self.received_value))
                if self.received_value is not None
                else None
            ),
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
    max_length: int | None = None
    max_items: int | None = None
    max_depth: int | None = None


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


def _parse_numeric_string(val: str) -> str:
    """Normalize a model-supplied numeric string into a parseable number.

    Local models routinely decorate numbers with units or group separators
    (``"60%"``, ``"1,500"``, ``"400px"``, ``"5 s"``).  A single trailing unit
    token is removed; everything else is left strict so a genuinely wrong value
    fails validation instead of silently parsing to something unexpected.

    Args:
        val: Raw numeric string from a tool argument.

    Returns:
        str: Text suitable for :func:`int` or :func:`float`.
    """
    text = val.strip().replace(",", "").replace("_", "")
    return _NUMERIC_SUFFIX_RE.sub("", text, count=1).strip()


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
                if not math.isfinite(val):
                    raise ValueError("non-finite float")
                return int(round(val))
            if isinstance(val, str):
                return int(_parse_numeric_string(val))
            raise ValueError()

        # 2. Float Coercion
        if target_type is float:
            if isinstance(val, (int, str)):
                number = float(_parse_numeric_string(val) if isinstance(val, str) else val)
                if not math.isfinite(number):
                    # NaN/Infinity silently propagate through arithmetic and
                    # poison durations, volumes and coordinates downstream.
                    raise ValueError("non-finite number")
                return number
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
            if isinstance(val, (int, float, bool)):
                return bound_text(str(val), MAX_STRING_COERCION_LENGTH)
            # Containers are not strings: ``str({'a': 1})`` would hand the
            # handler a Python repr it cannot parse.
            raise ValueError()

        # 5. List / Sequence Coercion
        if target_type in (list, tuple):
            if isinstance(val, str):
                val_s = val.strip()
                if val_s.startswith("[") and val_s.endswith("]"):
                    parsed = json.loads(val_s)
                    if not isinstance(parsed, list):
                        raise ValueError("JSON value is not an array")
                    return parsed
                return [item.strip() for item in val_s.split(",") if item.strip()]
            if isinstance(val, (set, tuple, list)):
                return list(val)
            raise ValueError()

        # 6. Dict Coercion
        if target_type is dict:
            if isinstance(val, str):
                parsed = json.loads(val)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON value is not an object")
                return parsed
            raise ValueError()

    except Exception as exc:
        raise ToolValidationError(
            f"Cannot coerce parameter '{param_name}' with value {val!r} to expected type {target_type.__name__}",
            field=param_name,
            received_value=val,
            expected_type=target_type.__name__,
        ) from exc

    return val


def _enforce_value_bounds(
    value: Any,
    param_name: str,
    constraint: ParameterConstraint,
) -> None:
    """Enforce maxLength, maxItems, and recursion-depth limits.

    Args:
        value: Coerced parameter value.
        param_name: Parameter name used in error messages.
        constraint: Constraint carrying per-schema overrides.

    Raises:
        ToolValidationError: If a string, container, or nesting limit is exceeded.
    """
    max_length = constraint.max_length if constraint.max_length is not None else MAX_STRING_LENGTH
    max_items = constraint.max_items if constraint.max_items is not None else MAX_ARRAY_ITEMS
    max_depth = constraint.max_depth if constraint.max_depth is not None else MAX_NESTING_DEPTH
    # Clamp schema-provided limits to the global hard ceilings.
    max_length = min(max_length, MAX_STRING_LENGTH)
    max_items = min(max_items, MAX_ARRAY_ITEMS)
    max_depth = min(max_depth, MAX_NESTING_DEPTH)

    def _walk(node: Any, depth: int) -> None:
        if depth > max_depth:
            raise ToolValidationError(
                f"Parameter '{param_name}' exceeds maximum nesting depth of {max_depth}",
                field=param_name,
                received_value="<nested>",
                code="value_too_deep",
            )
        if isinstance(node, str):
            if len(node) > max_length:
                raise ToolValidationError(
                    f"Parameter '{param_name}' string length {len(node)} exceeds maximum of {max_length}",
                    field=param_name,
                    received_value=f"<string len={len(node)}>",
                    code="value_too_long",
                )
            return
        if isinstance(node, dict):
            if len(node) > max_items:
                raise ToolValidationError(
                    f"Parameter '{param_name}' object has {len(node)} entries, exceeding maximum of {max_items}",
                    field=param_name,
                    received_value=f"<object len={len(node)}>",
                    code="too_many_items",
                )
            for item in node.values():
                _walk(item, depth + 1)
            return
        if isinstance(node, (list, tuple, set, frozenset)):
            if len(node) > max_items:
                raise ToolValidationError(
                    f"Parameter '{param_name}' array has {len(node)} items, exceeding maximum of {max_items}",
                    field=param_name,
                    received_value=f"<array len={len(node)}>",
                    code="too_many_items",
                )
            for item in node:
                _walk(item, depth + 1)

    _walk(value, 0)


def harden_tool_arguments(
    raw_args: dict[str, Any] | None,
    constraints: dict[str, ParameterConstraint] | None = None,
    custom_aliases: dict[str, list[str]] | None = None,
    schema: dict[str, Any] | None = None,
    reject_unknown: bool | None = None,
) -> dict[str, Any]:
    """Normalize, resolve aliases, detect conflicts, and validate tool input arguments.

    Unknown properties are rejected for function-calling schemas (the default)
    unless the schema explicitly sets ``additionalProperties: true``.  Explicit
    ``null`` is treated as missing and is rejected for required parameters.
    Strings, containers, and nesting depth are bounded by global safety caps.

    Args:
        raw_args: Raw arguments provided by caller or LLM tool call.
        constraints: Optional mapping of parameter constraints.
        custom_aliases: Optional parameter-specific alias mapping.
        schema: Optional JSON schema from ToolDefinition.
        reject_unknown: Override unknown-property handling.  ``None`` derives
            the behaviour from the schema (reject by default for function
            schemas with declared properties).

    Returns:
        Clean, validated, and type-coerced arguments dictionary.

    Raises:
        ToolValidationError: If arguments conflict, fail type coercion, or violate constraints.
    """
    if raw_args is not None and not isinstance(raw_args, dict):
        raise ToolValidationError(
            "Tool arguments must be a JSON object",
            field="arguments",
            received_value=raw_args,
            expected_type="object",
            code="invalid_arguments_type",
        )
    raw = dict(raw_args or {})
    result: dict[str, Any] = {}
    effective_aliases: dict[str, list[str]] = dict(GLOBAL_APPROVED_ALIASES)
    if custom_aliases:
        effective_aliases.update(custom_aliases)

    # 1. Extract schema-based constraints first so target parameter names are known
    effective_constraints: dict[str, ParameterConstraint] = dict(constraints or {})
    schema_declares_properties = bool(schema and "properties" in schema)
    additional_properties = schema.get("additionalProperties") if schema else None
    if schema:
        props = schema.get("properties", {})
        required_list = schema.get("required", [])
        for p_name, p_spec in props.items():
            if p_name not in effective_constraints:
                p_type_str = p_spec.get("type")
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                c = ParameterConstraint(
                    target_type=type_map.get(p_type_str) if p_type_str else None,
                    required=(p_name in required_list),
                    default=p_spec.get("default", None),
                    min_val=p_spec.get("minimum", None),
                    max_val=p_spec.get("maximum", None),
                    allowed_choices=p_spec.get("enum", None),
                    max_length=p_spec.get("maxLength", None),
                    max_items=p_spec.get("maxItems", None),
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

    # 3. Reject unknown properties.  Function-calling schemas are strict by
    # default; an explicit ``additionalProperties: true`` opts out.
    strict_unknown = (
        reject_unknown
        if reject_unknown is not None
        else bool(
            (schema_declares_properties and additional_properties is not True)
            or additional_properties is False
        )
    )
    if strict_unknown:
        unknown_keys = [k for k in result if k not in known_target_params]
        if unknown_keys:
            key = unknown_keys[0]
            raise ToolValidationError(
                f"Unknown parameter '{key}' is not permitted for this tool",
                field=key,
                received_value=result[key],
                code="unknown_parameter",
            )

    # 4. Type Coercion, Default Application, and Range Validation
    for p_name, constraint in effective_constraints.items():
        if p_name in result:
            val = result[p_name]
            if val is None:
                if constraint.required:
                    raise ToolValidationError(
                        f"Required parameter '{p_name}' cannot be null",
                        field=p_name,
                        received_value=None,
                        code="null_required_parameter",
                    )
                # Explicit null for an optional field behaves like an absent
                # field so handler defaults can apply.
                result.pop(p_name, None)
                if constraint.default is not None:
                    result[p_name] = constraint.default
                continue

            # Coerce type.  Numeric strings are only converted when the schema
            # explicitly declares number/integer; absent types stay untouched.
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

            # Size / item / depth ceilings
            _enforce_value_bounds(coerced, p_name, constraint)

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
