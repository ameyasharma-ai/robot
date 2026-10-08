---
trigger: always_on
description: Enforce comprehensive code documentation, docstrings, and strict type hints across all code.
---

# Code Documentation & Clarity Rule

Whenever you write, edit, or refactor code in this repository, you **MUST** follow these documentation standards:

1. **Mandatory Docstrings**:
   - Every new or modified Python function, method, and class must include a comprehensive docstring (Google / PEP 257 format) with `Args:`, `Returns:`, and `Raises:` where applicable.
   - Summaries must be written in the imperative mood (e.g., `"Calculate acoustic energy..."`, not `"Calculates..."`).

2. **Strict Type Annotations**:
   - Every parameter and return value must have explicit modern Python type hints (`str | None`, `dict[str, Any]`, `list[str]`, etc.).

3. **Frontend & JSDoc Standards**:
   - All JavaScript functions and methods in `src/robot/api/static/` must include JSDoc comments (`/** ... */`) documenting parameters, return values, and DOM interactions.
   - CSS files must feature clear sectional comment banners explaining layouts and responsive breakpoints.

4. **Document Architectural Rationale**:
   - Include inline comments explaining non-obvious design decisions, hardware constraints (e.g. RTX 3050 4GB VRAM, ALSA audio buffer limits, latency budgets), and concurrency guarantees.

5. **Preserve Documentation Integrity**:
   - Never remove or degrade existing docstrings or explanatory comments when refactoring code.
