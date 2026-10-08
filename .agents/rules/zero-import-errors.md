---
trigger: always_on
description: Mandatory zero import errors, zero missing dependencies, and zero syntax errors across the codebase.
---

# Zero Import Errors & Code Integrity Rule

To ensure rock-solid stability and prevent broken runtime dependencies:

1. **Mandatory Integrity Check**:
   - Before completing any task that edits Python code, you **MUST** run:
     ```bash
     .venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py
     ```
   - All modules must compile, resolve, and import cleanly with exit code `0`.

2. **No Unresolved Imports or Undefined Names**:
   - Never introduce code with missing imports, misspelled function names, or unhandled `NameError`/`ImportError` conditions.
   - For optional or hardware-dependent libraries, always provide clean `try / except ImportError:` fallbacks.

3. **No Circular Imports**:
   - Guard against circular module dependencies using `if TYPE_CHECKING:` or localized lazy imports.

4. **All Tests Must Pass**:
   - Always run `.venv/bin/pytest tests/` and verify that 100% of test cases pass before reporting completion.
