---
name: zero-import-errors
description: >-
  Strictly verify zero import errors, zero missing dependencies, zero syntax errors, and zero undefined symbols or dead code across the entire codebase.
---

# Zero Import Errors & Code Integrity Skill

Use this skill on **every single code change** before declaring any task complete. Runtime `ImportError`, `ModuleNotFoundError`, `NameError`, or syntax failures break the agentic pair programming loop, crash background processes, and degrade system stability.

---

## 1. Automated Verification Script

An automated, zero-dependency verification engine is provided in this skill:
```bash
.venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py
```

### What `verify_imports.py` Checks:
1. **Bytecode Compilation (`py_compile`)**:
   - Ensures all Python files compile cleanly into bytecode without syntax errors, tokenization errors, or indentation mistakes.
2. **Abstract Syntax Tree Validation (`ast.parse`)**:
   - Inspects AST nodes across all files to detect invalid constructs.
3. **Module Import Resolution**:
   - Dynamically resolves and imports all modules across `src/robot/` in the project Python environment.
   - Detects circular imports, broken `from robot... import ...` paths, missing packages, and typos in module names.
4. **Undefined Symbols & Variables**:
   - Analyzes variable scopes to flag unresolved variable names or references to unimported symbols.

---

## 2. Standard Verification Workflow

Whenever you add or modify code:

### Step 1: Run the Integrity Verifier
```bash
.venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py
```
- If any error is reported, read the filename and line number, inspect the offending code, and fix the import or undefined variable immediately.

### Step 2: Run the Pytest Suite
```bash
.venv/bin/pytest tests/
```
- Ensure all 100+ tests pass with zero failures or errors.

### Step 3: Verify Frontend Script Integrity
Inspect modified JavaScript files (`app.js`, `AnimationSystem.js`, etc.) to confirm:
- All referenced DOM elements exist in `index.html` or have null checks (`const el = document.getElementById(...); if (el) ...`).
- All dispatched API endpoints (`/voice/toggle`, `/voice/resume`, `/tools/execute`) match FastAPI routes in `engine.py`.

---

## 3. Common Import Anti-Patterns & How to Avoid Them

### ❌ Anti-Pattern 1: Hidden Circular Imports
**Problem**: Module A imports from Module B at top-level, and Module B imports from Module A.
**Solution**:
- Use lazy local imports inside functions for optional or circular dependencies.
- Or use `if TYPE_CHECKING:` guards from `typing` when imports are solely needed for type annotations.

### ❌ Anti-Pattern 2: Unused Star Imports (`from foo import *`)
**Problem**: Obscures symbol origin, pollutes the namespace, and confuses linters and AST visitors.
**Solution**:
- Always use explicit imports: `from robot.core.config import RobotConfig, get_settings`.

### ❌ Anti-Pattern 3: Assuming Non-Standard Libraries Exist Without Fallbacks
**Problem**: Importing external hardware packages (e.g. `sounddevice`, `serial`, `pactl`) that may fail on machines without audio/serial interfaces.
**Solution**:
- Wrap hardware library imports in `try / except ImportError:` with a graceful mock or log fallback.

---

## 4. Exit Gate
A task is **NOT complete** until:
1. `.venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py` returns code `0`.
2. `.venv/bin/pytest tests/` passes 100%.
