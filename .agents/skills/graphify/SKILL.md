---
name: graphify
description: Build a queryable knowledge graph from the robot codebase using Graphify
---

# Graphify Skill

Run `/graphify` to build a knowledge graph from this codebase.

## Setup

```bash
pip install graphifyy
graphify install
```

## Usage

### In Antigravity Chat (Slash Command):
- Type `/graphify` in the assistant prompt

### In Terminal (CLI):
```bash
.venv/bin/graphify extract . --code-only       # Graph entire codebase
.venv/bin/graphify update .                   # Update graph incrementally
.venv/bin/graphify export html                # Generate interactive graph.html
.venv/bin/graphify query "how does barge-in work?"
```

## Output

The graph is written to `graphify-out/`:
- `graph.html` — Interactive visual graph
- `graph.json` — Queryable knowledge graph
- `GRAPH_REPORT.md` — Architecture analysis
