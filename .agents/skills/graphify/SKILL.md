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

```bash
/graphify .                    # Graph entire codebase
/graphify src/robot            # Graph specific directory
/graphify . --mode deep        # Deep relationship extraction
/graphify query "how does the context engine work?"
```

## Output

The graph is written to `graphify-out/`:
- `graph.html` — Interactive visual graph
- `graph.json` — Queryable knowledge graph
- `GRAPH_REPORT.md` — Architecture analysis
