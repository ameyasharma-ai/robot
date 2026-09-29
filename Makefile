.PHONY: install dev run test lint clean

# Install production dependencies
install:
	python -m venv .venv
	.venv/bin/pip install -e .

# Install with development dependencies
dev:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

# Run Robot Core
run:
	.venv/bin/python -m robot

# Run with infrastructure services
run-full:
	bash scripts/run.sh

# Run tests
test:
	.venv/bin/pytest tests/ -v

# Lint
lint:
	.venv/bin/ruff check src/ tests/

# Format
format:
	.venv/bin/ruff format src/ tests/

# Build knowledge graph with Graphify
graphify:
	graphify install
	@echo "Run '/graphify' in your AI coding assistant"

# Clean build artifacts
clean:
	rm -rf __pycache__ src/*.egg-info dist build .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
