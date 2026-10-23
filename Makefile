.PHONY: install dev run run-full test lint format graphify clean docker-build docker-up coverage typecheck

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

# Build knowledge graph with Graphify (prefer local venv binary, fall back to global)
graphify:
	@if [ -x .venv/bin/graphify ]; then .venv/bin/graphify install; \
	elif command -v graphify >/dev/null 2>&1; then graphify install; \
	else echo "graphify not found - install with: .venv/bin/pip install graphifyy"; exit 1; fi
	@echo "Run '/graphify' in your AI coding assistant"

# Run tests with coverage (requires pytest-cov, included in the dev extra)
coverage:
	@if ! .venv/bin/python -c "import pytest_cov" 2>/dev/null; then \
		echo "pytest-cov is not installed. Run: .venv/bin/pip install pytest-cov (or: make dev)"; \
		exit 1; \
	fi
	.venv/bin/python -m pytest --cov=src/robot tests/

# Static type checking (uses mypy or pyright only if present in .venv)
typecheck:
	@if [ -x .venv/bin/mypy ]; then \
		.venv/bin/python -m mypy src/robot; \
	elif command -v pyright >/dev/null 2>&1 || [ -x .venv/bin/pyright ]; then \
		.venv/bin/pyright src/robot 2>/dev/null || pyright src/robot; \
	else \
		echo "No type checker installed in .venv (mypy/pyright not found)."; \
		echo "Enable with: .venv/bin/pip install mypy  (then: make typecheck)"; \
		exit 1; \
	fi

# Build the production Docker image
docker-build:
	docker build -t robot-core:latest .

# Start the Docker Compose stack (host networking + ALSA pass-through)
docker-up:
	docker compose up -d --build

# Clean build artifacts (never touches conversations.db, data/ or models/)
clean:
	rm -rf __pycache__ src/*.egg-info dist build .ruff_cache .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
