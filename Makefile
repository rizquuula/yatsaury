.DEFAULT_GOAL := help

.PHONY: help install test test-e2e lint fmt typecheck check clean web

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup"
	@echo "  install     Install all dependencies (dev + prod) via uv"
	@echo ""
	@echo "Testing"
	@echo "  test        Run unit tests (excludes e2e)"
	@echo "  test-e2e    Run end-to-end tests (requires local Ollama)"
	@echo ""
	@echo "Code quality"
	@echo "  lint        Run ruff linter"
	@echo "  fmt         Auto-format with ruff"
	@echo "  typecheck   Run mypy type checker"
	@echo "  check       lint + typecheck + tests (full CI gate)"
	@echo ""
	@echo "Run"
	@echo "  web         Launch the local web UI"
	@echo ""
	@echo "Maintenance"
	@echo "  clean       Remove build artefacts and caches"

install:
	uv sync --all-extras

test:
	uv run pytest

test-e2e:
	uv run pytest -m e2e

lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck test

web:
	uv run yatsaury web

clean:
	rm -rf .cache/ dist/ .mypy_cache/ .ruff_cache/ __pycache__/ \
	       src/yatsaury/__pycache__ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
