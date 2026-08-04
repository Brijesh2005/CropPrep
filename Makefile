.PHONY: help install install-dev test test-fast test-quality test-backend coverage lint format typecheck security build compose docs release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies + first-party packages
	pip install -r requirements.txt
	pip install -e ./ai/models ./ai/preprocessing ./ai/training ./ai/explainability
	pip install -e ./services/dataset_manager ./services/spatial_alignment

install-dev: install ## Install development tooling too
	pip install -r requirements-dev.txt
	cd frontend && npm ci

test: ## Run the full Python test suite
	pytest

test-fast: ## Fast subset (unit markers only)
	pytest -m "not slow and not performance"

test-quality: ## Quality assurance package tests
	pytest quality -q

test-backend: ## Backend tests
	pytest backend/app/tests -q

coverage: ## Run tests with coverage report
	pytest --cov=ai --cov=services --cov=quality --cov=backend/app --cov-report=term-missing

lint: ## Ruff lint on Python packages
	ruff check ai services quality backend scripts mlops

format: ## Ruff format
	ruff format ai services quality backend scripts mlops

typecheck: ## mypy type check
	mypy ai services quality backend/app scripts mlops

security: ## Security scanning (bandit + safety)
	bandit -r ai services quality backend mlops
	safety check -r requirements.txt

build: ## Build frontend production bundle
	cd frontend && npm ci && npm run build

compose: ## Validate docker-compose files
	docker compose config --quiet
	docker compose -f docker-compose.prod.yml config --quiet

docs: ## Build the static documentation site
	python scripts/build_docs.py --source docs --output build/docs

release: ## Validate everything before a release
	pytest -q && cd frontend && npm test && npm run build
