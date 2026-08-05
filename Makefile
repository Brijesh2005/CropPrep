.PHONY: help install install-dev test test-fast test-quality test-backend coverage lint format typecheck security build compose docs release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies + first-party packages
	pip install -r requirements.txt
	pip install -e ./training/models ./training/preprocessing ./training/training ./training/explainability
	pip install -e ./training/dataset_manager ./training/stam

install-dev: install ## Install development tooling too
	pip install -r requirements-dev.txt
	cd application/frontend && npm ci

test: ## Run the full Python test suite
	pytest

test-fast: ## Fast subset (unit markers only)
	pytest -m "not slow and not performance"

test-quality: ## Quality assurance package tests
	pytest training/quality -q

test-backend: ## Backend tests
	pytest application/backend/app/tests -q

coverage: ## Run tests with coverage report
	pytest --cov=training --cov=application/backend/app --cov-report=term-missing

lint: ## Ruff lint on Python packages
	ruff check training application/backend application/database application/tests shared scripts

format: ## Ruff format
	ruff format training application/backend application/database shared scripts

typecheck: ## mypy type check
	mypy training shared application/backend application/database scripts

security: ## Security scanning (bandit + safety)
	bandit -r training shared application/backend application/database
	safety check -r requirements.txt

build: ## Build frontend production bundle
	cd application/frontend && npm ci && npm run build

compose: ## Validate docker-compose files
	docker compose -f application/docker/docker-compose.yml config --quiet
	docker compose -f application/docker/docker-compose.prod.yml config --quiet

docs: ## Build the static documentation site
	python scripts/build_docs.py --source docs --output build/docs

release: ## Validate everything before a release
	pytest -q && cd application/frontend && npm test && npm run build
