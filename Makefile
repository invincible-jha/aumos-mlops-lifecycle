.PHONY: install test test-quick lint format typecheck clean docker-build docker-run all

all: lint typecheck test

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=aumos_mlops_lifecycle --cov-report=term-missing

test-quick:
	pytest tests/ -x -q --no-header

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/aumos_mlops_lifecycle/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info htmlcov/ .coverage

docker-build:
	docker build -t aumos/mlops-lifecycle:dev .

docker-run:
	docker compose -f docker-compose.dev.yml up -d
