# Zone Fade Detector - Makefile
# Common development tasks and commands

.PHONY: help install install-dev test test-unit test-integration lint format type-check clean run setup-venv

# Default target
help:
	@echo "Zone Fade Detector - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup-venv     Create and activate virtual environment"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run            Run the zone fade detector"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean up temporary files"
	@echo "  docs           Generate documentation"
	@echo "  pre-commit     Run pre-commit hooks"

# Setup virtual environment
setup-venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Run the application
run:
	python -m zone_fade_detector.main

# Run with custom config
run-config:
	python -m zone_fade_detector.main --config config/config.yaml

# Run all tests
test:
	pytest

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Run integration tests only
test-integration:
	pytest tests/integration/ -v

# Run tests with coverage
test-coverage:
	pytest --cov=zone_fade_detector --cov-report=html --cov-report=term-missing

# Run linting
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Type checking
type-check:
	mypy src/

# Run all quality checks
check: lint type-check test

# Clean up temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/

# Generate documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Run pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Build package
build:
	python -m build

# Install package in development mode
install-editable:
	pip install -e .

# Update dependencies
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Security check
security:
	bandit -r src/

# Performance profiling
profile:
	python -m cProfile -o profile.stats -m zone_fade_detector.main
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Database operations (if using database)
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Docker operations (if using Docker)
docker-build:
	docker build -t zone-fade-detector .

docker-run:
	docker run --env-file .env zone-fade-detector

# Backup and restore
backup:
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ logs/ config/

# Development server with auto-reload
dev:
	python -m zone_fade_detector.main --reload

# Production server
prod:
	python -m zone_fade_detector.main --config config/production.yaml

# Check system requirements
check-requirements:
	python --version
	pip --version
	@echo "Python version check:"
	@python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
	@echo "Required: Python 3.11+"
	@python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" || echo "ERROR: Python 3.11+ required"

# Initialize project (first time setup)
init: setup-venv install-dev
	@echo "Project initialized! Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Copy .env.example to .env and configure API keys"
	@echo "3. Copy config/config.example.yaml to config/config.yaml"
	@echo "4. Run tests: make test"
	@echo "5. Start the detector: make run"