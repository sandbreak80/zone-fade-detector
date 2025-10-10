# Zone Fade Detector - Docker Makefile
# Docker-based development tasks and commands

.PHONY: help build up down logs test clean run dev setup

# Default target
help:
	@echo "Zone Fade Detector - Docker Commands:"
	@echo ""
	@echo "⚠️  IMPORTANT: This project uses Docker exclusively!"
	@echo "   DO NOT install Python, pip, or virtual environments locally."
	@echo ""
	@echo "Docker Commands:"
	@echo "  build          Build Docker images"
	@echo "  up             Start the zone fade detector"
	@echo "  down           Stop all containers"
	@echo "  logs           View container logs"
	@echo "  test           Run tests in Docker"
	@echo "  clean          Clean up Docker resources"
	@echo "  run            Run detector in standard mode"
	@echo "  dev            Run in development mode with hot reload"
	@echo "  setup          Initial setup (copy config files)"
	@echo ""
	@echo "Testing:"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage  Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  shell          Open shell in container"
	@echo "  install-pkg    Install package in container"
	@echo "  format         Format code (in container)"
	@echo "  lint           Run linting (in container)"
	@echo "  type-check     Run type checking (in container)"
	@echo ""
	@echo "Operations:"
	@echo "  live           Run in live mode (RTH only)"
	@echo "  replay         Run in replay mode (requires START, END, SYMBOLS, PROVIDER)"
	@echo "  test-alerts    Test alert channels"
	@echo "  signals-today  View today's signals"

# Build Docker images
build:
	docker-compose build

# Start the application
up:
	docker-compose up zone-fade-detector

# Start in background
up-d:
	docker-compose up -d zone-fade-detector

# Stop all containers
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f zone-fade-detector

# Run all tests
test:
	docker-compose run --rm zone-fade-detector-test

# Run unit tests only
test-unit:
	docker-compose run --rm zone-fade-detector-test pytest tests/unit/ -v

# Run integration tests only
test-integration:
	docker-compose run --rm zone-fade-detector-test pytest tests/integration/ -v

# Run tests with coverage
test-coverage:
	docker-compose run --rm zone-fade-detector-test pytest --cov=zone_fade_detector --cov-report=html --cov-report=term-missing

# Clean up Docker resources
clean:
	docker-compose down
	docker system prune -f
	docker volume prune -f

# Run detector in standard mode
run:
	docker-compose run --rm zone-fade-detector

# Run in development mode with hot reload
dev:
	docker-compose up zone-fade-detector-dev

# Open shell in container
shell:
	docker-compose exec zone-fade-detector bash

# Install additional package in container
install-pkg:
	@if [ -z "$(PKG)" ]; then \
		echo "Usage: make install-pkg PKG=package-name"; \
		exit 1; \
	fi
	docker-compose exec zone-fade-detector pip install $(PKG)

# Format code (in container)
format:
	docker-compose run --rm zone-fade-detector-dev black src/ tests/
	docker-compose run --rm zone-fade-detector-dev isort src/ tests/

# Run linting (in container)
lint:
	docker-compose run --rm zone-fade-detector-dev flake8 src/ tests/
	docker-compose run --rm zone-fade-detector-dev black --check src/ tests/
	docker-compose run --rm zone-fade-detector-dev isort --check-only src/ tests/

# Run type checking (in container)
type-check:
	docker-compose run --rm zone-fade-detector-dev mypy src/

# Run all quality checks
check: lint type-check test

# Initial setup (copy config files)
setup:
	@echo "Setting up Zone Fade Detector..."
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file from .env.example"; \
		echo "⚠️  Please edit .env with your API credentials"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@if [ ! -f "config/config.yaml" ]; then \
		cp config/config.example.yaml config/config.yaml; \
		echo "✅ Created config/config.yaml from config/config.example.yaml"; \
		echo "⚠️  Please edit config/config.yaml with your preferences"; \
	else \
		echo "✅ config/config.yaml already exists"; \
	fi
	@echo ""
	@echo "🎉 Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your API keys (Alpaca, Polygon, Discord)"
	@echo "2. Edit config/config.yaml with your preferences"
	@echo "3. Build: make build"
	@echo "4. Test: make test-alerts"
	@echo "5. Run: make run"

# Run in live mode (RTH only)
live:
	docker-compose run --rm zone-fade-detector --mode live --verbose

# Run in replay mode (historical data)
replay:
	@if [ -z "$(START)" ] || [ -z "$(END)" ] || [ -z "$(SYMBOLS)" ] || [ -z "$(PROVIDER)" ]; then \
		echo "Usage: make replay START=2025-01-06 END=2025-01-10 SYMBOLS=SPY,QQQ,IWM PROVIDER=alpaca"; \
		exit 1; \
	fi
	docker-compose run --rm zone-fade-detector \
		--mode replay \
		--start-date $(START) \
		--end-date $(END) \
		--symbols $(SYMBOLS) \
		--provider $(PROVIDER)

# Test alert channels
test-alerts:
	docker-compose run --rm zone-fade-detector --test-alerts

# View today's signals
signals-today:
	@echo "Today's signals:"
	@if [ -f "signals/$(shell date +%Y-%m-%d).jsonl" ]; then \
		cat signals/$(shell date +%Y-%m-%d).jsonl | jq .; \
	else \
		echo "No signals found for today"; \
	fi

# Show container status
status:
	docker-compose ps

# Restart containers
restart:
	docker-compose restart zone-fade-detector

# View container resource usage
stats:
	docker stats zone-fade-detector

# Backup data
backup:
	@echo "Creating backup..."
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ logs/ signals/ config/config.yaml .env
	@echo "Backup created: backup-$(shell date +%Y%m%d-%H%M%S).tar.gz"

# Full development workflow
dev-workflow: setup build test
	@echo "Development workflow complete!"
	@echo "Ready for development. Use 'make dev' to start with hot reload."

# Production deployment
prod-deploy: build
	docker-compose -f docker-compose.yml up -d zone-fade-detector
	@echo "Production deployment complete!"

# Health check
health:
	docker-compose exec zone-fade-detector python -c "import zone_fade_detector; print('✅ Health check passed')"

# Show help for Docker commands
docker-help:
	@echo "Docker Compose Commands:"
	@echo "  docker-compose up zone-fade-detector     # Start detector"
	@echo "  docker-compose up -d zone-fade-detector  # Start in background"
	@echo "  docker-compose down                      # Stop all containers"
	@echo "  docker-compose logs -f zone-fade-detector # View logs"
	@echo "  docker-compose ps                        # Show container status"
	@echo "  docker-compose exec zone-fade-detector bash # Open shell"