# Zone Fade Detector - Setup Rules

## 🚨 CRITICAL: Docker-Only Development

This project is designed to run **exclusively** in Docker containers. Local Python installation is **NOT supported** and will cause issues.

## ❌ What NOT to do

### Never install Python locally:
- ❌ Do NOT run `sudo apt install python3` or `sudo apt install python3-pip`
- ❌ Do NOT run `python3 -m venv venv` or `python -m venv venv`
- ❌ Do NOT run `pip install -r requirements.txt` on your host system
- ❌ Do NOT run `pip install package-name` on your host system
- ❌ Do NOT run `python script.py` directly on your host system
- ❌ Do NOT create or activate virtual environments locally

### Never use local Python commands:
- ❌ Do NOT run `python -m zone_fade_detector.main`
- ❌ Do NOT run `pytest tests/`
- ❌ Do NOT run `python -c "import zone_fade_detector"`

## ✅ What TO do

### Always use Docker commands:
- ✅ Use `docker-compose up zone-fade-detector` to run the application
- ✅ Use `docker-compose run --rm zone-fade-detector` for one-time runs
- ✅ Use `docker-compose run --rm zone-fade-detector-test` for testing
- ✅ Use `docker-compose exec zone-fade-detector pip install package` for dependencies
- ✅ Use `docker-compose exec zone-fade-detector python script.py` for Python scripts

## 🐳 Docker Commands Reference

### Running the Application
```bash
# Standard mode (continuous monitoring)
docker-compose up zone-fade-detector

# Background mode
docker-compose up -d zone-fade-detector

# One-time run
docker-compose run --rm zone-fade-detector

# Live mode (RTH only)
docker-compose run --rm zone-fade-detector --mode live --verbose

# Replay mode (historical data)
docker-compose run --rm zone-fade-detector \
  --mode replay \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --provider alpaca
```

### Development
```bash
# Development mode with hot reload
docker-compose up zone-fade-detector-dev

# Install additional packages
docker-compose exec zone-fade-detector pip install package-name

# Run Python scripts
docker-compose exec zone-fade-detector python scripts/script.py

# Interactive shell
docker-compose exec zone-fade-detector bash
```

### Testing
```bash
# Run all tests
docker-compose run --rm zone-fade-detector-test

# Run specific test file
docker-compose run --rm zone-fade-detector-test pytest tests/test_specific.py

# Run with coverage
docker-compose run --rm zone-fade-detector-test pytest --cov=zone_fade_detector
```

### Debugging
```bash
# View logs
docker-compose logs -f zone-fade-detector

# Check container status
docker-compose ps

# Test imports
docker-compose run --rm zone-fade-detector python -c "import zone_fade_detector"

# Test alerts
docker-compose run --rm zone-fade-detector --test-alerts
```

## 🔧 Configuration

### Environment Setup
```bash
# Copy environment template
cp .env.docker .env

# Edit with your API credentials
nano .env
```

### Application Configuration
```bash
# Copy configuration template
cp config/config.docker.yaml config/config.yaml

# Edit configuration as needed
nano config/config.yaml
```

## 🚨 Common Mistakes to Avoid

1. **Installing Python locally** - This will cause conflicts and errors
2. **Running pip install on host** - Dependencies must be installed in containers
3. **Creating local virtual environments** - Not needed with Docker
4. **Running Python scripts directly** - Always use Docker commands
5. **Mixing local and Docker Python** - Stick to Docker-only approach

## 📚 Additional Resources

- [Main README](README.md) - Project overview and features
- [Docker Setup Guide](README.Docker.md) - Detailed Docker instructions
- [Discord Setup](docs/DISCORD_SETUP.md) - Discord webhook configuration
- [Product Requirements](docs/PRD.md) - Technical specifications

## 🆘 Getting Help

If you encounter issues:

1. **Check this file first** - Most issues are covered here
2. **Use Docker commands only** - Never install Python locally
3. **Check Docker logs** - `docker-compose logs zone-fade-detector`
4. **Verify configuration** - Ensure `.env` and `config.yaml` are correct
5. **Report issues** - Use GitHub Issues with Docker-specific details

## 🔄 Remember

**This project uses Docker exclusively. Any attempt to use local Python will cause issues and is not supported.**