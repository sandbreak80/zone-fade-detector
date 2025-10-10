# Zone Fade Detector - Docker Setup

This document provides comprehensive instructions for running the Zone Fade Detector in Docker containers.

## 🐳 Quick Start

### Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- At least 2GB RAM and 1 CPU core available

### 1. Clone and Setup

```bash
git clone <repository-url>
cd zone-fade-detector
```

### 2. Configure Environment

```bash
# Copy the Docker environment template
cp .env.docker .env

# Edit with your API credentials
nano .env
```

**Required Environment Variables:**
- `ALPACA_API_KEY` - Your Alpaca API key
- `ALPACA_SECRET_KEY` - Your Alpaca secret key  
- `POLYGON_API_KEY` - Your Polygon API key
- `DISCORD_WEBHOOK_URL` - Your Discord webhook URL for alerts

### 3. Configure Application

```bash
# Copy the Docker configuration template
cp config/config.docker.yaml config/config.yaml

# Edit configuration as needed
nano config/config.yaml
```

### 4. Build and Run

```bash
# Build the Docker image
docker-compose build

# Run in standard mode
docker-compose up zone-fade-detector

# Run in background
docker-compose up -d zone-fade-detector

# View logs
docker-compose logs -f zone-fade-detector
```

## 🚀 Running Modes

### Standard Mode (Continuous Monitoring)
```bash
docker-compose up zone-fade-detector
```

### Live Mode (RTH Only)
```bash
docker-compose run --rm zone-fade-detector \
  --mode live --verbose
```

### Replay Mode (Historical Data)
```bash
docker-compose run --rm zone-fade-detector \
  --mode replay \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --provider alpaca
```

### Test Alerts
```bash
docker-compose run --rm zone-fade-detector \
  --test-alerts
```

### Development Mode (Hot Reload)
```bash
docker-compose up zone-fade-detector-dev
```

### Run Tests
```bash
docker-compose run --rm zone-fade-detector-test
```

## 📁 Volume Mounts

The Docker setup uses the following volume mounts:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./config` | `/app/config` | Configuration files (read-only) |
| `./logs` | `/app/logs` | Application logs |
| `./signals` | `/app/signals` | Generated signal files |
| `./data` | `/app/data` | Persistent data and cache |

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# API Credentials (REQUIRED)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
POLYGON_API_KEY=your_key_here

# Discord Webhook (for alerts)
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN

# System Settings
LOG_LEVEL=INFO
POLL_INTERVAL=30
ENVIRONMENT=docker

# Docker-specific
PYTHONUNBUFFERED=1
TZ=America/New_York
```

### Configuration File

Main configuration in `config/config.yaml`:

```yaml
# Symbol Configuration
symbols:
  - SPY
  - QQQ  
  - IWM

# Polling Configuration
polling:
  interval_seconds: 30
  max_retries: 3
  timeout_seconds: 10

# Alert Configuration
alerts:
  channels: [console, file, webhook]
  min_qrs_score: 7
  deduplication_minutes: 5
  
  webhook:
    enabled: true
    url: ${DISCORD_WEBHOOK_URL}
    timeout: 5
```

## 📱 Discord Webhook Setup

The system supports Discord webhooks for real-time alerts. See the [Discord Setup Guide](docs/DISCORD_SETUP.md) for detailed instructions.

### Quick Setup

1. **Create a Discord webhook** in your server
2. **Add the webhook URL** to your `.env` file:
   ```bash
   DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
   ```
3. **Test the webhook**:
   ```bash
   docker-compose run --rm zone-fade-detector --test-alerts
   ```

### Alert Format

Discord alerts include:
- Symbol and direction (Long/Short)
- Zone level and type
- QRS score (quality rating)
- Rejection candle details
- Target levels and confirmation status

## 🛠️ Development

### Development Container

The development container includes:

- Hot reload for source code changes
- Development dependencies installed
- Volume mounts for live code editing
- Enhanced logging and debugging

```bash
# Start development container
docker-compose up zone-fade-detector-dev

# Install additional dev dependencies
docker-compose exec zone-fade-detector-dev pip install package-name

# Run specific commands
docker-compose exec zone-fade-detector-dev python -m pytest tests/
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

## 📊 Monitoring

### Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker-compose ps

# View health check logs
docker inspect zone-fade-detector | jq '.[0].State.Health'
```

### Logs

```bash
# View all logs
docker-compose logs zone-fade-detector

# Follow logs in real-time
docker-compose logs -f zone-fade-detector

# View specific log files
docker-compose exec zone-fade-detector tail -f /app/logs/zone_fade_detector.log
```

### Resource Usage

```bash
# Monitor resource usage
docker stats zone-fade-detector

# View container details
docker inspect zone-fade-detector
```

## 🔒 Security

### Non-Root User

The container runs as a non-root user (`appuser`) for security.

### Environment Variables

- Never commit `.env` files to version control
- Use Docker secrets for production deployments
- Rotate API keys regularly

### Network Security

```bash
# Run with custom network
docker-compose up --network custom-network

# Limit container capabilities
docker run --cap-drop=ALL --cap-add=NET_RAW zone-fade-detector
```

## 🚀 Production Deployment

### Resource Limits

The Docker Compose file includes resource limits:

```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

### Logging

Production logging configuration:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Health Checks

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import zone_fade_detector"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## 🐛 Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs zone-fade-detector

# Check configuration
docker-compose config

# Verify environment variables
docker-compose exec zone-fade-detector env | grep ALPACA
```

**Permission issues:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER logs signals data

# Check container user
docker-compose exec zone-fade-detector whoami
```

**API connection issues:**
```bash
# Test API connectivity
docker-compose exec zone-fade-detector python -c "
import requests
print(requests.get('https://api.alpaca.markets/v2/clock').json())
"
```

**Memory issues:**
```bash
# Monitor memory usage
docker stats zone-fade-detector

# Increase memory limits in docker-compose.yml
```

### Debug Mode

```bash
# Run with debug logging
docker-compose run --rm zone-fade-detector \
  --log-level DEBUG --verbose

# Interactive shell
docker-compose run --rm zone-fade-detector bash

# Check Python environment
docker-compose exec zone-fade-detector python -c "
import sys
print(sys.path)
import zone_fade_detector
print('Import successful')
"
```

## 📚 Advanced Usage

### Custom Entrypoint

The container uses a custom entrypoint script that supports various modes:

```bash
# Show help
docker-compose run --rm zone-fade-detector --help

# Custom configuration
docker-compose run --rm zone-fade-detector \
  --config /app/config/custom.yaml \
  --log-level DEBUG \
  --symbols SPY,QQQ
```

### Multi-Container Setup

For production, you might want separate containers for different components:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  detector:
    build: .
    environment:
      - MODE=detector
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
  
  alert-processor:
    build: .
    environment:
      - MODE=alerts
    volumes:
      - ./logs:/app/logs
      - ./signals:/app/signals
```

### Docker Swarm

For high availability:

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml zone-fade-detector

# Scale services
docker service scale zone-fade-detector_zone-fade-detector=3
```

## 📝 Maintenance

### Updates

```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose down && docker-compose up -d
```

### Cleanup

```bash
# Remove stopped containers
docker-compose down

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a
```

### Backup

```bash
# Backup configuration
tar -czf config-backup.tar.gz config/

# Backup logs
tar -czf logs-backup.tar.gz logs/

# Backup signals
tar -czf signals-backup.tar.gz signals/
```

## 🤝 Contributing

When contributing to the Docker setup:

1. Test changes with `docker-compose build`
2. Verify all modes work: standard, live, replay, test-alerts
3. Update documentation if needed
4. Test on different platforms (Linux, macOS, Windows)

## 📄 License

This Docker setup follows the same MIT license as the main project.