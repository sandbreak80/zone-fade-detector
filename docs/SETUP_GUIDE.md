# Setup Guide

This guide will help you set up the Zone Fade Detector system for backtesting and development.

## 🎯 Overview

The Zone Fade Detector is a sophisticated trading system designed for detecting Zone Fade setups. This guide covers:
- System requirements
- Installation steps
- Configuration setup
- Running backtesting
- Development setup

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Docker**: Docker Engine 20.10+ and Docker Compose 2.0+
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for data and containers
- **Network**: Internet connection for API access

### API Requirements
- **Alpaca API**: Free account with API key and secret
- **Polygon API**: Free account with API key (optional)
- **Discord Webhook**: Discord server with webhook URL (optional)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd zone-fade-detector
```

### 2. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Configure API Keys
Edit `.env` file with your API credentials:
```bash
# Alpaca API (Required)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Polygon API (Optional)
POLYGON_API_KEY=your_polygon_api_key

# Discord Webhook (Optional)
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Logging
LOG_LEVEL=INFO
```

### 4. Build and Run
```bash
# Build the Docker image
docker-compose build

# Run the system
docker-compose up
```

## 🔧 Detailed Setup

### Docker Installation

#### Ubuntu/Debian
```bash
# Update package index
sudo apt update

# Install Docker
sudo apt install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop
open /Applications/Docker.app
```

#### Windows (WSL2)
```bash
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop

# Enable WSL2 integration
# In Docker Desktop settings
```

### API Account Setup

#### Alpaca API
1. **Create Account**: Visit [alpaca.markets](https://alpaca.markets)
2. **Get API Keys**: Go to Account → API Keys
3. **Copy Credentials**: Save API key and secret key
4. **Note**: Free tier includes paper trading and historical data

#### Polygon API (Optional)
1. **Create Account**: Visit [polygon.io](https://polygon.io)
2. **Get API Key**: Go to Dashboard → API Keys
3. **Copy Key**: Save the API key
4. **Note**: Free tier includes limited historical data

#### Discord Webhook (Optional)
1. **Create Server**: Create a Discord server
2. **Create Webhook**: Server Settings → Integrations → Webhooks
3. **Copy URL**: Save the webhook URL
4. **Note**: Used for real-time alerts

## 📊 Data Setup

### Download Historical Data
```bash
# Download 2024 data for backtesting
docker-compose run zone-fade-detector python download_2024_data.py
```

This will create:
- `data/2024/SPY_2024.pkl` (~196k bars)
- `data/2024/QQQ_2024.pkl` (~210k bars)
- `data/2024/IWM_2024.pkl` (~186k bars)

### Data Storage
- **Location**: `data/` directory
- **Format**: Pickle files for efficient storage
- **Size**: ~500MB total for 2024 data
- **Persistence**: Data persists between container restarts

## 🧪 Running Backtesting

### Quick Validation
```bash
# Run efficient validation (last 50k bars per symbol)
docker-compose run zone-fade-detector python backtest_2024_efficient_validation.py
```

### Full 2024 Backtesting
```bash
# Run complete 2024 backtesting
docker-compose run zone-fade-detector python backtest_2024_full_validation.py
```

### Debug Mode
```bash
# Run December 2024 debug backtesting
docker-compose run zone-fade-detector python backtest_2024_december_debug.py
```

### Expected Results
- **Entry Points**: 160 high-quality setups
- **QRS Scores**: 6.23 average (above 7.0 threshold)
- **Entry Windows**: 28.9 minutes average duration
- **Output**: CSV files in `validation_output/` directory

## 🔧 Development Setup

### Local Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run specific test
pytest tests/unit/test_zone_fade_strategy.py
```

### Docker Development
```bash
# Run in development mode
docker-compose up zone-fade-detector-dev

# Run tests in container
docker-compose run zone-fade-detector-test pytest

# Access container shell
docker-compose exec zone-fade-detector bash
```

### Code Quality
```bash
# Run linting
flake8 src/

# Run type checking
mypy src/

# Run formatting
black src/
```

## 📁 Project Structure

```
zone-fade-detector/
├── src/                          # Source code
│   └── zone_fade_detector/
│       ├── core/                 # Core components
│       ├── strategies/           # Strategy implementations
│       ├── indicators/           # Technical indicators
│       └── data/                 # Data management
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── docs/                         # Documentation
├── data/                         # Historical data
├── config/                       # Configuration files
├── docker-compose.yml            # Docker configuration
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables
```

## ⚙️ Configuration

### Environment Variables
| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ALPACA_API_KEY` | Yes | Alpaca API key | - |
| `ALPACA_SECRET_KEY` | Yes | Alpaca secret key | - |
| `POLYGON_API_KEY` | No | Polygon API key | - |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook URL | - |
| `LOG_LEVEL` | No | Logging level | INFO |

### Strategy Parameters
Edit `src/zone_fade_detector/strategies/zone_fade_strategy.py`:
```python
# Rejection candle parameters
rejection_candle_min_wick_ratio = 0.3  # 30% wick ratio
volume_spike_threshold = 1.8           # 1.8x volume spike

# QRS scoring parameters
min_qrs_score = 7.0                    # Minimum QRS score
```

### Alert Configuration
Edit `src/zone_fade_detector/core/alert_system.py`:
```python
# Alert channels
console_alerts = True
file_alerts = True
discord_alerts = True

# Alert formatting
include_qrs_breakdown = True
include_volume_analysis = True
```

## 🚨 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Check Docker status
docker --version
docker-compose --version

# Restart Docker service
sudo systemctl restart docker

# Clean up containers
docker-compose down
docker system prune -a
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

#### API Issues
```bash
# Test API connectivity
docker-compose run zone-fade-detector python -c "
from zone_fade_detector.data.alpaca_client import AlpacaClient
client = AlpacaClient()
print('API connection successful')
"
```

#### Memory Issues
```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# In Docker Desktop settings
```

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG docker-compose up

# Run specific component
docker-compose run zone-fade-detector python -m zone_fade_detector.core.detector
```

### Logs
```bash
# View logs
docker-compose logs zone-fade-detector

# Follow logs
docker-compose logs -f zone-fade-detector

# View specific log
docker-compose logs zone-fade-detector | grep ERROR
```

## 📊 Performance Optimization

### System Optimization
- **Memory**: Increase Docker memory limit to 8GB+
- **CPU**: Use multi-core processing for backtesting
- **Storage**: Use SSD for better I/O performance
- **Network**: Ensure stable internet connection

### Backtesting Optimization
- **Sample Size**: Use efficient validation for quick testing
- **Parallel Processing**: Use 3-thread processing for full backtesting
- **Memory Management**: Monitor memory usage during processing
- **Caching**: Use persistent data caching for repeated runs

## 🔒 Security Considerations

### API Security
- **Environment Variables**: Never commit API keys to version control
- **API Limits**: Monitor API usage and limits
- **Rate Limiting**: Respect API rate limits
- **Key Rotation**: Regularly rotate API keys

### Data Security
- **Data Encryption**: Consider encrypting sensitive data
- **Access Control**: Limit access to data files
- **Backup**: Regular backup of important data
- **Audit**: Monitor data access and usage

## 📚 Next Steps

### After Setup
1. **Run Backtesting**: Execute the backtesting scripts
2. **Validate Results**: Check entry points manually
3. **Analyze Performance**: Review quality metrics
4. **Optimize Parameters**: Tune strategy parameters

### Development
1. **Read Documentation**: Review all documentation
2. **Run Tests**: Execute the test suite
3. **Explore Code**: Understand the codebase
4. **Make Changes**: Implement improvements

### Production
1. **Futures Integration**: Implement ES/NQ/RTY futures
2. **Live Trading**: Optimize for real-time execution
3. **Risk Management**: Add comprehensive risk controls
4. **Monitoring**: Implement real-time monitoring

## 📞 Support

### Getting Help
- **Documentation**: Check all documentation in `/docs`
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub discussions
- **Code Review**: Submit pull requests

### Resources
- [Backtesting Guide](BACKTESTING_GUIDE.md)
- [Manual Validation Guide](MANUAL_VALIDATION_GUIDE.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [2024 Results Summary](2024_RESULTS_SUMMARY.md)

---

*This setup guide provides comprehensive instructions for setting up the Zone Fade Detector system. For specific issues or questions, refer to the troubleshooting section or create an issue on GitHub.*