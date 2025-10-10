# Zone Fade Detector

A Python-based trading system that identifies high-probability Zone Fade reversal setups using 15-minute delayed market data from Alpaca and Polygon APIs.

## 🎯 Overview

The Zone Fade Detector automates the identification of reversal trading opportunities when price reaches higher-timeframe zones and shows signs of exhaustion. The system processes real-time market data for SPY, QQQ, and IWM (proxies for /ES, /NQ, /RTY futures) and generates alerts for A-grade setups based on a comprehensive quality rating system.

## 🏗️ Architecture

```
zone-fade-detector/
├── src/zone_fade_detector/          # Main package
│   ├── core/                        # Core trading logic
│   ├── data/                        # Data management and API integration
│   ├── indicators/                  # Technical indicators (VWAP, OR, swing structure)
│   ├── strategies/                  # Trading strategy implementations
│   └── utils/                       # Utility functions and helpers
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
├── docs/                            # Documentation
├── config/                          # Configuration files
├── scripts/                         # Utility scripts
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
└── .cursorrules                     # Cursor IDE rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Alpaca API credentials
- Polygon API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd zone-fade-detector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

5. **Configure the system**
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your preferences
   ```

### Quick Setup

```bash
# Run the setup script
./scripts/setup.sh

# Or manually:
make setup
```

### Quick Run Sequence

```bash
# 1. Smoke test
make test && make typecheck && make format

# 2. Replay (balanced/trend days)
make replay START=2025-01-06 END=2025-01-10 SYMBOLS=SPY,QQQ,IWM PROVIDER=alpaca

# 3. Inspect signals
make signals-today  # or view signals/*.jsonl files

# 4. Live (paper) run during RTH
make live
```

### Running the System

```bash
# Standard mode (continuous monitoring)
python -m zone_fade_detector.main

# Live mode (RTH only)
python -m zone_fade_detector.main --live

# Replay mode (historical data)
python -m zone_fade_detector.main --replay --start-date 2025-01-06 --end-date 2025-01-10 --symbols SPY,QQQ,IWM --provider alpaca

# Test alert channels
python -m zone_fade_detector.main --test-alerts

# Run with custom configuration
python -m zone_fade_detector.main --config config/production.yaml
```

## 📊 Zone Fade Strategy

### What is a Zone Fade?

A Zone Fade is a high-probability reversal setup that occurs when:

1. **Price approaches a higher-timeframe zone** (support/resistance, supply/demand)
2. **Market shows signs of exhaustion** (lack of initiative, low follow-through volume)
3. **Price fails to accept beyond the zone** and closes back inside
4. **A rejection candle forms** at or beyond the zone
5. **A Change of Character (CHoCH)** occurs in the opposite direction

### Entry Criteria

- **Trigger**: CHoCH opposite prior direction + close back inside zone
- **Confirmation**: Candle closes in favor of reversal
- **Entry Method**: Limit or market order at mid-zone or on CHoCH confirmation
- **Stop Placement**: 1-2 ticks beyond zone back or CHoCH invalidation swing
- **Targets**: T1 (VWAP/range mid), T2 (opposite range edge)

### Quality Rating System (QRS)

The system uses a 5-factor scoring system to rate setup quality:

| Factor | Points | Description |
|--------|--------|-------------|
| Zone Quality | 0-2 | HTF relevance and strength |
| Rejection Clarity | 0-2 | Clear rejection candle formation |
| Structure Flip | 0-2 | CHoCH confirmation |
| Context | 0-2 | Balanced market (not trend day) |
| Intermarket Divergence | 0-2 | ETF divergence confirmation |

**A-Setup Threshold**: ≥7 points (out of 10 total)

## 🔧 Configuration

### Environment Variables

```bash
# API Credentials
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
POLYGON_API_KEY=your_polygon_key

# System Configuration
LOG_LEVEL=INFO
POLL_INTERVAL=30
```

### Configuration File (config/config.yaml)

```yaml
symbols:
  - SPY
  - QQQ
  - IWM

polling:
  interval_seconds: 30
  max_retries: 3
  timeout_seconds: 10

indicators:
  vwap:
    enabled: true
    standard_deviations: [1, 2]
  opening_range:
    duration_minutes: 30
  swing_structure:
    lookback_bars: 20

zones:
  prior_day: true
  weekly: true
  value_area: true

alerts:
  channels: ['console', 'file']
  min_qrs_score: 7
  deduplication_minutes: 5
```

## 📈 Technical Indicators

### VWAP (Volume Weighted Average Price)
- Calculated from RTH open using minute bars
- Includes 1σ and 2σ standard deviation bands
- Used for trend identification and target levels

### Opening Range (OR)
- First 30 minutes of regular trading hours
- Tracks OR high and OR low levels
- Used for intraday support/resistance

### Swing Structure Detection
- Identifies local swing highs and lows
- Detects Change of Character (CHoCH) patterns
- Essential for reversal confirmation

### Initiative Analysis
- Volume expansion ratios (impulse vs pullback)
- Candle spread vs volume mismatch
- VWAP slope analysis near zones

## 🔌 API Integration

### Alpaca API
- **Data**: SPY, QQQ, IWM OHLCV bars
- **Frequency**: 1-minute bars (15-minute delayed)
- **Rate Limit**: 200 requests/minute
- **Library**: `alpaca-py`

### Polygon API
- **Data**: Aggregates and previous day bars
- **Frequency**: 1-minute bars (15-minute delayed)
- **Rate Limit**: 5 requests/minute (free tier)
- **Library**: `polygon-api-client`

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=zone_fade_detector

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v
```

### Test Structure

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test API integration and data flow
- **Mock Data**: Use factory-boy for generating test data
- **Coverage**: Aim for >90% code coverage

## 📝 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📚 Documentation

- **PRD**: [Product Requirements Document](docs/PRD.md)
- **API Reference**: Generated from docstrings
- **Strategy Guide**: Detailed trading strategy documentation
- **Configuration Guide**: Complete configuration options

## 🚨 Alerts

The system supports multiple alert channels:

- **Console**: Real-time console output
- **File**: Log file output
- **Email**: SMTP email alerts (future)
- **Webhook**: HTTP POST notifications (future)

### Alert Format

```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "symbol": "SPY",
  "direction": "long",
  "zone_level": 485.50,
  "zone_type": "prior_day_high",
  "qrs_score": 8,
  "rejection_candle": {
    "open": 485.20,
    "high": 485.80,
    "low": 484.90,
    "close": 485.10,
    "volume": 1500000
  },
  "choch_confirmed": true,
  "vwap_level": 484.75,
  "target_1": 485.25,
  "target_2": 486.00
}
```

## 🔒 Security

- API credentials stored in environment variables
- No hardcoded secrets in source code
- HTTPS for all external API calls
- Input validation for all external data
- Secure configuration file handling

## 📊 Performance

- **Data Processing**: <5 seconds from API response to alert
- **Memory Usage**: <1GB for typical operation
- **CPU Usage**: <50% on modern hardware
- **Uptime**: >99% during market hours

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consider consulting with a financial advisor before making trading decisions.

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check the docs/ directory for detailed guides
- **Discussions**: Use GitHub Discussions for questions and community support

## 🔄 Changelog

### Version 0.1.0 (Initial Release)
- Basic Zone Fade detection logic
- Alpaca and Polygon API integration
- VWAP and swing structure indicators
- Quality Rating System (QRS)
- Console and file alert channels
- Comprehensive test suite
- Documentation and configuration

## 🗺️ Roadmap

### Phase 2
- Web dashboard for setup visualization
- Historical backtesting capabilities
- Additional technical indicators
- Multi-timeframe analysis

### Phase 3
- Machine learning for setup quality prediction
- Integration with trading platforms
- Real-time data feed integration
- Advanced risk management features