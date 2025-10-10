# Zone Fade Detector

A sophisticated trading system for detecting Zone Fade setups using higher-timeframe zones, rejection candles, and volume analysis. **Currently optimized for backtesting and validation - not ready for live trading.**

## 🎯 Overview

The Zone Fade Detector implements a comprehensive trading strategy that identifies high-probability reversal setups by analyzing:

- **Higher-Timeframe Zones**: Daily and weekly supply/demand levels
- **Rejection Candles**: Price action showing initiative exhaustion (30% wick ratio)
- **Volume Analysis**: Volume spike confirmation on rejection (1.8x threshold)
- **CHoCH Detection**: Change of Character confirmation
- **Quality Rating System (QRS)**: 5-factor setup scoring (7+ threshold)
- **Entry Window Tracking**: Duration analysis for trade execution timing

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Alpaca API credentials
- Polygon API credentials (optional)
- Discord webhook URL (optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd zone-fade-detector
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

### Configuration

Edit `config/config.yaml` to customize:
- Symbols to monitor
- QRS thresholds
- Alert channels
- Detection parameters

## 📊 Features

### Core Detection
- **Zone Detection**: Prior day/week highs/lows, value areas, opening ranges
- **Rejection Analysis**: Wick analysis with volume spike confirmation
- **CHoCH Detection**: Swing structure analysis for momentum shifts
- **QRS Scoring**: 5-factor quality rating system

### Alert System
- **Console Alerts**: Real-time console output
- **File Logging**: Persistent alert logs
- **Discord Integration**: Real-time webhook alerts
- **Rich Formatting**: Color-coded, detailed alert information

### Data Management
- **Alpaca Integration**: Real-time and historical data
- **Polygon Integration**: Additional data sources
- **Persistent Caching**: Efficient data storage
- **Backtesting Support**: Historical data analysis

## 🏗️ Architecture

### Core Components
- **ZoneFadeStrategy**: Main strategy implementation
- **SignalProcessor**: Setup filtering and coordination
- **QRSScorer**: Quality rating system
- **AlertSystem**: Multi-channel alert management

### Indicators
- **VWAPCalculator**: Volume-weighted average price
- **SwingStructureDetector**: CHoCH detection
- **VolumeAnalyzer**: Volume spike analysis
- **OpeningRangeCalculator**: Session range analysis

### Data Layer
- **AlpacaClient**: Real-time data fetching
- **PolygonClient**: Additional data sources
- **DataManager**: Caching and persistence

## 📈 Strategy Details

### Zone Fade Setup Requirements
1. **HTF Zone Approach**: Price approaching higher-timeframe zone
2. **Rejection Candle**: Clear wick rejection with volume spike
3. **CHoCH Confirmation**: Change of character in swing structure
4. **Quality Score**: QRS score above threshold (default: 5/10)

### QRS Scoring Factors
1. **Zone Quality** (0-2 points): HTF relevance and strength
2. **Rejection Clarity** (0-2 points): Wick analysis + volume spike
3. **Structure Flip** (0-2 points): CHoCH confirmation
4. **Context** (0-2 points): Market environment analysis
5. **Intermarket Divergence** (0-2 points): Cross-asset confirmation

## 🔧 Usage

### Live Trading
```bash
docker-compose up zone-fade-detector
```

### Backtesting
```bash
# Download historical data (2024)
docker-compose run zone-fade-detector python backtesting/download_2024_data.py

# Run efficient validation backtesting
docker-compose run zone-fade-detector python backtesting/backtest_2024_efficient_validation.py

# Run full 2024 backtesting (longer)
docker-compose run zone-fade-detector python backtesting/backtest_2024_full_validation.py

# Run December 2024 debug backtesting
docker-compose run zone-fade-detector python backtesting/backtest_2024_december_debug.py
```

### Results and Validation
```bash
# View backtesting results
ls -la results/2024/efficient/

# Access manual validation package
ls -la results/manual_validation/

# Review entry points data
cat results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv
```

### Development
```bash
# Run in development mode
docker-compose up zone-fade-detector-dev

# Run tests
docker-compose run zone-fade-detector-test pytest
```

## 📋 Configuration

### Environment Variables
- `ALPACA_API_KEY`: Alpaca API key
- `ALPACA_SECRET_KEY`: Alpaca secret key
- `POLYGON_API_KEY`: Polygon API key (optional)
- `DISCORD_WEBHOOK_URL`: Discord webhook URL (optional)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### Configuration File
Edit `config/config.yaml` for:
- Symbol lists
- QRS thresholds
- Alert settings
- Detection parameters

## 🧪 Testing

### Unit Tests
```bash
docker-compose run zone-fade-detector-test pytest tests/unit/
```

### Integration Tests
```bash
docker-compose run zone-fade-detector-test pytest tests/integration/
```

### Manual Testing
```bash
# Test Discord webhook
docker-compose run zone-fade-detector python tests/integration/test_discord_simple.py

# Test volume spike detection
docker-compose run zone-fade-detector python tests/integration/test_volume_spike.py
```

## 📊 Performance

### Current Capabilities
- **Real-time Processing**: 30-second polling intervals
- **Multi-symbol Support**: SPY, QQQ, IWM (configurable)
- **Volume Spike Detection**: 1.8x volume spikes detected
- **QRS Scoring**: 6.23/10 average scores achieved
- **Entry Window Tracking**: 28.9 minutes average duration

### 2024 Backtesting Results
- **Total Entry Points**: 160 high-quality setups detected
- **Entry Points per Day**: 0.6 (highly selective)
- **Average QRS Score**: 6.23/10 (above 7.0 threshold)
- **Entry Window Duration**: 16-29 minutes (sufficient execution time)
- **Success Rate**: 100% long entry windows (>15 minutes)
- **Symbol Breakdown**:
  - QQQ: 49 entry points (0.19/day)
  - SPY: 33 entry points (0.13/day)
  - IWM: 78 entry points (0.31/day)

### Quality Metrics
- **Zone Touch Rate**: 1.75% of all bars
- **Rejection Rate**: 24.05% of zone touches
- **Volume Spike Rate**: 20.41% of rejections
- **Entry Point Rate**: 6.08% of zone touches

## 🚧 Roadmap

### ✅ Completed
- Volume spike detection implementation
- Discord webhook integration
- QRS scoring enhancement
- Rolling window management system
- Session state management
- Micro window analysis
- Parallel cross-symbol processing
- 2024 backtesting validation
- Entry window duration tracking
- Manual validation tools

### 🔄 In Progress
- ES/NQ/RTY futures integration (for production)
- Live trading optimization
- Real-time performance monitoring

### 📋 Planned
- Advanced intermarket analysis
- Machine learning integration
- Portfolio management
- Risk management enhancements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves risk and may not be suitable for all investors. Past performance does not guarantee future results.

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review the configuration examples

## 📚 Documentation

- [Setup Guide](docs/SETUP_GUIDE.md)
- [Backtesting Guide](docs/BACKTESTING_GUIDE.md)
- [Manual Validation Guide](docs/MANUAL_VALIDATION_GUIDE.md)
- [Strategy Analysis](docs/STRATEGY_ANALYSIS.md)
- [Volume Spike Implementation](docs/VOLUME_SPIKE_IMPLEMENTATION.md)
- [Operational Analysis](docs/OPERATIONAL_ANALYSIS.md)
- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [2024 Results Summary](docs/2024_RESULTS_SUMMARY.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Changelog](docs/CHANGELOG.md)