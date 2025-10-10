# Changelog

All notable changes to the Zone Fade Detector project will be documented in this file.

## [Unreleased]

### Added
- Volume spike detection for rejection candles
- Enhanced QRS scoring with volume confirmation
- Comprehensive Discord webhook integration
- Backtesting support with 2024 historical data
- Multi-channel alert system (console, file, Discord)
- Docker containerization with multi-stage builds
- Comprehensive test suite (unit and integration)
- Detailed documentation and analysis reports

### Changed
- Improved rejection candle validation with volume analysis
- Enhanced QRS scoring system (5-factor scoring)
- More permissive swing structure detection
- Optimized detection parameters for better results
- Improved error handling and logging

### Fixed
- Discord webhook formatting errors
- Timezone comparison issues in signal processor
- Permission issues with file logging
- QRS scoring threshold adjustments

### Technical Improvements
- Added volume spike detection methods
- Enhanced rejection candle validation
- Improved QRS scoring with volume bonus
- Better error handling and fallback mechanisms
- Comprehensive test coverage

## [0.1.0] - 2024-10-10

### Added
- Initial Zone Fade Detector implementation
- Core strategy components (ZoneFadeStrategy, SignalProcessor, QRSScorer)
- Alert system with multiple channels
- Data integration (Alpaca, Polygon)
- Docker containerization
- Basic test framework

### Features
- Zone detection (daily/weekly highs/lows, value areas)
- Rejection candle analysis
- CHoCH detection
- QRS scoring system
- Multi-symbol support (SPY, QQQ, IWM)
- Real-time and historical data processing