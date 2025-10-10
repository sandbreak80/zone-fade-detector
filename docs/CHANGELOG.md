# Changelog

All notable changes to the Zone Fade Detector project will be documented in this file.

## [Unreleased]

### Added
- **Rolling Window Manager**: Centralized time window management system
- **Session State Manager**: Comprehensive RTH session tracking and analysis
- **Micro Window Analyzer**: Pre/post zone touch analysis for initiative detection
- **Parallel Cross-Symbol Processor**: Real-time intermarket analysis across multiple symbols
- **Entry Window Duration Tracking**: Analysis of how long entry opportunities remain valid
- **2024 Backtesting Validation**: Comprehensive backtesting with 160 high-quality entry points
- **Manual Validation Tools**: CSV export and validation guides for entry point analysis
- **Efficient Backtesting Scripts**: Optimized scripts for different backtesting scenarios
- **Volume Spike Detection**: Enhanced rejection candle validation with volume confirmation
- **Enhanced QRS Scoring**: 5-factor quality rating system with strict thresholds
- **Comprehensive Discord Integration**: Real-time webhook alerts with detailed formatting
- **Multi-channel Alert System**: Console, file, and Discord alert channels
- **Docker Containerization**: Multi-stage builds with optimized performance
- **Comprehensive Test Suite**: Unit and integration tests for all components
- **Detailed Documentation**: Architecture guides, backtesting guides, and analysis reports

### Changed
- **Strategy Parameters**: Restored original strict values (30% wick ratio, 1.8x volume, QRS 7+)
- **QRS Scoring System**: Enhanced 5-factor scoring with volume spike integration
- **Swing Structure Detection**: Optimized parameters for better CHoCH detection
- **Backtesting Performance**: Parallel processing and memory optimization
- **Error Handling**: Improved error handling and logging throughout the system
- **Documentation Structure**: Organized all documentation in `/docs` directory

### Fixed
- **Discord Webhook Errors**: Fixed attribute errors and QRS field formatting
- **Timezone Issues**: Resolved timezone comparison problems in signal processing
- **Permission Issues**: Fixed file logging and CSV export permissions
- **Import Paths**: Corrected module import paths for all components
- **Memory Management**: Optimized memory usage for large dataset processing
- **CSV Export**: Fixed permission issues with validation output directory

### Performance Improvements
- **2024 Backtesting Results**: 160 high-quality entry points detected
- **Entry Window Analysis**: 28.9 minutes average duration with 100% long windows
- **Quality Metrics**: 6.23/10 average QRS score with 0.6 entry points per day
- **Parallel Processing**: 3-thread processing for efficient backtesting
- **Memory Optimization**: Chunked processing for large datasets
- **Real-time Updates**: Progress tracking and status updates during processing

### Architecture Enhancements
- **Operational Design**: Implemented complete rolling window architecture
- **Component Integration**: Seamless integration of all new components
- **Data Flow Optimization**: Efficient data flow between components
- **Scalability**: Designed for future expansion and optimization

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