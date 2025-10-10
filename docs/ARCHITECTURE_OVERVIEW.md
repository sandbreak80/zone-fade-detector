# Architecture Overview

This document provides a comprehensive overview of the Zone Fade Detector's architecture, including all components, data flow, and operational design.

## 🏗️ System Architecture

### High-Level Overview
The Zone Fade Detector is built on a modular, scalable architecture designed for both backtesting and live trading. The system processes market data through multiple analysis layers to identify high-quality Zone Fade setups.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Core Detector  │    │  Alert System   │
│                 │    │                 │    │                 │
│ • Alpaca API    │───▶│ • Zone Detection│───▶│ • Console       │
│ • Polygon API   │    │ • Rejection     │    │ • File Logging  │
│ • Historical    │    │ • QRS Scoring   │    │ • Discord       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧩 Core Components

### 1. Zone Fade Detector (`detector.py`)
**Purpose**: Main orchestrator that coordinates all components
**Responsibilities**:
- Data ingestion and processing
- Component coordination
- Detection cycle management
- Rolling window management

**Key Features**:
- 30-second polling intervals
- Multi-symbol support (SPY, QQQ, IWM)
- Rolling window integration
- Error handling and recovery

### 2. Zone Fade Strategy (`zone_fade_strategy.py`)
**Purpose**: Core strategy implementation
**Responsibilities**:
- Zone approach detection
- Rejection candle validation
- CHoCH confirmation
- Setup creation and scoring

**Key Features**:
- 30% wick ratio requirement
- 1.8x volume spike threshold
- CHoCH confirmation logic
- Quality setup filtering

### 3. Signal Processor (`signal_processor.py`)
**Purpose**: Setup filtering and coordination
**Responsibilities**:
- Setup validation
- Duplicate filtering
- Alert generation
- Quality control

**Key Features**:
- QRS threshold filtering
- Time-based deduplication
- Alert prioritization
- Performance tracking

### 4. QRS Scorer (`qrs_scorer.py`)
**Purpose**: Quality rating system
**Responsibilities**:
- 5-factor scoring
- Quality assessment
- Score calculation
- Performance metrics

**Key Features**:
- Zone quality scoring (0-2 points)
- Rejection clarity scoring (0-2 points)
- Structure flip scoring (0-2 points)
- Context scoring (0-2 points)
- Intermarket divergence scoring (0-2 points)

## 🔧 Operational Architecture

### Rolling Window Manager (`rolling_window_manager.py`)
**Purpose**: Centralized time window management
**Responsibilities**:
- Multiple time window management
- Memory optimization
- Session resets
- Performance tracking

**Window Types**:
- **VWAP**: Rolling VWAP calculation
- **Session Context**: RTH session tracking
- **Swing/CHoCH**: Short-term structure analysis
- **Initiative**: Pre/post zone touch analysis
- **Intermarket**: Cross-symbol analysis
- **QRS Accumulator**: Quality score accumulation
- **Opening Range**: First 30 minutes of session
- **HTF Zones**: Higher-timeframe zone tracking

### Session State Manager (`session_state_manager.py`)
**Purpose**: RTH session state tracking
**Responsibilities**:
- Session phase detection
- Boundary management
- Metrics calculation
- Market context analysis

**Session Phases**:
- **Pre-Market**: 4:00 AM - 9:30 AM ET
- **Opening Range**: 9:30 AM - 10:00 AM ET
- **Early Session**: 10:00 AM - 12:00 PM ET
- **Mid Session**: 12:00 PM - 2:00 PM ET
- **Late Session**: 2:00 PM - 4:00 PM ET
- **After Hours**: 4:00 PM - 8:00 PM ET

### Micro Window Analyzer (`micro_window_analyzer.py`)
**Purpose**: Pre/post zone touch analysis
**Responsibilities**:
- Initiative detection
- Absorption analysis
- Exhaustion patterns
- Volume analysis

**Analysis Types**:
- **Pre-Touch**: 15 minutes before zone touch
- **Post-Touch**: 10 minutes after zone touch
- **Initiative Patterns**: Volume and momentum analysis
- **Absorption Signals**: Lack of follow-through
- **Exhaustion Signals**: Volume and volatility spikes

### Parallel Cross-Symbol Processor (`parallel_cross_symbol_processor.py`)
**Purpose**: Real-time intermarket analysis
**Responsibilities**:
- Multi-symbol processing
- Correlation analysis
- Divergence detection
- Risk sentiment analysis

**Analysis Types**:
- **Price Change**: Percentage change analysis
- **Volume Ratio**: Volume comparison
- **Momentum**: Price momentum analysis
- **Volatility**: Volatility comparison
- **Relative Strength**: Cross-symbol strength
- **Trend Analysis**: Trend direction analysis
- **Outlier Detection**: Unusual price movements
- **Correlation**: Cross-symbol correlation

## 📊 Data Flow Architecture

### 1. Data Ingestion
```
Market Data → Alpaca/Polygon APIs → Data Manager → Rolling Window Manager
```

### 2. Analysis Pipeline
```
Rolling Windows → Strategy Components → Signal Processor → Alert System
```

### 3. Component Integration
```
Zone Fade Strategy ←→ QRS Scorer ←→ Signal Processor
       ↓
Rolling Window Manager ←→ Session State Manager
       ↓
Micro Window Analyzer ←→ Parallel Cross-Symbol Processor
```

## 🔄 Processing Flow

### 1. Data Collection
- **Real-time**: 30-second polling from Alpaca API
- **Historical**: Batch processing from Polygon API
- **Caching**: Persistent storage with diskcache
- **Validation**: Data quality checks and filtering

### 2. Window Management
- **Rolling Windows**: Continuous window updates
- **Session Tracking**: RTH session state management
- **Memory Management**: Efficient memory usage
- **Performance Monitoring**: Window performance metrics

### 3. Analysis Processing
- **Zone Detection**: HTF zone identification
- **Rejection Analysis**: Candle pattern analysis
- **Volume Analysis**: Volume spike detection
- **CHoCH Detection**: Swing structure analysis
- **QRS Scoring**: Quality assessment

### 4. Signal Generation
- **Setup Creation**: Zone fade setup generation
- **Quality Filtering**: QRS threshold filtering
- **Deduplication**: Duplicate setup removal
- **Alert Generation**: Multi-channel alerts

## 🎯 Strategy Implementation

### Zone Fade Setup Requirements
1. **HTF Zone Approach**: Price approaching higher-timeframe zone
2. **Rejection Candle**: Clear wick rejection (30%+ ratio)
3. **Volume Spike**: Significant volume confirmation (1.8x+)
4. **CHoCH Confirmation**: Change of character in swing structure
5. **Quality Score**: QRS score above threshold (7.0+)

### QRS Scoring System
The Quality Rating System (QRS) evaluates setups across 5 factors:

1. **Zone Quality** (0-2 points)
   - HTF relevance and strength
   - Prior touches and history
   - Zone type appropriateness

2. **Rejection Clarity** (0-2 points)
   - Wick ratio analysis
   - Volume spike confirmation
   - Price action clarity

3. **Structure Flip** (0-2 points)
   - CHoCH confirmation
   - Swing structure analysis
   - Momentum shift detection

4. **Context** (0-2 points)
   - Market environment
   - VWAP relationship
   - Session phase analysis

5. **Intermarket Divergence** (0-2 points)
   - Cross-symbol confirmation
   - Sector rotation analysis
   - Risk sentiment assessment

## 🚀 Performance Architecture

### Parallel Processing
- **Multi-threading**: 3-thread processing for backtesting
- **Async Operations**: Asynchronous data processing
- **Concurrent Analysis**: Parallel component execution
- **Resource Management**: Efficient resource utilization

### Memory Management
- **Rolling Windows**: Efficient window data management
- **Caching**: Persistent data caching
- **Garbage Collection**: Automatic memory cleanup
- **Performance Monitoring**: Memory usage tracking

### Error Handling
- **Graceful Degradation**: System continues on component failure
- **Error Recovery**: Automatic error recovery
- **Logging**: Comprehensive error logging
- **Monitoring**: Real-time error monitoring

## 🔧 Configuration Architecture

### Environment Configuration
- **API Keys**: Alpaca, Polygon, Discord
- **Logging**: Log level and output configuration
- **Database**: Caching and persistence settings
- **Alerts**: Alert channel configuration

### Strategy Configuration
- **Symbols**: Monitored symbol list
- **Parameters**: Strategy parameter tuning
- **Thresholds**: QRS and quality thresholds
- **Windows**: Time window configurations

### System Configuration
- **Polling**: Data collection intervals
- **Processing**: Analysis processing settings
- **Memory**: Memory management settings
- **Performance**: Performance monitoring settings

## 📈 Monitoring and Observability

### Performance Metrics
- **Processing Time**: Component execution times
- **Memory Usage**: Memory consumption tracking
- **Error Rates**: Error frequency and types
- **Throughput**: Data processing rates

### Quality Metrics
- **QRS Scores**: Quality score distribution
- **Entry Points**: Entry point frequency and quality
- **False Positives**: Incorrect setup detection
- **False Negatives**: Missed setup detection

### System Health
- **Component Status**: Individual component health
- **Data Quality**: Data source quality
- **Alert Delivery**: Alert system performance
- **Resource Usage**: System resource utilization

## 🚧 Future Architecture Enhancements

### Planned Improvements
- **Futures Integration**: ES/NQ/RTY futures data
- **Machine Learning**: ML-based parameter optimization
- **Advanced Analytics**: More sophisticated indicators
- **Portfolio Management**: Multi-asset portfolio management

### Scalability Considerations
- **Horizontal Scaling**: Multi-instance deployment
- **Load Balancing**: Distributed processing
- **Database Scaling**: Distributed data storage
- **API Scaling**: High-frequency data processing

### Production Readiness
- **Live Trading**: Real-time execution optimization
- **Risk Management**: Comprehensive risk controls
- **Monitoring**: Advanced monitoring and alerting
- **Deployment**: Production deployment automation

## 📚 Related Documentation

- [Backtesting Guide](BACKTESTING_GUIDE.md)
- [Manual Validation Guide](MANUAL_VALIDATION_GUIDE.md)
- [Strategy Analysis](STRATEGY_ANALYSIS.md)
- [2024 Results Summary](2024_RESULTS_SUMMARY.md)
- [Operational Analysis](OPERATIONAL_ANALYSIS.md)

## 🔧 Development Guidelines

### Code Organization
- **Modular Design**: Clear component separation
- **Interface Contracts**: Well-defined interfaces
- **Error Handling**: Comprehensive error management
- **Testing**: Unit and integration tests

### Performance Guidelines
- **Efficient Algorithms**: Optimized processing
- **Memory Management**: Efficient memory usage
- **Caching**: Strategic data caching
- **Monitoring**: Performance tracking

### Quality Guidelines
- **Code Quality**: High code standards
- **Documentation**: Comprehensive documentation
- **Testing**: Thorough testing coverage
- **Validation**: Manual validation processes

---

*This architecture overview provides a comprehensive understanding of the Zone Fade Detector system. For specific implementation details, refer to the individual component documentation and source code.*