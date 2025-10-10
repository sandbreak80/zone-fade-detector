# Zone Fade Detector - Product Requirements Document

## 1. Executive Summary

### 1.1 Project Overview
The Zone Fade Detector is a Python-based trading system that identifies high-probability reversal setups in equity markets using 15-minute delayed data from Alpaca and Polygon APIs. The system detects when price reaches higher-timeframe zones and shows signs of exhaustion before reversing, providing automated alerts for Zone Fade trading opportunities.

### 1.2 Business Objectives
- Automate the detection of Zone Fade trading setups
- Provide real-time alerts for high-probability reversal opportunities
- Support trading of SPY, QQQ, and IWM (proxies for /ES, /NQ, /RTY futures)
- Implement a quality rating system to filter only A-grade setups
- Enable systematic backtesting and performance analysis

### 1.3 Success Metrics
- Detection accuracy: >80% of A-Setup alerts result in profitable reversals
- False positive rate: <20% of alerts fail to produce expected price action
- System uptime: >99% during market hours
- Alert latency: <60 seconds from setup formation to alert generation

## 2. Technical Requirements

### 2.1 System Architecture
- **Language**: Python 3.11+
- **Environment**: Virtual environment (venv)
- **Architecture**: Modular, event-driven design
- **Data Flow**: REST API polling → Data processing → Signal detection → Alert generation
- **Deployment**: Local/cloud deployment with configuration management

### 2.2 Data Sources
| Source | Library | Data Type | Frequency | Delay |
|--------|---------|-----------|-----------|-------|
| Alpaca | alpaca-py | SPY, QQQ, IWM OHLCV | 1-minute bars | 15 minutes |
| Polygon | polygon.client | Aggregates, previous day bars | 1-minute bars | 15 minutes |

### 2.3 Core Components
1. **Data Manager**: Handles API integration and data caching
2. **Indicator Engine**: Calculates technical indicators (VWAP, OR, swing structure)
3. **Zone Detector**: Identifies higher-timeframe support/resistance zones
4. **Signal Processor**: Implements Zone Fade detection logic
5. **Quality Rater**: Applies QRS scoring system
6. **Alert System**: Generates and manages trading alerts

## 3. Functional Requirements

### 3.1 Data Management
- **FR-001**: Poll Alpaca and Polygon APIs every 30 seconds for latest data
- **FR-002**: Cache historical data for efficient indicator calculations
- **FR-003**: Validate data integrity before processing
- **FR-004**: Handle API rate limits and network errors gracefully
- **FR-005**: Support data backfill for historical analysis

### 3.2 Indicator Calculations
- **FR-006**: Calculate session VWAP from RTH open using minute bars
- **FR-007**: Compute VWAP standard deviation bands (1σ, 2σ)
- **FR-008**: Identify Opening Range (OR) high/low from first 30 minutes
- **FR-009**: Track Overnight High/Low (ONH/ONL) from prior close to current open
- **FR-010**: Detect swing structure and Change of Character (CHoCH)
- **FR-011**: Calculate volume expansion ratios and initiative proxies

### 3.3 Zone Detection
- **FR-012**: Identify prior day high/low levels
- **FR-013**: Calculate weekly high/low levels
- **FR-014**: Estimate Value Area High/Low using volume concentration
- **FR-015**: Track price approach to defined zones
- **FR-016**: Validate zone quality and relevance

### 3.4 Signal Detection
- **FR-017**: Detect first touch of higher-timeframe zones
- **FR-018**: Identify rejection candles (pin bars, engulfing, long wicks)
- **FR-019**: Confirm lack of initiative through volume analysis
- **FR-020**: Detect Change of Character in opposite direction
- **FR-021**: Validate market environment (balanced vs trend day)
- **FR-022**: Check intermarket divergence between SPY, QQQ, IWM

### 3.5 Quality Rating System
- **FR-023**: Implement 5-factor QRS scoring:
  - Zone quality (HTF relevance): 0-2 points
  - Rejection clarity: 0-2 points
  - Structure flip (CHoCH): 0-2 points
  - Context (balanced/not trend day): 0-2 points
  - Intermarket divergence: 0-2 points
- **FR-024**: Classify setups as A-Setup (≥7 points) or lower quality
- **FR-025**: Log detailed scoring breakdown for each setup

### 3.6 Alert Management
- **FR-026**: Generate alerts for A-Setup Zone Fade opportunities
- **FR-027**: Include setup details: symbol, direction, zone level, QRS score
- **FR-028**: Support multiple alert channels (console, file, email, webhook)
- **FR-029**: Implement alert deduplication to prevent spam
- **FR-030**: Provide alert history and performance tracking

## 4. Non-Functional Requirements

### 4.1 Performance
- **NFR-001**: Process data updates within 5 seconds of API response
- **NFR-002**: Support concurrent processing of multiple symbols
- **NFR-003**: Memory usage < 1GB for typical operation
- **NFR-004**: CPU usage < 50% on modern hardware during market hours

### 4.2 Reliability
- **NFR-005**: System uptime > 99% during market hours (9:30 AM - 4:00 PM ET)
- **NFR-006**: Graceful handling of API failures and network issues
- **NFR-007**: Automatic recovery from temporary errors
- **NFR-008**: Data persistence across system restarts

### 4.3 Scalability
- **NFR-009**: Support additional symbols without major code changes
- **NFR-010**: Configurable polling intervals (15s to 5min)
- **NFR-011**: Modular design for easy feature additions
- **NFR-012**: Efficient memory usage for extended operation

### 4.4 Security
- **NFR-013**: Secure storage of API credentials
- **NFR-014**: No hardcoded secrets in source code
- **NFR-015**: HTTPS for all external API calls
- **NFR-016**: Input validation for all external data

### 4.5 Maintainability
- **NFR-017**: Comprehensive test coverage (>90%)
- **NFR-018**: Clear documentation and code comments
- **NFR-019**: Modular architecture for easy updates
- **NFR-020**: Configuration-driven behavior

## 5. Trading Strategy Specifications

### 5.1 Zone Fade Definition
A Zone Fade is a high-probability reversal setup that occurs when:
1. Price approaches a pre-defined higher-timeframe zone
2. Market shows signs of exhaustion (lack of initiative)
3. Price fails to accept beyond the zone and closes back inside
4. A rejection candle forms at or beyond the zone
5. A Change of Character occurs in the opposite direction

### 5.2 Entry Criteria
- **Trigger**: CHoCH opposite prior direction + close back inside zone
- **Confirmation**: Candle closes in favor of reversal
- **Entry Method**: Limit or market order at mid-zone or on CHoCH confirmation
- **Stop Placement**: 1-2 ticks beyond zone back or CHoCH invalidation swing
- **Targets**: T1 (VWAP/range mid), T2 (opposite range edge)

### 5.3 Market Context Requirements
- Market must be balanced or neutral (not a trend day)
- VWAP slope should be flat or flattening
- Price should show overlapping value areas
- At least one ETF should diverge from others (intermarket signal)

### 5.4 Invalidation Criteria
- Trend day detected (strong unidirectional VWAP slope)
- Price accepts and holds beyond the zone
- Volume expands in breakout direction
- CHoCH invalidation occurs

## 6. Data Models

### 6.1 Core Data Structures
```python
@dataclass
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class Zone:
    level: float
    zone_type: str  # 'prior_day_high', 'prior_day_low', 'weekly_high', etc.
    quality: int  # 0-2
    created_at: datetime

@dataclass
class ZoneFadeSetup:
    symbol: str
    direction: str  # 'long' or 'short'
    zone: Zone
    rejection_candle: OHLCVBar
    choch_confirmed: bool
    qrs_score: int
    timestamp: datetime
    is_a_setup: bool
```

### 6.2 Indicator Data
```python
@dataclass
class VWAPData:
    vwap: float
    upper_1sigma: float
    lower_1sigma: float
    upper_2sigma: float
    lower_2sigma: float
    slope: float

@dataclass
class OpeningRange:
    high: float
    low: float
    start_time: datetime
    end_time: datetime

@dataclass
class SwingStructure:
    swing_highs: List[float]
    swing_lows: List[float]
    last_swing_high: float
    last_swing_low: float
    choch_detected: bool
```

## 7. API Integration Specifications

### 7.1 Alpaca API
- **Endpoint**: Market data for SPY, QQQ, IWM
- **Data Type**: 1-minute OHLCV bars
- **Rate Limit**: 200 requests/minute
- **Authentication**: API key + secret
- **Error Handling**: Exponential backoff with jitter

### 7.2 Polygon API
- **Endpoint**: Aggregates and previous day data
- **Data Type**: 1-minute bars and aggregates
- **Rate Limit**: 5 requests/minute (free tier)
- **Authentication**: API key
- **Error Handling**: Retry with exponential backoff

## 8. Configuration Management

### 8.1 Environment Variables
- `ALPACA_API_KEY`: Alpaca API key
- `ALPACA_SECRET_KEY`: Alpaca secret key
- `POLYGON_API_KEY`: Polygon API key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `POLL_INTERVAL`: Data polling interval in seconds (default: 30)

### 8.2 Configuration File
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

## 9. Testing Requirements

### 9.1 Unit Tests
- Test all indicator calculations with known data
- Test zone detection logic with various market conditions
- Test signal detection with mock price data
- Test QRS scoring system with edge cases
- Test API integration with mocked responses

### 9.2 Integration Tests
- Test end-to-end data flow from API to alert generation
- Test error handling and recovery scenarios
- Test configuration loading and validation
- Test alert generation and delivery

### 9.3 Performance Tests
- Test system performance under high data load
- Test memory usage over extended periods
- Test API rate limit handling
- Test concurrent symbol processing

## 10. Deployment Requirements

### 10.1 Development Environment
- Python 3.11+ with venv
- All dependencies from requirements.txt
- API keys in environment variables
- Local configuration file

### 10.2 Production Environment
- Linux server with Python 3.11+
- Process management (systemd or supervisor)
- Log rotation and monitoring
- Backup and recovery procedures
- Health check endpoints

## 11. Monitoring and Alerting

### 11.1 System Monitoring
- API response times and error rates
- Data processing latency
- Memory and CPU usage
- Alert generation frequency and accuracy

### 11.2 Business Metrics
- Number of A-Setup alerts generated
- QRS score distribution
- Setup success rate (manual tracking)
- System uptime during market hours

## 12. Future Enhancements

### 12.1 Phase 2 Features
- Web dashboard for setup visualization
- Historical backtesting capabilities
- Additional technical indicators
- Multi-timeframe analysis

### 12.2 Phase 3 Features
- Machine learning for setup quality prediction
- Integration with trading platforms
- Real-time data feed integration
- Advanced risk management features

## 13. Risk Assessment

### 13.1 Technical Risks
- API rate limiting and data availability
- System performance under high load
- Data quality and integrity issues
- Network connectivity problems

### 13.2 Mitigation Strategies
- Implement robust error handling and retry logic
- Use efficient data structures and algorithms
- Validate all incoming data
- Implement health checks and monitoring

### 13.3 Business Risks
- False positive alerts leading to poor trading decisions
- System downtime during critical market periods
- Regulatory compliance issues
- Data security and privacy concerns

## 14. Success Criteria

### 14.1 Technical Success
- System runs reliably during market hours
- Alerts are generated within 60 seconds of setup formation
- Data processing accuracy > 99.9%
- System uptime > 99% during market hours

### 14.2 Business Success
- A-Setup alerts show >80% accuracy in predicting reversals
- False positive rate <20%
- System provides consistent, actionable trading signals
- User satisfaction with alert quality and timing

## 15. Acceptance Criteria

### 15.1 Functional Acceptance
- All functional requirements (FR-001 through FR-030) are implemented and tested
- System correctly identifies Zone Fade setups according to specified criteria
- QRS scoring system produces consistent, accurate ratings
- Alert system delivers timely, relevant notifications

### 15.2 Non-Functional Acceptance
- All performance requirements (NFR-001 through NFR-004) are met
- Reliability requirements (NFR-005 through NFR-008) are satisfied
- Security requirements (NFR-013 through NFR-016) are implemented
- Maintainability requirements (NFR-017 through NFR-020) are achieved

### 15.3 User Acceptance
- System is easy to configure and deploy
- Alerts are clear, actionable, and timely
- Documentation is comprehensive and accurate
- System performs reliably in production environment