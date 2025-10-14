# Zone Fade Strategy Specification

## 🎯 Overview

The Zone Fade Strategy is a mean-reversion trading approach that identifies high-probability reversal setups by analyzing higher-timeframe zones, rejection candles, volume patterns, and market context. The strategy is designed for intraday trading with strict risk management and systematic exit procedures.

## 📊 Core Philosophy

### Trading Approach
- **Mean Reversion**: Fade moves away from equilibrium (VWAP)
- **Higher Timeframe Context**: Use daily/weekly zones for structure
- **Volume Confirmation**: Require volume spikes for validation
- **Quality Over Quantity**: Focus on high-probability setups only

### Guiding Principles
1. **Follow Structure, Not Emotion**: Exits triggered by quantifiable signals
2. **Preserve Capital First**: Hard stops are sacred - no averaging down
3. **Quality Dictates Patience**: Higher-quality setups deserve larger targets
4. **Adapt to Context**: Different market conditions require different approaches
5. **Determinism**: Same data and parameters yield same results

## 🎯 Entry Requirements

### Zone Requirements
- **Zone Types**: Prior day high/low, value area high/low, intraday structure
- **Zone Quality**: Minimum confluence score of 0.3
- **Zone Lifecycle**: Daily reset, maximum 8 zones per symbol per day
- **First Touch Only**: Only trade the first touch of each zone

### Rejection Candle Requirements
- **Wick Ratio**: Minimum 30% wick ratio
- **Volume Spike**: 1.8x average volume confirmation
- **Price Action**: Clear rejection from zone level
- **Timing**: Within 5 minutes of zone touch

### Market Context Requirements
- **Session Type**: Balanced or choppy days only (no trend days)
- **Time Filter**: Avoid first 30 minutes and last 30 minutes
- **Volatility**: Moderate volatility preferred
- **Momentum**: Fade momentum, not follow it

### Quality Rating System (QRS)
- **Minimum Score**: 7.0/10 for trade execution
- **Factors**:
  - Zone Quality (0-2 points)
  - Rejection Clarity (0-2 points)
  - Structure Flip (0-2 points)
  - Context (0-2 points)
  - Intermarket Divergence (0-2 points)

## 🎯 Exit Strategy

### Risk Unit (R) Calculation
```
R = |Entry Price - Initial Stop Price|
```

### Exit Hierarchy
1. **Hard Stop** (Highest Priority)
   - Zone invalidation (price breaks beyond zone)
   - CHoCH pivot violation
   - Immediate exit, no exceptions

2. **T1 Target** (Scale Out 40-50%)
   - Nearest of VWAP or 1R from entry
   - Scale out 45% of position
   - Move stop to breakeven
   - Continue to T2/T3

3. **T2 Target** (Scale Out 25%)
   - Opposite side of OR range or 2R
   - Scale out 25% of original position
   - Continue to T3

4. **T3 Target** (Close Remaining)
   - Opposite high-timeframe zone or 3R
   - Close remaining position
   - Trail or close completely

### Exit Logic Implementation
```python
# T1 Calculation
if direction == "SHORT":
    t1_price = max(vwap, entry_price - risk_unit)
else:  # LONG
    t1_price = min(vwap, entry_price + risk_unit)

# T2 Calculation
if direction == "SHORT":
    t2_price = max(opening_range_low, entry_price - 2*risk_unit)
else:  # LONG
    t2_price = min(opening_range_high, entry_price + 2*risk_unit)

# T3 Calculation
if direction == "SHORT":
    t3_price = entry_price - 3*risk_unit
else:  # LONG
    t3_price = entry_price + 3*risk_unit
```

## 🔧 Zone Management

### Zone Lifecycle
- **Creation**: Based on confluence scoring and daily limits
- **Persistence**: Daily reset at market close
- **Expiration**: Session-based (4 PM ET)
- **Touch Limit**: First touch only per zone

### Daily Limits
- **Total Zones**: 8 per symbol per day
- **Primary Zones**: 4 per symbol per day
- **Secondary Zones**: 4 per symbol per day
- **Zone Types**: Prior day, value area, intraday, VWAP deviation

### Confluence Scoring
- **Zone Type Priority**: 40% weight
- **QRS Score**: 25% weight
- **Zone Strength**: 15% weight
- **Volume Factor**: 10% weight
- **Time Factor**: 5% weight
- **Random Component**: 5% weight

## 📊 Market Context

### Session Classification
- **Trend Day**: Strong directional move, low volatility
- **Balanced Day**: Normal price action, moderate volatility
- **Choppy Day**: High volatility, erratic price action

### Context Detection
```python
def detect_market_context(bars, lookback=20):
    price_change = (bars[-1].close - bars[0].close) / bars[0].close * 100
    price_range = (max(b.high for b in bars) - min(b.low for b in bars)) / bars[0].close * 100
    
    if abs(price_change) > 0.5 and price_range < 2.0:
        return "trend"
    elif price_range > 3.0:
        return "choppy"
    else:
        return "balanced"
```

### Trading Rules by Context
- **Trend Days**: No fade trades (follow momentum)
- **Balanced Days**: Fade trades allowed
- **Choppy Days**: Fade trades allowed with caution

## 💰 Risk Management

### Position Sizing
- **Fixed Amount**: $10,000 per trade
- **Maximum Risk**: 2% of account per trade
- **Stop Loss**: Hard stop at zone invalidation
- **Position Size**: Calculated based on risk amount

### Commission and Slippage
- **Commission**: $5 per trade (flat rate)
- **Slippage**: 2 ticks per trade
- **Slippage Application**: Added for SHORT, subtracted for LONG

### Risk Controls
- **Maximum Drawdown**: 10% of account
- **Daily Loss Limit**: 5% of account
- **Position Limits**: Maximum 3 concurrent positions
- **Symbol Limits**: Maximum 1 position per symbol

## 📈 Performance Metrics

### Profitability Metrics
- **Net Profit**: Total P&L after all costs
- **Win Rate**: Percentage of profitable trades
- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Profit Factor**: Gross profit / Gross loss

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **VaR**: Value at Risk (95% confidence)

### Operational Metrics
- **Trade Frequency**: Trades per day/month
- **Execution Rate**: Percentage of signals executed
- **Rejection Rate**: Percentage of signals rejected
- **Average Hold Time**: Average time in position

## 🔍 Quality Assurance

### Entry Validation
- **Zone Quality Check**: Confluence score validation
- **Volume Confirmation**: Volume spike verification
- **Market Context**: Session type validation
- **Timing Check**: Entry window validation

### Exit Validation
- **Stop Loss**: Hard stop verification
- **Target Validation**: T1/T2/T3 calculation check
- **Scaling Logic**: Position scaling verification
- **Slippage Application**: Correct slippage calculation

### Performance Monitoring
- **Real-time Metrics**: Live performance tracking
- **Alert System**: Performance threshold alerts
- **Logging**: Comprehensive trade logging
- **Analysis**: Regular performance analysis

## 🎯 Implementation Notes

### Data Requirements
- **Resolution**: 1-minute bars minimum
- **Symbols**: SPY, QQQ, IWM (configurable)
- **History**: 5 years minimum for backtesting
- **Real-time**: Live data for trading

### Technical Requirements
- **Platform**: Docker containerized
- **APIs**: Alpaca, Polygon for data
- **Alerts**: Discord webhook integration
- **Storage**: Persistent data caching

### Configuration
- **Symbols**: Configurable symbol list
- **Thresholds**: Adjustable QRS and volume thresholds
- **Limits**: Configurable zone and position limits
- **Alerts**: Customizable alert channels

## 📚 Related Documentation

- [5-Year Backtesting Results](5YEAR_BACKTESTING_RESULTS.md)
- [Exit Logic Requirements](EXIT_LOGIC_REQUIREMENTS.md)
- [Backtesting Guide](BACKTESTING_GUIDE.md)
- [Changelog](CHANGELOG.md)

## 🔗 Implementation Files

- **Main Strategy**: `src/zone_fade_detector/strategies/zone_fade_strategy.py`
- **5-Year Backtest**: `backtesting/5year_zone_fade_backtest.py`
- **Data Download**: `backtesting/download_5year_data.py`
- **Hard Stop Analysis**: `backtesting/hard_stop_analysis.py`

---

*Last Updated: January 11, 2025*
*Version: 2.0.0*