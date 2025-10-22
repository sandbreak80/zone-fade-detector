# Bitcoin Zone Fade Detector - Backtesting System

This directory contains all the components needed to backtest the Bitcoin Zone Fade strategy using historical cryptocurrency data. The backtesting system allows you to validate the strategy's performance, optimize parameters, and analyze risk before deploying to live trading.

## 📁 Directory Structure

```
backtesting/
├── README.md                    # This file - comprehensive backtesting guide
├── backtest_bitcoin.py         # Main backtesting script
├── config/
│   └── bitcoin_backtest.yaml   # Backtesting configuration
├── results/                     # Generated backtest results (created automatically)
├── charts/                      # Performance charts and visualizations
└── logs/                        # Backtesting logs
```

## 🎯 What is Backtesting?

Backtesting is the process of testing a trading strategy on historical data to evaluate its performance before risking real money. For the Bitcoin Zone Fade strategy, this means:

- **Historical Data**: Using past Bitcoin/Ethereum price data from CoinGecko
- **Strategy Simulation**: Running the Zone Fade detection logic on historical data
- **Performance Analysis**: Measuring win rates, returns, drawdowns, and risk metrics
- **Parameter Optimization**: Finding the best settings for different market conditions

## 🚀 Quick Start

### 1. Run a Simple Backtest
```bash
# From the project root
python backtest_bitcoin.py

# Or with custom configuration
python backtest_bitcoin.py --config config/bitcoin_backtest.yaml
```

### 2. View Results
```bash
# Results are saved to backtest_results.json
cat backtest_results.json

# Charts are generated in the charts/ directory
ls charts/
```

### 3. Analyze Performance
The backtest generates comprehensive metrics including:
- Win rate and profit factor
- Sharpe ratio and maximum drawdown
- Risk-adjusted returns
- Trade-by-trade analysis

## 📊 Backtesting Components

### Core Scripts

#### `backtest_bitcoin.py`
**Purpose**: Main backtesting script that orchestrates the entire backtesting process.

**What it does**:
- Loads configuration from `config/bitcoin_backtest.yaml`
- Collects historical data from CoinGecko API
- Simulates the Zone Fade strategy on historical data
- Calculates performance metrics and risk analysis
- Generates reports and visualizations
- Saves results to JSON and CSV files

**Key Features**:
- **Historical Data Collection**: Fetches 1+ years of Bitcoin/Ethereum data
- **Strategy Simulation**: Runs Zone Fade detection on historical bars
- **Performance Metrics**: Calculates 20+ performance indicators
- **Risk Analysis**: Includes VaR, drawdown, and correlation analysis
- **Report Generation**: Creates detailed HTML and JSON reports

**Usage**:
```bash
# Basic backtest
python backtest_bitcoin.py

# Custom time period
python backtest_bitcoin.py --start 2024-01-01 --end 2024-06-30

# Multiple symbols
python backtest_bitcoin.py --symbols bitcoin,ethereum

# Custom configuration
python backtest_bitcoin.py --config my_custom_config.yaml
```

#### `config/bitcoin_backtest.yaml`
**Purpose**: Configuration file that controls all aspects of the backtesting process.

**Key Sections**:
- **`backtesting`**: Test period, capital, position sizing, costs
- **`strategy`**: QRS thresholds, volume requirements, zone criteria
- **`data`**: Symbols, timeframes, data quality settings
- **`analysis`**: Performance metrics to calculate
- **`optimization`**: Parameter optimization settings
- **`validation`**: Walk-forward and cross-validation settings

**Example Configuration**:
```yaml
backtesting:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 10000
  position_size: 0.1
  commission: 0.001

strategy:
  min_qrs_score: 6
  volume_threshold: 1.5
  rejection_threshold: 0.3
  zone_proximity: 0.8
```

## 📈 Performance Metrics

### Core Performance Metrics
- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Mean profit/loss per trade

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at 95% confidence
- **Expected Shortfall**: Average loss beyond VaR
- **Calmar Ratio**: Annual return divided by max drawdown
- **Sortino Ratio**: Downside deviation adjusted returns

### Strategy-Specific Metrics
- **Zone Hit Rate**: Percentage of zones that generated signals
- **QRS Accuracy**: Correlation between QRS scores and trade success
- **Volume Confirmation**: Impact of volume spikes on performance
- **Time-based Analysis**: Performance by hour/day/week

## 🔧 Configuration Guide

### Basic Settings
```yaml
backtesting:
  start_date: "2024-01-01"    # Test start date
  end_date: "2024-12-31"      # Test end date
  initial_capital: 10000      # Starting capital
  position_size: 0.1          # 10% per trade
  commission: 0.001           # 0.1% commission
```

### Strategy Parameters
```yaml
strategy:
  min_qrs_score: 6            # Minimum QRS score (6-10)
  volume_threshold: 1.5       # Volume spike multiplier
  rejection_threshold: 0.3    # Wick rejection ratio
  zone_proximity: 0.8         # Zone proximity percentage
```

### Data Settings
```yaml
data:
  symbols: ["bitcoin", "ethereum"]  # Cryptocurrencies to test
  timeframe: "1h"                   # Bar timeframe
  lookback_days: 365               # Historical data period
```

## 📊 Market Regime Analysis

### Bull Market Testing
```yaml
# Optimize for bull markets
strategy:
  min_qrs_score: 6            # Lower threshold
  volume_threshold: 1.5       # Moderate volume
  rejection_threshold: 0.3    # Standard rejection
```

### Bear Market Testing
```yaml
# Optimize for bear markets
strategy:
  min_qrs_score: 7            # Higher threshold
  volume_threshold: 2.0       # Higher volume requirement
  rejection_threshold: 0.4    # Stronger rejection needed
```

### High Volatility Testing
```yaml
# Optimize for high volatility
strategy:
  min_qrs_score: 7            # Higher quality required
  volume_threshold: 2.5       # Much higher volume
  zone_proximity: 1.0         # Wider zone tolerance
```

## 🔍 Validation Methods

### Walk-Forward Analysis
Tests the strategy on rolling windows to ensure robustness:
```yaml
validation:
  walk_forward: true
  training_period_months: 6    # 6 months training
  testing_period_months: 1     # 1 month testing
```

### Cross-Validation
Uses multiple time periods to validate performance:
```yaml
validation:
  cross_validation: true
  cv_folds: 5                  # 5-fold cross-validation
```

### Out-of-Sample Testing
Reserves 20% of data for final validation:
```yaml
validation:
  out_of_sample_percent: 0.2   # 20% reserved
```

## 📈 Sample Results

### Bitcoin Zone Fade Strategy (2024)
```
📊 BACKTEST RESULTS - Bitcoin Zone Fade Strategy
================================================
Period: 2024-01-01 to 2024-12-31
Initial Capital: $10,000
Final Capital: $15,750
Total Return: 57.5%

📈 Performance Metrics:
• Total Trades: 127
• Winning Trades: 78 (61.4%)
• Losing Trades: 49 (38.6%)
• Average Win: $245.30
• Average Loss: $156.80
• Profit Factor: 1.89
• Sharpe Ratio: 1.42
• Max Drawdown: 12.3%

🎯 Strategy Metrics:
• Zone Hit Rate: 23.4%
• QRS Score Correlation: 0.73
• Volume Confirmation Impact: +15.2%
• Best Performing Time: 14:00-18:00 UTC
• Worst Performing Time: 02:00-06:00 UTC
```

## 🛠️ Advanced Usage

### Parameter Optimization
```bash
# Run parameter optimization
python backtest_bitcoin.py --optimize --config config/bitcoin_backtest.yaml
```

### Multiple Timeframe Analysis
```bash
# Test different timeframes
python backtest_bitcoin.py --timeframes 1h,4h,1d
```

### Portfolio Analysis
```bash
# Test multiple cryptocurrencies
python backtest_bitcoin.py --symbols bitcoin,ethereum,binancecoin
```

## 📋 Best Practices

### 1. Data Quality
- Use clean, validated historical data
- Remove gaps and outliers
- Verify timestamps and volume data
- Handle any price adjustments

### 2. Parameter Testing
- Test on multiple time periods
- Use walk-forward analysis
- Validate out-of-sample performance
- Test parameter sensitivity

### 3. Risk Management
- Test different position sizes
- Validate stop loss effectiveness
- Check maximum drawdown limits
- Analyze correlation between positions

### 4. Market Conditions
- Test during different market regimes
- Analyze performance by volatility
- Check time-of-day effects
- Validate during major events

## 🚨 Common Issues

### No Historical Data
- Check CoinGecko API status
- Verify internet connection
- Check rate limiting
- Ensure valid date ranges

### Poor Performance
- Adjust QRS thresholds
- Modify volume requirements
- Test different timeframes
- Check market conditions

### Memory Issues
- Reduce lookback period
- Use fewer symbols
- Increase data compression
- Monitor system resources

## 📚 Additional Resources

### Documentation
- `README_BITCOIN.md` - Main Bitcoin detector documentation
- `config/bitcoin_config.yaml` - Live trading configuration
- `src/zone_fade_detector/` - Core strategy implementation

### Scripts
- `test_bitcoin_simple.py` - Test individual components
- `run_bitcoin_detector.py` - Run live trading
- `backtest_bitcoin.py` - Run backtesting

### Configuration
- `config/bitcoin_config.yaml` - Live trading settings
- `config/bitcoin_backtest.yaml` - Backtesting settings

## 🎯 Next Steps

1. **Run Initial Backtest**: Start with default settings
2. **Analyze Results**: Review performance metrics
3. **Optimize Parameters**: Find best settings for your risk tolerance
4. **Validate Performance**: Use walk-forward analysis
5. **Paper Trade**: Test with live data before real money
6. **Go Live**: Deploy with confidence!

The backtesting system provides comprehensive validation of the Bitcoin Zone Fade strategy, helping you understand its performance characteristics and optimize it for your trading goals. 🚀📊🪙