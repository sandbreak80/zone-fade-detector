# Backtesting Guide

This guide explains how to run backtesting on the Zone Fade Detector system using historical 2024 data.

## 🎯 Overview

The Zone Fade Detector includes comprehensive backtesting capabilities that allow you to:
- Validate strategy performance on historical data
- Analyze entry point frequency and quality
- Track entry window duration for execution timing
- Export results for manual validation
- Test different parameter configurations

## 📊 2024 Backtesting Results

### Key Performance Metrics
- **Total Entry Points**: 160 high-quality setups detected
- **Entry Points per Day**: 0.6 (highly selective)
- **Average QRS Score**: 6.23/10 (above 7.0 threshold)
- **Entry Window Duration**: 16-29 minutes (sufficient execution time)
- **Success Rate**: 100% long entry windows (>15 minutes)

### Symbol Breakdown
| Symbol | Entry Points | Per Day | Zone Touch Rate | Rejection Rate | Volume Spike Rate |
|--------|--------------|---------|-----------------|----------------|-------------------|
| QQQ    | 49           | 0.19    | 1.79%           | 26.82%         | 16.67%            |
| SPY    | 33           | 0.13    | 1.06%           | 22.12%         | 19.66%            |
| IWM    | 78           | 0.31    | 2.39%           | 23.20%         | 24.91%            |

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- 2024 historical data downloaded (see Data Preparation section)
- Sufficient disk space for data and output files

### 1. Data Preparation

First, download the 2024 historical data:

```bash
# Download 2024 data for SPY, QQQ, IWM
docker-compose run zone-fade-detector python backtesting/download_2024_data.py
```

This will create the following data files:
- `data/2024/SPY_2024.pkl` (~196k bars)
- `data/2024/QQQ_2024.pkl` (~210k bars)
- `data/2024/IWM_2024.pkl` (~186k bars)

### 2. Run Backtesting

Choose the appropriate backtesting script based on your needs:

#### Efficient Validation (Recommended)
```bash
# Fast backtesting with last 50k bars per symbol
docker-compose run zone-fade-detector python backtesting/backtest_2024_efficient_validation.py
```

#### Full 2024 Dataset
```bash
# Complete 2024 backtesting (longer runtime)
docker-compose run zone-fade-detector python backtesting/backtest_2024_full_validation.py
```

#### December 2024 Debug
```bash
# Single month debug backtesting
docker-compose run zone-fade-detector python backtesting/backtest_2024_december_debug.py
```

## 📋 Available Backtesting Scripts

### 1. `backtest_2024_efficient_validation.py`
**Purpose**: Fast validation with sample data
- **Data**: Last 50,000 bars per symbol
- **Runtime**: ~5-10 minutes
- **Output**: CSV files with entry points and analysis
- **Use Case**: Quick validation and testing

### 2. `backtest_2024_full_validation.py`
**Purpose**: Complete 2024 analysis
- **Data**: Full 2024 dataset (~200k bars per symbol)
- **Runtime**: ~30-60 minutes
- **Output**: Comprehensive CSV files
- **Use Case**: Full year analysis and validation

### 3. `backtest_2024_december_debug.py`
**Purpose**: Single month debugging
- **Data**: December 2024 only
- **Runtime**: ~2-5 minutes
- **Output**: Detailed debug information
- **Use Case**: Parameter tuning and debugging

## 📊 Understanding the Results

### Entry Point Data
Each entry point includes:
- **entry_id**: Unique identifier
- **symbol**: Asset symbol (SPY, QQQ, IWM)
- **timestamp**: Entry point timestamp
- **price**: Entry price
- **zone_level**: Zone level that was touched
- **zone_type**: Type of zone (supply/demand)
- **qrs_score**: Quality rating score (0-10)
- **rejection_candle**: Whether it was a rejection candle
- **volume_spike**: Whether volume spike was detected
- **zone_strength**: Zone strength (0-1)
- **zone_quality**: Zone quality (0-3)

### Entry Window Analysis
- **window_duration_minutes**: How long the entry window lasted
- **window_bars**: Number of bars the window was valid
- **max_price_deviation**: Maximum price deviation during window
- **min_price_deviation**: Minimum price deviation during window
- **entry_window_ended**: Whether window ended before timeout

### Performance Metrics
- **zone_touch_rate**: Percentage of bars that touched zones
- **rejection_rate**: Percentage of zone touches that were rejections
- **volume_spike_rate**: Percentage of rejections with volume spikes
- **entry_point_rate**: Percentage of zone touches that became entry points
- **avg_qrs_score**: Average QRS score across all entry points

## 🔧 Configuration Options

### Strategy Parameters
The backtesting scripts use the following default parameters:
- **Rejection Candle Wick Ratio**: 30% (strict)
- **Volume Spike Threshold**: 1.8x average volume
- **QRS Threshold**: 7.0 (high quality)
- **Swing Structure**: 20 bars lookback, 0.1 min swing size
- **Zone Quality**: 2+ for valid zones

### Customization
To modify parameters, edit the backtesting scripts:
```python
# In the backtesting script
def is_rejection_candle(bar: OHLCVBar, zone: Zone) -> bool:
    # Modify wick ratio threshold
    if upper_wick / body_size > 0.3:  # Change 0.3 to desired ratio
        return True

def calculate_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar]) -> float:
    # Modify volume spike threshold
    if bar.volume > avg_volume * 1.8:  # Change 1.8 to desired multiplier
        score += 1.0
```

## 📁 Output Files

### CSV Files
All backtesting scripts generate CSV files in the `validation_output/` directory:

1. **`zone_fade_entry_points_2024_efficient.csv`**
   - Complete entry point data
   - Entry window analysis
   - Performance metrics

2. **`zone_fade_summary_2024.csv`**
   - Summary statistics per symbol
   - Overall performance metrics
   - Quality analysis

### Manual Validation
Use the CSV files to manually validate entry points:
1. Open the CSV file in Excel or similar tool
2. Sort by timestamp to see chronological order
3. Use timestamp to locate on your charts
4. Verify rejection candles and volume spikes
5. Check entry window duration

## 🎯 Manual Validation Process

### 1. Chart Analysis
- Open your charting platform
- Load the symbol and timeframe (1-minute)
- Navigate to the entry point timestamp
- Verify the rejection candle pattern
- Check volume spike confirmation

### 2. Entry Window Validation
- Note the entry window duration
- Check if price stayed within the zone
- Verify the entry opportunity timing
- Analyze price action during the window

### 3. Quality Assessment
- Evaluate the QRS score components
- Check zone strength and quality
- Verify rejection candle clarity
- Assess volume spike significance

## ⚠️ Important Notes

### Data Quality
- Historical data is from Alpaca/Polygon APIs
- 1-minute bar resolution
- Includes pre-market and after-hours data
- Data quality may vary by symbol and time period

### Limitations
- **Not Live Trading Ready**: System is optimized for backtesting
- **ETF Proxies**: Uses SPY/QQQ/IWM instead of futures
- **No Slippage**: Assumes perfect execution
- **No Commission**: Does not include trading costs

### Performance Considerations
- Full 2024 backtesting requires significant memory
- Use efficient validation for quick testing
- Monitor system resources during processing
- Consider running overnight for full dataset

## 🚀 Next Steps

### After Backtesting
1. **Validate Results**: Manually check entry points on charts
2. **Analyze Performance**: Review entry window durations
3. **Parameter Tuning**: Adjust parameters based on results
4. **Strategy Refinement**: Improve based on validation findings

### Production Preparation
1. **Futures Integration**: Implement ES/NQ/RTY futures data
2. **Live Trading**: Optimize for real-time execution
3. **Risk Management**: Add position sizing and risk controls
4. **Monitoring**: Implement real-time performance tracking

## 📞 Support

For questions or issues:
- Check the [Manual Validation Guide](MANUAL_VALIDATION_GUIDE.md)
- Review the [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- Create an issue on GitHub
- Check the configuration examples

## 📚 Related Documentation

- [Manual Validation Guide](MANUAL_VALIDATION_GUIDE.md)
- [Strategy Analysis](STRATEGY_ANALYSIS.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [2024 Results Summary](2024_RESULTS_SUMMARY.md)