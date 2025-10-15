# Zone Fade Entry Point Visualizer

A comprehensive visualization tool for manually verifying the accuracy of entry point detections by visual inspection. This tool generates publication-quality charts for each detected entry point, enabling thorough manual validation of the Zone Fade strategy.

## 🎯 Overview

The Entry Visualizer creates detailed charts showing:
- **4-hour time window** (2 hours before + 2 hours after entry)
- **Price action** with candlestick charts
- **VWAP overlay** for volume-weighted analysis
- **Volume analysis** with color-coded bars
- **Entry point highlighting** with clear annotations
- **Zone level visualization** for context
- **Setup metrics** and quality indicators

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Backtesting data files in `data/2024/`
- Entry points CSV in `results/manual_validation/entry_points/`

### Basic Usage

```bash
# Test mode (first 5 entries)
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --test

# Full mode (all 160 entries)
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --full

# Custom mode with parameters
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --limit 10 --hours-before 3 --hours-after 3
```

## 📊 Features

### Core Visualization Components

1. **Price Chart (Top Panel)**
   - Candlestick bars with green/red color coding
   - VWAP line in orange
   - Zone level line in purple (dashed)
   - Entry point marker (red star)
   - Entry window duration (yellow dotted line)
   - Detailed info panel with setup metrics

2. **Volume Chart (Bottom Panel)**
   - Volume bars with semi-transparent color coding
   - Entry point volume highlight (red)
   - Volume spike indicators
   - Color coding based on price direction

### Visual Design

- **Clean, minimalist design** with no clutter
- **Publication-quality output** at 300 DPI
- **Professional color scheme** optimized for analysis
- **Clear annotations** with setup details
- **Quality indicators** with QRS score color coding

## ⚙️ Configuration Options

### Time Window Settings
- `--hours-before`: Hours before entry point (default: 2)
- `--hours-after`: Hours after entry point (default: 2)

### Chart Settings
- `--width`: Chart width in inches (default: 16)
- `--height`: Chart height in inches (default: 10)
- `--dpi`: Chart DPI (default: 300)
- `--style`: Matplotlib style (default: seaborn-v0_8-whitegrid)
- `--no-grid`: Disable grid lines
- `--volume-alpha`: Volume bar transparency (default: 0.6)

### Filtering Options
- `--min-qrs`: Minimum QRS score filter
- `--symbols`: Symbols to include (SPY, QQQ, IWM)
- `--limit`: Limit number of entries to process

## 📁 Output Structure

```
outputs/
├── visuals/                    # Full visualizations
│   ├── SPY_20241004_1230_entry_visual.png
│   ├── QQQ_20241007_1200_entry_visual.png
│   ├── IWM_20241008_1944_entry_visual.png
│   └── entry_points_visual_report.html
├── visuals_test/               # Test visualizations
└── visuals_custom/             # Custom visualizations
```

### Filename Pattern
`{SYMBOL}_{YYYYMMDD}_{HHMM}_entry_visual.png`

## 🔍 Manual Verification Process

1. **Load Entry Points**: Read from CSV with 160 detected entries
2. **Extract Data Window**: Get 4-hour window of OHLCV data
3. **Calculate VWAP**: Volume-weighted average price for the period
4. **Create Visualization**: Generate candlestick + volume chart
5. **Add Annotations**: Include setup metrics and quality indicators
6. **Save Output**: Export as high-quality PNG
7. **Generate Report**: Create HTML report combining all visualizations

## 📈 Quality Indicators

### QRS Score Color Coding
- **High Quality (8+):** Green - Excellent setup
- **Good Quality (6-7):** Orange - Good setup
- **Moderate Quality (<6):** Red - Moderate setup

### Setup Metrics Displayed
- Entry ID and timestamp
- QRS score and quality rating
- Zone type and level
- Entry price and distance from zone
- Window duration and rejection candle status
- Volume spike confirmation
- Zone strength and quality
- Price range analysis
- Volume analysis

## 🎨 Visual Elements

### Color Scheme
- **Price Up:** `#2e7d32` (Green)
- **Price Down:** `#c62828` (Red)
- **VWAP:** `#ff6f00` (Orange)
- **Entry Line:** `#d32f2f` (Red)
- **Zone Level:** `#7b1fa2` (Purple)
- **Volume Bars:** Semi-transparent with price-based coloring

### Chart Layout
```
┌─────────────────────────────────────────────────────────┐
│  PRICE CHART (Top Panel)                               │
│  ├── Candlestick bars (green/red)                      │
│  ├── VWAP line (orange)                               │
│  ├── Zone level (purple dashed)                       │
│  ├── Entry point (red star)                           │
│  ├── Entry window (yellow dotted line)                │
│  └── Info panel (setup details)                       │
├─────────────────────────────────────────────────────────┤
│  VOLUME CHART (Bottom Panel)                           │
│  ├── Volume bars (semi-transparent)                   │
│  ├── Color coding (green/red)                         │
│  └── Entry point highlight (red)                      │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Advanced Usage

### Custom Time Windows
```bash
# 3 hours before, 3 hours after
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --hours-before 3 --hours-after 3
```

### Filter by Quality
```bash
# Only high-quality setups (QRS >= 8)
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --min-qrs 8
```

### Symbol-Specific Analysis
```bash
# Only SPY and QQQ entries
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --symbols SPY QQQ
```

### Limited Processing
```bash
# Process only first 20 entries
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --limit 20
```

## 📊 HTML Report

The visualizer automatically generates an HTML report that:
- Combines all visualizations in a grid layout
- Shows summary statistics
- Provides interactive navigation
- Includes detailed entry point information
- Offers professional presentation format

## 🔧 Technical Details

### Data Requirements
- **OHLCV Data**: Pickle files with 15-minute bars
- **Entry Points**: CSV with detection results
- **VWAP Calculation**: Volume-weighted average price
- **Time Alignment**: Precise timestamp matching

### Performance
- **Processing Speed**: ~2-3 seconds per visualization
- **Memory Usage**: Optimized for Docker container limits
- **Output Quality**: 300 DPI publication-ready
- **File Sizes**: ~200-500KB per PNG

### Docker Integration
- **Container**: `zone-fade-detector-test`
- **Volume Mounts**: All necessary directories included
- **Dependencies**: All Python packages pre-installed
- **Environment**: Isolated and reproducible

## 🎯 Use Cases

1. **Manual Validation**: Verify entry point accuracy
2. **Strategy Analysis**: Understand setup patterns
3. **Quality Assessment**: Evaluate QRS scoring
4. **Presentation**: Create professional reports
5. **Research**: Analyze market behavior around entries
6. **Documentation**: Generate visual evidence

## 📝 Example Commands

```bash
# Quick test with 5 entries
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --test

# Full analysis of all entries
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --full

# Custom analysis with 3-hour windows
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --hours-before 3 --hours-after 3 --limit 10

# High-quality setups only
docker compose run --rm zone-fade-detector-test python backtesting/run_visualizations.py --custom --min-qrs 8 --symbols SPY QQQ
```

## 🎉 Results

The visualizer successfully generates:
- **160 individual visualizations** (one per entry point)
- **Publication-quality charts** at 300 DPI
- **Comprehensive HTML report** with all visualizations
- **Structured filename pattern** for easy organization
- **Professional presentation** ready for analysis

Each visualization clearly shows how price and VWAP behaved before and after the detected entry, enabling accurate manual verification of the Zone Fade strategy's entry point detection logic.