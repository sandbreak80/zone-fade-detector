# Clean Zone Fade Entry Point Visualizer

A clean, professional visualization tool that generates TradingView/Yahoo Finance style charts for manual verification of entry point detections. This tool creates publication-quality stock charts with minimal clutter and maximum readability.

## 🎯 Overview

The Clean Entry Visualizer creates professional financial charts showing:
- **Clean candlestick charts** with minimal overlays
- **Professional color scheme** (teal/red for price, orange for VWAP, blue for highlights)
- **Properly scaled volume subplot** with subtle coloring
- **Clear entry markers** without visual clutter
- **Minimal annotations** showing only essential information
- **Publication-quality output** at 300 DPI

## ✅ Problems Fixed

### ❌ Previous Issues
- **Cluttered Overlay**: Excessive overlays obscuring price data
- **Poor Layout Balance**: Popup boxes overlapping price axis
- **Volume Scaling Issues**: Oversized, bright red volume bars
- **Unclear Axis Design**: Overcrowded ticks and inconsistent gridlines
- **Non-Standard Elements**: Zone windows and setup metrics boxes

### ✅ Clean Solutions
- **Minimal Overlays**: Only essential elements (VWAP, zone level, entry marker)
- **Balanced Layout**: Small info box in top-right corner, no overlaps
- **Proper Volume Scaling**: Smaller subplot with subtle gray/teal coloring
- **Clean Axis Design**: Hourly ticks, consistent gridlines, professional styling
- **Standard Elements**: Focus on candlesticks, volume, and key markers only

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Backtesting data files in `data/2024/`
- Entry points CSV in `results/manual_validation/entry_points/`

### Basic Usage

```bash
# Test mode (first 5 entries)
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --test

# Full mode (all 160 entries)
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --full

# Custom mode with parameters
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --limit 10 --hours-before 3 --hours-after 3
```

## 📊 Clean Chart Design

### Professional Color Scheme
- **Price Up**: `#26a69a` (Teal green)
- **Price Down**: `#ef5350` (Red)
- **VWAP**: `#ff9800` (Orange)
- **Entry Marker**: `#1976d2` (Blue)
- **Zone Level**: `#9c27b0` (Purple)
- **Volume**: `#90a4ae` (Gray, semi-transparent)

### Chart Layout
```
┌─────────────────────────────────────────────────────────┐
│  PRICE CHART (Top Panel - 4:1 ratio)                   │
│  ├── Clean candlesticks (teal/red)                     │
│  ├── VWAP line (orange)                               │
│  ├── Zone level (purple dashed)                       │
│  ├── Entry marker (blue star)                         │
│  └── Minimal info box (top-right)                     │
├─────────────────────────────────────────────────────────┤
│  VOLUME CHART (Bottom Panel - 1:1 ratio)               │
│  ├── Subtle volume bars (gray/teal)                   │
│  ├── Entry highlight (blue)                           │
│  └── Clean time axis                                  │
└─────────────────────────────────────────────────────────┘
```

### Key Features
- **Clean Candlesticks**: Professional OHLC bars with proper edges
- **Minimal Overlays**: Only VWAP and zone level lines
- **Balanced Layout**: 4:1 ratio for price:volume
- **Subtle Volume**: Semi-transparent bars with price-based coloring
- **Clear Markers**: Blue star for entry, clean vertical line
- **Essential Info**: Small info box with key metrics only

## ⚙️ Configuration Options

### Time Window Settings
- `--hours-before`: Hours before entry point (default: 2)
- `--hours-after`: Hours after entry point (default: 2)

### Chart Settings
- `--width`: Chart width in inches (default: 14)
- `--height`: Chart height in inches (default: 8)
- `--dpi`: Chart DPI (default: 300)
- `--no-grid`: Disable grid lines
- `--volume-alpha`: Volume bar transparency (default: 0.6)

### Filtering Options
- `--min-qrs`: Minimum QRS score filter
- `--symbols`: Symbols to include (SPY, QQQ, IWM)
- `--limit`: Limit number of entries to process

## 📁 Output Structure

```
outputs/
├── visuals_clean/                    # Full clean visualizations
│   ├── SPY_20241004_1230_entry_visual.png
│   ├── QQQ_20241007_1200_entry_visual.png
│   └── clean_entry_points_visual_report.html
├── visuals_clean_test/               # Test clean visualizations
└── visuals_clean_custom/             # Custom clean visualizations
```

### Filename Pattern
`{SYMBOL}_{YYYYMMDD}_{HHMM}_entry_visual.png`

## 🎨 Visual Elements

### Candlestick Styling
- **Body**: Filled rectangles with proper edges
- **Wicks**: Clean black lines with appropriate thickness
- **Colors**: Teal for up, red for down
- **Spacing**: Proper bar width (0.6) for clean appearance

### Volume Styling
- **Bars**: Semi-transparent with price-based coloring
- **Height**: Properly scaled to subplot
- **Entry Highlight**: Blue bar for entry point
- **Alpha**: 0.6 for subtle appearance

### Annotations
- **Info Box**: Small, top-right corner
- **Content**: Entry price, zone level, QRS score, zone type
- **Styling**: Clean white background with gray border
- **Font**: Small, readable size (8pt)

## 🔍 Manual Verification Process

1. **Load Entry Points**: Read from CSV with 160 detected entries
2. **Extract Data Window**: Get 4-hour window of OHLCV data
3. **Calculate VWAP**: Volume-weighted average price for the period
4. **Create Clean Chart**: Generate professional candlestick + volume chart
5. **Add Essential Markers**: Entry point, zone level, VWAP only
6. **Minimal Annotations**: Small info box with key metrics
7. **Save Output**: Export as high-quality PNG
8. **Generate Report**: Create clean HTML report

## 📈 Quality Indicators

### QRS Score Color Coding
- **High Quality (8+):** Teal - Excellent setup
- **Good Quality (6-7):** Orange - Good setup
- **Moderate Quality (<6):** Red - Moderate setup

### Essential Metrics Displayed
- Entry price and zone level
- QRS score and zone type
- Clean, readable format
- No visual clutter

## 🚀 Advanced Usage

### Custom Time Windows
```bash
# 3 hours before, 3 hours after
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --hours-before 3 --hours-after 3
```

### Filter by Quality
```bash
# Only high-quality setups (QRS >= 8)
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --min-qrs 8
```

### Symbol-Specific Analysis
```bash
# Only SPY and QQQ entries
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --symbols SPY QQQ
```

### No Grid Option
```bash
# Clean charts without grid lines
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --no-grid
```

## 📊 HTML Report

The clean visualizer generates an HTML report that:
- Combines all visualizations in a clean grid layout
- Shows summary statistics
- Provides professional presentation format
- Includes essential entry point information
- Uses clean, modern styling

## 🔧 Technical Details

### Data Requirements
- **OHLCV Data**: Pickle files with 15-minute bars
- **Entry Points**: CSV with detection results
- **VWAP Calculation**: Volume-weighted average price
- **Time Alignment**: Precise timestamp matching

### Performance
- **Processing Speed**: ~1-2 seconds per visualization
- **Memory Usage**: Optimized for clean output
- **Output Quality**: 300 DPI publication-ready
- **File Sizes**: ~80-100KB per PNG (cleaner, smaller files)

### Docker Integration
- **Container**: `zone-fade-detector-test`
- **Volume Mounts**: All necessary directories included
- **Dependencies**: All Python packages pre-installed
- **Environment**: Isolated and reproducible

## 🎯 Use Cases

1. **Manual Validation**: Verify entry point accuracy with clean charts
2. **Strategy Analysis**: Understand setup patterns without clutter
3. **Quality Assessment**: Evaluate QRS scoring with clear visuals
4. **Presentation**: Create professional reports for stakeholders
5. **Research**: Analyze market behavior around entries
6. **Documentation**: Generate clean visual evidence

## 📝 Example Commands

```bash
# Quick test with 5 entries
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --test

# Full analysis of all entries
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --full

# Custom analysis with 3-hour windows
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --hours-before 3 --hours-after 3 --limit 10

# High-quality setups only, no grid
docker compose run --rm zone-fade-detector-test python backtesting/run_clean_visualizations.py --custom --min-qrs 8 --symbols SPY QQQ --no-grid
```

## 🎉 Results

The clean visualizer successfully generates:
- **160 individual clean visualizations** (one per entry point)
- **Publication-quality charts** at 300 DPI
- **Professional appearance** similar to TradingView/Yahoo Finance
- **Comprehensive HTML report** with all visualizations
- **Structured filename pattern** for easy organization
- **Clean, readable format** perfect for manual verification

Each visualization clearly shows how price and VWAP behaved before and after the detected entry, enabling accurate manual verification of the Zone Fade strategy's entry point detection logic with a clean, professional appearance.

## 🔄 Comparison

### Before (Cluttered)
- Excessive overlays obscuring price data
- Large popup boxes overlapping axes
- Bright red volume bars dominating the chart
- Non-standard elements and visual noise
- Poor readability and unprofessional appearance

### After (Clean)
- Minimal overlays focusing on essential elements
- Small info box in corner, no overlaps
- Subtle volume bars with proper scaling
- Standard financial chart elements only
- Professional, publication-quality appearance

The clean visualizer transforms cluttered, hard-to-read charts into professional, TradingView-style visualizations that are perfect for manual verification and presentation.