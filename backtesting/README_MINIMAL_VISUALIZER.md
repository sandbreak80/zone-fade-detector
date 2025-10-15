# Minimal Entry Point Visualizer

A clean, minimal visualization tool using **matplotlib + seaborn only** that renders ONLY the essential elements for manual verification of entry point detections.

## 🎯 Overview

This visualizer creates clean, minimal charts with **ONLY**:
- **OHLC candlesticks**
- **VWAP line**
- **Volume bars in second subplot**

**NO overlays, zones, annotations, or other shapes.**

## ✅ What It Renders

### Price Chart (Top Panel)
- **OHLC candlesticks** with teal/red coloring
- **VWAP line** in orange
- **Clean grid** and time axis

### Volume Chart (Bottom Panel)
- **Volume bars** with price-based coloring
- **Clean scaling** and time axis

## ❌ What It Does NOT Render

- ❌ No translucent rectangles
- ❌ No overlays or spans
- ❌ No filled areas
- ❌ No zones or zone levels
- ❌ No annotations or text boxes
- ❌ No entry point markers
- ❌ No legends
- ❌ No extra shapes

## 📊 Data Contract

### Input Requirements
```python
# pandas DataFrame with columns:
df = pd.DataFrame({
    'timestamp': pd.DatetimeIndex,  # tz-aware or naive
    'open': float,                  # Opening price
    'high': float,                  # High price
    'low': float,                   # Low price
    'close': float,                 # Close price
    'volume': int/float,            # Volume
    'vwap': float                   # Optional: precomputed VWAP
})
```

### VWAP Calculation
If `vwap` column is missing, automatically computes rolling session VWAP:
```python
typical_price = (high + low + close) / 3
cum_volume = volume.cumsum()
cum_pv = (typical_price * volume).cumsum()
vwap = cum_pv / cum_volume
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Backtesting data files in `data/2024/`
- Entry points CSV in `results/manual_validation/entry_points/`

### Basic Usage

```bash
# Test mode (first 5 entries)
docker compose run --rm zone-fade-detector-test python backtesting/minimal_entry_visualizer.py

# The visualizer automatically:
# - Loads your backtesting data
# - Computes VWAP if not provided
# - Creates minimal charts with ONLY OHLC, VWAP, Volume
# - Saves as high-quality PNG files
```

## 🎨 Visual Design

### Color Scheme
- **Price Up**: `#26a69a` (Teal green)
- **Price Down**: `#ef5350` (Red)
- **VWAP**: `#ff9800` (Orange)
- **Volume**: Price-based coloring (teal/red)

### Chart Layout
```
┌─────────────────────────────────────────────────────────┐
│  PRICE CHART (Top Panel - 4:1 ratio)                   │
│  ├── OHLC candlesticks (teal/red)                      │
│  └── VWAP line (orange)                               │
├─────────────────────────────────────────────────────────┤
│  VOLUME CHART (Bottom Panel - 1:1 ratio)               │
│  └── Volume bars (price-based coloring)               │
└─────────────────────────────────────────────────────────┘
```

### Styling Features
- **Clean candlesticks**: Professional OHLC bars with proper edges
- **Minimal design**: Only essential elements
- **Balanced layout**: 4:1 ratio for price:volume
- **Price-based volume**: Teal for up days, red for down days
- **No clutter**: No overlays, annotations, or extra shapes

## ⚙️ Configuration Options

### Time Window Settings
```python
visualizer = MinimalEntryVisualizer(
    hours_before=2,    # Hours before entry to show
    hours_after=2,     # Hours after entry to show
    figsize=(14, 8),   # Chart size (width, height)
    dpi=300           # Output DPI
)
```

### Chart Customization
- **Time window**: Configurable hours before/after entry
- **Chart size**: Width and height in inches
- **DPI**: Output resolution (300 for publication quality)
- **Colors**: Predefined for consistency

## 📁 Output Structure

```
outputs/
└── visuals_minimal/
    ├── SPY_20241004_1230_minimal.png
    ├── QQQ_20241007_1200_minimal.png
    ├── IWM_20241008_1944_minimal.png
    └── ...
```

### Filename Pattern
`{SYMBOL}_{YYYYMMDD}_{HHMM}_minimal.png`

## 🔍 Manual Verification Process

1. **Load Entry Points**: Read from CSV with 160 detected entries
2. **Extract Data Window**: Get 4-hour window of OHLCV data
3. **Compute VWAP**: Calculate rolling session VWAP
4. **Create Minimal Chart**: Generate chart with ONLY OHLC, VWAP, Volume
5. **Save Output**: Export as high-quality PNG
6. **Review**: Manual verification of entry detection accuracy

## 📈 Chart Components

### Price Chart (Top Panel)
- **OHLC Candlesticks**: Professional bars with teal/red coloring
- **VWAP Line**: Orange line showing volume-weighted average price
- **Clean Grid**: Subtle gridlines for readability
- **Time Axis**: Hourly ticks with proper formatting

### Volume Chart (Bottom Panel)
- **Volume Bars**: Price-based coloring (teal for up, red for down)
- **Clean Scaling**: Properly sized for readability
- **Time Axis**: Aligned with price chart

### Minimal Design
- **No overlays** - Clean price action visibility
- **No annotations** - Focus on data only
- **No extra shapes** - Minimal visual noise
- **No clutter** - Professional appearance

## 🔧 Technical Implementation

### Dependencies
- **matplotlib**: Chart creation and styling
- **seaborn**: Clean styling and color palettes
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Key Functions
```python
# Main visualization function
create_entry_visualization(entry_point, bars, symbol, output_dir)

# VWAP calculation
compute_session_vwap(df)

# Chart components
_plot_ohlc_candles(ax, df)
_plot_vwap_line(ax, df)
_plot_volume_bars(ax, df)
```

### Data Processing
1. **Load OHLCV data** from pickle files
2. **Filter time window** around entry point
3. **Compute VWAP** if not provided
4. **Create minimal chart** with matplotlib + seaborn
5. **Save PNG** with high quality

## 🎯 Use Cases

1. **Manual Validation**: Verify entry point accuracy with clean charts
2. **Strategy Analysis**: Understand price action patterns
3. **Quality Assessment**: Evaluate entry detection logic
4. **Presentation**: Create professional reports
5. **Research**: Analyze market behavior around entries
6. **Documentation**: Generate visual evidence

## 📊 Performance

### Processing Speed
- **~1-2 seconds per visualization**
- **Optimized for Docker container limits**
- **Efficient data filtering and processing**

### Output Quality
- **300 DPI publication-ready**
- **Clean, minimal appearance**
- **Professional financial charts**
- **File sizes: ~290KB per PNG**

### Memory Usage
- **Optimized for container limits**
- **Efficient data structures**
- **Minimal memory footprint**

## 🎉 Results

The minimal visualizer successfully generates:
- **160 individual visualizations** (one per entry point)
- **Publication-quality charts** at 300 DPI
- **Clean, minimal design** with no clutter
- **Structured filename pattern** for easy organization
- **Perfect for manual verification** of entry point detection logic

## 🔄 Comparison with Other Visualizers

### Minimal Visualizer (This)
- ✅ **ONLY OHLC, VWAP, Volume**
- ✅ **No overlays or annotations**
- ✅ **Clean, minimal design**
- ✅ **Professional appearance**
- ✅ **Focus on data only**

### Previous Visualizers
- ❌ **Cluttered overlays**
- ❌ **Excessive annotations**
- ❌ **Non-standard elements**
- ❌ **Visual noise**

## 📝 Example Usage

```python
from minimal_entry_visualizer import MinimalEntryVisualizer

# Initialize visualizer
visualizer = MinimalEntryVisualizer(
    hours_before=2,
    hours_after=2,
    figsize=(14, 8),
    dpi=300
)

# Load data
symbols_data = visualizer.load_backtest_data(Path("data/2024"))
entry_points_df = visualizer.load_entry_points(Path("results/.../entry_points.csv"))

# Generate visualizations
results = visualizer.generate_entry_visuals(
    entry_points_df, 
    symbols_data, 
    Path("outputs/visuals_minimal")
)
```

## 🎨 Visual Examples

Each generated chart shows:
- **Clean OHLC candlesticks** with professional styling
- **VWAP line** in orange for volume context
- **Volume bars** with price-based coloring
- **No overlays, zones, annotations, or other shapes**

The result is a clean, minimal chart that focuses purely on the data, perfect for manual verification of your Zone Fade strategy's entry point detections.

## 🚀 Ready to Use

The minimal visualizer is ready for production use and generates clean, minimal charts that are perfect for manual verification of entry point detection accuracy. Each visualization shows only the essential elements: OHLC candlesticks, VWAP line, and volume bars - nothing more, nothing less.