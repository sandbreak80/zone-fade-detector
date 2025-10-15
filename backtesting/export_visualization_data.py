#!/usr/bin/env python3
"""
Zone Fade Entry Points Data Export for Visualization

This script exports detailed data around each entry point for visualization in external tools:
- Price data (OHLCV) for 60 minutes before + 2 hours after entry
- VWAP calculations
- Zone levels and entry markers
- Setup metrics and annotations
- CSV files ready for import into TradingView, Excel, or other charting tools
"""

import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


def load_2024_data():
    """Load the 2024 data for visualization."""
    print("📊 Loading 2024 Data...")
    
    data_dir = Path("data/2024")
    symbols_data = {}
    
    for symbol in ["SPY", "QQQ", "IWM"]:
        file_path = data_dir / f"{symbol}_2024.pkl"
        if file_path.exists():
            print(f"   Loading {symbol} data...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                symbols_data[symbol] = data
                print(f"     ✅ {symbol}: {len(data)} bars")
        else:
            print(f"     ❌ {symbol}: File not found")
    
    return symbols_data


def load_entry_points():
    """Load the entry points CSV data."""
    print("📋 Loading Entry Points...")
    
    csv_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    
    if not csv_file.exists():
        print(f"❌ Entry points file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"   ✅ Loaded {len(df)} entry points")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def calculate_vwap(bars: List[OHLCVBar], start_idx: int, end_idx: int) -> List[float]:
    """Calculate VWAP for a range of bars."""
    if start_idx < 0 or end_idx >= len(bars) or start_idx >= end_idx:
        return []
    
    vwap_values = []
    cumulative_volume = 0
    cumulative_pv = 0
    
    for i in range(start_idx, end_idx + 1):
        bar = bars[i]
        typical_price = (bar.high + bar.low + bar.close) / 3
        cumulative_pv += typical_price * bar.volume
        cumulative_volume += bar.volume
        
        if cumulative_volume > 0:
            vwap = cumulative_pv / cumulative_volume
        else:
            vwap = typical_price
        
        vwap_values.append(vwap)
    
    return vwap_values


def calculate_volume_metrics(bars: List[OHLCVBar], start_idx: int, end_idx: int) -> List[Dict]:
    """Calculate volume metrics for visualization."""
    if start_idx < 0 or end_idx >= len(bars) or start_idx >= end_idx:
        return []
    
    volume_metrics = []
    
    for i in range(start_idx, end_idx + 1):
        bar = bars[i]
        
        # Calculate volume ratios
        recent_volume = [b.volume for b in bars[max(0, i-19):i+1]]  # Last 20 bars
        avg_volume = np.mean(recent_volume) if recent_volume else bar.volume
        volume_ratio = bar.volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike detection
        is_volume_spike = volume_ratio > 1.8
        
        volume_metrics.append({
            'volume_ratio': volume_ratio,
            'is_volume_spike': is_volume_spike,
            'avg_volume': avg_volume
        })
    
    return volume_metrics


def export_entry_point_data(
    entry_point: pd.Series,
    bars: List[OHLCVBar],
    symbol: str,
    output_dir: Path
) -> bool:
    """Export detailed data for a single entry point."""
    
    # Find the entry point in the bars data
    entry_timestamp = entry_point['timestamp']
    entry_price = entry_point['price']
    zone_level = entry_point['zone_level']
    qrs_score = entry_point['qrs_score']
    window_duration = entry_point['window_duration_minutes']
    
    # Find the bar index for this entry point
    entry_idx = None
    for i, bar in enumerate(bars):
        if abs((bar.timestamp - entry_timestamp).total_seconds()) < 60:  # Within 1 minute
            entry_idx = i
            break
    
    if entry_idx is None:
        print(f"   ❌ Could not find entry point {entry_point['entry_id']} in bars data")
        return False
    
    # Define time window: 60 minutes before + 2 hours after
    bars_before = 60  # 60 minutes before
    bars_after = 120  # 2 hours after
    
    start_idx = max(0, entry_idx - bars_before)
    end_idx = min(len(bars) - 1, entry_idx + bars_after)
    
    # Extract data for visualization
    vis_bars = bars[start_idx:end_idx + 1]
    
    # Calculate VWAP
    vwap_values = calculate_vwap(bars, start_idx, end_idx)
    
    # Calculate volume metrics
    volume_metrics = calculate_volume_metrics(bars, start_idx, end_idx)
    
    # Find entry point index in visualization data
    vis_entry_idx = entry_idx - start_idx
    
    # Create comprehensive data export
    export_data = []
    
    for i, (bar, vwap, vol_metrics) in enumerate(zip(vis_bars, vwap_values, volume_metrics)):
        # Calculate relative time from entry point
        time_from_entry = i - vis_entry_idx
        
        # Determine if this is the entry bar
        is_entry_bar = (i == vis_entry_idx)
        
        # Calculate distance from zone level
        distance_from_zone = abs(bar.close - zone_level)
        
        # Determine if price is above or below zone
        above_zone = bar.close > zone_level
        
        export_data.append({
            'timestamp': bar.timestamp,
            'time_from_entry_minutes': time_from_entry,
            'is_entry_bar': is_entry_bar,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'vwap': vwap,
            'zone_level': zone_level,
            'distance_from_zone': distance_from_zone,
            'above_zone': above_zone,
            'volume_ratio': vol_metrics['volume_ratio'],
            'is_volume_spike': vol_metrics['is_volume_spike'],
            'avg_volume': vol_metrics['avg_volume'],
            'typical_price': (bar.high + bar.low + bar.close) / 3,
            'body_size': abs(bar.close - bar.open),
            'upper_wick': bar.high - max(bar.open, bar.close),
            'lower_wick': min(bar.open, bar.close) - bar.low,
            'wick_ratio': (bar.high - min(bar.open, bar.close)) / abs(bar.close - bar.open) if abs(bar.close - bar.open) > 0 else 0
        })
    
    # Create DataFrame
    df = pd.DataFrame(export_data)
    
    # Add entry point metadata
    entry_metadata = {
        'entry_id': entry_point['entry_id'],
        'symbol': symbol,
        'entry_timestamp': entry_timestamp,
        'entry_price': entry_price,
        'zone_level': zone_level,
        'zone_type': entry_point['zone_type'],
        'qrs_score': qrs_score,
        'window_duration_minutes': window_duration,
        'rejection_candle': entry_point['rejection_candle'],
        'volume_spike': entry_point.get('volume_spike', False),
        'zone_strength': entry_point['zone_strength'],
        'zone_quality': entry_point['zone_quality'],
        'max_price_deviation': entry_point.get('max_price_deviation', 0),
        'min_price_deviation': entry_point.get('min_price_deviation', 0),
        'entry_window_ended': entry_point.get('entry_window_ended', True)
    }
    
    # Save detailed CSV
    csv_file = output_dir / f"{symbol}_{entry_point['entry_id']}_detailed_data.csv"
    df.to_csv(csv_file, index=False)
    
    # Save entry metadata
    metadata_file = output_dir / f"{symbol}_{entry_point['entry_id']}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(entry_metadata, f, indent=2, default=str)
    
    # Create TradingView import format
    tv_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    tv_data['timestamp'] = tv_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    tv_file = output_dir / f"{symbol}_{entry_point['entry_id']}_tradingview.csv"
    tv_data.to_csv(tv_file, index=False)
    
    # Create summary for this entry point
    summary = {
        'entry_id': entry_point['entry_id'],
        'symbol': symbol,
        'entry_timestamp': entry_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'entry_price': entry_price,
        'zone_level': zone_level,
        'qrs_score': qrs_score,
        'window_duration_minutes': window_duration,
        'total_bars': len(df),
        'bars_before_entry': vis_entry_idx,
        'bars_after_entry': len(df) - vis_entry_idx - 1,
        'price_range': {
            'min_price': df['low'].min(),
            'max_price': df['high'].max(),
            'price_range': df['high'].max() - df['low'].min()
        },
        'volume_stats': {
            'avg_volume': df['volume'].mean(),
            'max_volume': df['volume'].max(),
            'volume_spikes': df['is_volume_spike'].sum()
        },
        'vwap_stats': {
            'entry_vwap': vwap_values[vis_entry_idx] if vis_entry_idx < len(vwap_values) else None,
            'vwap_at_entry': df.iloc[vis_entry_idx]['vwap'] if vis_entry_idx < len(df) else None
        }
    }
    
    summary_file = output_dir / f"{symbol}_{entry_point['entry_id']}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return True


def create_visualization_guide(output_dir: Path):
    """Create a guide for using the exported data."""
    guide_content = """# Zone Fade Entry Points Visualization Guide

## Overview
This directory contains detailed data exports for each Zone Fade entry point, ready for visualization in external charting tools.

## File Types

### 1. Detailed Data Files (`{symbol}_{entry_id}_detailed_data.csv`)
Complete data for each entry point including:
- **Price Data**: OHLCV for 60 minutes before + 2 hours after entry
- **VWAP**: Volume-weighted average price
- **Zone Information**: Zone level and distance calculations
- **Volume Metrics**: Volume ratios and spike detection
- **Wick Analysis**: Upper/lower wick calculations
- **Time References**: Minutes from entry point

### 2. TradingView Format (`{symbol}_{entry_id}_tradingview.csv`)
Simplified format for TradingView import:
- Standard OHLCV format
- Timestamp in TradingView-compatible format
- Ready for direct import

### 3. Metadata Files (`{symbol}_{entry_id}_metadata.json`)
Entry point details and setup information:
- QRS score and quality metrics
- Zone strength and type
- Entry window duration
- Rejection candle and volume spike flags

### 4. Summary Files (`{symbol}_{entry_id}_summary.json`)
Statistical summary for each entry point:
- Price range and volume statistics
- VWAP calculations
- Time window analysis

## How to Use

### For TradingView:
1. Download the `{symbol}_{entry_id}_tradingview.csv` file
2. Import into TradingView as a custom dataset
3. Add VWAP indicator
4. Draw horizontal line at zone level
5. Mark entry point with annotation

### For Excel/Google Sheets:
1. Open the `{symbol}_{entry_id}_detailed_data.csv` file
2. Create candlestick charts using OHLC data
3. Add VWAP line using the vwap column
4. Use conditional formatting for volume spikes
5. Filter by `is_entry_bar` to highlight entry point

### For Python/Matplotlib:
1. Load the detailed CSV file
2. Use the provided data for custom visualizations
3. Access all calculated metrics and indicators

## Key Columns Explained

- **time_from_entry_minutes**: Minutes relative to entry point (negative = before, positive = after)
- **is_entry_bar**: True for the actual entry bar
- **distance_from_zone**: Absolute distance from zone level
- **above_zone**: True if price is above zone level
- **volume_ratio**: Current volume / average volume (20 bars)
- **is_volume_spike**: True if volume > 1.8x average
- **wick_ratio**: Total wick size / body size
- **typical_price**: (High + Low + Close) / 3

## Visualization Tips

1. **Highlight Entry Point**: Use `is_entry_bar` column to mark the entry
2. **Show Zone Level**: Draw horizontal line at `zone_level`
3. **Volume Spikes**: Use `is_volume_spike` for conditional formatting
4. **VWAP**: Plot `vwap` column as a line
5. **Time Window**: Use `time_from_entry_minutes` for x-axis

## Entry Point Quality Assessment

Use the metadata files to assess entry quality:
- **QRS Score**: Higher is better (7+ is good)
- **Zone Strength**: 0-1 scale, higher is better
- **Window Duration**: Longer is better for execution
- **Rejection Candle**: Should be True for valid setups
- **Volume Spike**: Confirms rejection with volume

## Example Analysis

1. **Load Data**: Import detailed CSV into your preferred tool
2. **Create Chart**: OHLC candlestick chart with volume
3. **Add Indicators**: VWAP line, zone level line
4. **Mark Entry**: Highlight the entry bar
5. **Analyze**: Check rejection pattern, volume spike, zone quality
6. **Validate**: Compare with metadata for quality assessment

## Files Generated

Total entry points processed: {total_entries}
- QQQ: {qqq_count} entry points
- SPY: {spy_count} entry points  
- IWM: {iwm_count} entry points

Each entry point has 4 files:
- Detailed data CSV
- TradingView format CSV
- Metadata JSON
- Summary JSON

Ready for comprehensive visual analysis!
"""
    
    with open(output_dir / "VISUALIZATION_GUIDE.md", 'w') as f:
        f.write(guide_content)


def main():
    """Main export function."""
    print("📊 Zone Fade Entry Points Data Export Tool")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/visualization_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    symbols_data = load_2024_data()
    entry_points_df = load_entry_points()
    
    if not symbols_data or entry_points_df is None:
        print("❌ Failed to load required data")
        return
    
    print(f"\n🎯 Processing {len(entry_points_df)} entry points...")
    
    # Process each entry point
    successful_exports = 0
    failed_exports = 0
    symbol_counts = {'QQQ': 0, 'SPY': 0, 'IWM': 0}
    
    for idx, entry_point in entry_points_df.iterrows():
        symbol = entry_point['symbol']
        entry_id = entry_point['entry_id']
        
        print(f"   📊 Exporting data for {entry_id} ({symbol})...")
        
        if symbol in symbols_data:
            success = export_entry_point_data(
                entry_point, 
                symbols_data[symbol], 
                symbol, 
                output_dir
            )
            
            if success:
                successful_exports += 1
                symbol_counts[symbol] += 1
                print(f"     ✅ Exported data for {entry_id}")
            else:
                failed_exports += 1
                print(f"     ❌ Failed to export data for {entry_id}")
        else:
            print(f"     ❌ No data available for {symbol}")
            failed_exports += 1
    
    # Create visualization guide
    create_visualization_guide(output_dir)
    
    # Print results
    print(f"\n📊 Export Results:")
    print(f"   ✅ Successful: {successful_exports}")
    print(f"   ❌ Failed: {failed_exports}")
    print(f"   📁 Output Directory: {output_dir.absolute()}")
    print(f"\n📈 Per Symbol:")
    for symbol, count in symbol_counts.items():
        print(f"   {symbol}: {count} entry points")
    
    print(f"\n🎉 Data export complete! Check {output_dir.absolute()}")
    print(f"\n📋 Next Steps:")
    print(f"   1. Download CSV files from {output_dir}")
    print(f"   2. Import into TradingView, Excel, or your preferred tool")
    print(f"   3. Follow the VISUALIZATION_GUIDE.md for instructions")
    print(f"   4. Analyze entry points visually to validate setups")


if __name__ == "__main__":
    main()