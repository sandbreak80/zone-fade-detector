#!/usr/bin/env python3
"""
Zone Fade Entry Points Visualization Tool

This script generates comprehensive visualizations for each entry point showing:
- Price action (OHLC candlesticks)
- Volume bars
- VWAP line
- Zone levels
- Entry point markers
- Setup metrics and annotations

Timeframe: 60 minutes before entry + 2 hours after entry
"""

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

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


def create_entry_visualization(
    entry_point: pd.Series,
    bars: List[OHLCVBar],
    symbol: str,
    output_dir: Path
) -> bool:
    """Create a comprehensive visualization for a single entry point."""
    
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
    timestamps = [bar.timestamp for bar in vis_bars]
    opens = [bar.open for bar in vis_bars]
    highs = [bar.high for bar in vis_bars]
    lows = [bar.low for bar in vis_bars]
    closes = [bar.close for bar in vis_bars]
    volumes = [bar.volume for bar in vis_bars]
    
    # Calculate VWAP
    vwap_values = calculate_vwap(bars, start_idx, end_idx)
    
    # Find entry point index in visualization data
    vis_entry_idx = entry_idx - start_idx
    
    # Create the figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    
    # Configure the plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # === PRICE CHART (Top Panel) ===
    
    # Plot candlesticks
    for i, (timestamp, open_price, high, low, close) in enumerate(zip(timestamps, opens, highs, lows, closes)):
        color = 'green' if close >= open_price else 'red'
        alpha = 0.7
        
        # Body
        body_height = abs(close - open_price)
        body_bottom = min(open_price, close)
        ax1.bar(timestamp, body_height, bottom=body_bottom, width=0.8, 
                color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        
        # Wicks
        ax1.plot([timestamp, timestamp], [low, high], color='black', linewidth=1)
        ax1.plot([timestamp, timestamp], [low, min(open_price, close)], color=color, linewidth=2)
        ax1.plot([timestamp, timestamp], [max(open_price, close), high], color=color, linewidth=2)
    
    # Plot VWAP
    if vwap_values:
        ax1.plot(timestamps, vwap_values, color='purple', linewidth=2, 
                label='VWAP', alpha=0.8)
    
    # Plot zone level
    ax1.axhline(y=zone_level, color='orange', linestyle='--', linewidth=2, 
               label=f'Zone Level ({zone_level:.2f})', alpha=0.8)
    
    # Highlight entry point
    entry_timestamp_vis = timestamps[vis_entry_idx]
    ax1.axvline(x=entry_timestamp_vis, color='red', linestyle='-', linewidth=2, 
               alpha=0.8, label='Entry Point')
    
    # Mark entry point with a star
    ax1.scatter(entry_timestamp_vis, entry_price, color='red', s=200, 
               marker='*', edgecolor='black', linewidth=2, zorder=10)
    
    # Add entry window duration
    window_end_time = entry_timestamp_vis + timedelta(minutes=window_duration)
    ax1.axvspan(entry_timestamp_vis, window_end_time, alpha=0.2, color='yellow', 
               label=f'Entry Window ({window_duration} min)')
    
    # Configure price chart
    ax1.set_title(f'{symbol} - Entry Point {entry_point["entry_id"]} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # === VOLUME CHART (Bottom Panel) ===
    
    # Plot volume bars
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(closes, opens)]
    ax2.bar(timestamps, volumes, color=colors, alpha=0.7, width=0.8)
    
    # Highlight entry point volume
    ax2.bar(timestamps[vis_entry_idx], volumes[vis_entry_idx], 
           color='red', alpha=1.0, width=0.8, edgecolor='black', linewidth=2)
    
    # Add volume spike indicator if applicable
    if entry_point.get('volume_spike', False):
        avg_volume = np.mean(volumes)
        spike_threshold = avg_volume * 1.8
        ax2.axhline(y=spike_threshold, color='orange', linestyle='--', 
                   alpha=0.8, label='Volume Spike Threshold')
    
    # Configure volume chart
    ax2.set_title('Volume', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add annotations
    annotation_text = f"""
Entry Details:
• Symbol: {symbol}
• Entry ID: {entry_point['entry_id']}
• QRS Score: {qrs_score:.1f}/10
• Zone Type: {entry_point['zone_type']}
• Zone Level: {zone_level:.2f}
• Entry Price: {entry_price:.2f}
• Window Duration: {window_duration} min
• Rejection Candle: {entry_point['rejection_candle']}
• Volume Spike: {entry_point.get('volume_spike', False)}
• Zone Strength: {entry_point['zone_strength']:.2f}
• Zone Quality: {entry_point['zone_quality']}
    """
    
    ax1.text(0.02, 0.98, annotation_text, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f"{symbol}_{entry_point['entry_id']}_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_summary_visualization(entry_points_df: pd.DataFrame, output_dir: Path):
    """Create summary visualizations across all entry points."""
    print("📊 Creating Summary Visualizations...")
    
    # QRS Score Distribution
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(entry_points_df['qrs_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('QRS Score Distribution', fontweight='bold')
    plt.xlabel('QRS Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Entry Points by Symbol
    plt.subplot(2, 2, 2)
    symbol_counts = entry_points_df['symbol'].value_counts()
    plt.pie(symbol_counts.values, labels=symbol_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Entry Points by Symbol', fontweight='bold')
    
    # Window Duration Distribution
    plt.subplot(2, 2, 3)
    plt.hist(entry_points_df['window_duration_minutes'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Entry Window Duration Distribution', fontweight='bold')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # QRS Score vs Window Duration
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(entry_points_df['qrs_score'], entry_points_df['window_duration_minutes'], 
                         c=entry_points_df['zone_strength'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Zone Strength')
    plt.title('QRS Score vs Window Duration', fontweight='bold')
    plt.xlabel('QRS Score')
    plt.ylabel('Window Duration (minutes)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_visualizations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Summary visualizations created")


def main():
    """Main visualization function."""
    print("🎨 Zone Fade Entry Points Visualization Tool")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    symbols_data = load_2024_data()
    entry_points_df = load_entry_points()
    
    if not symbols_data or entry_points_df is None:
        print("❌ Failed to load required data")
        return
    
    print(f"\n🎯 Processing {len(entry_points_df)} entry points...")
    
    # Process each entry point
    successful_visualizations = 0
    failed_visualizations = 0
    
    for idx, entry_point in entry_points_df.iterrows():
        symbol = entry_point['symbol']
        entry_id = entry_point['entry_id']
        
        print(f"   📊 Creating visualization for {entry_id} ({symbol})...")
        
        if symbol in symbols_data:
            success = create_entry_visualization(
                entry_point, 
                symbols_data[symbol], 
                symbol, 
                output_dir
            )
            
            if success:
                successful_visualizations += 1
                print(f"     ✅ Created visualization for {entry_id}")
            else:
                failed_visualizations += 1
                print(f"     ❌ Failed to create visualization for {entry_id}")
        else:
            print(f"     ❌ No data available for {symbol}")
            failed_visualizations += 1
    
    # Create summary visualizations
    create_summary_visualization(entry_points_df, output_dir)
    
    # Print results
    print(f"\n📊 Visualization Results:")
    print(f"   ✅ Successful: {successful_visualizations}")
    print(f"   ❌ Failed: {failed_visualizations}")
    print(f"   📁 Output Directory: {output_dir.absolute()}")
    
    # Create an index file
    index_file = output_dir / "README.md"
    with open(index_file, 'w') as f:
        f.write("# Zone Fade Entry Points Visualizations\n\n")
        f.write(f"Generated visualizations for {successful_visualizations} entry points.\n\n")
        f.write("## Files:\n")
        f.write("- Individual entry point visualizations: `{symbol}_{entry_id}_visualization.png`\n")
        f.write("- Summary visualizations: `summary_visualizations.png`\n\n")
        f.write("## Visualization Details:\n")
        f.write("- Time window: 60 minutes before entry + 2 hours after entry\n")
        f.write("- Shows: Price action, Volume, VWAP, Zone levels, Entry markers\n")
        f.write("- Setup metrics: QRS score, rejection candles, volume spikes\n")
    
    print(f"\n🎉 Visualization complete! Check {output_dir.absolute()}")


if __name__ == "__main__":
    main()