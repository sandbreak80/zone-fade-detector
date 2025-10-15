#!/usr/bin/env python3
"""
Improved Zone Fade Entry Points Visualization Tool

This script generates cleaner, more readable visualizations for each entry point:
- Simplified price action with clear candlesticks
- Cleaner zone representation
- Better color scheme and contrast
- Improved layout and annotations
- More intuitive volume representation

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


def create_improved_entry_visualization(
    entry_point: pd.Series,
    bars: List[OHLCVBar],
    symbol: str,
    output_dir: Path
) -> bool:
    """Create an improved, cleaner visualization for a single entry point."""
    
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
    
    # Create the figure with improved layout
    fig = plt.figure(figsize=(16, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 0.8], width_ratios=[4, 1], 
                         hspace=0.3, wspace=0.1)
    
    # Main price chart (top, left)
    ax_price = fig.add_subplot(gs[0, 0])
    
    # Volume chart (middle, left)
    ax_volume = fig.add_subplot(gs[1, 0])
    
    # Info panel (right side)
    ax_info = fig.add_subplot(gs[0:2, 1])
    
    # === IMPROVED PRICE CHART ===
    
    # Plot candlesticks with better styling
    for i, (timestamp, open_price, high, low, close) in enumerate(zip(timestamps, opens, highs, lows, closes)):
        # Determine candle color
        is_green = close >= open_price
        color = '#2E8B57' if is_green else '#DC143C'  # Forest green / Crimson red
        alpha = 0.8
        
        # Body
        body_height = abs(close - open_price)
        body_bottom = min(open_price, close)
        if body_height > 0:
            ax_price.bar(timestamp, body_height, bottom=body_bottom, width=0.6, 
                        color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        
        # Wicks
        ax_price.plot([timestamp, timestamp], [low, high], color='black', linewidth=1.2)
        ax_price.plot([timestamp, timestamp], [low, min(open_price, close)], color=color, linewidth=2)
        ax_price.plot([timestamp, timestamp], [max(open_price, close), high], color=color, linewidth=2)
    
    # Plot VWAP with better styling
    if vwap_values:
        ax_price.plot(timestamps, vwap_values, color='#8A2BE2', linewidth=2.5, 
                     label='VWAP', alpha=0.9, linestyle='-')
    
    # Plot zone level with better visibility
    ax_price.axhline(y=zone_level, color='#FF8C00', linestyle='--', linewidth=3, 
                    label=f'Zone Level: ${zone_level:.2f}', alpha=0.9)
    
    # Highlight entry point with better styling
    entry_timestamp_vis = timestamps[vis_entry_idx]
    ax_price.axvline(x=entry_timestamp_vis, color='#FF0000', linestyle='-', linewidth=3, 
                    alpha=0.8, label='Entry Point')
    
    # Mark entry point with a prominent marker
    ax_price.scatter(entry_timestamp_vis, entry_price, color='#FF0000', s=300, 
                    marker='*', edgecolor='black', linewidth=2, zorder=10, label='Entry Price')
    
    # Add entry window duration with subtle highlighting
    window_end_time = entry_timestamp_vis + timedelta(minutes=window_duration)
    ax_price.axvspan(entry_timestamp_vis, window_end_time, alpha=0.15, color='#FFD700', 
                    label=f'Entry Window ({window_duration} min)')
    
    # Configure price chart
    ax_price.set_title(f'{symbol} - Entry Point {entry_point["entry_id"]} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime("%Y-%m-%d %H:%M")}', 
                      fontsize=16, fontweight='bold', pad=20)
    ax_price.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax_price.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format x-axis
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_price.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
    
    # === IMPROVED VOLUME CHART ===
    
    # Calculate volume colors based on price direction
    volume_colors = []
    for i, (close, open_price) in enumerate(zip(closes, opens)):
        if close >= open_price:
            volume_colors.append('#2E8B57')  # Green for up bars
        else:
            volume_colors.append('#DC143C')  # Red for down bars
    
    # Plot volume bars with better styling
    ax_volume.bar(timestamps, volumes, color=volume_colors, alpha=0.7, width=0.6, 
                 edgecolor='black', linewidth=0.3)
    
    # Highlight entry point volume
    ax_volume.bar(timestamps[vis_entry_idx], volumes[vis_entry_idx], 
                 color='#FF0000', alpha=1.0, width=0.6, edgecolor='black', linewidth=2)
    
    # Add volume spike indicator if applicable
    if entry_point.get('volume_spike', False):
        avg_volume = np.mean(volumes)
        spike_threshold = avg_volume * 1.8
        ax_volume.axhline(y=spike_threshold, color='#FF8C00', linestyle='--', 
                         alpha=0.8, linewidth=2, label='Volume Spike Threshold')
        ax_volume.legend(fontsize=10)
    
    # Configure volume chart
    ax_volume.set_title('Volume', fontsize=14, fontweight='bold', pad=15)
    ax_volume.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax_volume.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax_volume.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format x-axis
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_volume.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
    
    # === IMPROVED INFO PANEL ===
    
    # Clear the info panel
    ax_info.clear()
    ax_info.axis('off')
    
    # Create a clean info display
    info_text = f"""
ENTRY DETAILS
═══════════════════════════════════════

Symbol: {symbol}
Entry ID: {entry_point['entry_id']}
Date: {entry_timestamp.strftime('%Y-%m-%d')}
Time: {entry_timestamp.strftime('%H:%M:%S')}

QUALITY METRICS
═══════════════════════════════════════
QRS Score: {qrs_score:.1f}/10
Zone Strength: {entry_point['zone_strength']:.2f}
Zone Quality: {entry_point['zone_quality']}

ZONE INFORMATION
═══════════════════════════════════════
Zone Type: {entry_point['zone_type']}
Zone Level: ${zone_level:.2f}
Entry Price: ${entry_price:.2f}
Distance: ${abs(entry_price - zone_level):.2f}

ENTRY WINDOW
═══════════════════════════════════════
Duration: {window_duration} minutes
Rejection Candle: {entry_point['rejection_candle']}
Volume Spike: {entry_point.get('volume_spike', False)}

PRICE RANGE
═══════════════════════════════════════
High: ${max(highs):.2f}
Low: ${min(lows):.2f}
Range: ${max(highs) - min(lows):.2f}

VOLUME ANALYSIS
═══════════════════════════════════════
Avg Volume: {np.mean(volumes):,.0f}
Max Volume: {max(volumes):,.0f}
Entry Volume: {volumes[vis_entry_idx]:,.0f}
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', 
                         edgecolor='#4682B4', linewidth=2))
    
    # Add a quality indicator
    if qrs_score >= 8:
        quality_color = '#32CD32'  # Green
        quality_text = "HIGH QUALITY"
    elif qrs_score >= 6:
        quality_color = '#FFD700'  # Gold
        quality_text = "GOOD QUALITY"
    else:
        quality_color = '#FF6347'  # Tomato
        quality_text = "MODERATE QUALITY"
    
    ax_info.text(0.5, 0.05, quality_text, transform=ax_info.transAxes, 
                fontsize=14, fontweight='bold', ha='center', color=quality_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=quality_color, linewidth=2))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with high quality
    output_file = output_dir / f"{symbol}_{entry_point['entry_id']}_improved.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.2)
    plt.close()
    
    return True


def create_improved_summary_visualization(entry_points_df: pd.DataFrame, output_dir: Path):
    """Create improved summary visualizations."""
    print("📊 Creating Improved Summary Visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Zone Fade Entry Points - Summary Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # QRS Score Distribution
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(entry_points_df['qrs_score'], bins=20, alpha=0.7, 
                               color='skyblue', edgecolor='black', linewidth=1)
    ax1.set_title('QRS Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('QRS Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(entry_points_df['qrs_score'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {entry_points_df["qrs_score"].mean():.1f}')
    ax1.legend()
    
    # Entry Points by Symbol
    ax2 = axes[0, 1]
    symbol_counts = entry_points_df['symbol'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    wedges, texts, autotexts = ax2.pie(symbol_counts.values, labels=symbol_counts.index, 
                                      autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Entry Points by Symbol', fontsize=14, fontweight='bold')
    
    # Window Duration Distribution
    ax3 = axes[0, 2]
    ax3.hist(entry_points_df['window_duration_minutes'], bins=20, alpha=0.7, 
            color='lightgreen', edgecolor='black', linewidth=1)
    ax3.set_title('Entry Window Duration Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Duration (minutes)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # QRS Score vs Window Duration
    ax4 = axes[1, 0]
    scatter = ax4.scatter(entry_points_df['qrs_score'], entry_points_df['window_duration_minutes'], 
                         c=entry_points_df['zone_strength'], cmap='viridis', alpha=0.7, s=60)
    ax4.set_title('QRS Score vs Window Duration', fontsize=14, fontweight='bold')
    ax4.set_xlabel('QRS Score', fontsize=12)
    ax4.set_ylabel('Window Duration (minutes)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Zone Strength')
    
    # Zone Strength Distribution
    ax5 = axes[1, 1]
    ax5.hist(entry_points_df['zone_strength'], bins=20, alpha=0.7, 
            color='orange', edgecolor='black', linewidth=1)
    ax5.set_title('Zone Strength Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Zone Strength', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Monthly Distribution
    ax6 = axes[1, 2]
    entry_points_df['month'] = entry_points_df['timestamp'].dt.month
    monthly_counts = entry_points_df['month'].value_counts().sort_index()
    ax6.bar(monthly_counts.index, monthly_counts.values, alpha=0.7, 
           color='purple', edgecolor='black', linewidth=1)
    ax6.set_title('Entry Points by Month', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Month', fontsize=12)
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_xticks(range(1, 13))
    ax6.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improved_summary_visualizations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Improved summary visualizations created")


def main():
    """Main visualization function."""
    print("🎨 Improved Zone Fade Entry Points Visualization Tool")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/visualizations_improved")
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
        
        print(f"   📊 Creating improved visualization for {entry_id} ({symbol})...")
        
        if symbol in symbols_data:
            success = create_improved_entry_visualization(
                entry_point, 
                symbols_data[symbol], 
                symbol, 
                output_dir
            )
            
            if success:
                successful_visualizations += 1
                print(f"     ✅ Created improved visualization for {entry_id}")
            else:
                failed_visualizations += 1
                print(f"     ❌ Failed to create improved visualization for {entry_id}")
        else:
            print(f"     ❌ No data available for {symbol}")
            failed_visualizations += 1
    
    # Create improved summary visualizations
    create_improved_summary_visualization(entry_points_df, output_dir)
    
    # Print results
    print(f"\n📊 Improved Visualization Results:")
    print(f"   ✅ Successful: {successful_visualizations}")
    print(f"   ❌ Failed: {failed_visualizations}")
    print(f"   📁 Output Directory: {output_dir.absolute()}")
    
    # Create an improved index file
    index_file = output_dir / "README.md"
    with open(index_file, 'w') as f:
        f.write("# Improved Zone Fade Entry Points Visualizations\n\n")
        f.write(f"Generated improved visualizations for {successful_visualizations} entry points.\n\n")
        f.write("## Improvements Made:\n")
        f.write("- **Cleaner Design**: Removed visual clutter and overlapping elements\n")
        f.write("- **Better Color Scheme**: High contrast, professional colors\n")
        f.write("- **Clearer Candlesticks**: Proper OHLC representation\n")
        f.write("- **Simplified Zones**: Single zone level line instead of overlapping bands\n")
        f.write("- **Improved Volume**: Traditional bar chart instead of stacked areas\n")
        f.write("- **Better Layout**: Organized info panel with clear sections\n")
        f.write("- **Quality Indicators**: Visual quality assessment\n\n")
        f.write("## Files:\n")
        f.write("- Individual entry point visualizations: `{symbol}_{entry_id}_improved.png`\n")
        f.write("- Summary visualizations: `improved_summary_visualizations.png`\n\n")
        f.write("## Visualization Details:\n")
        f.write("- Time window: 60 minutes before entry + 2 hours after entry\n")
        f.write("- Shows: Price action, Volume, VWAP, Zone levels, Entry markers\n")
        f.write("- Setup metrics: QRS score, rejection candles, volume spikes\n")
        f.write("- Quality assessment: Visual quality indicators\n")
    
    print(f"\n🎉 Improved visualization complete! Check {output_dir.absolute()}")


if __name__ == "__main__":
    main()