#!/usr/bin/env python3
"""
Clean Professional Zone Fade Entry Points Visualization Tool

This script generates clean, professional visualizations for each entry point:
- NO background rectangles or visual clutter
- Clean candlestick charts with proper OHLC representation
- Professional color scheme and styling
- Focus on actual data: price, volume, VWAP
- Precise time window: 1 hour before + 2 hours after entry
- Professional financial chart appearance

Timeframe: 60 minutes before entry + 120 minutes after entry
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


def create_clean_professional_visualization(
    entry_point: pd.Series,
    bars: List[OHLCVBar],
    symbol: str,
    output_dir: Path
) -> bool:
    """Create a clean, professional visualization for a single entry point."""
    
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
    
    # Define time window: 60 minutes before + 120 minutes after (exactly as requested)
    bars_before = 60  # 60 minutes before
    bars_after = 120  # 120 minutes after (2 hours)
    
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
    
    # Create the figure with professional layout
    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(16, 10), 
                                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    
    # Set professional styling
    plt.style.use('default')
    
    # === CLEAN PROFESSIONAL PRICE CHART ===
    
    # Plot candlesticks with professional styling
    for i, (timestamp, open_price, high, low, close) in enumerate(zip(timestamps, opens, highs, lows, closes)):
        # Professional candlestick colors
        is_green = close >= open_price
        color = '#26a69a' if is_green else '#ef5350'  # Professional green/red
        edge_color = '#1b5e20' if is_green else '#b71c1c'  # Darker edges
        
        # Body
        body_height = abs(close - open_price)
        body_bottom = min(open_price, close)
        if body_height > 0:
            ax_price.bar(timestamp, body_height, bottom=body_bottom, width=0.8, 
                        color=color, alpha=0.8, edgecolor=edge_color, linewidth=1)
        else:
            # Doji - draw a horizontal line
            ax_price.plot([timestamp, timestamp], [open_price, open_price], 
                         color=edge_color, linewidth=2)
        
        # Wicks
        ax_price.plot([timestamp, timestamp], [low, high], color='black', linewidth=1.5)
        ax_price.plot([timestamp, timestamp], [low, min(open_price, close)], color=color, linewidth=2)
        ax_price.plot([timestamp, timestamp], [max(open_price, close), high], color=color, linewidth=2)
    
    # Plot VWAP with professional styling
    if vwap_values:
        ax_price.plot(timestamps, vwap_values, color='#9c27b0', linewidth=2.5, 
                     label='VWAP', alpha=0.9, linestyle='-')
    
    # Plot zone level with professional styling
    ax_price.axhline(y=zone_level, color='#ff9800', linestyle='--', linewidth=2.5, 
                    label=f'Zone Level: ${zone_level:.2f}', alpha=0.9)
    
    # Highlight entry point with professional styling
    entry_timestamp_vis = timestamps[vis_entry_idx]
    ax_price.axvline(x=entry_timestamp_vis, color='#f44336', linestyle='-', linewidth=2, 
                    alpha=0.8, label='Entry Time')
    
    # Mark entry point with professional marker
    ax_price.scatter(entry_timestamp_vis, entry_price, color='#f44336', s=200, 
                    marker='*', edgecolor='white', linewidth=2, zorder=10, label='Entry Price')
    
    # Add entry window with subtle highlighting
    window_end_time = entry_timestamp_vis + timedelta(minutes=window_duration)
    ax_price.axvspan(entry_timestamp_vis, window_end_time, alpha=0.1, color='#ffeb3b', 
                    label=f'Entry Window ({window_duration} min)')
    
    # Configure price chart professionally
    ax_price.set_title(f'{symbol} - Entry Point {entry_point["entry_id"]} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime("%Y-%m-%d %H:%M")}', 
                      fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax_price.set_ylabel('Price ($)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax_price.legend(loc='upper left', fontsize=11, framealpha=0.95, fancybox=True, shadow=True)
    ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#bdc3c7')
    ax_price.set_facecolor('#f8f9fa')
    
    # Format x-axis professionally
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_price.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
    
    # === CLEAN PROFESSIONAL VOLUME CHART ===
    
    # Calculate volume colors based on price direction
    volume_colors = []
    for i, (close, open_price) in enumerate(zip(closes, opens)):
        if close >= open_price:
            volume_colors.append('#26a69a')  # Professional green
        else:
            volume_colors.append('#ef5350')  # Professional red
    
    # Plot volume bars with professional styling
    ax_volume.bar(timestamps, volumes, color=volume_colors, alpha=0.7, width=0.8, 
                 edgecolor='black', linewidth=0.5)
    
    # Highlight entry point volume
    ax_volume.bar(timestamps[vis_entry_idx], volumes[vis_entry_idx], 
                 color='#f44336', alpha=1.0, width=0.8, edgecolor='white', linewidth=2)
    
    # Add volume spike indicator if applicable
    if entry_point.get('volume_spike', False):
        avg_volume = np.mean(volumes)
        spike_threshold = avg_volume * 1.8
        ax_volume.axhline(y=spike_threshold, color='#ff9800', linestyle='--', 
                         alpha=0.8, linewidth=2, label='Volume Spike Threshold')
        ax_volume.legend(fontsize=10, framealpha=0.95)
    
    # Configure volume chart professionally
    ax_volume.set_title('Volume', fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    ax_volume.set_ylabel('Volume', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_volume.set_xlabel('Time', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_volume.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#bdc3c7')
    ax_volume.set_facecolor('#f8f9fa')
    
    # Format x-axis professionally
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_volume.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
    
    # Add professional info box
    info_text = f"""ENTRY ANALYSIS
═══════════════════════════════════════
Symbol: {symbol} | ID: {entry_point['entry_id']}
Date: {entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
QRS Score: {qrs_score:.1f}/10

ZONE DETAILS
═══════════════════════════════════════
Type: {entry_point['zone_type']}
Level: ${zone_level:.2f}
Entry Price: ${entry_price:.2f}
Distance: ${abs(entry_price - zone_level):.2f}

SETUP METRICS
═══════════════════════════════════════
Window Duration: {window_duration} min
Rejection Candle: {entry_point['rejection_candle']}
Volume Spike: {entry_point.get('volume_spike', False)}
Zone Strength: {entry_point['zone_strength']:.2f}
Zone Quality: {entry_point['zone_quality']}

PRICE RANGE
═══════════════════════════════════════
High: ${max(highs):.2f} | Low: ${min(lows):.2f}
Range: ${max(highs) - min(lows):.2f}

VOLUME ANALYSIS
═══════════════════════════════════════
Avg: {np.mean(volumes):,.0f} | Max: {max(volumes):,.0f}
Entry: {volumes[vis_entry_idx]:,.0f}"""
    
    # Add info box to the price chart
    ax_price.text(0.02, 0.98, info_text, transform=ax_price.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          edgecolor='#34495e', linewidth=1.5, alpha=0.95))
    
    # Add quality indicator
    if qrs_score >= 8:
        quality_color = '#27ae60'  # Green
        quality_text = "HIGH QUALITY"
    elif qrs_score >= 6:
        quality_color = '#f39c12'  # Orange
        quality_text = "GOOD QUALITY"
    else:
        quality_color = '#e74c3c'  # Red
        quality_text = "MODERATE QUALITY"
    
    ax_price.text(0.98, 0.02, quality_text, transform=ax_price.transAxes, 
                 fontsize=12, fontweight='bold', ha='right', va='bottom', color=quality_color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                          edgecolor=quality_color, linewidth=2, alpha=0.95))
    
    # Adjust layout professionally
    plt.tight_layout()
    
    # Save the plot with high quality
    output_file = output_dir / f"{symbol}_{entry_point['entry_id']}_clean.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.2)
    plt.close()
    
    return True


def create_clean_summary_visualization(entry_points_df: pd.DataFrame, output_dir: Path):
    """Create clean professional summary visualizations."""
    print("📊 Creating Clean Professional Summary Visualizations...")
    
    # Set professional styling
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Zone Fade Entry Points - Professional Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Professional color scheme
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # QRS Score Distribution
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(entry_points_df['qrs_score'], bins=20, alpha=0.7, 
                               color=colors[0], edgecolor='white', linewidth=1)
    ax1.set_title('QRS Score Distribution', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_xlabel('QRS Score', fontsize=12, color='#2c3e50')
    ax1.set_ylabel('Frequency', fontsize=12, color='#2c3e50')
    ax1.grid(True, alpha=0.3, color='#bdc3c7')
    ax1.axvline(entry_points_df['qrs_score'].mean(), color=colors[1], linestyle='--', 
               linewidth=2, label=f'Mean: {entry_points_df["qrs_score"].mean():.1f}')
    ax1.legend()
    ax1.set_facecolor('#f8f9fa')
    
    # Entry Points by Symbol
    ax2 = axes[0, 1]
    symbol_counts = entry_points_df['symbol'].value_counts()
    wedges, texts, autotexts = ax2.pie(symbol_counts.values, labels=symbol_counts.index, 
                                      autopct='%1.1f%%', startangle=90, colors=colors[:3])
    ax2.set_title('Entry Points by Symbol', fontsize=14, fontweight='bold', color='#2c3e50')
    
    # Window Duration Distribution
    ax3 = axes[0, 2]
    ax3.hist(entry_points_df['window_duration_minutes'], bins=20, alpha=0.7, 
            color=colors[2], edgecolor='white', linewidth=1)
    ax3.set_title('Entry Window Duration Distribution', fontsize=14, fontweight='bold', color='#2c3e50')
    ax3.set_xlabel('Duration (minutes)', fontsize=12, color='#2c3e50')
    ax3.set_ylabel('Frequency', fontsize=12, color='#2c3e50')
    ax3.grid(True, alpha=0.3, color='#bdc3c7')
    ax3.set_facecolor('#f8f9fa')
    
    # QRS Score vs Window Duration
    ax4 = axes[1, 0]
    scatter = ax4.scatter(entry_points_df['qrs_score'], entry_points_df['window_duration_minutes'], 
                         c=entry_points_df['zone_strength'], cmap='viridis', alpha=0.7, s=60)
    ax4.set_title('QRS Score vs Window Duration', fontsize=14, fontweight='bold', color='#2c3e50')
    ax4.set_xlabel('QRS Score', fontsize=12, color='#2c3e50')
    ax4.set_ylabel('Window Duration (minutes)', fontsize=12, color='#2c3e50')
    ax4.grid(True, alpha=0.3, color='#bdc3c7')
    ax4.set_facecolor('#f8f9fa')
    plt.colorbar(scatter, ax=ax4, label='Zone Strength')
    
    # Zone Strength Distribution
    ax5 = axes[1, 1]
    ax5.hist(entry_points_df['zone_strength'], bins=20, alpha=0.7, 
            color=colors[3], edgecolor='white', linewidth=1)
    ax5.set_title('Zone Strength Distribution', fontsize=14, fontweight='bold', color='#2c3e50')
    ax5.set_xlabel('Zone Strength', fontsize=12, color='#2c3e50')
    ax5.set_ylabel('Frequency', fontsize=12, color='#2c3e50')
    ax5.grid(True, alpha=0.3, color='#bdc3c7')
    ax5.set_facecolor('#f8f9fa')
    
    # Monthly Distribution
    ax6 = axes[1, 2]
    entry_points_df['month'] = entry_points_df['timestamp'].dt.month
    monthly_counts = entry_points_df['month'].value_counts().sort_index()
    ax6.bar(monthly_counts.index, monthly_counts.values, alpha=0.7, 
           color=colors[4], edgecolor='white', linewidth=1)
    ax6.set_title('Entry Points by Month', fontsize=14, fontweight='bold', color='#2c3e50')
    ax6.set_xlabel('Month', fontsize=12, color='#2c3e50')
    ax6.set_ylabel('Count', fontsize=12, color='#2c3e50')
    ax6.set_xticks(range(1, 13))
    ax6.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax6.grid(True, alpha=0.3, color='#bdc3c7')
    ax6.set_facecolor('#f8f9fa')
    
    # Set background color for the entire figure
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_dir / "clean_professional_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Clean professional summary visualizations created")


def main():
    """Main visualization function."""
    print("🎨 Clean Professional Zone Fade Entry Points Visualization Tool")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("results/visualizations_clean")
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
        
        print(f"   📊 Creating clean professional visualization for {entry_id} ({symbol})...")
        
        if symbol in symbols_data:
            success = create_clean_professional_visualization(
                entry_point, 
                symbols_data[symbol], 
                symbol, 
                output_dir
            )
            
            if success:
                successful_visualizations += 1
                print(f"     ✅ Created clean professional visualization for {entry_id}")
            else:
                failed_visualizations += 1
                print(f"     ❌ Failed to create clean professional visualization for {entry_id}")
        else:
            print(f"     ❌ No data available for {symbol}")
            failed_visualizations += 1
    
    # Create clean professional summary visualizations
    create_clean_summary_visualization(entry_points_df, output_dir)
    
    # Print results
    print(f"\n📊 Clean Professional Visualization Results:")
    print(f"   ✅ Successful: {successful_visualizations}")
    print(f"   ❌ Failed: {failed_visualizations}")
    print(f"   📁 Output Directory: {output_dir.absolute()}")
    
    # Create a professional index file
    index_file = output_dir / "README.md"
    with open(index_file, 'w') as f:
        f.write("# Clean Professional Zone Fade Entry Points Visualizations\n\n")
        f.write(f"Generated clean, professional visualizations for {successful_visualizations} entry points.\n\n")
        f.write("## Key Features:\n")
        f.write("- **NO Background Clutter**: Removed all distracting rectangles and visual noise\n")
        f.write("- **Professional Styling**: Clean, financial-grade appearance\n")
        f.write("- **Clear Data Focus**: Price, Volume, VWAP clearly visible\n")
        f.write("- **Precise Time Window**: Exactly 1 hour before + 2 hours after entry\n")
        f.write("- **Professional Colors**: Financial industry standard color scheme\n")
        f.write("- **Clean Layout**: Organized, easy-to-read information panels\n")
        f.write("- **High Quality**: 300 DPI, publication-ready charts\n\n")
        f.write("## Files:\n")
        f.write("- Individual entry point visualizations: `{symbol}_{entry_id}_clean.png`\n")
        f.write("- Summary visualizations: `clean_professional_summary.png`\n\n")
        f.write("## Visualization Details:\n")
        f.write("- Time window: 60 minutes before entry + 120 minutes after entry\n")
        f.write("- Shows: Clean price action, Volume bars, VWAP line, Zone level\n")
        f.write("- Setup metrics: QRS score, rejection candles, volume spikes\n")
        f.write("- Professional quality: Ready for analysis and presentation\n")
    
    print(f"\n🎉 Clean professional visualization complete! Check {output_dir.absolute()}")


if __name__ == "__main__":
    main()