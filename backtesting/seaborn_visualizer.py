#!/usr/bin/env python3
"""
Seaborn-based Zone Fade Visualizer
Creates comprehensive trading visualizations using Seaborn and Matplotlib
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle
from dataclasses import dataclass

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class OHLCVBar:
    """OHLCV bar data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class SeabornZoneFadeVisualizer:
    """Seaborn-based visualizer for Zone Fade trading setups."""
    
    def __init__(self):
        """Initialize the visualizer with styling."""
        # Set Seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        # Color scheme
        self.colors = {
            'up': '#00ff88',
            'down': '#ff4444',
            'vwap': '#ffaa00',
            'entry': '#00aaff',
            'zone': '#ff00ff',
            'volume_up': '#88ff88',
            'volume_down': '#ff8888'
        }
    
    def create_candlestick_chart(self, ax, timestamps, opens, highs, lows, closes, 
                                entry_idx: int, symbol: str, entry_id: str, 
                                qrs_score: float, entry_timestamp_str: str):
        """Create a candlestick chart using matplotlib."""
        
        # Convert timestamps to matplotlib dates
        dates = mdates.date2num(timestamps)
        
        # Calculate candlestick dimensions
        width = 0.8
        for i, (date, open_price, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            # Determine color
            color = self.colors['up'] if close >= open_price else self.colors['down']
            
            # Draw the high-low line
            ax.plot([date, date], [low, high], color='black', linewidth=1)
            
            # Draw the open-close rectangle
            height = abs(close - open_price)
            bottom = min(open_price, close)
            
            rect = Rectangle((date - width/2, bottom), width, height, 
                           facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        
        # Add entry point markers
        entry_timestamp = timestamps[entry_idx]
        entry_price = closes[entry_idx]
        entry_date = mdates.date2num(entry_timestamp)
        
        # Entry start marker
        ax.scatter(entry_date, entry_price, color=self.colors['entry'], 
                  s=100, marker='^', zorder=5, label='Entry Start')
        
        # Entry end marker (assuming 5 minutes later)
        entry_end_idx = min(entry_idx + 5, len(timestamps) - 1)
        entry_end_timestamp = timestamps[entry_end_idx]
        entry_end_date = mdates.date2num(entry_end_timestamp)
        ax.scatter(entry_end_date, entry_price, color='red', 
                  s=100, marker='v', zorder=5, label='Entry End')
        
        # Entry range line
        ax.plot([entry_date, entry_end_date], [entry_price, entry_price], 
               color='orange', linewidth=3, linestyle='--', alpha=0.8, label='Entry Range')
        
        # Formatting
        ax.set_title(f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f} | {entry_timestamp_str}", 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10)
        
        return ax
    
    def create_volume_chart(self, ax, timestamps, opens, closes, volumes):
        """Create a volume chart using matplotlib."""
        
        # Convert timestamps to matplotlib dates
        dates = mdates.date2num(timestamps)
        
        # Create volume bars
        colors = [self.colors['volume_up'] if close >= open_price else self.colors['volume_down'] 
                 for close, open_price in zip(closes, opens)]
        
        ax.bar(dates, volumes, width=0.8, color=colors, alpha=0.7)
        
        # Formatting
        ax.set_title('Volume', fontsize=14, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        return ax
    
    def create_data_table(self, ax, analysis_data: List[List[str]], symbol: str, entry_id: str):
        """Create a data table using matplotlib table."""
        
        # Remove the axes
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=analysis_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the header
        for i in range(2):
            table[(0, i)].set_facecolor('#1f77b4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(analysis_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        # Set title
        ax.set_title(f'Trade Analysis - {symbol} Entry {entry_id}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        return ax
    
    def create_visualization(self, timestamps, opens, highs, lows, closes, volumes, vwaps,
                           entry_idx: int, symbol: str, entry_id: str, qrs_score: float,
                           entry_timestamp_str: str, entry: Dict[str, Any]) -> plt.Figure:
        """Create a complete visualization with candlestick, volume, and data table."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 20))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 4], hspace=0.3)
        
        # Calculate analysis data
        analysis_data = self._calculate_analysis_data(timestamps, opens, highs, lows, closes, 
                                                    volumes, vwaps, entry_idx, symbol, entry_id, 
                                                    qrs_score, entry_timestamp_str, entry)
        
        # Create candlestick chart
        ax1 = fig.add_subplot(gs[0])
        self.create_candlestick_chart(ax1, timestamps, opens, highs, lows, closes,
                                    entry_idx, symbol, entry_id, qrs_score, entry_timestamp_str)
        
        # Create volume chart
        ax2 = fig.add_subplot(gs[1])
        self.create_volume_chart(ax2, timestamps, opens, closes, volumes)
        
        # Create data table
        ax3 = fig.add_subplot(gs[2:])
        self.create_data_table(ax3, analysis_data, symbol, entry_id)
        
        # Add overall title
        fig.suptitle(f'Zone Fade Analysis - {symbol} Entry {entry_id}', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _calculate_analysis_data(self, timestamps, opens, highs, lows, closes, volumes, vwaps,
                               entry_idx: int, symbol: str, entry_id: str, qrs_score: float,
                               entry_timestamp_str: str, entry: Dict[str, Any]) -> List[List[str]]:
        """Calculate comprehensive analysis data for the table."""
        
        # Basic calculations
        entry_price = closes[entry_idx]
        entry_timestamp = timestamps[entry_idx]
        entry_volume = volumes[entry_idx]
        entry_vwap = vwaps[entry_idx]
        
        # Price analysis
        price_high = max(highs)
        price_low = min(lows)
        price_range = price_high - price_low
        price_range_pct = (price_range / entry_price) * 100
        
        # Zone analysis (simplified)
        zone_high = price_high
        zone_low = price_low
        zone_range = zone_high - zone_low
        zone_mid = (zone_high + zone_low) / 2
        entry_zone_position = ((entry_price - zone_low) / zone_range) * 100
        
        # Volume analysis
        avg_volume = np.mean(volumes)
        volume_ratio = entry_volume / avg_volume if avg_volume > 0 else 1
        
        # VWAP analysis
        vwap_distance = ((entry_price - entry_vwap) / entry_vwap) * 100
        
        # Risk management
        stop_loss = zone_low
        take_profit = zone_high
        risk_amount = entry_price - stop_loss
        reward_amount = take_profit - entry_price
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Time analysis
        entry_start_idx = max(0, entry_idx - 2)
        entry_end_idx = min(len(timestamps) - 1, entry_idx + 2)
        entry_start_time = timestamps[entry_start_idx]
        entry_end_time = timestamps[entry_end_idx]
        entry_duration = (entry_end_time - entry_start_time).total_seconds() / 60
        
        # Convert to ET
        et_tz = pytz.timezone('US/Eastern')
        entry_timestamp_et = entry_timestamp.astimezone(et_tz)
        entry_start_et = entry_start_time.astimezone(et_tz)
        entry_end_et = entry_end_time.astimezone(et_tz)
        
        entry_timestamp_str_et = entry_timestamp_et.strftime('%Y-%m-%d %H:%M:%S ET')
        entry_start_str = entry_start_et.strftime('%H:%M:%S ET')
        entry_end_str = entry_end_et.strftime('%H:%M:%S ET')
        
        # QRS calculations (simplified)
        quality_score = min(10, max(1, (entry_zone_position / 20) + (min(volume_ratio, 2) * 1.5) + 2))
        risk_score = min(10, max(1, 10 - (risk_reward_ratio * 3) + 0.5))
        setup_score = min(10, max(1, (abs(vwap_distance) * 10) + (min(volume_ratio, 1.5) * 2) + 1))
        
        # Create analysis data
        analysis_data = [
            ['ENTRY DETAILS', ''],
            ['Symbol', symbol],
            ['Entry Price', f"${entry_price:.2f}"],
            ['Direction', 'LONG'],
            ['Entry Time', entry_timestamp_str_et],
            ['', ''],
            ['QRS BREAKDOWN', ''],
            ['Quality Score', f"{quality_score:.1f}"],
            ['Risk Score', f"{risk_score:.1f}"],
            ['Setup Score', f"{setup_score:.1f}"],
            ['Overall QRS', f"{qrs_score:.1f}/10"],
            ['', ''],
            ['ENTRY POINT TIMING', ''],
            ['Entry Start Time', entry_start_str],
            ['Entry End Time', entry_end_str],
            ['Entry Duration', f"{entry_duration:.1f} min"],
            ['Entry Start Index', str(entry_start_idx)],
            ['Entry End Index', str(entry_end_idx)],
            ['', ''],
            ['PRICE ANALYSIS', ''],
            ['Current Price', f"${entry_price:.2f}"],
            ['Price Range', f"${price_range:.2f} ({price_range_pct:.2f}%)"],
            ['High', f"${price_high:.2f}"],
            ['Low', f"${price_low:.2f}"],
            ['', ''],
            ['ZONE ANALYSIS', ''],
            ['Zone High', f"${zone_high:.2f}"],
            ['Zone Low', f"${zone_low:.2f}"],
            ['Zone Range', f"${zone_range:.2f}"],
            ['Zone Mid', f"${zone_mid:.2f}"],
            ['Entry Zone Position', f"{entry_zone_position:.1f}%"],
            ['', ''],
            ['RISK MANAGEMENT', ''],
            ['Stop Loss', f"${stop_loss:.2f}"],
            ['Take Profit', f"${take_profit:.2f}"],
            ['Risk Amount', f"${risk_amount:.2f}"],
            ['Reward Amount', f"${reward_amount:.2f}"],
            ['Risk/Reward Ratio', f"1:{risk_reward_ratio:.2f}"],
            ['', ''],
            ['VOLUME & VWAP', ''],
            ['Entry Volume', f"{entry_volume:,.0f}"],
            ['Average Volume', f"{avg_volume:,.0f}"],
            ['Volume Ratio', f"{volume_ratio:.2f}x"],
            ['Entry VWAP', f"${entry_vwap:.2f}"],
            ['VWAP Distance', f"{vwap_distance:+.2f}%"],
            ['', ''],
            ['MARKET CONDITIONS', ''],
            ['Time Window', f"{len(timestamps)} bars"],
            ['Bars Before Entry', str(entry_start_idx)],
            ['Bars After Entry', str(len(timestamps) - entry_end_idx - 1)],
            ['Total Data Points', str(len(timestamps))]
        ]
        
        return analysis_data
    
    def create_visualizations(self, data: Dict[str, List[OHLCVBar]], 
                            entry_points: List[Dict[str, Any]], 
                            output_dir: Path) -> bool:
        """Create visualizations for all entry points."""
        
        print("🎨 Creating Seaborn visualizations...")
        
        successful = 0
        failed = 0
        
        for i, entry in enumerate(entry_points[:20]):  # Generate 20 graphs for testing
            try:
                symbol = entry['symbol']
                entry_id = entry['entry_id']
                entry_timestamp = pd.to_datetime(entry['timestamp'])
                
                print(f"   📊 Creating visualization for {entry_id} ({symbol})...")
                
                # Get data for this symbol
                if symbol not in data:
                    print(f"     ❌ No data found for {symbol}")
                    failed += 1
                    continue
                
                bars = data[symbol]
                
                # Find entry point in data
                entry_idx = None
                for j, bar in enumerate(bars):
                    if abs((bar.timestamp - entry_timestamp).total_seconds()) < 60:  # Within 1 minute
                        entry_idx = j
                        break
                
                if entry_idx is None:
                    print(f"     ❌ Entry point not found in data for {entry_id}")
                    failed += 1
                    continue
                
                # Extract data around entry point (2 hours before, 12 hours after)
                start_idx = max(0, entry_idx - 120)  # 2 hours before (assuming 1-minute bars)
                end_idx = min(len(bars), entry_idx + 720)  # 12 hours after
                
                selected_bars = bars[start_idx:end_idx]
                
                if len(selected_bars) < 10:
                    print(f"     ❌ Insufficient data for {entry_id}")
                    failed += 1
                    continue
                
                # Extract arrays
                timestamps = [bar.timestamp for bar in selected_bars]
                opens = [bar.open for bar in selected_bars]
                highs = [bar.high for bar in selected_bars]
                lows = [bar.low for bar in selected_bars]
                closes = [bar.close for bar in selected_bars]
                volumes = [bar.volume for bar in selected_bars]
                
                # Calculate VWAP (simplified)
                vwaps = []
                cumulative_volume = 0
                cumulative_price_volume = 0
                for j, (close, volume) in enumerate(zip(closes, volumes)):
                    cumulative_volume += volume
                    cumulative_price_volume += close * volume
                    vwap = cumulative_price_volume / cumulative_volume if cumulative_volume > 0 else close
                    vwaps.append(vwap)
                
                # Adjust entry index for selected data
                adjusted_entry_idx = entry_idx - start_idx
                
                # Calculate QRS score (simplified)
                qrs_score = 7.5  # Placeholder
                
                # Create visualization
                fig = self.create_visualization(
                    timestamps, opens, highs, lows, closes, volumes, vwaps,
                    adjusted_entry_idx, symbol, entry_id, qrs_score,
                    entry_timestamp.strftime('%Y-%m-%d %H:%M:%S'), entry
                )
                
                # Save visualization
                filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_seaborn_chart.png"
                filepath = output_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"     ✅ Created visualization for {entry_id}")
                successful += 1
                
            except Exception as e:
                print(f"     ❌ Error creating visualization for {entry_id}: {str(e)}")
                failed += 1
                continue
        
        print(f"📊 Seaborn visualization results:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        
        return successful > 0

def main():
    """Main function to run the Seaborn visualizer."""
    print("🚀 Seaborn Zone Fade Visualizer")
    print("=" * 60)
    
    # Load data
    print("📊 Loading backtesting data...")
    data = {}
    data_dir = Path("data")
    
    for symbol in ['SPY', 'QQQ', 'IWM']:
        file_path = data_dir / "2024" / f"{symbol}_2024.pkl"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                bars = pickle.load(f)
            data[symbol] = bars
            print(f"   ✅ {symbol}: {len(bars):,} bars")
        else:
            print(f"   ❌ {symbol}: File not found")
    
    # Load entry points
    print("📋 Loading entry points...")
    file_path = Path("results/2024/efficient/zone_fade_entry_points_2024_efficient.csv")
    if file_path.exists():
        df = pd.read_csv(file_path)
        entry_points = df.to_dict('records')
        print(f"   ✅ Loaded {len(entry_points)} entry points from CSV")
    else:
        print(f"   ❌ Entry points file not found")
        return
    
    # Create output directory
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    visualizer = SeabornZoneFadeVisualizer()
    success = visualizer.create_visualizations(data, entry_points, output_dir)
    
    if success:
        print("🎉 Seaborn visualization complete!")
        print(f"📁 Visualizations saved to: {output_dir}")
        print("🖼️  Open the PNG files directly to view the charts!")
        print("📄 Open index.html in your browser to see all visualizations in a gallery")
    else:
        print("❌ No visualizations were created successfully")

if __name__ == "__main__":
    main()