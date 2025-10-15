#!/usr/bin/env python3
"""
Zone Fade Entry Point Visualizer

A comprehensive visualization tool for manually verifying the accuracy of entry point detections
by visual inspection. Generates publication-quality charts for each detected entry point.

Features:
- 4-hour time window (2 hours before + 2 hours after entry)
- Clean, minimalist design with no clutter
- Candlestick charts with VWAP overlay
- Volume analysis with semi-transparent bars
- Entry point highlighting with clear annotations
- Automatic PNG saving with structured filenames
- Optional HTML report generation

Author: Zone Fade Detector Team
Version: 1.0.0
"""

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import json
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    # Time window settings
    hours_before: int = 2
    hours_after: int = 2
    
    # Chart styling
    figure_size: Tuple[int, int] = (16, 10)
    dpi: int = 300
    style: str = 'seaborn-v0_8-whitegrid'
    
    # Colors
    price_up_color: str = '#2e7d32'      # Green for up candles
    price_down_color: str = '#c62828'    # Red for down candles
    vwap_color: str = '#ff6f00'          # Orange for VWAP
    entry_line_color: str = '#d32f2f'    # Red for entry line
    zone_line_color: str = '#7b1fa2'     # Purple for zone level
    volume_alpha: float = 0.6
    
    # Font settings
    title_fontsize: int = 16
    label_fontsize: int = 12
    annotation_fontsize: int = 10
    
    # Grid and background
    show_grid: bool = True
    grid_alpha: float = 0.3
    background_color: str = 'white'


class EntryVisualizer:
    """Main class for generating entry point visualizations."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer with configuration."""
        self.config = config or VisualizationConfig()
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        plt.style.use(self.config.style)
        plt.rcParams.update({
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
            'font.size': self.config.annotation_fontsize,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.annotation_fontsize,
            'ytick.labelsize': self.config.annotation_fontsize,
            'legend.fontsize': self.config.annotation_fontsize,
        })
    
    def load_backtest_data(self, data_dir: Path) -> Dict[str, List[OHLCVBar]]:
        """Load backtesting data from pickle files."""
        print("📊 Loading backtesting data...")
        
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
    
    def load_entry_points(self, csv_file: Path) -> pd.DataFrame:
        """Load entry points from CSV file."""
        print("📋 Loading entry points...")
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Entry points file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        print(f"   ✅ Loaded {len(df)} entry points")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def calculate_vwap(self, bars: List[OHLCVBar], start_idx: int, end_idx: int) -> List[float]:
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
    
    def find_entry_bar_index(self, bars: List[OHLCVBar], entry_timestamp: datetime) -> Optional[int]:
        """Find the bar index closest to the entry timestamp."""
        for i, bar in enumerate(bars):
            if abs((bar.timestamp - entry_timestamp).total_seconds()) < 60:  # Within 1 minute
                return i
        return None
    
    def create_entry_visualization(
        self,
        entry_point: pd.Series,
        bars: List[OHLCVBar],
        symbol: str,
        output_dir: Path
    ) -> bool:
        """Create a single entry point visualization."""
        
        # Extract entry point data
        entry_timestamp = entry_point['timestamp']
        entry_price = entry_point['price']
        zone_level = entry_point['zone_level']
        qrs_score = entry_point['qrs_score']
        window_duration = entry_point['window_duration_minutes']
        entry_id = entry_point['entry_id']
        
        # Find the entry bar index
        entry_idx = self.find_entry_bar_index(bars, entry_timestamp)
        if entry_idx is None:
            print(f"   ❌ Could not find entry point {entry_id} in bars data")
            return False
        
        # Calculate time window
        bars_before = self.config.hours_before * 4  # 4 bars per hour (15-min bars)
        bars_after = self.config.hours_after * 4
        
        start_idx = max(0, entry_idx - bars_before)
        end_idx = min(len(bars) - 1, entry_idx + bars_after)
        
        # Extract visualization data
        vis_bars = bars[start_idx:end_idx + 1]
        timestamps = [bar.timestamp for bar in vis_bars]
        opens = [bar.open for bar in vis_bars]
        highs = [bar.high for bar in vis_bars]
        lows = [bar.low for bar in vis_bars]
        closes = [bar.close for bar in vis_bars]
        volumes = [bar.volume for bar in vis_bars]
        
        # Calculate VWAP
        vwap_values = self.calculate_vwap(bars, start_idx, end_idx)
        
        # Find entry point index in visualization data
        vis_entry_idx = entry_idx - start_idx
        
        # Create the figure
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, 
            figsize=self.config.figure_size,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15}
        )
        
        # Set background color
        fig.patch.set_facecolor(self.config.background_color)
        ax_price.set_facecolor(self.config.background_color)
        ax_volume.set_facecolor(self.config.background_color)
        
        # === PRICE CHART ===
        self._plot_candlesticks(ax_price, timestamps, opens, highs, lows, closes)
        self._plot_vwap(ax_price, timestamps, vwap_values)
        self._plot_zone_level(ax_price, zone_level)
        self._plot_entry_point(ax_price, timestamps[vis_entry_idx], entry_price, entry_timestamp)
        self._plot_entry_window(ax_price, timestamps[vis_entry_idx], window_duration)
        self._configure_price_chart(ax_price, symbol, entry_id, qrs_score, entry_timestamp)
        
        # === VOLUME CHART ===
        self._plot_volume(ax_volume, timestamps, volumes, closes, opens, vis_entry_idx)
        self._configure_volume_chart(ax_volume)
        
        # === ANNOTATIONS ===
        self._add_annotations(ax_price, entry_point, symbol, vis_bars, vis_entry_idx)
        
        # Save the plot
        filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_entry_visual.png"
        output_file = output_dir / filename
        
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor=self.config.background_color, edgecolor='none')
        plt.close()
        
        return True
    
    def _plot_candlesticks(self, ax, timestamps, opens, highs, lows, closes):
        """Plot candlestick chart."""
        for i, (timestamp, open_price, high, low, close) in enumerate(zip(timestamps, opens, highs, lows, closes)):
            is_green = close >= open_price
            color = self.config.price_up_color if is_green else self.config.price_down_color
            edge_color = '#1b5e20' if is_green else '#b71c1c'
            
            # Body
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            
            if body_height > 0:
                # Draw body as rectangle
                ax.bar(timestamp, body_height, bottom=body_bottom, width=0.8, 
                      color=color, alpha=0.8, edgecolor=edge_color, linewidth=1)
            else:
                # Doji - draw horizontal line
                ax.plot([timestamp, timestamp], [open_price, open_price], 
                       color=edge_color, linewidth=2)
            
            # Wicks
            ax.plot([timestamp, timestamp], [low, high], color='black', linewidth=1, alpha=0.8)
    
    def _plot_vwap(self, ax, timestamps, vwap_values):
        """Plot VWAP line."""
        if vwap_values:
            ax.plot(timestamps, vwap_values, color=self.config.vwap_color, 
                   linewidth=2.5, label='VWAP', alpha=0.9, linestyle='-')
    
    def _plot_zone_level(self, ax, zone_level):
        """Plot zone level line."""
        ax.axhline(y=zone_level, color=self.config.zone_line_color, 
                  linestyle='--', linewidth=2, alpha=0.8, label=f'Zone Level: ${zone_level:.2f}')
    
    def _plot_entry_point(self, ax, entry_timestamp, entry_price, entry_datetime):
        """Plot entry point marker and line."""
        # Entry line
        ax.axvline(x=entry_timestamp, color=self.config.entry_line_color, 
                  linestyle='-', linewidth=2, alpha=0.8, label='Entry Time')
        
        # Entry point marker
        ax.scatter(entry_timestamp, entry_price, color=self.config.entry_line_color, 
                  s=200, marker='*', edgecolor='white', linewidth=2, zorder=10, 
                  label=f'Entry: ${entry_price:.2f}')
    
    def _plot_entry_window(self, ax, entry_timestamp, window_duration):
        """Plot entry window end marker."""
        window_end_time = entry_timestamp + timedelta(minutes=window_duration)
        ax.axvline(x=window_end_time, color='#ffa000', linestyle=':', linewidth=2, 
                  alpha=0.7, label=f'Window End ({window_duration} min)')
    
    def _configure_price_chart(self, ax, symbol, entry_id, qrs_score, entry_timestamp):
        """Configure price chart appearance."""
        ax.set_title(f'{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=self.config.title_fontsize, fontweight='bold', pad=20)
        ax.set_ylabel('Price ($)', fontsize=self.config.label_fontsize, fontweight='bold')
        ax.legend(loc='upper left', fontsize=self.config.annotation_fontsize, 
                 framealpha=0.95, fancybox=True, shadow=True)
        
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, linestyle='-', linewidth=0.5)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=self.config.annotation_fontsize)
    
    def _plot_volume(self, ax, timestamps, volumes, closes, opens, entry_idx):
        """Plot volume bars."""
        # Calculate volume colors based on price direction
        volume_colors = []
        for close, open_price in zip(closes, opens):
            if close >= open_price:
                volume_colors.append(self.config.price_up_color)
            else:
                volume_colors.append(self.config.price_down_color)
        
        # Plot volume bars
        for i, (timestamp, volume, color) in enumerate(zip(timestamps, volumes, volume_colors)):
            ax.bar(timestamp, volume, width=0.8, color=color, alpha=self.config.volume_alpha, 
                  edgecolor=color, linewidth=0.5)
        
        # Highlight entry point volume
        ax.bar(timestamps[entry_idx], volumes[entry_idx], width=0.8, 
              color=self.config.entry_line_color, alpha=1.0, edgecolor='white', linewidth=2)
    
    def _configure_volume_chart(self, ax):
        """Configure volume chart appearance."""
        ax.set_title('Volume', fontsize=self.config.label_fontsize, fontweight='bold', pad=15)
        ax.set_ylabel('Volume', fontsize=self.config.annotation_fontsize, fontweight='bold')
        ax.set_xlabel('Time', fontsize=self.config.annotation_fontsize, fontweight='bold')
        
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, linestyle='-', linewidth=0.5)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=self.config.annotation_fontsize)
    
    def _add_annotations(self, ax, entry_point, symbol, vis_bars, entry_idx):
        """Add detailed annotations to the chart."""
        # Extract data for annotations
        entry_timestamp = entry_point['timestamp']
        entry_price = entry_point['price']
        zone_level = entry_point['zone_level']
        qrs_score = entry_point['qrs_score']
        window_duration = entry_point['window_duration_minutes']
        
        # Calculate price range
        highs = [bar.high for bar in vis_bars]
        lows = [bar.low for bar in vis_bars]
        volumes = [bar.volume for bar in vis_bars]
        
        # Create info text
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
Entry: {volumes[entry_idx]:,.0f}"""
        
        # Add info box
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='#34495e', linewidth=1.5, alpha=0.95))
        
        # Add quality indicator
        if qrs_score >= 8:
            quality_color = '#27ae60'
            quality_text = "HIGH QUALITY"
        elif qrs_score >= 6:
            quality_color = '#f39c12'
            quality_text = "GOOD QUALITY"
        else:
            quality_color = '#e74c3c'
            quality_text = "MODERATE QUALITY"
        
        ax.text(0.98, 0.02, quality_text, transform=ax.transAxes, 
               fontsize=12, fontweight='bold', ha='right', va='bottom', color=quality_color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=quality_color, linewidth=2, alpha=0.95))
    
    def generate_entry_visuals(
        self, 
        entry_points_df: pd.DataFrame, 
        symbols_data: Dict[str, List[OHLCVBar]], 
        output_dir: Path
    ) -> Dict[str, int]:
        """Generate visualizations for all entry points."""
        print(f"\n🎯 Generating visualizations for {len(entry_points_df)} entry points...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        results = {'successful': 0, 'failed': 0, 'no_data': 0}
        
        # Process each entry point
        for idx, entry_point in entry_points_df.iterrows():
            symbol = entry_point['symbol']
            entry_id = entry_point['entry_id']
            
            print(f"   📊 Creating visualization for {entry_id} ({symbol})...")
            
            if symbol in symbols_data:
                success = self.create_entry_visualization(
                    entry_point, 
                    symbols_data[symbol], 
                    symbol, 
                    output_dir
                )
                
                if success:
                    results['successful'] += 1
                    print(f"     ✅ Created visualization for {entry_id}")
                else:
                    results['failed'] += 1
                    print(f"     ❌ Failed to create visualization for {entry_id}")
            else:
                results['no_data'] += 1
                print(f"     ❌ No data available for {symbol}")
        
        return results
    
    def create_html_report(self, entry_points_df: pd.DataFrame, output_dir: Path) -> str:
        """Create an HTML report combining all visualizations."""
        print("📄 Creating HTML report...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zone Fade Entry Points - Visual Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .visualization-item {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .visualization-item img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .entry-info {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .entry-info h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .entry-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.9em;
        }}
        .entry-details div {{
            padding: 5px;
            background: white;
            border-radius: 3px;
        }}
        .qrs-high {{ color: #27ae60; font-weight: bold; }}
        .qrs-good {{ color: #f39c12; font-weight: bold; }}
        .qrs-moderate {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Zone Fade Entry Points - Visual Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Manual verification of entry point detection accuracy</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{len(entry_points_df)}</div>
            <div>Total Entry Points</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{entry_points_df['symbol'].nunique()}</div>
            <div>Symbols</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{entry_points_df['qrs_score'].mean():.1f}</div>
            <div>Avg QRS Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{entry_points_df['qrs_score'].max():.1f}</div>
            <div>Max QRS Score</div>
        </div>
    </div>
    
    <div class="visualization-grid">
"""
        
        # Add each visualization to the HTML
        for idx, entry_point in entry_points_df.iterrows():
            symbol = entry_point['symbol']
            entry_id = entry_point['entry_id']
            qrs_score = entry_point['qrs_score']
            entry_timestamp = entry_point['timestamp']
            
            # Determine QRS class for styling
            if qrs_score >= 8:
                qrs_class = "qrs-high"
            elif qrs_score >= 6:
                qrs_class = "qrs-good"
            else:
                qrs_class = "qrs-moderate"
            
            # Create filename
            filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_entry_visual.png"
            
            html_content += f"""
        <div class="visualization-item">
            <img src="{filename}" alt="Entry Point {entry_id}">
            <div class="entry-info">
                <h3>{symbol} - Entry {entry_id}</h3>
                <div class="entry-details">
                    <div><strong>Date:</strong> {entry_timestamp.strftime('%Y-%m-%d %H:%M')}</div>
                    <div><strong>QRS Score:</strong> <span class="{qrs_class}">{qrs_score:.1f}/10</span></div>
                    <div><strong>Price:</strong> ${entry_point['price']:.2f}</div>
                    <div><strong>Zone Level:</strong> ${entry_point['zone_level']:.2f}</div>
                    <div><strong>Zone Type:</strong> {entry_point['zone_type']}</div>
                    <div><strong>Window:</strong> {entry_point['window_duration_minutes']} min</div>
                    <div><strong>Rejection:</strong> {entry_point['rejection_candle']}</div>
                    <div><strong>Volume Spike:</strong> {entry_point.get('volume_spike', False)}</div>
                </div>
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_file = output_dir / "entry_points_visual_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"   ✅ HTML report created: {html_file}")
        return str(html_file)


def main():
    """Main function to run the entry visualizer."""
    print("🎨 Zone Fade Entry Point Visualizer")
    print("=" * 50)
    
    # Configuration
    config = VisualizationConfig()
    
    # Initialize visualizer
    visualizer = EntryVisualizer(config)
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals")
    
    try:
        # Load data
        symbols_data = visualizer.load_backtest_data(data_dir)
        entry_points_df = visualizer.load_entry_points(entry_points_file)
        
        if not symbols_data:
            print("❌ No backtesting data found")
            return
        
        # Generate visualizations
        results = visualizer.generate_entry_visuals(entry_points_df, symbols_data, output_dir)
        
        # Create HTML report
        html_file = visualizer.create_html_report(entry_points_df, output_dir)
        
        # Print results
        print(f"\n📊 Visualization Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        print(f"   📄 HTML Report: {html_file}")
        
        print(f"\n🎉 Visualization complete! Check {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()