#!/usr/bin/env python3
"""
Clean Zone Fade Entry Point Visualizer

A clean, professional visualization tool that generates TradingView/Yahoo Finance style
charts for manual verification of entry point detections.

Features:
- Clean candlestick charts with minimal overlays
- Professional color scheme (green/red for price, blue for highlights)
- Properly scaled volume subplot
- Clear entry/exit markers
- Minimal, essential annotations only
- Publication-quality output

Author: Zone Fade Detector Team
Version: 2.0.0
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
class CleanVisualizationConfig:
    """Configuration for clean visualization settings."""
    # Time window settings
    hours_before: int = 2
    hours_after: int = 2
    
    # Chart styling
    figure_size: Tuple[int, int] = (14, 8)
    dpi: int = 300
    
    # Colors - Professional financial palette
    price_up_color: str = '#26a69a'      # Teal green for up candles
    price_down_color: str = '#ef5350'    # Red for down candles
    price_up_edge: str = '#00695c'       # Darker teal for edges
    price_down_edge: str = '#c62828'     # Darker red for edges
    vwap_color: str = '#ff9800'          # Orange for VWAP
    entry_color: str = '#1976d2'         # Blue for entry markers
    zone_color: str = '#9c27b0'          # Purple for zone level
    volume_color: str = '#90a4ae'        # Gray for volume
    volume_alpha: float = 0.6
    
    # Font settings
    title_fontsize: int = 14
    label_fontsize: int = 11
    tick_fontsize: int = 9
    legend_fontsize: int = 9
    
    # Layout
    show_grid: bool = True
    grid_alpha: float = 0.3
    background_color: str = 'white'


class CleanEntryVisualizer:
    """Clean, professional entry point visualizer."""
    
    def __init__(self, config: Optional[CleanVisualizationConfig] = None):
        """Initialize the visualizer with configuration."""
        self.config = config or CleanVisualizationConfig()
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for clean, professional output."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': self.config.tick_fontsize,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.tick_fontsize,
            'ytick.labelsize': self.config.tick_fontsize,
            'legend.fontsize': self.config.legend_fontsize,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': self.config.grid_alpha,
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
    
    def create_clean_visualization(
        self,
        entry_point: pd.Series,
        bars: List[OHLCVBar],
        symbol: str,
        output_dir: Path
    ) -> bool:
        """Create a clean, professional entry point visualization."""
        
        # Extract entry point data
        entry_timestamp = entry_point['timestamp']
        entry_price = entry_point['price']
        zone_level = entry_point['zone_level']
        qrs_score = entry_point['qrs_score']
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
        
        # Create the figure with clean layout
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, 
            figsize=self.config.figure_size,
            gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}
        )
        
        # Set background color
        fig.patch.set_facecolor(self.config.background_color)
        ax_price.set_facecolor(self.config.background_color)
        ax_volume.set_facecolor(self.config.background_color)
        
        # === CLEAN PRICE CHART ===
        self._plot_clean_candlesticks(ax_price, timestamps, opens, highs, lows, closes)
        self._plot_vwap_line(ax_price, timestamps, vwap_values)
        self._plot_zone_level(ax_price, zone_level)
        self._plot_entry_marker(ax_price, timestamps[vis_entry_idx], entry_price)
        self._configure_price_chart(ax_price, symbol, entry_id, qrs_score, entry_timestamp)
        
        # === CLEAN VOLUME CHART ===
        self._plot_clean_volume(ax_volume, timestamps, volumes, closes, opens, vis_entry_idx)
        self._configure_volume_chart(ax_volume)
        
        # === MINIMAL ANNOTATIONS ===
        self._add_minimal_annotations(ax_price, entry_point, symbol)
        
        # Clean layout
        plt.tight_layout()
        
        # Save the plot
        filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_entry_visual.png"
        output_file = output_dir / filename
        
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor=self.config.background_color, edgecolor='none')
        plt.close()
        
        return True
    
    def _plot_clean_candlesticks(self, ax, timestamps, opens, highs, lows, closes):
        """Plot clean candlestick chart with professional styling."""
        for i, (timestamp, open_price, high, low, close) in enumerate(zip(timestamps, opens, highs, lows, closes)):
            is_green = close >= open_price
            color = self.config.price_up_color if is_green else self.config.price_down_color
            edge_color = self.config.price_up_edge if is_green else self.config.price_down_edge
            
            # Body
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            
            if body_height > 0:
                # Draw body as filled rectangle
                ax.bar(timestamp, body_height, bottom=body_bottom, width=0.3, 
                      color=color, alpha=0.8, edgecolor=edge_color, linewidth=0.5)
            else:
                # Doji - draw horizontal line
                ax.plot([timestamp, timestamp], [open_price, open_price], 
                       color=edge_color, linewidth=1.5)
            
            # Wicks - clean lines
            ax.plot([timestamp, timestamp], [low, high], color='black', linewidth=0.8, alpha=0.7)
    
    def _plot_vwap_line(self, ax, timestamps, vwap_values):
        """Plot VWAP line with clean styling."""
        if vwap_values:
            ax.plot(timestamps, vwap_values, color=self.config.vwap_color, 
                   linewidth=2, label='VWAP', alpha=0.8)
    
    def _plot_zone_level(self, ax, zone_level):
        """Plot zone level with subtle styling."""
        ax.axhline(y=zone_level, color=self.config.zone_color, 
                  linestyle='--', linewidth=1.5, alpha=0.7, label=f'Zone: ${zone_level:.2f}')
    
    def _plot_entry_marker(self, ax, entry_timestamp, entry_price):
        """Plot entry point marker with clean styling."""
        # Entry line
        ax.axvline(x=entry_timestamp, color=self.config.entry_color, 
                  linestyle='-', linewidth=1.5, alpha=0.8)
        
        # Entry point marker
        ax.scatter(entry_timestamp, entry_price, color=self.config.entry_color, 
                  s=100, marker='*', edgecolor='white', linewidth=1, zorder=10)
    
    def _configure_price_chart(self, ax, symbol, entry_id, qrs_score, entry_timestamp):
        """Configure price chart with clean styling."""
        ax.set_title(f'{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=self.config.title_fontsize, fontweight='bold', pad=15)
        ax.set_ylabel('Price ($)', fontsize=self.config.label_fontsize, fontweight='bold')
        
        # Clean legend
        ax.legend(loc='upper left', fontsize=self.config.legend_fontsize, 
                 framealpha=0.9, fancybox=False, shadow=False, edgecolor='gray')
        
        # Clean grid
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, linestyle='-', linewidth=0.5)
        
        # Format x-axis cleanly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))  # Every hour
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=self.config.tick_fontsize)
    
    def _plot_clean_volume(self, ax, timestamps, volumes, closes, opens, entry_idx):
        """Plot clean volume bars with subtle styling."""
        # Calculate volume colors based on price direction
        volume_colors = []
        for close, open_price in zip(closes, opens):
            if close >= open_price:
                volume_colors.append(self.config.price_up_color)
            else:
                volume_colors.append(self.config.price_down_color)
        
        # Plot volume bars with clean styling
        for i, (timestamp, volume, color) in enumerate(zip(timestamps, volumes, volume_colors)):
            ax.bar(timestamp, volume, width=0.3, color=color, alpha=self.config.volume_alpha, 
                  edgecolor=color, linewidth=0.3)
        
        # Highlight entry point volume subtly
        ax.bar(timestamps[entry_idx], volumes[entry_idx], width=0.3, 
              color=self.config.entry_color, alpha=0.8, edgecolor='white', linewidth=1)
    
    def _configure_volume_chart(self, ax):
        """Configure volume chart with clean styling."""
        ax.set_ylabel('Volume', fontsize=self.config.label_fontsize, fontweight='bold')
        ax.set_xlabel('Time', fontsize=self.config.label_fontsize, fontweight='bold')
        
        # Clean grid
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, linestyle='-', linewidth=0.5)
        
        # Format x-axis cleanly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))  # Every hour
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=self.config.tick_fontsize)
    
    def _add_minimal_annotations(self, ax, entry_point, symbol):
        """Add minimal, essential annotations only."""
        # Small info box in top right corner
        info_text = f"""Entry: ${entry_point['price']:.2f}
Zone: ${entry_point['zone_level']:.2f}
QRS: {entry_point['qrs_score']:.1f}
Type: {entry_point['zone_type']}"""
        
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', linewidth=0.5, alpha=0.9))
    
    def generate_entry_visuals(
        self, 
        entry_points_df: pd.DataFrame, 
        symbols_data: Dict[str, List[OHLCVBar]], 
        output_dir: Path
    ) -> Dict[str, int]:
        """Generate clean visualizations for all entry points."""
        print(f"\n🎯 Generating clean visualizations for {len(entry_points_df)} entry points...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        results = {'successful': 0, 'failed': 0, 'no_data': 0}
        
        # Process each entry point
        for idx, entry_point in entry_points_df.iterrows():
            symbol = entry_point['symbol']
            entry_id = entry_point['entry_id']
            
            print(f"   📊 Creating clean visualization for {entry_id} ({symbol})...")
            
            if symbol in symbols_data:
                success = self.create_clean_visualization(
                    entry_point, 
                    symbols_data[symbol], 
                    symbol, 
                    output_dir
                )
                
                if success:
                    results['successful'] += 1
                    print(f"     ✅ Created clean visualization for {entry_id}")
                else:
                    results['failed'] += 1
                    print(f"     ❌ Failed to create clean visualization for {entry_id}")
            else:
                results['no_data'] += 1
                print(f"     ❌ No data available for {symbol}")
        
        return results
    
    def create_html_report(self, entry_points_df: pd.DataFrame, output_dir: Path) -> str:
        """Create a clean HTML report combining all visualizations."""
        print("📄 Creating clean HTML report...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zone Fade Entry Points - Clean Visual Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #1976d2 0%, #9c27b0 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
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
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #1976d2;
        }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .visualization-item {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .visualization-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .entry-info {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
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
        .qrs-high {{ color: #26a69a; font-weight: bold; }}
        .qrs-good {{ color: #ff9800; font-weight: bold; }}
        .qrs-moderate {{ color: #ef5350; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Zone Fade Entry Points - Clean Visual Analysis</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Professional stock chart visualizations for manual verification</p>
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
        html_file = output_dir / "clean_entry_points_visual_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"   ✅ Clean HTML report created: {html_file}")
        return str(html_file)


def main():
    """Main function to run the clean entry visualizer."""
    print("🎨 Clean Zone Fade Entry Point Visualizer")
    print("=" * 50)
    
    # Configuration
    config = CleanVisualizationConfig()
    
    # Initialize visualizer
    visualizer = CleanEntryVisualizer(config)
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals_clean")
    
    try:
        # Load data
        symbols_data = visualizer.load_backtest_data(data_dir)
        entry_points_df = visualizer.load_entry_points(entry_points_file)
        
        if not symbols_data:
            print("❌ No backtesting data found")
            return
        
        # Test with first 5 entries
        test_entries = entry_points_df.head(5)
        print(f"🎯 Testing with {len(test_entries)} entry points...")
        
        # Generate visualizations
        results = visualizer.generate_entry_visuals(test_entries, symbols_data, output_dir)
        
        # Create HTML report
        html_file = visualizer.create_html_report(test_entries, output_dir)
        
        # Print results
        print(f"\n📊 Clean Visualization Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        print(f"   📄 HTML Report: {html_file}")
        
        print(f"\n🎉 Clean visualization complete! Check {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()