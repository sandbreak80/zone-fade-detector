#!/usr/bin/env python3
"""
TradingView-Style Entry Point Visualizer

A clean, professional visualization tool using matplotlib + seaborn only.
Generates TradingView-style charts for manual verification of entry point detections.

Data Contract:
- Input: pandas DataFrame with columns: timestamp, open, high, low, close, volume
- Optional: vwap (precomputed)
- Entry point data: timestamp, price, zone_level, qrs_score, etc.
"""

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


class TradingViewVisualizer:
    """TradingView-style entry point visualizer using matplotlib + seaborn."""
    
    def __init__(self, 
                 hours_before: int = 2, 
                 hours_after: int = 2,
                 figsize: tuple = (14, 8),
                 dpi: int = 300):
        """Initialize the visualizer."""
        self.hours_before = hours_before
        self.hours_after = hours_after
        self.figsize = figsize
        self.dpi = dpi
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup matplotlib and seaborn styling."""
        # Set seaborn style for clean appearance
        sns.set_style("whitegrid")
        
        # Configure matplotlib for financial charts
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 9,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
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
    
    def bars_to_dataframe(self, bars: List[OHLCVBar]) -> pd.DataFrame:
        """Convert OHLCVBar list to pandas DataFrame."""
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        return pd.DataFrame(data)
    
    def compute_session_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Compute rolling session VWAP for the DataFrame."""
        # Calculate typical price (HLC/3)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative volume and cumulative price*volume
        cum_volume = df['volume'].cumsum()
        cum_pv = (typical_price * df['volume']).cumsum()
        
        # Calculate VWAP
        vwap = cum_pv / cum_volume
        
        return vwap
    
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
        """Create a TradingView-style entry point visualization."""
        
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
        
        # Convert bars to DataFrame
        df = self.bars_to_dataframe(bars)
        
        # Filter data to time window around entry
        start_time = entry_timestamp - timedelta(hours=self.hours_before)
        end_time = entry_timestamp + timedelta(hours=self.hours_after)
        
        chart_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
        
        if len(chart_df) == 0:
            print(f"   ❌ No data found in time window for {entry_id}")
            return False
        
        # Compute VWAP
        chart_df['vwap'] = self.compute_session_vwap(chart_df)
        
        # Find entry point index in filtered data
        chart_entry_idx = None
        for i, row in chart_df.iterrows():
            if abs((row['timestamp'] - entry_timestamp).total_seconds()) < 60:
                chart_entry_idx = i
                break
        
        if chart_entry_idx is None:
            chart_entry_idx = 0  # Fallback to first bar
        
        # Create chart
        fig = self._create_tradingview_chart(
            chart_df, 
            entry_timestamp, 
            entry_price, 
            zone_level, 
            qrs_score,
            symbol,
            entry_id,
            chart_entry_idx
        )
        
        # Save chart
        filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_tradingview.png"
        output_file = output_dir / filename
        
        fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
    
    def _create_tradingview_chart(self, df: pd.DataFrame, entry_timestamp: datetime, 
                                 entry_price: float, zone_level: float, qrs_score: float,
                                 symbol: str, entry_id: str, entry_idx: int) -> plt.Figure:
        """Create the actual TradingView-style chart."""
        
        # Create figure with subplots
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, 
            figsize=self.figsize,
            gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}
        )
        
        # Set background colors
        fig.patch.set_facecolor('white')
        ax_price.set_facecolor('white')
        ax_volume.set_facecolor('white')
        
        # === PRICE CHART ===
        self._plot_ohlc_candles(ax_price, df)
        self._plot_vwap_line(ax_price, df)
        self._plot_zone_level(ax_price, zone_level)
        self._plot_entry_point(ax_price, df, entry_idx, entry_price)
        self._configure_price_chart(ax_price, symbol, entry_id, qrs_score, entry_timestamp)
        
        # === VOLUME CHART ===
        self._plot_volume_bars(ax_volume, df)
        self._highlight_entry_volume(ax_volume, df, entry_idx)
        self._configure_volume_chart(ax_volume)
        
        # Clean layout
        plt.tight_layout()
        
        return fig
    
    def _plot_ohlc_candles(self, ax, df: pd.DataFrame):
        """Plot OHLC candlesticks with TradingView styling."""
        # TradingView colors
        up_color = '#26a69a'      # Teal green
        down_color = '#ef5350'    # Red
        up_edge = '#1b5e20'       # Dark teal
        down_edge = '#b71c1c'     # Dark red
        
        for i, row in df.iterrows():
            timestamp = row['timestamp']
            open_price = row['open']
            high = row['high']
            low = row['low']
            close = row['close']
            
            # Determine candle color
            is_up = close >= open_price
            color = up_color if is_up else down_color
            edge_color = up_edge if is_up else down_edge
            
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
            
            # Wicks
            ax.plot([timestamp, timestamp], [low, high], color='black', linewidth=0.8, alpha=0.7)
    
    def _plot_vwap_line(self, ax, df: pd.DataFrame):
        """Plot VWAP line with TradingView styling."""
        vwap_color = '#ff9800'  # Orange
        ax.plot(df['timestamp'], df['vwap'], color=vwap_color, linewidth=2, alpha=0.8)
    
    def _plot_zone_level(self, ax, zone_level: float):
        """Plot zone level line."""
        zone_color = '#9c27b0'  # Purple
        ax.axhline(y=zone_level, color=zone_color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    def _plot_entry_point(self, ax, df: pd.DataFrame, entry_idx: int, entry_price: float):
        """Plot entry point marker."""
        entry_color = '#1976d2'  # Blue
        
        # Get entry timestamp from the original entry point, not the filtered DataFrame
        # Find the closest timestamp in the filtered data
        entry_timestamp = None
        for i, row in df.iterrows():
            if i == entry_idx:
                entry_timestamp = row['timestamp']
                break
        
        if entry_timestamp is None:
            # Fallback to first timestamp if entry_idx is out of bounds
            entry_timestamp = df.iloc[0]['timestamp']
        
        # Entry line
        ax.axvline(x=entry_timestamp, color=entry_color, linewidth=1.5, alpha=0.8)
        
        # Entry point marker
        ax.scatter(entry_timestamp, entry_price, color=entry_color, s=100, 
                  marker='*', edgecolor='white', linewidth=1, zorder=10)
    
    def _plot_volume_bars(self, ax, df: pd.DataFrame):
        """Plot volume bars with price-based coloring."""
        up_color = '#26a69a'      # Teal green
        down_color = '#ef5350'    # Red
        volume_alpha = 0.6
        
        for i, row in df.iterrows():
            timestamp = row['timestamp']
            volume = row['volume']
            close = row['close']
            open_price = row['open']
            
            # Determine volume color based on price direction
            is_up = close >= open_price
            color = up_color if is_up else down_color
            
            # Plot volume bar
            ax.bar(timestamp, volume, width=0.3, color=color, alpha=volume_alpha, 
                  edgecolor=color, linewidth=0.3)
    
    def _highlight_entry_volume(self, ax, df: pd.DataFrame, entry_idx: int):
        """Highlight entry point volume."""
        entry_color = '#1976d2'  # Blue
        
        # Find the entry timestamp and volume safely
        entry_timestamp = None
        entry_volume = None
        for i, row in df.iterrows():
            if i == entry_idx:
                entry_timestamp = row['timestamp']
                entry_volume = row['volume']
                break
        
        if entry_timestamp is not None and entry_volume is not None:
            # Highlight entry volume bar
            ax.bar(entry_timestamp, entry_volume, width=0.3, color=entry_color, 
                  alpha=0.8, edgecolor='white', linewidth=1)
    
    def _configure_price_chart(self, ax, symbol: str, entry_id: str, qrs_score: float, entry_timestamp: datetime):
        """Configure price chart appearance."""
        title = f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime('%Y-%m-%d %H:%M')}"
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_ylabel('Price ($)', fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _configure_volume_chart(self, ax):
        """Configure volume chart appearance."""
        ax.set_ylabel('Volume', fontweight='bold')
        ax.set_xlabel('Time', fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def generate_entry_visuals(
        self, 
        entry_points_df: pd.DataFrame, 
        symbols_data: Dict[str, List[OHLCVBar]], 
        output_dir: Path
    ) -> Dict[str, int]:
        """Generate TradingView-style visualizations for all entry points."""
        print(f"\n🎯 Generating TradingView-style visualizations for {len(entry_points_df)} entry points...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        results = {'successful': 0, 'failed': 0, 'no_data': 0}
        
        # Process each entry point
        for idx, entry_point in entry_points_df.iterrows():
            symbol = entry_point['symbol']
            entry_id = entry_point['entry_id']
            
            print(f"   📊 Creating TradingView visualization for {entry_id} ({symbol})...")
            
            if symbol in symbols_data:
                success = self.create_entry_visualization(
                    entry_point, 
                    symbols_data[symbol], 
                    symbol, 
                    output_dir
                )
                
                if success:
                    results['successful'] += 1
                    print(f"     ✅ Created TradingView visualization for {entry_id}")
                else:
                    results['failed'] += 1
                    print(f"     ❌ Failed to create TradingView visualization for {entry_id}")
            else:
                results['no_data'] += 1
                print(f"     ❌ No data available for {symbol}")
        
        return results


def main():
    """Main function to run the TradingView visualizer."""
    print("📊 TradingView-Style Entry Point Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = TradingViewVisualizer(
        hours_before=2,
        hours_after=2,
        figsize=(14, 8),
        dpi=300
    )
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals_tradingview")
    
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
        
        # Print results
        print(f"\n📊 TradingView Visualization Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        
        print(f"\n🎉 TradingView visualization complete! Check {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()