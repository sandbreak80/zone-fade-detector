#!/usr/bin/env python3
"""
Simple Plotly Zone Fade Entry Point Visualizer

A simplified version that focuses on working functionality.
"""

import sys
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


class SimplePlotlyVisualizer:
    """Simple visualization using Plotly."""
    
    def __init__(self):
        """Initialize the simple visualizer."""
        self.colors = {
            'up': '#00C853',
            'down': '#FF1744',
            'vwap': '#FF9800',
            'entry': '#9C27B0'
        }
    
    def load_data(self, data_dir: Path) -> Dict[str, List[OHLCVBar]]:
        """Load backtesting data from pickle files."""
        print("📊 Loading backtesting data...")
        
        data = {}
        symbols = ['SPY', 'QQQ', 'IWM']
        
        for symbol in symbols:
            file_path = data_dir / "2024" / f"{symbol}_2024.pkl"
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        data[symbol] = pickle.load(f)
                    print(f"   ✅ {symbol}: {len(data[symbol]):,} bars")
                except Exception as e:
                    print(f"   ❌ {symbol}: Error loading data - {e}")
            else:
                print(f"   ❌ {symbol}: File not found - {file_path}")
        
        return data
    
    def load_entry_points(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Load entry points from CSV file."""
        print("📋 Loading entry points...")
        
        file_path = Path("results/2024/efficient/zone_fade_entry_points_2024_efficient.csv")
        if not file_path.exists():
            print(f"   ❌ Entry points file not found: {file_path}")
            return []
        
        try:
            df = pd.read_csv(file_path)
            entry_points = df.to_dict('records')
            print(f"   ✅ Loaded {len(entry_points)} entry points from CSV")
            return entry_points
        except Exception as e:
            print(f"   ❌ Error loading entry points: {e}")
            return []
    
    def create_simple_chart(self, df: pd.DataFrame, entry_idx: int, 
                           symbol: str, entry_id: str, qrs_score: float) -> go.Figure:
        """Create a simple Plotly candlestick chart."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f}", "Volume")
        )
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color=self.colors['up'],
            decreasing_line_color=self.colors['down'],
            increasing_fillcolor=self.colors['up'],
            decreasing_fillcolor=self.colors['down']
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add VWAP line
        vwap_trace = go.Scatter(
            x=df['timestamp'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color=self.colors['vwap'], width=2)
        )
        
        fig.add_trace(vwap_trace, row=1, col=1)
        
        # Add entry point marker
        entry_timestamp = df.iloc[entry_idx]['timestamp']
        entry_price = df.iloc[entry_idx]['close']
        
        # Entry line
        fig.add_vline(
            x=entry_timestamp,
            line=dict(color=self.colors['entry'], width=2, dash="dash"),
            annotation_text="Entry",
            row=1, col=1
        )
        
        # Entry point marker
        entry_marker = go.Scatter(
            x=[entry_timestamp],
            y=[entry_price],
            mode='markers',
            name='Entry Point',
            marker=dict(
                color=self.colors['entry'],
                size=12,
                symbol='star'
            )
        )
        
        fig.add_trace(entry_marker, row=1, col=1)
        
        # Add volume bars
        volume_colors = []
        for i, row in df.iterrows():
            if row['close'] >= row['open']:
                volume_colors.append(self.colors['up'])
            else:
                volume_colors.append(self.colors['down'])
        
        volume_trace = go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7
        )
        
        fig.add_trace(volume_trace, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f}",
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_visualizations(self, data: Dict[str, List[OHLCVBar]], 
                            entry_points: List[Dict[str, Any]], 
                            output_dir: Path) -> bool:
        """Create simple visualizations for entry points."""
        print("🎨 Creating simple Plotly visualizations...")
        
        # Create output directory
        plotly_dir = output_dir / "simple_plotly_charts"
        plotly_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = len(entry_points)
        
        for i, entry in enumerate(entry_points[:5]):  # Limit to first 5 for testing
            try:
                symbol = entry['symbol']
                entry_id = entry['entry_id']
                qrs_score = entry['qrs_score']
                entry_timestamp = pd.to_datetime(entry['timestamp'])
                
                print(f"   📊 Creating visualization for {entry_id} ({symbol})...")
                
                # Get data for this symbol
                if symbol not in data:
                    print(f"     ❌ No data available for {symbol}")
                    continue
                
                symbol_data = data[symbol]
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                } for bar in symbol_data])
                
                # Convert timestamps to pandas datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Calculate VWAP
                df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
                
                # Find entry point in data (simplified approach)
                entry_idx = None
                for idx, row in df.iterrows():
                    if abs((row['timestamp'] - entry_timestamp).total_seconds()) < 300:  # Within 5 minutes
                        entry_idx = idx
                        break
                
                if entry_idx is None:
                    print(f"     ❌ Entry point not found in data for {entry_id}")
                    continue
                
                # Create a simple time window around entry point
                start_idx = max(0, entry_idx - 50)  # 50 bars before
                end_idx = min(len(df), entry_idx + 50)  # 50 bars after
                window_df = df.iloc[start_idx:end_idx].copy()
                
                if len(window_df) == 0:
                    print(f"     ❌ No data in window for {entry_id}")
                    continue
                
                # Find new entry index in filtered data
                new_entry_idx = None
                for idx, row in window_df.iterrows():
                    if abs((row['timestamp'] - entry_timestamp).total_seconds()) < 300:
                        new_entry_idx = list(window_df.index).index(idx)
                        break
                
                if new_entry_idx is None:
                    print(f"     ❌ Entry point not found in filtered data for {entry_id}")
                    continue
                
                # Create Plotly chart
                fig = self.create_simple_chart(
                    window_df, new_entry_idx, symbol, entry_id, qrs_score
                )
                
                # Save as HTML
                filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_simple_chart.html"
                output_file = plotly_dir / filename
                fig.write_html(output_file)
                
                print(f"     ✅ Created visualization for {entry_id}")
                success_count += 1
                
            except Exception as e:
                print(f"     ❌ Error creating visualization for {entry.get('entry_id', 'unknown')}: {e}")
                continue
        
        print(f"📊 Simple visualization results:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {total_count - success_count}")
        print(f"   📁 Output Directory: {plotly_dir}")
        
        return success_count > 0
    
    def run_visualization(self, data_dir: Path, output_dir: Path) -> bool:
        """Run the simple visualization process."""
        print("🚀 Simple Plotly Zone Fade Visualizer")
        print("=" * 50)
        
        # Load data
        data = self.load_data(data_dir)
        if not data:
            print("❌ No data loaded. Exiting.")
            return False
        
        entry_points = self.load_entry_points(data_dir)
        if not entry_points:
            print("❌ No entry points loaded. Exiting.")
            return False
        
        # Create visualizations
        success = self.create_visualizations(data, entry_points, output_dir)
        
        if success:
            print("🎉 Simple visualization complete!")
        else:
            print("❌ Simple visualization failed!")
        
        return success


def main():
    """Main function for command-line usage."""
    visualizer = SimplePlotlyVisualizer()
    
    # Set up paths
    data_dir = Path('data')
    output_dir = Path('outputs/simple_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = visualizer.run_visualization(data_dir, output_dir)
    
    if success:
        print(f"\n📁 Visualizations saved to: {output_dir}")
        print("🌐 Open the HTML files in your browser for interactive charts!")
    else:
        print("\n❌ Visualization failed. Check the logs above for details.")


if __name__ == "__main__":
    main()