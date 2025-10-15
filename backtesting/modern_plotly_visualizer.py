#!/usr/bin/env python3
"""
Modern Plotly + Seaborn Zone Fade Entry Point Visualizer

A cutting-edge visualization tool using Plotly for interactive candlestick charts
and Seaborn for statistical analysis of trade entry points.

Features:
- Interactive Plotly candlestick charts with OHLCV + VWAP + Volume
- Seaborn statistical analysis and box plots
- QuantConnect-style professional aesthetics
- Real-time zoom, pan, and hover functionality
- Export to HTML for sharing and analysis

Author: Zone Fade Detector Team
Version: 3.0.0
"""

import sys
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import json
from dataclasses import dataclass
import plotly.io as pio

warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class ModernVisualizationConfig:
    """Configuration for modern visualization settings."""
    # Time window settings
    hours_before: int = 2
    hours_after: int = 2
    
    # Chart styling
    figure_width: int = 1200
    figure_height: int = 800
    dpi: int = 300
    
    # Colors (QuantConnect inspired)
    price_up_color: str = '#00C853'      # Green
    price_down_color: str = '#FF1744'    # Red
    vwap_color: str = '#FF9800'          # Orange
    volume_color: str = '#2196F3'        # Blue
    entry_color: str = '#9C27B0'         # Purple
    zone_color: str = '#607D8B'          # Blue Grey
    
    # Styling
    background_color: str = '#FFFFFF'
    grid_color: str = '#E0E0E0'
    text_color: str = '#212121'
    
    # Fonts
    title_fontsize: int = 16
    label_fontsize: int = 12
    annotation_fontsize: int = 10
    
    # Seaborn styling
    seaborn_style: str = 'whitegrid'
    seaborn_palette: str = 'Set2'


class ModernPlotlyVisualizer:
    """Modern visualization using Plotly and Seaborn."""
    
    def __init__(self, config: ModernVisualizationConfig = None):
        """Initialize the modern visualizer."""
        self.config = config or ModernVisualizationConfig()
        self._setup_plotly_theme()
        self._setup_seaborn_style()
    
    def _setup_plotly_theme(self):
        """Setup Plotly theme for professional appearance."""
        pio.templates.default = "plotly_white"
        
        # Custom theme
        custom_theme = {
            'layout': {
                'paper_bgcolor': self.config.background_color,
                'plot_bgcolor': self.config.background_color,
                'font': {
                    'family': 'Arial, sans-serif',
                    'size': self.config.label_fontsize,
                    'color': self.config.text_color
                },
                'xaxis': {
                    'gridcolor': self.config.grid_color,
                    'showgrid': True,
                    'gridwidth': 1
                },
                'yaxis': {
                    'gridcolor': self.config.grid_color,
                    'showgrid': True,
                    'gridwidth': 1
                }
            }
        }
        
        pio.templates["custom"] = custom_theme
        pio.templates.default = "custom"
    
    def _setup_seaborn_style(self):
        """Setup Seaborn styling."""
        sns.set_style(self.config.seaborn_style)
        sns.set_palette(self.config.seaborn_palette)
        plt.rcParams.update({
            'font.size': self.config.label_fontsize,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.annotation_fontsize,
            'ytick.labelsize': self.config.annotation_fontsize,
        })
    
    def load_backtest_data(self, data_dir: Path) -> Dict[str, List[OHLCVBar]]:
        """Load backtesting data from pickle files."""
        print("📊 Loading backtesting data...")
        
        data = {}
        symbols = ['SPY', 'QQQ', 'IWM']
        
        for symbol in symbols:
            # Try multiple possible file locations
            possible_paths = [
                data_dir / f"{symbol}_2024_data.pkl",
                data_dir / f"{symbol}_2024.pkl",
                data_dir / "2024" / f"{symbol}_2024.pkl"
            ]
            
            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
            
            if file_path:
                try:
                    with open(file_path, 'rb') as f:
                        data[symbol] = pickle.load(f)
                    print(f"   ✅ {symbol}: {len(data[symbol]):,} bars")
                except Exception as e:
                    print(f"   ❌ {symbol}: Error loading data - {e}")
            else:
                print(f"   ❌ {symbol}: File not found in any expected location")
        
        return data
    
    def load_entry_points(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Load entry points from JSON or CSV file."""
        print("📋 Loading entry points...")
        
        # Try multiple possible file locations and formats
        possible_paths = [
            data_dir / "entry_points_2024.json",
            data_dir / "2024" / "entry_points_2024.json",
            data_dir / "results" / "2024" / "efficient" / "zone_fade_entry_points_2024_efficient.csv",
            data_dir / "results" / "summaries" / "zone_fade_entry_points_2024_efficient.csv",
            Path("results/2024/efficient/zone_fade_entry_points_2024_efficient.csv"),
            Path("results/summaries/zone_fade_entry_points_2024_efficient.csv")
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            print(f"   ❌ Entry points file not found in any expected location")
            return []
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    entry_points = json.load(f)
                print(f"   ✅ Loaded {len(entry_points)} entry points from JSON")
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                # Convert DataFrame to list of dictionaries
                entry_points = df.to_dict('records')
                print(f"   ✅ Loaded {len(entry_points)} entry points from CSV")
            else:
                print(f"   ❌ Unsupported file format: {file_path.suffix}")
                return []
            
            return entry_points
        except Exception as e:
            print(f"   ❌ Error loading entry points: {e}")
            return []
    
    def create_plotly_candlestick_chart(self, df: pd.DataFrame, entry_idx: int, 
                                      symbol: str, entry_id: str, qrs_score: float) -> go.Figure:
        """Create interactive Plotly candlestick chart with OHLCV + VWAP + Volume."""
        
        # Create subplots with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f}", "Volume"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color=self.config.price_up_color,
            decreasing_line_color=self.config.price_down_color,
            increasing_fillcolor=self.config.price_up_color,
            decreasing_fillcolor=self.config.price_down_color,
            line=dict(width=1)
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add VWAP line
        vwap_trace = go.Scatter(
            x=df['timestamp'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color=self.config.vwap_color, width=2),
            hovertemplate="<b>%{x}</b><br>VWAP: $%{y:.2f}<extra></extra>"
        )
        
        fig.add_trace(vwap_trace, row=1, col=1)
        
        # Add entry point marker
        entry_timestamp = df.iloc[entry_idx]['timestamp']
        entry_price = df.iloc[entry_idx]['close']
        
        # Entry line
        fig.add_vline(
            x=entry_timestamp,
            line=dict(color=self.config.entry_color, width=2, dash="dash"),
            annotation_text="Entry",
            annotation_position="top",
            row=1, col=1
        )
        
        # Entry point marker
        entry_marker = go.Scatter(
            x=[entry_timestamp],
            y=[entry_price],
            mode='markers',
            name='Entry Point',
            marker=dict(
                color=self.config.entry_color,
                size=12,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            hovertemplate="<b>Entry Point</b><br>" +
                         "Time: %{x}<br>" +
                         "Price: $%{y:.2f}<extra></extra>"
        )
        
        fig.add_trace(entry_marker, row=1, col=1)
        
        # Add volume bars
        volume_colors = []
        for i, row in df.iterrows():
            if row['close'] >= row['open']:
                volume_colors.append(self.config.price_up_color)
            else:
                volume_colors.append(self.config.price_down_color)
        
        volume_trace = go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>"
        )
        
        fig.add_trace(volume_trace, row=2, col=1)
        
        # Highlight entry volume
        entry_volume = df.iloc[entry_idx]['volume']
        entry_volume_trace = go.Bar(
            x=[entry_timestamp],
            y=[entry_volume],
            name='Entry Volume',
            marker_color=self.config.entry_color,
            opacity=0.9,
            hovertemplate="<b>Entry Volume</b><br>Volume: %{y:,.0f}<extra></extra>"
        )
        
        fig.add_trace(entry_volume_trace, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f} | {entry_timestamp.strftime('%Y-%m-%d %H:%M')}",
            title_font_size=self.config.title_fontsize,
            width=self.config.figure_width,
            height=self.config.figure_height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Format x-axis
        fig.update_xaxes(
            tickformat="%H:%M",
            tickmode='auto',
            nticks=10
        )
        
        return fig
    
    def create_seaborn_analysis(self, entry_points: List[Dict[str, Any]], 
                              output_dir: Path) -> None:
        """Create Seaborn statistical analysis charts."""
        print("📊 Creating Seaborn analysis charts...")
        
        if not entry_points:
            print("   ❌ No entry points available for analysis")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(entry_points)
        
        # Create analysis directory
        analysis_dir = output_dir / "seaborn_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # 1. QRS Score Distribution by Symbol
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='symbol', y='qrs_score', palette='Set2')
        plt.title('QRS Score Distribution by Symbol', fontsize=16, fontweight='bold')
        plt.xlabel('Symbol', fontsize=12)
        plt.ylabel('QRS Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'qrs_score_distribution.png', dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        # 2. Entry Time Analysis
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        plt.figure(figsize=(14, 6))
        sns.countplot(data=df, x='hour', palette='viridis')
        plt.title('Entry Points by Hour of Day', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Entries', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'entry_time_analysis.png', dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        # 3. QRS Score vs Symbol Heatmap
        pivot_data = df.pivot_table(values='qrs_score', index='symbol', columns='hour', aggfunc='mean')
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Average QRS Score'})
        plt.title('Average QRS Score by Symbol and Hour', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Symbol', fontsize=12)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'qrs_heatmap.png', dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        # 4. Entry Point Quality Analysis
        df['quality_category'] = pd.cut(df['qrs_score'], 
                                      bins=[0, 5, 7, 9, 10], 
                                      labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        plt.figure(figsize=(12, 8))
        quality_counts = df['quality_category'].value_counts()
        colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1']
        plt.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Entry Point Quality Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(analysis_dir / 'quality_distribution.png', dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Seaborn analysis charts saved to {analysis_dir}")
    
    def create_visualization(self, data: Dict[str, List[OHLCVBar]], 
                           entry_points: List[Dict[str, Any]], 
                           output_dir: Path) -> bool:
        """Create modern visualizations for entry points."""
        print("🎨 Creating modern Plotly visualizations...")
        
        # Create output directory
        plotly_dir = output_dir / "plotly_charts"
        plotly_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = len(entry_points)
        
        for i, entry in enumerate(entry_points):
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
                
                
                # Find entry point in data
                entry_idx = None
                for idx, row in df.iterrows():
                    if abs((row['timestamp'] - entry_timestamp).total_seconds()) < 60:  # Within 1 minute
                        entry_idx = idx
                        break
                
                if entry_idx is None:
                    print(f"     ❌ Entry point not found in data for {entry_id}")
                    continue
                
                # Create time window using pandas Timedelta
                start_time = entry_timestamp - pd.Timedelta(hours=self.config.hours_before)
                end_time = entry_timestamp + pd.Timedelta(hours=self.config.hours_after)
                
                # Filter data to time window
                mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                window_df = df[mask].copy()
                
                if len(window_df) == 0:
                    print(f"     ❌ No data in time window for {entry_id}")
                    continue
                
                # Find new entry index in filtered data
                new_entry_idx = None
                for idx, row in window_df.iterrows():
                    if abs((row['timestamp'] - entry_timestamp).total_seconds()) < 60:
                        new_entry_idx = list(window_df.index).index(idx)
                        break
                
                if new_entry_idx is None:
                    print(f"     ❌ Entry point not found in filtered data for {entry_id}")
                    continue
                
                # Create Plotly chart
                fig = self.create_plotly_candlestick_chart(
                    window_df, new_entry_idx, symbol, entry_id, qrs_score
                )
                
                # Save as HTML
                filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_modern_chart.html"
                output_file = plotly_dir / filename
                fig.write_html(output_file)
                
                # Also save as PNG for static viewing
                png_filename = f"{symbol}_{entry_timestamp.strftime('%Y%m%d_%H%M')}_modern_chart.png"
                png_file = plotly_dir / png_filename
                fig.write_image(png_file, width=self.config.figure_width, height=self.config.figure_height)
                
                print(f"     ✅ Created visualization for {entry_id}")
                success_count += 1
                
            except Exception as e:
                print(f"     ❌ Error creating visualization for {entry.get('entry_id', 'unknown')}: {e}")
                continue
        
        print(f"📊 Modern visualization results:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {total_count - success_count}")
        print(f"   📁 Output Directory: {plotly_dir}")
        
        return success_count > 0
    
    def run_visualization(self, data_dir: Path, output_dir: Path, 
                         limit: Optional[int] = None) -> bool:
        """Run the complete modern visualization process."""
        print("🚀 Modern Plotly + Seaborn Zone Fade Visualizer")
        print("=" * 60)
        
        # Load data
        data = self.load_backtest_data(data_dir)
        if not data:
            print("❌ No data loaded. Exiting.")
            return False
        
        entry_points = self.load_entry_points(data_dir)
        if not entry_points:
            print("❌ No entry points loaded. Exiting.")
            return False
        
        # Limit entry points if specified
        if limit:
            entry_points = entry_points[:limit]
            print(f"🎯 Processing {len(entry_points)} entry points (limited)")
        
        # Create visualizations
        success = self.create_visualization(data, entry_points, output_dir)
        
        # Create Seaborn analysis
        self.create_seaborn_analysis(entry_points, output_dir)
        
        if success:
            print("🎉 Modern visualization complete!")
        else:
            print("❌ Modern visualization failed!")
        
        return success


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Modern Plotly + Seaborn Zone Fade Visualizer')
    parser.add_argument('--data-dir', type=str, default='data/backtesting', 
                       help='Directory containing backtesting data')
    parser.add_argument('--output-dir', type=str, default='outputs/modern_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--limit', type=int, help='Limit number of entry points to process')
    parser.add_argument('--hours-before', type=int, default=2, 
                       help='Hours before entry point to show')
    parser.add_argument('--hours-after', type=int, default=2,
                       help='Hours after entry point to show')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ModernVisualizationConfig(
        hours_before=args.hours_before,
        hours_after=args.hours_after
    )
    
    # Create visualizer
    visualizer = ModernPlotlyVisualizer(config)
    
    # Run visualization
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = visualizer.run_visualization(data_dir, output_dir, args.limit)
    
    if success:
        print(f"\n📁 Visualizations saved to: {output_dir}")
        print("🌐 Open the HTML files in your browser for interactive charts!")
    else:
        print("\n❌ Visualization failed. Check the logs above for details.")


if __name__ == "__main__":
    main()