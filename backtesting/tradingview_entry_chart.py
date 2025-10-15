#!/usr/bin/env python3
"""
TradingView-Style Entry Point Chart

A clean, professional intraday chart for visualizing entry points using matplotlib + seaborn.
Plots OHLC candles, VWAP line, and Volume bars with entry point highlighting.

Data Contract:
- Input: pandas DataFrame with columns: timestamp, open, high, low, close, volume
- Optional: vwap (precomputed)
- Entry point data: timestamp, price, zone_level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute rolling session VWAP for the DataFrame.
    
    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    Returns:
        Series with VWAP values
    """
    # Calculate typical price (HLC/3)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate cumulative volume and cumulative price*volume
    cum_volume = df['volume'].cumsum()
    cum_pv = (typical_price * df['volume']).cumsum()
    
    # Calculate VWAP
    vwap = cum_pv / cum_volume
    
    return vwap


def plot_tradingview_entry_chart(
    df: pd.DataFrame,
    entry_timestamp: datetime,
    entry_price: float,
    zone_level: float,
    title: str = "Entry Point Chart",
    hours_before: int = 2,
    hours_after: int = 2,
    figsize: tuple = (14, 8),
    dpi: int = 300
) -> plt.Figure:
    """
    Create a clean TradingView-style chart with entry point highlighting.
    
    Args:
        df: DataFrame with OHLCV data
        entry_timestamp: Entry point timestamp
        entry_price: Entry point price
        zone_level: Zone level price
        title: Chart title
        hours_before: Hours before entry to show
        hours_after: Hours after entry to show
        figsize: Figure size (width, height)
        dpi: DPI for output
    
    Returns:
        matplotlib Figure object
    """
    
    # Set seaborn style for clean appearance
    sns.set_style("whitegrid")
    
    # Configure matplotlib for financial charts
    plt.rcParams.update({
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
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
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter data to time window around entry
    start_time = entry_timestamp - timedelta(hours=hours_before)
    end_time = entry_timestamp + timedelta(hours=hours_after)
    
    chart_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    if len(chart_df) == 0:
        raise ValueError("No data found in the specified time window")
    
    # Compute VWAP if not provided
    if 'vwap' not in chart_df.columns:
        chart_df['vwap'] = compute_session_vwap(chart_df)
    
    # Find entry point index in filtered data
    entry_idx = None
    for i, row in chart_df.iterrows():
        if abs((row['timestamp'] - entry_timestamp).total_seconds()) < 60:  # Within 1 minute
            entry_idx = i
            break
    
    if entry_idx is None:
        print("Warning: Entry point not found in filtered data, using closest timestamp")
        entry_idx = chart_df.index[0]
    
    # Create figure with subplots
    fig, (ax_price, ax_volume) = plt.subplots(
        2, 1, 
        figsize=figsize,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1}
    )
    
    # Set background colors
    fig.patch.set_facecolor('white')
    ax_price.set_facecolor('white')
    ax_volume.set_facecolor('white')
    
    # === PRICE CHART ===
    plot_ohlc_candles(ax_price, chart_df)
    plot_vwap_line(ax_price, chart_df)
    plot_zone_level(ax_price, chart_df, zone_level)
    plot_entry_point(ax_price, chart_df, entry_idx, entry_price)
    configure_price_chart(ax_price, title)
    
    # === VOLUME CHART ===
    plot_volume_bars(ax_volume, chart_df)
    highlight_entry_volume(ax_volume, chart_df, entry_idx)
    configure_volume_chart(ax_volume)
    
    # Clean layout
    plt.tight_layout()
    
    return fig


def plot_ohlc_candles(ax, df: pd.DataFrame):
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
            ax.bar(timestamp, body_height, bottom=body_bottom, width=0.6, 
                  color=color, alpha=0.8, edgecolor=edge_color, linewidth=0.5)
        else:
            # Doji - draw horizontal line
            ax.plot([timestamp, timestamp], [open_price, open_price], 
                   color=edge_color, linewidth=1.5)
        
        # Wicks
        ax.plot([timestamp, timestamp], [low, high], color='black', linewidth=0.8, alpha=0.7)


def plot_vwap_line(ax, df: pd.DataFrame):
    """Plot VWAP line with TradingView styling."""
    vwap_color = '#ff9800'  # Orange
    ax.plot(df['timestamp'], df['vwap'], color=vwap_color, linewidth=2, alpha=0.8)


def plot_zone_level(ax, df: pd.DataFrame, zone_level: float):
    """Plot zone level line."""
    zone_color = '#9c27b0'  # Purple
    ax.axhline(y=zone_level, color=zone_color, linestyle='--', linewidth=1.5, alpha=0.7)


def plot_entry_point(ax, df: pd.DataFrame, entry_idx, entry_price: float):
    """Plot entry point marker."""
    entry_color = '#1976d2'  # Blue
    entry_timestamp = df.iloc[entry_idx]['timestamp']
    
    # Entry line
    ax.axvline(x=entry_timestamp, color=entry_color, linewidth=1.5, alpha=0.8)
    
    # Entry point marker
    ax.scatter(entry_timestamp, entry_price, color=entry_color, s=100, 
              marker='*', edgecolor='white', linewidth=1, zorder=10)


def plot_volume_bars(ax, df: pd.DataFrame):
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
        ax.bar(timestamp, volume, width=0.6, color=color, alpha=volume_alpha, 
              edgecolor=color, linewidth=0.3)


def highlight_entry_volume(ax, df: pd.DataFrame, entry_idx):
    """Highlight entry point volume."""
    entry_color = '#1976d2'  # Blue
    entry_timestamp = df.iloc[entry_idx]['timestamp']
    entry_volume = df.iloc[entry_idx]['volume']
    
    # Highlight entry volume bar
    ax.bar(entry_timestamp, entry_volume, width=0.6, color=entry_color, 
          alpha=0.8, edgecolor='white', linewidth=1)


def configure_price_chart(ax, title: str):
    """Configure price chart appearance."""
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel('Price ($)', fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def configure_volume_chart(ax):
    """Configure volume chart appearance."""
    ax.set_ylabel('Volume', fontweight='bold')
    ax.set_xlabel('Time', fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def create_sample_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    # Generate sample data
    dates = pd.date_range('2024-01-15 09:30', periods=200, freq='15min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0, 0.001, 200)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from price
        volatility = 0.002
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
        close = price * (1 + np.random.uniform(-volatility/2, volatility/2))
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


def main():
    """Test the TradingView-style entry chart."""
    print("📊 Creating TradingView-style entry point chart...")
    
    # Create sample data
    df = create_sample_data()
    print(f"   ✅ Generated {len(df)} bars of sample data")
    
    # Define entry point
    entry_timestamp = datetime(2024, 1, 15, 12, 30)  # 12:30 PM
    entry_price = 100.50
    zone_level = 100.25
    
    # Create chart
    fig = plot_tradingview_entry_chart(
        df, 
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        zone_level=zone_level,
        title="Sample Entry Point Chart - SPY",
        hours_before=2,
        hours_after=2,
        figsize=(14, 8),
        dpi=150  # Lower DPI for testing
    )
    
    # Save chart
    output_file = "tradingview_entry_chart.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Chart saved as: {output_file}")
    print("   📊 Chart features:")
    print("     - Clean OHLC candlesticks (teal/red)")
    print("     - VWAP line (orange)")
    print("     - Zone level (purple dashed)")
    print("     - Entry point marker (blue star)")
    print("     - Volume bars with price-based coloring")
    print("     - Entry volume highlight (blue)")
    print("     - No legends or extra annotations")
    print("     - TradingView-style appearance")


if __name__ == "__main__":
    main()