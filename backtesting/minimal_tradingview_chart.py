#!/usr/bin/env python3
"""
Minimal TradingView-Style Chart

ONLY renders:
- OHLC candlesticks
- VWAP line
- Volume bars in second subplot

No overlays, zones, annotations, or other shapes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from typing import Optional
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


def plot_minimal_tradingview_chart(
    df: pd.DataFrame,
    title: str = "Intraday Chart",
    figsize: tuple = (14, 8),
    dpi: int = 300
) -> plt.Figure:
    """
    Create a minimal TradingView-style chart with ONLY:
    - OHLC candlesticks
    - VWAP line
    - Volume bars
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
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
    
    # Compute VWAP if not provided
    if 'vwap' not in df.columns:
        df = df.copy()
        df['vwap'] = compute_session_vwap(df)
    
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
    
    # === OHLC CANDLESTICKS ONLY ===
    plot_ohlc_candles(ax_price, df)
    
    # === VWAP LINE ONLY ===
    plot_vwap_line(ax_price, df)
    
    # === VOLUME BARS ONLY ===
    plot_volume_bars(ax_volume, df)
    
    # Configure charts
    configure_price_chart(ax_price, title)
    configure_volume_chart(ax_volume)
    
    # Clean layout
    plt.tight_layout()
    
    return fig


def plot_ohlc_candles(ax, df: pd.DataFrame):
    """Plot OHLC candlesticks with clean styling."""
    # Clean colors
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
    """Plot VWAP line only."""
    vwap_color = '#ff9800'  # Orange
    ax.plot(df['timestamp'], df['vwap'], color=vwap_color, linewidth=2, alpha=0.8)


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
    dates = pd.date_range('2024-01-15 09:30', periods=100, freq='15min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0, 0.001, 100)
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
    """Test the minimal TradingView-style chart."""
    print("📊 Creating minimal TradingView-style chart...")
    print("   ONLY: OHLC candlesticks + VWAP line + Volume bars")
    
    # Create sample data
    df = create_sample_data()
    print(f"   ✅ Generated {len(df)} bars of sample data")
    
    # Create chart
    fig = plot_minimal_tradingview_chart(
        df, 
        title="Minimal Intraday Chart - SPY",
        figsize=(14, 8),
        dpi=150  # Lower DPI for testing
    )
    
    # Save chart
    output_file = "minimal_tradingview_chart.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Chart saved as: {output_file}")
    print("   📊 Chart contains ONLY:")
    print("     - OHLC candlesticks (teal/red)")
    print("     - VWAP line (orange)")
    print("     - Volume bars (price-based coloring)")
    print("     - NO overlays, zones, annotations, or other shapes")


if __name__ == "__main__":
    main()