#!/usr/bin/env python3
"""
Ultra Simple Plotly Zone Fade Entry Point Visualizer

A version that completely avoids pandas datetime issues.
"""

import sys
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


class UltraSimplePlotlyVisualizer:
    """Ultra simple visualization using Plotly."""
    
    def __init__(self):
        """Initialize the ultra simple visualizer."""
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
    
    def create_ultra_simple_chart(self, timestamps, opens, highs, lows, closes, volumes, vwaps, 
                                 entry_idx: int, symbol: str, entry_id: str, qrs_score: float, 
                                 entry_timestamp_str: str, entry: Dict[str, Any]) -> go.Figure:
        """Create an ultra simple Plotly candlestick chart."""
        
        # Create subplots with space for data table
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,  # Reduced spacing between subplots
            row_heights=[0.25, 0.15, 0.6],  # Price chart, Volume, Data table (more space for table)
            subplot_titles=(f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f}", "Volume", "Trade Analysis"),
            specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]]
        )
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            showlegend=False,  # Remove legend
            increasing_line_color=self.colors['up'],
            decreasing_line_color=self.colors['down'],
            increasing_fillcolor=self.colors['up'],
            decreasing_fillcolor=self.colors['down']
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add VWAP line
        vwap_trace = go.Scatter(
            x=timestamps,
            y=vwaps,
            mode='lines',
            name='VWAP',
            showlegend=False,  # Remove legend
            line=dict(color=self.colors['vwap'], width=2)
        )
        
        fig.add_trace(vwap_trace, row=1, col=1)
        
        # Add entry point marker and time range
        entry_timestamp = timestamps[entry_idx]
        entry_price = closes[entry_idx]
        
        # Calculate entry time range using actual backtest data
        # Use the actual entry window duration from the backtest
        entry_timestamp = timestamps[entry_idx]
        
        # Get the actual window duration from the entry point data
        window_duration_minutes = entry.get('window_duration_minutes', 5)  # Default to 5 minutes if not available
        window_bars = entry.get('window_bars', 5)  # Default to 5 bars if not available
        
        # Use actual backtest data for entry window
        
        # Calculate entry window based on actual backtest data
        # Use half the duration before and half after the entry point
        half_duration = window_duration_minutes / 2
        start_time = entry_timestamp - pd.Timedelta(minutes=half_duration)
        end_time = entry_timestamp + pd.Timedelta(minutes=half_duration)
        
        # Find the closest indices
        entry_start_idx = 0
        entry_end_idx = len(timestamps) - 1
        
        for i, ts in enumerate(timestamps):
            if ts >= start_time and entry_start_idx == 0:
                entry_start_idx = i
            if ts > end_time:
                entry_end_idx = i
                break
        
        # Add entry start marker
        entry_start_marker = go.Scatter(
            x=[timestamps[entry_start_idx]],
            y=[entry_price],
            mode='markers',
            name='Entry Start',
            showlegend=False,
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        )
        fig.add_trace(entry_start_marker, row=1, col=1)
        
        # Add entry end marker
        entry_end_marker = go.Scatter(
            x=[timestamps[entry_end_idx]],
            y=[entry_price],
            mode='markers',
            name='Entry End',
            showlegend=False,
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        )
        fig.add_trace(entry_end_marker, row=1, col=1)
        
        # Add entry time range line
        entry_range_line = go.Scatter(
            x=[timestamps[entry_start_idx], timestamps[entry_end_idx]],
            y=[entry_price, entry_price],
            mode='lines',
            name='Entry Range',
            showlegend=False,
            line=dict(
                color='orange',
                width=3,
                dash='dash'
            )
        )
        fig.add_trace(entry_range_line, row=1, col=1)
        
        # Entry point marker
        entry_marker = go.Scatter(
            x=[entry_timestamp],
            y=[entry_price],
            mode='markers',
            name='Entry Point',
            showlegend=False,  # Remove legend
            marker=dict(
                color=self.colors['entry'],
                size=12,
                symbol='star'
            )
        )
        
        fig.add_trace(entry_marker, row=1, col=1)
        
        # Add volume bars
        volume_colors = []
        for i in range(len(opens)):
            if closes[i] >= opens[i]:
                volume_colors.append(self.colors['up'])
            else:
                volume_colors.append(self.colors['down'])
        
        volume_trace = go.Bar(
            x=timestamps,
            y=volumes,
            name='Volume',
            showlegend=False,  # Remove legend
            marker_color=volume_colors,
            opacity=0.7
        )
        
        fig.add_trace(volume_trace, row=2, col=1)
        
        # Create detailed title with strategy information
        title = f"{symbol} - Entry {entry_id} | QRS: {qrs_score:.1f} | {entry_timestamp_str}"
        
        # Update layout without legends
        fig.update_layout(
            title=title,
            width=1400,
            height=1000,  # Reduced height since no legends
            showlegend=False,  # Remove all legends
            margin=dict(l=50, r=50, t=100, b=100)  # Normal margins
        )
        
        # Calculate price range for auto-scaling
        price_min = min(min(opens), min(highs), min(lows), min(closes))
        price_max = max(max(opens), max(highs), max(lows), max(closes))
        price_range = price_max - price_min
        price_padding = price_range * 0.05  # 5% padding
        
        # Calculate volume range for auto-scaling
        volume_min = min(volumes)
        volume_max = max(volumes)
        volume_range = volume_max - volume_min
        volume_padding = volume_range * 0.1  # 10% padding
        
        # Debug output for range calculation
        print(f"     📊 Price range: ${price_min:.2f} - ${price_max:.2f} (range: ${price_range:.2f})")
        print(f"     📊 Volume range: {volume_min:,.0f} - {volume_max:,.0f} (range: {volume_range:,.0f})")
        print(f"     📊 Y-axis range: ${price_min - price_padding:.2f} - ${price_max + price_padding:.2f}")
        
        # Update axes with calculated ranges
        fig.update_xaxes(
            title_text="Time", 
            row=2, col=1,
            type="date",  # Date axis for proper time formatting
            tickformat="%H:%M",  # Show hours:minutes
            tickmode="auto"  # Auto-tick placement
        )
        
        # Also update the top x-axis (shared)
        fig.update_xaxes(
            type="date",  # Date axis for proper time formatting
            tickformat="%H:%M",  # Show hours:minutes
            tickmode="auto",  # Auto-tick placement
            showticklabels=False,  # Hide labels on top axis
            row=1, col=1
        )
        
        # Price axis - explicit range based on calculated min/max
        fig.update_yaxes(
            title_text="Price ($)", 
            row=1, col=1,
            range=[price_min - price_padding, price_max + price_padding],  # Explicit range
            type="linear"   # Linear scale for price
        )
        
        # Volume axis - explicit range based on calculated min/max (no title, use subplot title)
        fig.update_yaxes(
            row=2, col=1,
            range=[max(0, volume_min - volume_padding), volume_max + volume_padding],  # Explicit range, min 0
            type="linear"   # Linear scale for volume
        )
        
        # Add comprehensive strategy information as annotations
        self._add_strategy_annotations(fig, entry, entry_timestamp_str, qrs_score, symbol, timestamps, entry_idx, opens, highs, lows, closes, volumes, vwaps, entry_start_idx, entry_end_idx, window_duration_minutes)
        
        return fig
    
    def _add_strategy_annotations(self, fig, entry: Dict[str, Any], 
                                entry_timestamp_str: str, qrs_score: float, symbol: str, timestamps: List, entry_idx: int,
                                opens: List, highs: List, lows: List, closes: List, volumes: List, vwaps: List,
                                entry_start_idx: int, entry_end_idx: int, window_duration_minutes: float):
        """Add comprehensive strategy information as annotations."""
        
        # Extract strategy data from entry point with better defaults
        entry_price = entry.get('entry_price', entry.get('price', 'N/A'))
        zone_high = entry.get('zone_high', entry.get('resistance', 'N/A'))
        zone_low = entry.get('zone_low', entry.get('support', 'N/A'))
        rejection_candle_high = entry.get('rejection_candle_high', entry.get('candle_high', 'N/A'))
        rejection_candle_low = entry.get('rejection_candle_low', entry.get('candle_low', 'N/A'))
        rejection_candle_volume = entry.get('rejection_candle_volume', entry.get('volume', 'N/A'))
        volume_spike_ratio = entry.get('volume_spike_ratio', entry.get('volume_ratio', 'N/A'))
        wick_rejection_ratio = entry.get('wick_rejection_ratio', entry.get('wick_ratio', 'N/A'))
        setup_type = entry.get('setup_type', entry.get('type', 'Zone Fade'))
        timeframe = entry.get('timeframe', entry.get('tf', '1m'))
        
        # Additional trade validation data
        direction = entry.get('direction', entry.get('side', 'N/A'))
        confidence = entry.get('confidence', qrs_score)
        risk_reward = entry.get('risk_reward', entry.get('rr', 'N/A'))
        stop_loss = entry.get('stop_loss', entry.get('sl', 'N/A'))
        take_profit = entry.get('take_profit', entry.get('tp', 'N/A'))
        market_cap = entry.get('market_cap', 'N/A')
        sector = entry.get('sector', 'N/A')
        volatility = entry.get('volatility', entry.get('vol', 'N/A'))
        trend_strength = entry.get('trend_strength', entry.get('trend', 'N/A'))
        momentum = entry.get('momentum', 'N/A')
        rsi = entry.get('rsi', 'N/A')
        macd = entry.get('macd', 'N/A')
        bollinger_position = entry.get('bollinger_position', 'N/A')
        
        # Calculate zone range safely
        zone_range = "N/A"
        if zone_high != 'N/A' and zone_low != 'N/A':
            try:
                zone_range = f"${float(zone_high) - float(zone_low):.2f}"
            except (ValueError, TypeError):
                zone_range = "N/A"
        
        # Format volume safely
        volume_str = "N/A"
        if rejection_candle_volume != 'N/A':
            try:
                volume_str = f"{int(rejection_candle_volume):,}"
            except (ValueError, TypeError):
                volume_str = str(rejection_candle_volume)
        
        # Format numeric values safely
        wick_rejection_str = "N/A"
        if wick_rejection_ratio != 'N/A':
            try:
                wick_rejection_str = f"{float(wick_rejection_ratio):.2f}"
            except (ValueError, TypeError):
                wick_rejection_str = str(wick_rejection_ratio)
        
        volume_spike_str = "N/A"
        if volume_spike_ratio != 'N/A':
            try:
                volume_spike_str = f"{float(volume_spike_ratio):.2f}x"
            except (ValueError, TypeError):
                volume_spike_str = str(volume_spike_ratio)
        
        # Create comprehensive strategy information text
        strategy_info = f"""
<b>🎯 ZONE FADE SETUP DETAILS</b><br>
<b>Entry Price:</b> ${entry_price}<br>
<b>Direction:</b> {direction}<br>
<b>QRS Score:</b> {qrs_score:.1f}/10<br>
<b>Confidence:</b> {confidence:.1f}/10<br>
<b>Setup Type:</b> {setup_type}<br>
<b>Timeframe:</b> {timeframe}<br><br>

<b>📊 ZONE ANALYSIS</b><br>
<b>Zone High:</b> ${zone_high}<br>
<b>Zone Low:</b> ${zone_low}<br>
<b>Zone Range:</b> {zone_range}<br>
<b>Zone Strength:</b> {trend_strength}<br><br>

<b>🕯️ REJECTION CANDLE</b><br>
<b>High:</b> ${rejection_candle_high}<br>
<b>Low:</b> ${rejection_candle_low}<br>
<b>Volume:</b> {volume_str}<br>
<b>Wick Rejection:</b> {wick_rejection_str}<br>
<b>Momentum:</b> {momentum}<br><br>

<b>📈 VOLUME & MOMENTUM</b><br>
<b>Volume Spike:</b> {volume_spike_str}<br>
<b>Volatility:</b> {volatility}<br>
<b>RSI:</b> {rsi}<br>
<b>MACD:</b> {macd}<br>
<b>Bollinger Position:</b> {bollinger_position}<br><br>

<b>💰 RISK MANAGEMENT</b><br>
<b>Stop Loss:</b> ${stop_loss}<br>
<b>Take Profit:</b> ${take_profit}<br>
<b>Risk/Reward:</b> {risk_reward}<br>
<b>Entry Time:</b> {entry_timestamp_str}<br>
        """
        
        # Add strategy information annotation - REMOVED (now using data tables)
        # fig.add_annotation(...)
        
        # Add comprehensive entry point analysis
        entry_details = f"""
<b>🎯 ENTRY POINT ANALYSIS</b><br>
<b>Symbol:</b> {symbol}<br>
<b>Entry ID:</b> {entry.get('entry_id', 'N/A')}<br>
<b>Strategy:</b> Zone Fade<br>
<b>Direction:</b> {direction}<br>
<b>Confidence:</b> {qrs_score:.1f}/10<br>
<b>Market Cap:</b> {market_cap}<br>
<b>Sector:</b> {sector}<br><br>

<b>📋 SETUP VALIDATION</b><br>
✅ Higher Timeframe Zone<br>
✅ Rejection Candle Pattern<br>
✅ Volume Spike Confirmation<br>
✅ QRS Score: {qrs_score:.1f}<br>
✅ Risk/Reward: {risk_reward}<br>
✅ Stop Loss Set: ${stop_loss}<br>
✅ Take Profit Set: ${take_profit}<br><br>

<b>🔍 TECHNICAL ANALYSIS</b><br>
<b>Trend Strength:</b> {trend_strength}<br>
<b>Momentum:</b> {momentum}<br>
<b>RSI Level:</b> {rsi}<br>
<b>MACD Signal:</b> {macd}<br>
<b>Bollinger Position:</b> {bollinger_position}<br>
<b>Volatility:</b> {volatility}<br><br>

<b>⚠️ RISK ASSESSMENT</b><br>
<b>Entry Risk:</b> ${entry_price} → ${stop_loss}<br>
<b>Potential Reward:</b> ${entry_price} → ${take_profit}<br>
<b>Risk/Reward Ratio:</b> {risk_reward}<br>
<b>Position Size:</b> Based on risk tolerance<br>
        """
        
        # Add entry details annotation - REMOVED (now using data tables)
        # fig.add_annotation(...)
        
        # Add zone lines to the chart
        self._add_zone_lines(fig, entry, timestamps)
        
        # Add comprehensive trade analysis table below the chart
        self._add_trade_analysis_table(fig, entry, entry_timestamp_str, qrs_score, symbol, timestamps, entry_idx, opens, highs, lows, closes, volumes, vwaps, entry_start_idx, entry_end_idx, window_duration_minutes)
    
    def _add_zone_lines(self, fig, entry: Dict[str, Any], timestamps: List):
        """Add zone lines to visualize the trading zone."""
        
        zone_high = entry.get('zone_high')
        zone_low = entry.get('zone_low')
        
        if zone_high and zone_low:
            try:
                zone_high_val = float(zone_high)
                zone_low_val = float(zone_low)
                
                # Add zone high line
                fig.add_hline(
                    y=zone_high_val,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    annotation_text=f"Zone High: ${zone_high_val:.2f}",
                    annotation_position="top right"
                )
                
                # Add zone low line
                fig.add_hline(
                    y=zone_low_val,
                    line_dash="dash", 
                    line_color="green",
                    opacity=0.7,
                    annotation_text=f"Zone Low: ${zone_low_val:.2f}",
                    annotation_position="bottom right"
                )
                
                # Add zone fill
                fig.add_hrect(
                    y0=zone_low_val,
                    y1=zone_high_val,
                    fillcolor="rgba(255,255,0,0.1)",
                    layer="below",
                    line_width=0
                )
                
            except (ValueError, TypeError):
                pass  # Skip if zone values are not valid numbers
    
    def _add_trade_analysis_table(self, fig, entry: Dict[str, Any], 
                                 entry_timestamp_str: str, qrs_score: float, symbol: str,
                                 timestamps: List, entry_idx: int, opens: List, highs: List, 
                                 lows: List, closes: List, volumes: List, vwaps: List,
                                 entry_start_idx: int, entry_end_idx: int, window_duration_minutes: float):
        """Add comprehensive trade analysis table with calculated values only."""
        
        # Calculate all required values from the data
        entry_price = float(entry.get('entry_price', entry.get('price', closes[entry_idx])))
        direction = entry.get('direction', entry.get('side', 'LONG' if closes[entry_idx] > opens[entry_idx] else 'SHORT'))
        
        # Convert main entry timestamp to Eastern Time
        import pytz
        utc = pytz.UTC
        eastern = pytz.timezone('US/Eastern')
        entry_timestamp_utc = timestamps[entry_idx]
        entry_timestamp_et = entry_timestamp_utc.replace(tzinfo=utc).astimezone(eastern)
        entry_timestamp_str_et = entry_timestamp_et.strftime('%Y-%m-%d %H:%M:%S ET')
        
        # Price analysis
        current_price = closes[entry_idx]
        price_change = current_price - opens[entry_idx]
        price_change_pct = (price_change / opens[entry_idx]) * 100
        
        # Zone analysis
        zone_high = float(entry.get('zone_high', entry.get('resistance', max(highs))))
        zone_low = float(entry.get('zone_low', entry.get('support', min(lows))))
        zone_range = zone_high - zone_low
        zone_mid = (zone_high + zone_low) / 2
        
        # Entry position within zone
        if zone_range > 0:
            entry_zone_position = ((entry_price - zone_low) / zone_range) * 100
        else:
            entry_zone_position = 50.0
        
        # Risk management calculations
        stop_loss = float(entry.get('stop_loss', entry.get('sl', zone_low if direction == 'LONG' else zone_high)))
        take_profit = float(entry.get('take_profit', entry.get('tp', zone_high if direction == 'LONG' else zone_low)))
        
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)
        # Risk/Reward ratio should be reward:risk (higher is better)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Volume analysis
        avg_volume = sum(volumes) / len(volumes)
        entry_volume = volumes[entry_idx]
        volume_ratio = entry_volume / avg_volume if avg_volume > 0 else 1
        
        # VWAP analysis
        entry_vwap = vwaps[entry_idx]
        vwap_distance = ((entry_price - entry_vwap) / entry_vwap) * 100
        
        # Price range analysis
        price_high = max(highs)
        price_low = min(lows)
        price_range = price_high - price_low
        price_range_pct = (price_range / price_low) * 100
        
        # Volatility calculation (standard deviation of returns)
        returns = []
        for i in range(1, len(closes)):
            ret = (closes[i] - closes[i-1]) / closes[i-1]
            returns.append(ret)
        volatility = (sum([(r - sum(returns)/len(returns))**2 for r in returns]) / len(returns))**0.5 * 100 if returns else 0
        
        # Time analysis
        time_window_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        bars_before_entry = entry_idx
        bars_after_entry = len(timestamps) - entry_idx - 1
        
        # Entry point timing details (convert to US Eastern Time)
        import pytz
        
        # Convert to US Eastern Time
        utc = pytz.UTC
        eastern = pytz.timezone('US/Eastern')
        
        entry_start_time_utc = timestamps[entry_start_idx]
        entry_end_time_utc = timestamps[entry_end_idx]
        
        # Convert to Eastern Time
        entry_start_time = entry_start_time_utc.replace(tzinfo=utc).astimezone(eastern)
        entry_end_time = entry_end_time_utc.replace(tzinfo=utc).astimezone(eastern)
        
        # Use the actual window duration from the backtest data
        entry_duration_minutes = window_duration_minutes
        entry_start_str = entry_start_time.strftime('%H:%M:%S ET')
        entry_end_str = entry_end_time.strftime('%H:%M:%S ET')
        
        # Calculate QRS sub-components (more realistic scoring)
        # Quality: Based on zone strength, volume confirmation, and VWAP alignment
        quality_score = min(10, max(1, 
            (entry_zone_position / 20) +  # Zone position (0-5, scaled down)
            (min(volume_ratio, 2) * 1.5) +  # Volume spike (capped at 2x, scaled down)
            (max(0, 3 - abs(vwap_distance)/2)) +  # VWAP alignment (closer = better)
            2  # Base quality score
        ))
        
        # Risk: Based on risk/reward ratio (higher ratio = lower risk score) and volatility
        # Risk score should be LOWER for better risk/reward ratios
        risk_score = min(10, max(1, 
            10 - (risk_reward_ratio * 3) +  # Higher risk/reward = lower risk score
            (volatility * 2)  # Higher volatility = higher risk score
        ))
        
        # Setup: Based on price action, momentum, and zone strength
        setup_score = min(10, max(1, 
            (abs(price_change_pct) * 10) +  # Price momentum (scaled up)
            (min(volume_ratio, 1.5) * 2) +   # Volume confirmation (scaled down)
            (entry_zone_position / 20) +   # Zone position (scaled down)
            1  # Base setup score
        ))
        
        # Calculate QRS components
        qrs_quality = f"{quality_score:.1f}"
        qrs_risk = f"{risk_score:.1f}"
        qrs_setup = f"{setup_score:.1f}"
        
        # Calculate setup score components for detailed breakdown
        price_momentum_score = min(10, abs(price_change_pct) * 3)
        volume_confirmation_score = min(10, min(volume_ratio, 2) * 2)
        zone_position_score = min(10, entry_zone_position / 10)
        base_setup_score = 2.0
        
        # Format setup components
        setup_price_momentum = f"{price_momentum_score:.1f}"
        setup_volume_conf = f"{volume_confirmation_score:.1f}"
        setup_zone_pos = f"{zone_position_score:.1f}"
        setup_base = f"{base_setup_score:.1f}"
        
        # Create simple 2-column table for easy text selection
        analysis_data = [
            ['ENTRY DETAILS', ''],
            ['Symbol', symbol],
            ['Entry Price', f"${entry_price:.2f}"],
            ['Direction', direction],
            ['Entry Time', entry_timestamp_str_et],
            ['', ''],
            ['QRS BREAKDOWN', ''],
            ['Quality Score', qrs_quality],
            ['Risk Score', qrs_risk],
            ['Setup Score', qrs_setup],
            ['Overall QRS', f"{qrs_score:.1f}/10"],
            ['', ''],
            ['SETUP SCORE BREAKDOWN', ''],
            ['Price Momentum', f"{price_change_pct:+.2f}%"],
            ['Volume Confirmation', f"{volume_ratio:.2f}x"],
            ['Zone Position', f"{entry_zone_position:.1f}%"],
            ['Base Setup', 'Fixed'],
            ['', ''],
            ['COMPONENT SCORES', ''],
            ['Momentum Score', setup_price_momentum],
            ['Volume Score', setup_volume_conf],
            ['Zone Score', setup_zone_pos],
            ['Base Score', setup_base],
            ['', ''],
            ['ENTRY POINT TIMING', ''],
            ['Entry Start Time', entry_start_str],
            ['Entry End Time', entry_end_str],
            ['Entry Duration', f"{entry_duration_minutes:.1f} min"],
            ['Entry Start Index', str(entry_start_idx)],
            ['Entry End Index', str(entry_end_idx)],
            ['', ''],
            ['PRICE ANALYSIS', ''],
            ['Current Price', f"${current_price:.2f}"],
            ['Price Change', f"${price_change:.2f} ({price_change_pct:+.2f}%)"],
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
            ['Volatility', f"{volatility:.2f}%"],
            ['Time Window', f"{time_window_hours:.1f} hours"],
            ['Bars Before Entry', str(bars_before_entry)],
            ['Bars After Entry', str(bars_after_entry)],
            ['Total Data Points', str(len(timestamps))]
        ]
        
        # Create simple 2-column table for easy text selection
        column1_data = [row[0] for row in analysis_data]  # All first column values
        column2_data = [row[1] for row in analysis_data]  # All second column values
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='#1f77b4',
                    align='center',
                    font=dict(color='white', size=18, family="Arial Black"),
                    height=40
                ),
                cells=dict(
                    values=[column1_data, column2_data],
                    fill_color=[
                        ['#f8f9fa' if i % 2 == 0 else '#ffffff' for i in range(len(analysis_data))],
                        ['#f8f9fa' if i % 2 == 0 else '#ffffff' for i in range(len(analysis_data))]
                    ],
                    align=['left', 'right'],
                    font=dict(color='black', size=14, family="Arial"),
                    height=32,
                    line=dict(width=1, color='#dee2e6')
                ),
                domain=dict(x=[0, 1], y=[0, 1])  # Full table space
            ),
            row=3, col=1
        )
        
        # Update layout to accommodate the comprehensive table without scrolling
        fig.update_layout(
            height=3000,  # Even larger height to prevent scrolling
            margin=dict(l=50, r=50, t=100, b=50)  # Adequate margins
        )
        
        # Configure simple 2-column table
        fig.update_traces(
            selector=dict(type="table"),
            columnwidth=[300, 200]  # Simple 2-column layout
        )
    
    def create_visualizations(self, data: Dict[str, List[OHLCVBar]], 
                            entry_points: List[Dict[str, Any]], 
                            output_dir: Path) -> bool:
        """Create ultra simple visualizations for entry points."""
        print("🎨 Creating ultra simple Plotly visualizations...")
        
        # Create output directory
        plotly_dir = output_dir / "ultra_simple_plotly_charts"
        plotly_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = len(entry_points)
        
        for i, entry in enumerate(entry_points[:20]):  # Generate 20 graphs for testing
            try:
                symbol = entry['symbol']
                entry_id = entry['entry_id']
                qrs_score = entry['qrs_score']
                entry_timestamp_str = entry['timestamp']
                
                print(f"   📊 Creating visualization for {entry_id} ({symbol})...")
                
                # Get data for this symbol
                if symbol not in data:
                    print(f"     ❌ No data available for {symbol}")
                    continue
                
                symbol_data = data[symbol]
                
                # Convert to simple lists to avoid pandas datetime issues
                timestamps = [bar.timestamp for bar in symbol_data]
                opens = [bar.open for bar in symbol_data]
                highs = [bar.high for bar in symbol_data]
                lows = [bar.low for bar in symbol_data]
                closes = [bar.close for bar in symbol_data]
                volumes = [bar.volume for bar in symbol_data]
                
                # Calculate VWAP
                typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
                vwaps = []
                cumulative_volume = 0.0
                cumulative_vwap = 0.0
                
                for i, (tp, vol) in enumerate(zip(typical_prices, volumes)):
                    cumulative_volume += float(vol)
                    cumulative_vwap += float(tp) * float(vol)
                    if cumulative_volume > 0:
                        vwaps.append(cumulative_vwap / cumulative_volume)
                    else:
                        vwaps.append(float(tp))
                
                # Find entry point using string matching
                entry_idx = None
                entry_timestamp_str_clean = entry_timestamp_str.replace('+00:00', '').replace('Z', '')
                
                for idx, timestamp in enumerate(timestamps):
                    timestamp_str = str(timestamp).replace('+00:00', '').replace('Z', '')
                    if entry_timestamp_str_clean in timestamp_str or timestamp_str in entry_timestamp_str_clean:
                        entry_idx = idx
                        break
                
                if entry_idx is None:
                    print(f"     ❌ Entry point not found in data for {entry_id}")
                    continue
                
                # Create time window around entry point: 2 hours before, 12 hours after
                entry_timestamp = timestamps[entry_idx]
                
                # Calculate time window
                start_time = entry_timestamp - pd.Timedelta(hours=2)
                end_time = entry_timestamp + pd.Timedelta(hours=12)
                
                # Find indices for time window
                start_idx = 0
                end_idx = len(timestamps)
                
                for i, ts in enumerate(timestamps):
                    if ts >= start_time and start_idx == 0:
                        start_idx = i
                    if ts > end_time:
                        end_idx = i
                        break
                
                # Ensure we have some data
                if start_idx >= end_idx:
                    start_idx = max(0, entry_idx - 50)
                    end_idx = min(len(timestamps), entry_idx + 50)
                
                window_timestamps = timestamps[start_idx:end_idx]
                window_opens = opens[start_idx:end_idx]
                window_highs = highs[start_idx:end_idx]
                window_lows = lows[start_idx:end_idx]
                
                # Debug output for time window
                window_start = window_timestamps[0] if window_timestamps else "N/A"
                window_end = window_timestamps[-1] if window_timestamps else "N/A"
                print(f"     ⏰ Time window: {window_start} to {window_end}")
                print(f"     📊 Data points: {len(window_timestamps)} bars")
                window_closes = closes[start_idx:end_idx]
                window_volumes = volumes[start_idx:end_idx]
                window_vwaps = vwaps[start_idx:end_idx]
                
                if len(window_timestamps) == 0:
                    print(f"     ❌ No data in window for {entry_id}")
                    continue
                
                # Find new entry index in filtered data
                new_entry_idx = None
                for idx, timestamp in enumerate(window_timestamps):
                    timestamp_str = str(timestamp).replace('+00:00', '').replace('Z', '')
                    if entry_timestamp_str_clean in timestamp_str or timestamp_str in entry_timestamp_str_clean:
                        new_entry_idx = idx
                        break
                
                if new_entry_idx is None:
                    print(f"     ❌ Entry point not found in filtered data for {entry_id}")
                    continue
                
                # Create Plotly chart
                fig = self.create_ultra_simple_chart(
                    window_timestamps, window_opens, window_highs, window_lows, 
                    window_closes, window_volumes, window_vwaps,
                    new_entry_idx, symbol, entry_id, qrs_score, entry_timestamp_str, entry
                )
                
                # Save as HTML with proper structure
                entry_timestamp = timestamps[entry_idx]
                # Use string representation instead of strftime to avoid datetime arithmetic
                timestamp_str = str(entry_timestamp).replace(' ', '_').replace(':', '').replace('-', '').replace('+00:00', '')[:13]
                filename = f"{symbol}_{timestamp_str}_ultra_simple_chart.html"
                output_file = plotly_dir / filename
                
                # Write HTML with custom CSS for text selection
                html_content = fig.to_html(
                    include_plotlyjs=True,
                    config={
                        'displayModeBar': True, 
                        'displaylogo': False,
                        'scrollZoom': False,
                        'doubleClick': False,
                        'editable': False,
                        'selectable': True
                    }
                )
                
                # Add custom CSS to enable text selection and improve table styling
                custom_css = """
                <style>
                /* Enable text selection for all elements */
                * {
                    user-select: text !important;
                    -webkit-user-select: text !important;
                    -moz-user-select: text !important;
                    -ms-user-select: text !important;
                }
                
                /* Specific styling for Plotly tables */
                .plotly .table-container table {
                    user-select: text !important;
                    -webkit-user-select: text !important;
                    -moz-user-select: text !important;
                    -ms-user-select: text !important;
                }
                
                .plotly .table-container td, .plotly .table-container th {
                    user-select: text !important;
                    -webkit-user-select: text !important;
                    -moz-user-select: text !important;
                    -ms-user-select: text !important;
                    cursor: text !important;
                }
                
                .plotly .table-container {
                    user-select: text !important;
                    -webkit-user-select: text !important;
                    -moz-user-select: text !important;
                    -ms-user-select: text !important;
                }
                
                /* Force text selection on all table elements */
                .plotly .table-container * {
                    user-select: text !important;
                    -webkit-user-select: text !important;
                    -moz-user-select: text !important;
                    -ms-user-select: text !important;
                    cursor: text !important;
                }
                </style>
                """
                
                # Insert custom CSS before closing head tag
                html_content = html_content.replace('</head>', custom_css + '</head>')
                
                # Write the modified HTML
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"     ✅ Created visualization for {entry_id}")
                success_count += 1
                
            except Exception as e:
                import traceback
                print(f"     ❌ Error creating visualization for {entry.get('entry_id', 'unknown')}: {e}")
                print(f"     📍 Traceback: {traceback.format_exc()}")
                continue
        
        print(f"📊 Ultra simple visualization results:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {total_count - success_count}")
        print(f"   📁 Output Directory: {plotly_dir}")
        
        return success_count > 0
    
    def run_visualization(self, data_dir: Path, output_dir: Path) -> bool:
        """Run the ultra simple visualization process."""
        print("🚀 Ultra Simple Plotly Zone Fade Visualizer")
        print("=" * 60)
        
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
            print("🎉 Ultra simple visualization complete!")
        else:
            print("❌ Ultra simple visualization failed!")
        
        return success


def main():
    """Main function for command-line usage."""
    visualizer = UltraSimplePlotlyVisualizer()
    
    # Set up paths
    data_dir = Path('data')
    output_dir = Path('outputs/ultra_simple_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = visualizer.run_visualization(data_dir, output_dir)
    
    if success:
        print(f"\n📁 Visualizations saved to: {output_dir}")
        print("🌐 Open the HTML files in your browser for interactive charts!")
    else:
        print("\n❌ Visualization failed. Check the logs above for details.")


if __name__ == "__main__":
    main()