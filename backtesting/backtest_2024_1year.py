#!/usr/bin/env python3
"""
1-Year Zone Fade Backtest (2024)

This version runs the backtest using 1 year of 2024 data with the proper
Zone Fade exit strategy for all 3 assets (SPY, QQQ, IWM).
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
import json
from pathlib import Path

def load_2024_data():
    """Load the 2024 data for all symbols."""
    print("📊 Loading 2024 data (1-year)...")
    
    symbols = ['SPY', 'QQQ', 'IWM']
    data = {}
    
    # Try both paths (Docker and local)
    base_paths = [Path("/app/data/2024"), Path("data/2024")]
    
    for symbol in symbols:
        loaded = False
        for base_path in base_paths:
            try:
                file_path = base_path / f"{symbol}_2024.pkl"
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        symbol_data = pickle.load(f)
                    data[symbol] = symbol_data
                    print(f"   ✅ {symbol}: {len(symbol_data):,} bars loaded")
                    loaded = True
                    break
            except Exception as e:
                continue
        
        if not loaded:
            print(f"   ❌ {symbol}: Error loading data")
    
    return data


def calculate_risk_unit(entry_price: float, hard_stop: float) -> float:
    """Calculate Risk Unit (R) as per Zone Fade Strategy requirements."""
    return abs(entry_price - hard_stop)


def calculate_proper_t1_price(entry_price: float, hard_stop: float, direction: str, 
                             vwap: float, qrs_score: float) -> float:
    """
    Calculate T1 price as per Zone Fade Strategy:
    T1 = Nearest of VWAP or 1R from entry
    """
    
    # Calculate Risk Unit (R)
    risk_unit = calculate_risk_unit(entry_price, hard_stop)
    
    # Calculate 1R target
    if direction == "SHORT":
        t1_r_target = entry_price - risk_unit
    else:  # LONG
        t1_r_target = entry_price + risk_unit
    
    # T1 = Nearest of VWAP or 1R from entry
    if direction == "SHORT":
        # For SHORT: choose the higher of VWAP or 1R (closer to entry = better)
        t1_price = max(vwap, t1_r_target)
    else:  # LONG
        # For LONG: choose the lower of VWAP or 1R (closer to entry = better)
        t1_price = min(vwap, t1_r_target)
    
    return t1_price


def calculate_proper_t2_price(entry_price: float, hard_stop: float, direction: str, 
                             opening_range_high: float, opening_range_low: float) -> float:
    """
    Calculate T2 price as per Zone Fade Strategy:
    T2 = Opposite side of OR range or next structural level (+2R)
    """
    
    # Calculate Risk Unit (R)
    risk_unit = calculate_risk_unit(entry_price, hard_stop)
    
    # Calculate 2R target
    if direction == "SHORT":
        t2_r_target = entry_price - (2 * risk_unit)
    else:  # LONG
        t2_r_target = entry_price + (2 * risk_unit)
    
    # T2 = Opposite side of OR range or 2R (whichever is closer to entry)
    if direction == "SHORT":
        # For SHORT: choose the higher of OR low or 2R (closer to entry = better)
        t2_price = max(opening_range_low, t2_r_target)
    else:  # LONG
        # For LONG: choose the lower of OR high or 2R (closer to entry = better)
        t2_price = min(opening_range_high, t2_r_target)
    
    return t2_price


def calculate_proper_t3_price(entry_price: float, hard_stop: float, direction: str) -> float:
    """
    Calculate T3 price as per Zone Fade Strategy:
    T3 = Opposite high-timeframe zone (+3R+)
    """
    
    # Calculate Risk Unit (R)
    risk_unit = calculate_risk_unit(entry_price, hard_stop)
    
    # Calculate 3R target
    if direction == "SHORT":
        t3_price = entry_price - (3 * risk_unit)
    else:  # LONG
        t3_price = entry_price + (3 * risk_unit)
    
    return t3_price


class Zone:
    """Represents a zone for 1-year backtest."""
    
    def __init__(self, zone_type: str, level: float, symbol: str, created_at: datetime):
        self.zone_type = zone_type
        self.level = level
        self.symbol = symbol
        self.created_at = created_at
        self.zone_id = f"{symbol}_{zone_type}_{level:.2f}"
        self.touch_count = 0
        self.is_active = True
        self.strength = 1.0
        self.quality = 1
        
    def contains_price(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is within the zone range."""
        zone_range = self.level * tolerance
        return abs(price - self.level) <= zone_range


class ZoneManager:
    """Manages zones for 1-year backtest."""
    
    def __init__(self):
        self.zones: Dict[str, List[Zone]] = {}
        self.daily_zone_counts: Dict[str, Dict[str, int]] = {}
        self.max_zones_per_day = 8  # Enhanced: 8 zones per symbol per day
        
    def create_daily_zones(self, symbol: str, date: datetime, daily_data: List[dict]) -> List[Zone]:
        """Create zones for a trading day."""
        date_key = date.strftime("%Y-%m-%d")
        
        # Initialize daily count if needed
        if date_key not in self.daily_zone_counts:
            self.daily_zone_counts[date_key] = {}
        if symbol not in self.daily_zone_counts[date_key]:
            self.daily_zone_counts[date_key][symbol] = 0
        
        # Check if we've reached the daily limit
        if self.daily_zone_counts[date_key][symbol] >= self.max_zones_per_day:
            return []
        
        zones = []
        
        # Calculate zones from daily data
        if daily_data:
            daily_high = max([bar.high for bar in daily_data])
            daily_low = min([bar.low for bar in daily_data])
            
            # Prior day high/low zones
            zones.append(Zone("PRIOR_DAY_HIGH", daily_high, symbol, date))
            zones.append(Zone("PRIOR_DAY_LOW", daily_low, symbol, date))
            
            # Update count
            self.daily_zone_counts[date_key][symbol] += 2
        
        return zones
    
    def get_active_zones(self, symbol: str, timestamp: datetime) -> List[Zone]:
        """Get active zones for a symbol."""
        if symbol not in self.zones:
            return []
        
        active = [z for z in self.zones[symbol] if z.is_active]
        return active
    
    def add_zones(self, symbol: str, zones: List[Zone]):
        """Add zones to the manager."""
        if symbol not in self.zones:
            self.zones[symbol] = []
        self.zones[symbol].extend(zones)


def generate_entry_points_from_bars(bars_data: Dict[str, List[dict]], 
                                    zone_manager: ZoneManager) -> List[dict]:
    """Generate entry points from bar data using zone fade detection logic."""
    print("\n🔍 Generating entry points from bar data...")
    
    entry_points = []
    
    for symbol in ['SPY', 'QQQ', 'IWM']:
        if symbol not in bars_data:
            continue
        
        bars = bars_data[symbol]
        print(f"\n📊 Processing {symbol}: {len(bars):,} bars")
        
        current_day = None
        daily_bars = []
        
        for i, bar in enumerate(bars):
            timestamp = bar.timestamp
            bar_date = timestamp.date()
            
            # Track daily bars for zone creation
            if current_day != bar_date:
                # Create zones from previous day if we have data
                if current_day and daily_bars:
                    zones = zone_manager.create_daily_zones(symbol, timestamp, daily_bars)
                    if zones:
                        zone_manager.add_zones(symbol, zones)
                
                current_day = bar_date
                daily_bars = []
            
            daily_bars.append(bar)
            
            # Skip if not enough data for analysis
            if i < 20:
                continue
            
            # Get active zones
            active_zones = zone_manager.get_active_zones(symbol, timestamp)
            if not active_zones:
                continue
            
            # Check for zone approaches and rejections
            close_price = bar.close
            high_price = bar.high
            low_price = bar.low
            
            for zone in active_zones:
                # Check if price is touching the zone
                if zone.contains_price(high_price) or zone.contains_price(low_price):
                    zone.touch_count += 1
                    
                    # Only consider first touch
                    if zone.touch_count > 1:
                        continue
                    
                    # Calculate wick analysis
                    candle_range = high_price - low_price
                    if candle_range < 0.01:
                        continue
                    
                    upper_wick = high_price - max(bar.open, close_price)
                    lower_wick = min(bar.open, close_price) - low_price
                    
                    upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
                    lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
                    
                    # Check for rejection (30% wick threshold)
                    is_upper_rejection = upper_wick_ratio > 0.3
                    is_lower_rejection = lower_wick_ratio > 0.3
                    
                    # Calculate volume spike (using previous 20 bars)
                    recent_bars = bars[max(0, i-20):i]
                    if len(recent_bars) < 10:
                        continue
                    
                    avg_volume = sum([b.volume for b in recent_bars]) / len(recent_bars)
                    volume_spike = bar.volume / avg_volume if avg_volume > 0 else 0
                    
                    # Check for volume spike (1.8x threshold)
                    has_volume_spike = volume_spike > 1.8
                    
                    # Determine direction based on zone type and rejection
                    direction = None
                    if 'HIGH' in zone.zone_type and is_upper_rejection:
                        direction = 'SHORT'
                    elif 'LOW' in zone.zone_type and is_lower_rejection:
                        direction = 'LONG'
                    
                    if not direction:
                        continue
                    
                    # Calculate QRS score (simplified)
                    qrs_score = 5.0  # Base score
                    if has_volume_spike:
                        qrs_score += 2.0
                    if volume_spike > 2.0:
                        qrs_score += 1.0
                    
                    # Calculate VWAP (simplified - using close prices)
                    vwap_window = bars[max(0, i-50):i+1]
                    vwap = sum([b.close * b.volume for b in vwap_window]) / sum([b.volume for b in vwap_window]) if vwap_window else close_price
                    
                    # Calculate opening range (first 30 mins of trading day)
                    day_start = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
                    or_end = day_start + timedelta(minutes=30)
                    or_bars = [b for b in bars if day_start <= b.timestamp <= or_end]
                    
                    or_high = max([b.high for b in or_bars]) if or_bars else high_price
                    or_low = min([b.low for b in or_bars]) if or_bars else low_price
                    
                    # Create entry point
                    entry_point = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'date': timestamp.strftime('%Y-%m-%d'),
                        'time': timestamp.strftime('%H:%M:%S'),
                        'direction': direction,
                        'zone_type': zone.zone_type,
                        'zone_level': zone.level,
                        'entry_price': close_price,
                        'qrs_score': qrs_score,
                        'volume_spike': volume_spike,
                        'wick_ratio': upper_wick_ratio if direction == 'SHORT' else lower_wick_ratio,
                        'vwap': vwap,
                        'or_high': or_high,
                        'or_low': or_low,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': bar.volume
                    }
                    
                    entry_points.append(entry_point)
                    print(f"   ✅ {symbol} {direction} entry at {timestamp} (QRS: {qrs_score:.1f})")
    
    print(f"\n✅ Generated {len(entry_points)} entry points")
    return entry_points


def run_2024_zone_fade_backtest():
    """Run the backtest using 2024 data with proper Zone Fade exit strategy."""
    
    print("=" * 80)
    print("🎯 1-YEAR ZONE FADE BACKTEST (2024) - ENHANCED")
    print("=" * 80)
    print("This version uses 1 year of data (2024) with the complete Zone Fade Strategy:")
    print("• Hard stops at zone invalidation")
    print("• T1: Nearest of VWAP or 1R (scale out 40-50%, move stop to breakeven)")
    print("• T2: Opposite side of OR range or 2R (scale another 25%)")
    print("• T3: Opposite high-timeframe zone or 3R (trail or close remaining)")
    print("=" * 80)
    
    # Load 2024 data
    bars_data = load_2024_data()
    
    if not bars_data:
        print("❌ No 2024 data loaded. Please run download_2024_data.py first.")
        return None
    
    # Initialize zone manager
    zone_manager = ZoneManager()
    
    # Generate entry points from bar data
    entry_points = generate_entry_points_from_bars(bars_data, zone_manager)
    
    if not entry_points:
        print("❌ No entry points generated.")
        return None
    
    print(f"\n📊 Processing {len(entry_points)} entry points...")
    
    # Trading parameters
    initial_capital = 100000.0
    position_size_pct = 0.02  # 2% of capital per trade
    commission = 0.0  # $0 commission (most brokers are zero commission now)
    slippage = 0.01  # 1 cent slippage
    
    # Results tracking
    trades = []
    current_capital = initial_capital
    
    # Process each entry point
    for idx, entry in enumerate(entry_points):
        symbol = entry['symbol']
        entry_price = entry['entry_price']
        direction = entry['direction']
        timestamp = entry['timestamp']
        qrs_score = entry['qrs_score']
        vwap = entry['vwap']
        or_high = entry['or_high']
        or_low = entry['or_low']
        zone_level = entry['zone_level']
        
        # Calculate position size
        position_value = current_capital * position_size_pct
        shares = int(position_value / entry_price)
        
        if shares <= 0:
            continue
        
        # Calculate hard stop (zone invalidation)
        if direction == "SHORT":
            # For SHORT: stop is above the zone (zone high + buffer)
            hard_stop = zone_level * 1.002  # 0.2% above zone
        else:  # LONG
            # For LONG: stop is below the zone (zone low - buffer)
            hard_stop = zone_level * 0.998  # 0.2% below zone
        
        # Calculate targets
        t1_price = calculate_proper_t1_price(entry_price, hard_stop, direction, vwap, qrs_score)
        t2_price = calculate_proper_t2_price(entry_price, hard_stop, direction, or_high, or_low)
        t3_price = calculate_proper_t3_price(entry_price, hard_stop, direction)
        
        # Apply slippage
        actual_entry_price = entry_price + (slippage if direction == "LONG" else -slippage)
        
        # Get bars after entry for simulation
        symbol_bars = bars_data[symbol]
        entry_idx = next((i for i, bar in enumerate(symbol_bars) 
                         if bar.timestamp == timestamp), None)
        
        if entry_idx is None:
            continue
        
        # Simulate trade execution
        position_shares = shares
        t1_shares = int(shares * 0.45)  # 45% at T1
        t2_shares = int(shares * 0.25)  # 25% at T2
        t3_shares = shares - t1_shares - t2_shares  # Remaining at T3
        
        trade_result = {
            'symbol': symbol,
            'direction': direction,
            'entry_time': timestamp,
            'entry_price': actual_entry_price,
            'shares': shares,
            'hard_stop': hard_stop,
            't1_price': t1_price,
            't2_price': t2_price,
            't3_price': t3_price,
            'qrs_score': qrs_score,
            'exit_type': None,
            'exit_price': None,
            'exit_time': None,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'bars_in_trade': 0
        }
        
        # Simulate trade from entry onwards
        stop_moved_to_be = False
        t1_hit = False
        t2_hit = False
        
        max_bars_to_check = min(len(symbol_bars) - entry_idx - 1, 390)  # Max 1 trading day
        
        for i in range(1, max_bars_to_check + 1):
            if entry_idx + i >= len(symbol_bars):
                break
            
            bar = symbol_bars[entry_idx + i]
            high = bar.high
            low = bar.low
            close = bar.close
            
            trade_result['bars_in_trade'] = i
            
            # Check exits based on direction
            if direction == "SHORT":
                # Check hard stop first
                if high >= hard_stop:
                    trade_result['exit_type'] = 'HARD_STOP'
                    trade_result['exit_price'] = hard_stop
                    trade_result['exit_time'] = bar.timestamp
                    trade_result['pnl'] = (actual_entry_price - hard_stop) * position_shares
                    break
                
                # Check T1
                if not t1_hit and low <= t1_price:
                    t1_hit = True
                    position_shares -= t1_shares
                    trade_result['pnl'] += (actual_entry_price - t1_price) * t1_shares
                    hard_stop = actual_entry_price  # Move stop to breakeven
                    stop_moved_to_be = True
                
                # Check T2
                if t1_hit and not t2_hit and low <= t2_price:
                    t2_hit = True
                    position_shares -= t2_shares
                    trade_result['pnl'] += (actual_entry_price - t2_price) * t2_shares
                
                # Check T3
                if t2_hit and low <= t3_price:
                    trade_result['exit_type'] = 'T3'
                    trade_result['exit_price'] = t3_price
                    trade_result['exit_time'] = bar.timestamp
                    trade_result['pnl'] += (actual_entry_price - t3_price) * position_shares
                    break
            
            else:  # LONG
                # Check hard stop first
                if low <= hard_stop:
                    trade_result['exit_type'] = 'HARD_STOP'
                    trade_result['exit_price'] = hard_stop
                    trade_result['exit_time'] = bar.timestamp
                    trade_result['pnl'] = (hard_stop - actual_entry_price) * position_shares
                    break
                
                # Check T1
                if not t1_hit and high >= t1_price:
                    t1_hit = True
                    position_shares -= t1_shares
                    trade_result['pnl'] += (t1_price - actual_entry_price) * t1_shares
                    hard_stop = actual_entry_price  # Move stop to breakeven
                    stop_moved_to_be = True
                
                # Check T2
                if t1_hit and not t2_hit and high >= t2_price:
                    t2_hit = True
                    position_shares -= t2_shares
                    trade_result['pnl'] += (t2_price - actual_entry_price) * t2_shares
                
                # Check T3
                if t2_hit and high >= t3_price:
                    trade_result['exit_type'] = 'T3'
                    trade_result['exit_price'] = t3_price
                    trade_result['exit_time'] = bar.timestamp
                    trade_result['pnl'] += (t3_price - actual_entry_price) * position_shares
                    break
        
        # If still in trade at end of day, close at market
        if not trade_result['exit_type']:
            final_bar = symbol_bars[min(entry_idx + max_bars_to_check, len(symbol_bars) - 1)]
            close_price = final_bar.close
            
            if direction == "SHORT":
                trade_result['pnl'] += (actual_entry_price - close_price) * position_shares
            else:
                trade_result['pnl'] += (close_price - actual_entry_price) * position_shares
            
            trade_result['exit_type'] = 'EOD'
            trade_result['exit_price'] = close_price
            trade_result['exit_time'] = final_bar.timestamp
        
        # Calculate P&L percentage
        trade_result['pnl_pct'] = (trade_result['pnl'] / (actual_entry_price * shares)) * 100
        
        # Update capital
        current_capital += trade_result['pnl']
        
        trades.append(trade_result)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("📊 BACKTEST RESULTS - 2024 (1 YEAR)")
    print("=" * 80)
    
    if not trades:
        print("❌ No trades executed")
        return None
    
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    breakeven_trades = [t for t in trades if t['pnl'] == 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = sum([t['pnl'] for t in trades])
    total_wins = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
    total_losses = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    avg_win = total_wins / len(winning_trades) if winning_trades else 0
    avg_loss = total_losses / len(losing_trades) if losing_trades else 0
    
    # Exit type breakdown
    hard_stops = len([t for t in trades if t['exit_type'] == 'HARD_STOP'])
    t3_exits = len([t for t in trades if t['exit_type'] == 'T3'])
    eod_exits = len([t for t in trades if t['exit_type'] == 'EOD'])
    
    # Symbol breakdown
    spy_trades = [t for t in trades if t['symbol'] == 'SPY']
    qqq_trades = [t for t in trades if t['symbol'] == 'QQQ']
    iwm_trades = [t for t in trades if t['symbol'] == 'IWM']
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"   Losing Trades: {len(losing_trades)}")
    print(f"   Breakeven Trades: {len(breakeven_trades)}")
    print(f"\n💰 Profitability:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital: ${current_capital:,.2f}")
    print(f"   Total P&L: ${total_pnl:,.2f}")
    print(f"   Return: {((current_capital - initial_capital) / initial_capital * 100):.2f}%")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"\n📊 Trade Statistics:")
    print(f"   Average Win: ${avg_win:.2f}")
    print(f"   Average Loss: ${avg_loss:.2f}")
    print(f"   Win/Loss Ratio: {(avg_win / avg_loss if avg_loss > 0 else 0):.2f}")
    print(f"\n🎯 Exit Analysis:")
    print(f"   Hard Stops: {hard_stops} ({hard_stops/total_trades*100:.1f}%)")
    print(f"   T3 Exits: {t3_exits} ({t3_exits/total_trades*100:.1f}%)")
    print(f"   EOD Exits: {eod_exits} ({eod_exits/total_trades*100:.1f}%)")
    print(f"\n📊 Symbol Performance:")
    for symbol_trades, symbol in [(spy_trades, 'SPY'), (qqq_trades, 'QQQ'), (iwm_trades, 'IWM')]:
        if symbol_trades:
            sym_pnl = sum([t['pnl'] for t in symbol_trades])
            sym_wins = len([t for t in symbol_trades if t['pnl'] > 0])
            sym_win_rate = sym_wins / len(symbol_trades) * 100
            print(f"   {symbol}: {len(symbol_trades)} trades, Win Rate: {sym_win_rate:.1f}%, P&L: ${sym_pnl:.2f}")
    
    print("\n" + "=" * 80)
    
    # Save results
    results = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'initial_capital': initial_capital,
        'final_capital': current_capital,
        'trades': trades,
        'entry_points_count': len(entry_points)
    }
    
    # Save to file
    output_dir = Path("results/2024/1year_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "backtest_results_2024.json"
    with open(results_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        serializable_trades = []
        for trade in trades:
            trade_copy = trade.copy()
            if isinstance(trade_copy['entry_time'], datetime):
                trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
            if isinstance(trade_copy['exit_time'], datetime):
                trade_copy['exit_time'] = trade_copy['exit_time'].isoformat()
            serializable_trades.append(trade_copy)
        
        serializable_results = {
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'total_pnl': results['total_pnl'],
            'initial_capital': results['initial_capital'],
            'final_capital': results['final_capital'],
            'entry_points_count': results['entry_points_count'],
            'trades': serializable_trades
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_2024_zone_fade_backtest()
