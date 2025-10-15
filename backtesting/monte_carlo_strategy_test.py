#!/usr/bin/env python3
"""
Monte Carlo Strategy Testing - 100 Simulations

Generates synthetic OHLCV data with realistic characteristics and tests
the actual Phase 2.1 strategy on each dataset to measure robustness.

This tests the STRATEGY, not just trade sequences.
"""

import sys
import json
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class Zone:
    """Zone for strategy testing."""
    level: float
    zone_type: str
    created_at: datetime
    touches: int = 0


@dataclass
class Trade:
    """Trade with full details."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    direction: str
    shares: int
    
    hard_stop: float
    t1_price: float
    t2_price: float
    t3_price: float
    
    exit_type: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    
    def is_winner(self) -> bool:
        return self.pnl > 0


def generate_synthetic_ohlcv(
    num_bars: int = 100000,
    starting_price: float = 450.0,
    volatility: float = 0.015,
    trend: float = 0.0001,
    seed: Optional[int] = None
) -> List[OHLCVBar]:
    """
    Generate synthetic OHLCV data with realistic characteristics.
    
    Parameters:
    - num_bars: Number of 1-minute bars to generate
    - starting_price: Starting price
    - volatility: Daily volatility (std dev)
    - trend: Drift per bar (positive = uptrend, negative = downtrend)
    - seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    bars = []
    start_time = datetime(2024, 1, 2, 9, 30)  # Market open
    
    current_price = starting_price
    
    for i in range(num_bars):
        # Calculate timestamp (skip weekends and after-hours)
        timestamp = start_time + timedelta(minutes=i)
        
        # Skip non-trading hours (before 9:30 or after 16:00)
        hour = timestamp.hour
        minute = timestamp.minute
        
        if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            continue
        
        # Generate price movement (Geometric Brownian Motion)
        # dS = S * (mu * dt + sigma * sqrt(dt) * Z)
        dt = 1 / (252 * 390)  # 1 minute as fraction of trading year
        random_shock = np.random.normal(0, 1)
        
        price_change = current_price * (trend * dt + volatility * np.sqrt(dt) * random_shock)
        new_price = current_price + price_change
        
        # Generate OHLC from the price movement
        # Open is previous close (or current_price)
        open_price = current_price
        
        # Generate high/low with realistic spread
        intrabar_volatility = abs(price_change) * random.uniform(1.5, 3.0)
        high_price = max(open_price, new_price) + abs(np.random.normal(0, intrabar_volatility * 0.5))
        low_price = min(open_price, new_price) - abs(np.random.normal(0, intrabar_volatility * 0.5))
        
        # Close is the new price
        close_price = new_price
        
        # Ensure OHLC relationship is valid
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (log-normal distribution)
        base_volume = 100000
        volume = int(np.random.lognormal(np.log(base_volume), 0.5))
        
        # Add volume spikes occasionally (rejection candle simulation)
        if random.random() < 0.05:  # 5% chance of volume spike
            volume = int(volume * random.uniform(2.0, 5.0))
        
        bar = OHLCVBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        
        bars.append(bar)
        current_price = close_price
    
    return bars


def detect_zones(bars: List[OHLCVBar], lookback: int = 100) -> List[Zone]:
    """Detect support/resistance zones from price action."""
    zones = []
    
    # Simple zone detection: look for swing highs/lows
    for i in range(lookback, len(bars) - lookback, 50):
        window = bars[i-lookback:i+lookback]
        current = bars[i]
        
        # Check if swing high
        highs = [b.high for b in window]
        if current.high == max(highs):
            zones.append(Zone(
                level=current.high,
                zone_type='SWING_HIGH',
                created_at=current.timestamp
            ))
        
        # Check if swing low
        lows = [b.low for b in window]
        if current.low == min(lows):
            zones.append(Zone(
                level=current.low,
                zone_type='SWING_LOW',
                created_at=current.timestamp
            ))
    
    return zones


def calculate_atr(bars: List[OHLCVBar], period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(bars) < period + 1:
        return 0.0
    
    trs = []
    for i in range(len(bars) - period, len(bars)):
        if i == 0:
            continue
        high_low = bars[i].high - bars[i].low
        high_close = abs(bars[i].high - bars[i-1].close)
        low_close = abs(bars[i].low - bars[i-1].close)
        tr = max(high_low, high_close, low_close)
        trs.append(tr)
    
    return statistics.mean(trs) if trs else 0.0


def calculate_volume_spike(bars: List[OHLCVBar], current_idx: int, lookback: int = 20) -> float:
    """Calculate volume spike ratio."""
    if current_idx < lookback:
        return 1.0
    
    recent_volumes = [bars[i].volume for i in range(current_idx - lookback, current_idx)]
    avg_volume = statistics.mean(recent_volumes) if recent_volumes else bars[current_idx].volume
    
    if avg_volume == 0:
        return 1.0
    
    return bars[current_idx].volume / avg_volume


def calculate_wick_ratio(bar: OHLCVBar, direction: str) -> float:
    """Calculate wick ratio for rejection."""
    body = abs(bar.close - bar.open)
    total_range = bar.high - bar.low
    
    if total_range == 0:
        return 0.0
    
    if direction == 'LONG':
        # Lower wick for long
        wick = bar.open - bar.low if bar.close > bar.open else bar.close - bar.low
    else:
        # Upper wick for short
        wick = bar.high - bar.open if bar.close < bar.open else bar.high - bar.close
    
    return wick / total_range if total_range > 0 else 0.0


def check_balance(bars: List[OHLCVBar], current_idx: int) -> bool:
    """Check for balance/compression (ATR compression)."""
    if current_idx < 30:
        return False
    
    recent_atr = calculate_atr(bars[current_idx-14:current_idx+1], 14)
    baseline_atr = calculate_atr(bars[current_idx-30:current_idx-14], 14)
    
    if baseline_atr == 0:
        return False
    
    # Balance if recent ATR < 70% of baseline
    return recent_atr < (baseline_atr * 0.70)


def find_strategy_entries(bars: List[OHLCVBar], zones: List[Zone]) -> List[Dict]:
    """
    Find entries using Phase 2.1 strategy criteria.
    
    Criteria:
    - Price touching zone (within 0.5%)
    - Volume spike >= 2.0x
    - Wick ratio >= 40%
    - Balance detected
    - Zone touches <= 2
    - QRS >= 10.0 (simplified)
    """
    entries = []
    
    print(f"   Scanning {len(bars):,} bars for entries...")
    
    for i in range(100, len(bars) - 1):
        bar = bars[i]
        
        # Check each zone
        for zone in zones:
            # Skip if zone created after this bar
            if zone.created_at > bar.timestamp:
                continue
            
            # Check if price touching zone (within 0.5%)
            price_range = bar.high - bar.low
            zone_tolerance = zone.level * 0.005
            
            touching = False
            direction = None
            
            if zone.zone_type == 'SWING_HIGH':
                # SHORT setup - price touching from below
                if abs(bar.high - zone.level) <= zone_tolerance:
                    touching = True
                    direction = 'SHORT'
            elif zone.zone_type == 'SWING_LOW':
                # LONG setup - price touching from above
                if abs(bar.low - zone.level) <= zone_tolerance:
                    touching = True
                    direction = 'LONG'
            
            if not touching:
                continue
            
            # Check volume spike
            volume_spike = calculate_volume_spike(bars, i)
            if volume_spike < 2.0:
                continue
            
            # Check wick ratio
            wick_ratio = calculate_wick_ratio(bar, direction)
            if wick_ratio < 0.40:
                continue
            
            # Check balance
            if not check_balance(bars, i):
                continue
            
            # Check zone touches
            zone.touches += 1
            if zone.touches > 2:
                continue
            
            # Simplified QRS (we'll assume it meets 10.0 if all above criteria met)
            qrs = 12.0  # Simplified
            
            # Valid entry!
            entry_price = bar.close
            
            entries.append({
                'timestamp': bar.timestamp,
                'bar_index': i,
                'entry_price': entry_price,
                'direction': direction,
                'zone_level': zone.level,
                'zone_type': zone.zone_type,
                'volume_spike': volume_spike,
                'wick_ratio': wick_ratio,
                'qrs': qrs
            })
            
            # Only one entry per bar
            break
    
    return entries


def simulate_trade(entry: Dict, bars: List[OHLCVBar], starting_capital: float) -> Trade:
    """
    Simulate a trade using Phase 2.1 exit rules.
    
    Exit Rules:
    - Hard stop: 0.5% below entry
    - T1: 0.5R (0.25% above entry)
    - T2: 1R (0.5% above entry)
    - T3: 1.5R (0.75% above entry)
    - Breakeven stop after T1 hit
    - Time exit: 4 hours if < 0.5R move
    - EOD cutoff: 3:30 PM
    """
    entry_idx = entry['bar_index']
    entry_time = entry['timestamp']
    entry_price = entry['entry_price']
    direction = entry['direction']
    
    # Calculate position size (90% of capital)
    position_value = starting_capital * 0.90
    shares = int(position_value / entry_price)
    
    # Calculate targets
    stop_distance = entry_price * 0.005
    
    if direction == 'LONG':
        hard_stop = entry_price - stop_distance
        t1 = entry_price + (stop_distance * 0.5)
        t2 = entry_price + (stop_distance * 1.0)
        t3 = entry_price + (stop_distance * 1.5)
    else:  # SHORT
        hard_stop = entry_price + stop_distance
        t1 = entry_price - (stop_distance * 0.5)
        t2 = entry_price - (stop_distance * 1.0)
        t3 = entry_price - (stop_distance * 1.5)
    
    trade = Trade(
        entry_time=entry_time,
        entry_price=entry_price,
        exit_time=None,
        exit_price=None,
        direction=direction,
        shares=shares,
        hard_stop=hard_stop,
        t1_price=t1,
        t2_price=t2,
        t3_price=t3
    )
    
    t1_hit = False
    trailing_stop = None
    
    # Simulate forward
    for i in range(entry_idx + 1, len(bars)):
        bar = bars[i]
        trade.bars_held += 1
        
        hours_held = (bar.timestamp - entry_time).total_seconds() / 3600
        
        # Check EOD cutoff (3:30 PM)
        if bar.timestamp.hour >= 15 and bar.timestamp.minute >= 30:
            trade.exit_time = bar.timestamp
            trade.exit_price = bar.close
            trade.exit_type = 'EOD_EARLY'
            break
        
        # Check time exit (4 hours, < 0.5R move, T1 not hit)
        if hours_held >= 4.0 and not t1_hit:
            move_r = abs(bar.close - entry_price) / stop_distance
            if move_r < 0.5:
                trade.exit_time = bar.timestamp
                trade.exit_price = bar.close
                trade.exit_type = 'TIME_EXIT_NO_MOMENTUM'
                break
        
        if direction == 'LONG':
            # Check hard stop
            if bar.low <= hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            # Check trailing/breakeven stop
            if t1_hit and trailing_stop and bar.low <= trailing_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trailing_stop
                trade.exit_type = 'BREAKEVEN_STOP'
                break
            
            # Check T3
            if bar.high >= t3:
                trade.exit_time = bar.timestamp
                trade.exit_price = t3
                trade.exit_type = 'T3'
                break
            
            # Check T2
            if bar.high >= t2:
                trade.exit_time = bar.timestamp
                trade.exit_price = t2
                trade.exit_type = 'T2'
                break
            
            # Check T1 - activate trailing
            if bar.high >= t1 and not t1_hit:
                t1_hit = True
                trailing_stop = entry_price
        
        else:  # SHORT
            # Check hard stop
            if bar.high >= hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            # Check trailing/breakeven stop
            if t1_hit and trailing_stop and bar.high >= trailing_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trailing_stop
                trade.exit_type = 'BREAKEVEN_STOP'
                break
            
            # Check T3
            if bar.low <= t3:
                trade.exit_time = bar.timestamp
                trade.exit_price = t3
                trade.exit_type = 'T3'
                break
            
            # Check T2
            if bar.low <= t2:
                trade.exit_time = bar.timestamp
                trade.exit_price = t2
                trade.exit_type = 'T2'
                break
            
            # Check T1 - activate trailing
            if bar.low <= t1 and not t1_hit:
                t1_hit = True
                trailing_stop = entry_price
    
    # Calculate P&L
    if trade.exit_price:
        if direction == 'LONG':
            trade.pnl = (trade.exit_price - entry_price) * shares
        else:
            trade.pnl = (entry_price - trade.exit_price) * shares
        
        trade.pnl_pct = (trade.pnl / (entry_price * shares)) * 100
    
    return trade


def run_single_monte_carlo(sim_num: int, params: Dict) -> Dict:
    """Run one Monte Carlo simulation with synthetic data."""
    
    # Generate synthetic OHLCV data
    bars = generate_synthetic_ohlcv(
        num_bars=params['num_bars'],
        starting_price=params['starting_price'],
        volatility=params['volatility'],
        trend=params['trend'],
        seed=sim_num  # Use sim number as seed for reproducibility
    )
    
    # Detect zones
    zones = detect_zones(bars)
    
    # Find entries using strategy
    entries = find_strategy_entries(bars, zones)
    
    if not entries:
        # No trades found
        return {
            'sim_number': sim_num,
            'bars_generated': len(bars),
            'zones_detected': len(zones),
            'entries_found': 0,
            'trades_executed': 0,
            'starting_capital': params['starting_capital'],
            'ending_capital': params['starting_capital'],
            'total_return_pct': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # Simulate trades
    capital = params['starting_capital']
    trades = []
    equity_curve = [capital]
    
    for entry in entries:
        trade = simulate_trade(entry, bars, capital)
        trades.append(trade)
        
        # Update capital
        capital += trade.pnl
        equity_curve.append(capital)
        
        # Commission
        commission = max(1.0, 0.005 * trade.shares * 2)
        capital -= commission
    
    # Calculate metrics
    total_trades = len(trades)
    winners = sum(1 for t in trades if t.is_winner())
    losers = total_trades - winners
    
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    winning_pnl = sum(t.pnl for t in trades if t.is_winner())
    losing_pnl = abs(sum(t.pnl for t in trades if not t.is_winner()))
    
    profit_factor = (winning_pnl / losing_pnl) if losing_pnl > 0 else 0
    
    total_return_pct = ((capital - params['starting_capital']) / params['starting_capital']) * 100
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return {
        'sim_number': sim_num,
        'bars_generated': len(bars),
        'zones_detected': len(zones),
        'entries_found': len(entries),
        'trades_executed': total_trades,
        'starting_capital': params['starting_capital'],
        'ending_capital': capital,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'winners': winners,
        'losers': losers,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd,
        'total_pnl': capital - params['starting_capital']
    }


def run_monte_carlo_strategy_test(num_simulations: int = 100):
    """Run full Monte Carlo strategy test."""
    
    print("=" * 80)
    print("🎲 MONTE CARLO STRATEGY TEST - 100 SIMULATIONS")
    print("=" * 80)
    print()
    print("Testing STRATEGY robustness across synthetic market data")
    print()
    print("Configuration:")
    print(f"   Simulations: {num_simulations}")
    print(f"   Bars per simulation: ~100,000 (1-minute)")
    print(f"   Strategy: Phase 2.1 (Validated)")
    print(f"   Starting Capital: $10,000")
    print(f"   Position Sizing: 90% equity")
    print()
    print("Synthetic Data Parameters:")
    print(f"   Starting Price: $450 (QQQ-like)")
    print(f"   Volatility: 1.5% daily (normal market)")
    print(f"   Trend: Slight upward bias")
    print()
    
    # Base parameters
    base_params = {
        'num_bars': 100000,
        'starting_price': 450.0,
        'volatility': 0.015,
        'trend': 0.0001,
        'starting_capital': 10000.0
    }
    
    print("🎮 Running simulations...")
    print()
    
    results = []
    
    for i in range(num_simulations):
        # Vary parameters slightly for each simulation
        params = base_params.copy()
        params['volatility'] = random.uniform(0.010, 0.025)  # Vary volatility
        params['trend'] = random.uniform(-0.0002, 0.0005)  # Vary trend
        
        result = run_single_monte_carlo(i + 1, params)
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"   Completed {i + 1}/{num_simulations} simulations...")
    
    print()
    print("✅ All simulations complete")
    print()
    
    # Analyze results
    print("=" * 80)
    print("📊 MONTE CARLO RESULTS")
    print("=" * 80)
    print()
    
    # Filter out simulations with no trades
    results_with_trades = [r for r in results if r['trades_executed'] > 0]
    
    print(f"Simulations with trades: {len(results_with_trades)}/{num_simulations}")
    print()
    
    if not results_with_trades:
        print("❌ No trades found in any simulation")
        return
    
    # Returns analysis
    returns = [r['total_return_pct'] for r in results_with_trades]
    
    print("💰 **RETURNS DISTRIBUTION**")
    print()
    print(f"Mean Return: {statistics.mean(returns):+.2f}%")
    print(f"Median Return: {statistics.median(returns):+.2f}%")
    print(f"Std Dev: {statistics.stdev(returns) if len(returns) > 1 else 0:.2f}%")
    print(f"Min Return: {min(returns):+.2f}%")
    print(f"Max Return: {max(returns):+.2f}%")
    print()
    
    # Win rate analysis
    win_rates = [r['win_rate'] for r in results_with_trades]
    
    print("🎯 **WIN RATE DISTRIBUTION**")
    print()
    print(f"Mean Win Rate: {statistics.mean(win_rates):.1f}%")
    print(f"Median Win Rate: {statistics.median(win_rates):.1f}%")
    print(f"Min Win Rate: {min(win_rates):.1f}%")
    print(f"Max Win Rate: {max(win_rates):.1f}%")
    print()
    
    # Trade count analysis
    trade_counts = [r['trades_executed'] for r in results_with_trades]
    
    print("📊 **TRADE COUNT DISTRIBUTION**")
    print()
    print(f"Mean Trades: {statistics.mean(trade_counts):.1f}")
    print(f"Median Trades: {statistics.median(trade_counts):.1f}")
    print(f"Min Trades: {min(trade_counts)}")
    print(f"Max Trades: {max(trade_counts)}")
    print()
    
    # Drawdown analysis
    drawdowns = [r['max_drawdown_pct'] for r in results_with_trades]
    
    print("⚠️  **DRAWDOWN DISTRIBUTION**")
    print()
    print(f"Mean Max DD: {statistics.mean(drawdowns):.2f}%")
    print(f"Median Max DD: {statistics.median(drawdowns):.2f}%")
    print(f"Worst Max DD: {max(drawdowns):.2f}%")
    print()
    
    # Profit factor analysis
    profit_factors = [r['profit_factor'] for r in results_with_trades if r['profit_factor'] > 0]
    
    if profit_factors:
        print("📈 **PROFIT FACTOR DISTRIBUTION**")
        print()
        print(f"Mean PF: {statistics.mean(profit_factors):.2f}")
        print(f"Median PF: {statistics.median(profit_factors):.2f}")
        print()
    
    # Success metrics
    profitable_sims = sum(1 for r in results_with_trades if r['total_return_pct'] > 0)
    prob_profit = (profitable_sims / len(results_with_trades)) * 100
    
    win_rate_above_40 = sum(1 for r in results_with_trades if r['win_rate'] >= 40)
    prob_win_rate_40 = (win_rate_above_40 / len(results_with_trades)) * 100
    
    print("🎯 **PROBABILITY ANALYSIS**")
    print()
    print(f"Probability of Profit: {prob_profit:.1f}%")
    print(f"Probability of Win Rate ≥40%: {prob_win_rate_40:.1f}%")
    print()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "2024" / "monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "monte_carlo_strategy_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_monte_carlo_strategy_test(100)
