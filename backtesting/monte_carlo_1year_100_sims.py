#!/usr/bin/env python3
"""
Monte Carlo - 100 Full 1-Year Backtests

Generates 100 synthetic 1-year datasets (210k bars each like real QQQ)
and runs the complete Phase 2.1 strategy backtest on each one.

This provides statistical distribution of outcomes across different
market scenarios.
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
    """Zone for strategy."""
    level: float
    zone_type: str
    created_at: datetime
    touches: int = 0
    session_touches: Dict[str, int] = None
    
    def __post_init__(self):
        if self.session_touches is None:
            self.session_touches = {}


@dataclass
class Trade:
    """Complete trade."""
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
    bars_held: int = 0
    t1_hit: bool = False
    
    def is_winner(self) -> bool:
        return self.pnl > 0


def generate_full_year_synthetic_ohlcv(
    starting_price: float = 450.0,
    volatility: float = 0.015,
    trend: float = 0.0001,
    seed: Optional[int] = None
) -> List[OHLCVBar]:
    """
    Generate full year of synthetic 1-minute OHLCV data (~210k bars).
    
    Parameters:
    - starting_price: Starting price
    - volatility: Daily volatility (annualized std dev)
    - trend: Daily drift (positive = uptrend)
    - seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    bars = []
    start_time = datetime(2024, 1, 2, 9, 30)
    current_price = starting_price
    
    # Generate 252 trading days * 390 minutes = ~98,280 bars
    # We'll generate more to account for randomness and get ~210k
    num_trading_days = 252
    minutes_per_day = 390  # 9:30 AM to 4:00 PM
    
    current_date = start_time.date()
    bar_count = 0
    
    for day in range(num_trading_days):
        # Skip weekends
        while current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            current_date += timedelta(days=1)
        
        # Generate bars for this trading day
        for minute in range(minutes_per_day):
            hour = 9 + (minute // 60)
            min_in_hour = minute % 60
            
            if hour == 9 and min_in_hour < 30:
                continue
            if hour >= 16:
                break
            
            timestamp = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour, minutes=min_in_hour)
            
            # Price movement (Geometric Brownian Motion)
            dt = 1 / (252 * 390)  # 1 minute as fraction of year
            random_shock = np.random.normal(0, 1)
            
            price_change = current_price * (trend * dt + volatility * np.sqrt(dt) * random_shock)
            new_price = max(1.0, current_price + price_change)  # Prevent negative prices
            
            # Generate OHLC
            open_price = current_price
            close_price = new_price
            
            # Realistic high/low
            intrabar_vol = abs(price_change) * random.uniform(1.5, 3.0)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, intrabar_vol * 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, intrabar_vol * 0.5))
            
            # Ensure valid OHLC
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume (log-normal)
            base_volume = 100000
            volume = int(np.random.lognormal(np.log(base_volume), 0.5))
            
            # Occasional volume spikes (5% of bars)
            if random.random() < 0.05:
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
            bar_count += 1
        
        # Move to next trading day
        current_date += timedelta(days=1)
    
    return bars


def detect_zones_improved(bars: List[OHLCVBar]) -> List[Zone]:
    """Detect zones using swing highs/lows and prior day high/low."""
    zones = []
    
    # Detect swing highs/lows
    lookback = 50
    for i in range(lookback, len(bars) - lookback, 100):
        window = bars[i-lookback:i+lookback]
        current = bars[i]
        
        highs = [b.high for b in window]
        lows = [b.low for b in window]
        
        if current.high >= max(highs) * 0.999:  # Within 0.1%
            zones.append(Zone(
                level=current.high,
                zone_type='SWING_HIGH',
                created_at=current.timestamp
            ))
        
        if current.low <= min(lows) * 1.001:
            zones.append(Zone(
                level=current.low,
                zone_type='SWING_LOW',
                created_at=current.timestamp
            ))
    
    # Add daily high/low zones
    daily_data = {}
    for bar in bars:
        date_key = bar.timestamp.date()
        if date_key not in daily_data:
            daily_data[date_key] = {'high': bar.high, 'low': bar.low}
        else:
            daily_data[date_key]['high'] = max(daily_data[date_key]['high'], bar.high)
            daily_data[date_key]['low'] = min(daily_data[date_key]['low'], bar.low)
    
    for date, data in daily_data.items():
        zones.append(Zone(
            level=data['high'],
            zone_type='PRIOR_DAY_HIGH',
            created_at=datetime.combine(date, datetime.min.time())
        ))
        zones.append(Zone(
            level=data['low'],
            zone_type='PRIOR_DAY_LOW',
            created_at=datetime.combine(date, datetime.min.time())
        ))
    
    return zones


def calculate_atr(bars: List[OHLCVBar], period: int = 14) -> float:
    """Calculate ATR."""
    if len(bars) < period + 1:
        return bars[-1].high - bars[-1].low if bars else 0.0
    
    trs = []
    for i in range(len(bars) - period, len(bars)):
        if i == 0:
            continue
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        trs.append(tr)
    
    return statistics.mean(trs) if trs else 0.0


def calculate_volume_spike(bars: List[OHLCVBar], idx: int, lookback: int = 20) -> float:
    """Calculate volume spike."""
    if idx < lookback:
        return 1.0
    
    recent = [bars[i].volume for i in range(idx - lookback, idx)]
    avg = statistics.mean(recent) if recent else bars[idx].volume
    
    return bars[idx].volume / avg if avg > 0 else 1.0


def calculate_wick_ratio(bar: OHLCVBar, direction: str) -> float:
    """Calculate wick ratio."""
    total_range = bar.high - bar.low
    if total_range == 0:
        return 0.0
    
    if direction == 'LONG':
        wick = bar.open - bar.low if bar.close > bar.open else bar.close - bar.low
    else:
        wick = bar.high - bar.open if bar.close < bar.open else bar.high - bar.close
    
    return wick / total_range


def check_balance(bars: List[OHLCVBar], idx: int) -> bool:
    """Check for balance/compression."""
    if idx < 30:
        return False
    
    recent_atr = calculate_atr(bars[idx-14:idx+1], 14)
    baseline_atr = calculate_atr(bars[idx-30:idx-14], 14)
    
    if baseline_atr == 0:
        return False
    
    return recent_atr < (baseline_atr * 0.70)


def find_entries_phase21(bars: List[OHLCVBar], zones: List[Zone]) -> List[Dict]:
    """Find entries using Phase 2.1 criteria."""
    entries = []
    
    for i in range(100, len(bars) - 1):
        bar = bars[i]
        current_date = bar.timestamp.date()
        
        # Reset zone session touches at 9:30 AM
        if bar.timestamp.hour == 9 and bar.timestamp.minute == 30:
            for zone in zones:
                session_key = str(current_date)
                if session_key not in zone.session_touches:
                    zone.session_touches[session_key] = 0
        
        for zone in zones:
            if zone.created_at > bar.timestamp:
                continue
            
            # Check if touching zone
            zone_tolerance = zone.level * 0.005
            touching = False
            direction = None
            
            if zone.zone_type in ['SWING_HIGH', 'PRIOR_DAY_HIGH']:
                if abs(bar.high - zone.level) <= zone_tolerance:
                    touching = True
                    direction = 'SHORT'
            elif zone.zone_type in ['SWING_LOW', 'PRIOR_DAY_LOW']:
                if abs(bar.low - zone.level) <= zone_tolerance:
                    touching = True
                    direction = 'LONG'
            
            if not touching:
                continue
            
            # Check criteria
            volume_spike = calculate_volume_spike(bars, i)
            if volume_spike < 2.0:
                continue
            
            wick_ratio = calculate_wick_ratio(bar, direction)
            if wick_ratio < 0.40:
                continue
            
            if not check_balance(bars, i):
                continue
            
            # Check session touches
            session_key = str(current_date)
            if session_key not in zone.session_touches:
                zone.session_touches[session_key] = 0
            
            if zone.session_touches[session_key] >= 2:
                continue
            
            zone.session_touches[session_key] += 1
            
            entries.append({
                'timestamp': bar.timestamp,
                'bar_index': i,
                'entry_price': bar.close,
                'direction': direction,
                'zone_level': zone.level,
                'volume_spike': volume_spike,
                'wick_ratio': wick_ratio
            })
            
            break
    
    return entries


def simulate_trade_phase21(entry: Dict, bars: List[OHLCVBar], capital: float) -> Trade:
    """Simulate trade with Phase 2.1 rules."""
    entry_idx = entry['bar_index']
    entry_time = entry['timestamp']
    entry_price = entry['entry_price']
    direction = entry['direction']
    
    # Position size
    position_value = capital * 0.90
    shares = int(position_value / entry_price)
    
    # Targets
    stop_distance = entry_price * 0.005
    
    if direction == 'LONG':
        hard_stop = entry_price - stop_distance
        t1 = entry_price + (stop_distance * 0.5)
        t2 = entry_price + (stop_distance * 1.0)
        t3 = entry_price + (stop_distance * 1.5)
    else:
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
    
    trailing_stop = None
    
    for i in range(entry_idx + 1, len(bars)):
        bar = bars[i]
        trade.bars_held += 1
        
        hours_held = (bar.timestamp - entry_time).total_seconds() / 3600
        
        # EOD cutoff 3:30 PM
        if bar.timestamp.hour >= 15 and bar.timestamp.minute >= 30:
            trade.exit_time = bar.timestamp
            trade.exit_price = bar.close
            trade.exit_type = 'EOD_EARLY'
            break
        
        # Time exit: 4 hours, <0.5R move, T1 not hit
        if hours_held >= 4.0 and not trade.t1_hit:
            move_r = abs(bar.close - entry_price) / stop_distance
            if move_r < 0.5:
                trade.exit_time = bar.timestamp
                trade.exit_price = bar.close
                trade.exit_type = 'TIME_EXIT_NO_MOMENTUM'
                break
        
        if direction == 'LONG':
            if bar.low <= hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            if trade.t1_hit and trailing_stop and bar.low <= trailing_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trailing_stop
                trade.exit_type = 'BREAKEVEN_STOP'
                break
            
            if bar.high >= t3:
                trade.exit_time = bar.timestamp
                trade.exit_price = t3
                trade.exit_type = 'T3'
                break
            
            if bar.high >= t2:
                trade.exit_time = bar.timestamp
                trade.exit_price = t2
                trade.exit_type = 'T2'
                break
            
            if bar.high >= t1 and not trade.t1_hit:
                trade.t1_hit = True
                trailing_stop = entry_price
        
        else:  # SHORT
            if bar.high >= hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            if trade.t1_hit and trailing_stop and bar.high >= trailing_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trailing_stop
                trade.exit_type = 'BREAKEVEN_STOP'
                break
            
            if bar.low <= t3:
                trade.exit_time = bar.timestamp
                trade.exit_price = t3
                trade.exit_type = 'T3'
                break
            
            if bar.low <= t2:
                trade.exit_time = bar.timestamp
                trade.exit_price = t2
                trade.exit_type = 'T2'
                break
            
            if bar.low <= t1 and not trade.t1_hit:
                trade.t1_hit = True
                trailing_stop = entry_price
    
    # Calculate P&L
    if trade.exit_price:
        if direction == 'LONG':
            trade.pnl = (trade.exit_price - entry_price) * shares
        else:
            trade.pnl = (entry_price - trade.exit_price) * shares
        
        # Commission
        commission = max(1.0, 0.005 * shares * 2)
        trade.pnl -= commission
    
    return trade


def run_single_simulation(sim_num: int, params: Dict) -> Dict:
    """Run one full 1-year simulation."""
    print(f"   Simulation {sim_num}/100...")
    
    # Generate synthetic data
    bars = generate_full_year_synthetic_ohlcv(
        starting_price=params['starting_price'],
        volatility=params['volatility'],
        trend=params['trend'],
        seed=sim_num
    )
    
    # Detect zones
    zones = detect_zones_improved(bars)
    
    # Find entries
    entries = find_entries_phase21(bars, zones)
    
    if not entries:
        return {
            'sim_number': sim_num,
            'bars': len(bars),
            'zones': len(zones),
            'entries': 0,
            'trades': 0,
            'starting_capital': params['starting_capital'],
            'ending_capital': params['starting_capital'],
            'return_pct': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_dd': 0.0
        }
    
    # Simulate trades
    capital = params['starting_capital']
    trades = []
    equity_curve = [capital]
    
    for entry in entries:
        trade = simulate_trade_phase21(entry, bars, capital)
        trades.append(trade)
        capital += trade.pnl
        equity_curve.append(capital)
    
    # Calculate metrics
    total_trades = len(trades)
    winners = sum(1 for t in trades if t.is_winner())
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    winning_pnl = sum(t.pnl for t in trades if t.is_winner())
    losing_pnl = abs(sum(t.pnl for t in trades if not t.is_winner()))
    profit_factor = (winning_pnl / losing_pnl) if losing_pnl > 0 else 0
    
    return_pct = ((capital - params['starting_capital']) / params['starting_capital']) * 100
    
    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {
        'sim_number': sim_num,
        'bars': len(bars),
        'zones': len(zones),
        'entries': len(entries),
        'trades': total_trades,
        'starting_capital': params['starting_capital'],
        'ending_capital': capital,
        'return_pct': return_pct,
        'win_rate': win_rate,
        'winners': winners,
        'losers': total_trades - winners,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'total_pnl': capital - params['starting_capital']
    }


def run_monte_carlo_100_years():
    """Run 100 full 1-year Monte Carlo simulations."""
    
    print("=" * 80)
    print("🎲 MONTE CARLO - 100 FULL 1-YEAR BACKTESTS")
    print("=" * 80)
    print()
    print("Configuration:")
    print("   Simulations: 100")
    print("   Bars per simulation: ~98,000 (full year of 1-min data)")
    print("   Strategy: Phase 2.1 (Validated)")
    print("   Starting Capital: $10,000")
    print("   Position Sizing: 90% equity")
    print()
    print("This will take 5-10 minutes...")
    print()
    
    base_params = {
        'starting_price': 450.0,
        'volatility': 0.018,  # ~1.8% daily
        'trend': 0.00015,  # Slight upward bias
        'starting_capital': 10000.0
    }
    
    print("🎮 Running 100 full-year simulations...")
    print()
    
    results = []
    
    for i in range(100):
        params = base_params.copy()
        # Vary parameters
        params['volatility'] = random.uniform(0.012, 0.025)
        params['trend'] = random.uniform(-0.0003, 0.0006)
        params['starting_price'] = random.uniform(400, 500)
        
        result = run_single_simulation(i + 1, params)
        results.append(result)
    
    print()
    print("✅ All simulations complete!")
    print()
    
    # Analyze
    results_with_trades = [r for r in results if r['trades'] > 0]
    
    print("=" * 80)
    print("📊 MONTE CARLO RESULTS")
    print("=" * 80)
    print()
    
    print(f"Simulations with trades: {len(results_with_trades)}/100")
    print()
    
    if not results_with_trades:
        print("❌ No trades in any simulation")
        return
    
    # Returns
    returns = [r['return_pct'] for r in results_with_trades]
    returns_sorted = sorted(returns)
    
    print("💰 **RETURNS DISTRIBUTION**")
    print()
    print(f"Mean: {statistics.mean(returns):+.2f}%")
    print(f"Median: {statistics.median(returns):+.2f}%")
    print(f"Std Dev: {statistics.stdev(returns):.2f}%")
    print(f"Min: {min(returns):+.2f}%")
    print(f"Max: {max(returns):+.2f}%")
    print()
    print(f" 5th percentile: {returns_sorted[int(len(returns_sorted)*0.05)]:+.2f}%")
    print(f"25th percentile: {returns_sorted[int(len(returns_sorted)*0.25)]:+.2f}%")
    print(f"75th percentile: {returns_sorted[int(len(returns_sorted)*0.75)]:+.2f}%")
    print(f"95th percentile: {returns_sorted[int(len(returns_sorted)*0.95)]:+.2f}%")
    print()
    
    # Win rate
    win_rates = [r['win_rate'] for r in results_with_trades]
    
    print("🎯 **WIN RATE DISTRIBUTION**")
    print()
    print(f"Mean: {statistics.mean(win_rates):.1f}%")
    print(f"Median: {statistics.median(win_rates):.1f}%")
    print(f"Min: {min(win_rates):.1f}%")
    print(f"Max: {max(win_rates):.1f}%")
    print()
    
    # Trades
    trades = [r['trades'] for r in results_with_trades]
    
    print("📊 **TRADE COUNT**")
    print()
    print(f"Mean: {statistics.mean(trades):.1f}")
    print(f"Median: {statistics.median(trades):.1f}")
    print(f"Min: {min(trades)}")
    print(f"Max: {max(trades)}")
    print()
    
    # Drawdown
    drawdowns = [r['max_dd'] for r in results_with_trades]
    
    print("⚠️  **MAX DRAWDOWN**")
    print()
    print(f"Mean: {statistics.mean(drawdowns):.2f}%")
    print(f"Median: {statistics.median(drawdowns):.2f}%")
    print(f"Worst: {max(drawdowns):.2f}%")
    print()
    
    # Probabilities
    profitable = sum(1 for r in results_with_trades if r['return_pct'] > 0)
    win_rate_above_40 = sum(1 for r in results_with_trades if r['win_rate'] >= 40)
    
    print("🎯 **PROBABILITIES**")
    print()
    print(f"Probability of Profit: {(profitable/len(results_with_trades))*100:.1f}%")
    print(f"Probability of Win Rate ≥40%: {(win_rate_above_40/len(results_with_trades))*100:.1f}%")
    print()
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "2024" / "monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "monte_carlo_100_year_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_monte_carlo_100_years()
