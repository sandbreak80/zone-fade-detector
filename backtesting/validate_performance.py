#!/usr/bin/env python3
"""
Performance Validation Script

Runs full simulation on 290 improved entries to validate:
- Hard stop rate reduction (85% → <50%)
- Win rate improvement (15.9% → >40%)
- Profit factor improvement (0.70 → >1.5)
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class Trade:
    """Represents a complete trade."""
    entry_time: datetime
    entry_price: float
    direction: str
    symbol: str
    qrs: float
    volume_spike: float
    wick_ratio: float
    
    # Targets and stops
    hard_stop: float
    t1_price: float
    t2_price: float
    t3_price: float
    
    # Exit information
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None  # 'HARD_STOP', 'T1', 'T2', 'T3', 'EOD'
    bars_held: int = 0
    pnl: float = 0.0
    
    # Trade details
    risk_units: float = 0.0
    
    def is_winner(self) -> bool:
        """Check if trade was a winner."""
        return self.exit_type in ['T1', 'T2', 'T3']
    
    def hit_hard_stop(self) -> bool:
        """Check if trade hit hard stop."""
        return self.exit_type == 'HARD_STOP'


def load_2024_data() -> Dict[str, List[OHLCVBar]]:
    """Load all 2024 1-minute data."""
    print("📊 Loading 2024 data...")
    
    data_dir = Path(__file__).parent.parent / "data" / "2024"
    all_data = {}
    
    for symbol in ['SPY', 'QQQ', 'IWM']:
        file_path = data_dir / f"{symbol}_2024.pkl"
        
        if not file_path.exists():
            print(f"   ❌ {symbol} data not found at {file_path}")
            continue
        
        with open(file_path, 'rb') as f:
            bars = pickle.load(f)
        
        all_data[symbol] = bars
        print(f"   ✅ {symbol}: {len(bars):,} bars")
    
    print(f"✅ Loaded {sum(len(bars) for bars in all_data.values()):,} total bars\n")
    return all_data


def load_improved_entries() -> List[Dict]:
    """Load the 290 improved entry points."""
    print("📋 Loading improved entry points...")
    
    file_path = Path(__file__).parent.parent / "results" / "2024" / "improved_backtest" / "improved_entry_points.json"
    
    if not file_path.exists():
        print(f"   ❌ Entry points not found at {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract entry_points from the JSON structure
    if isinstance(data, dict) and 'entry_points' in data:
        entries = data['entry_points']
    else:
        entries = data
    
    print(f"✅ Loaded {len(entries)} improved entries\n")
    return entries


def calculate_targets_and_stops(entry_price: float, direction: str, atr: float = 5.0) -> Dict[str, float]:
    """Calculate target and stop prices."""
    
    # Hard stop: 0.5% minimum (improved from original tight stops)
    stop_distance = max(entry_price * 0.005, atr * 0.5)
    
    if direction == 'LONG':
        hard_stop = entry_price - stop_distance
        t1 = entry_price + (stop_distance * 1.0)  # 1R
        t2 = entry_price + (stop_distance * 2.0)  # 2R
        t3 = entry_price + (stop_distance * 3.0)  # 3R
    else:  # SHORT
        hard_stop = entry_price + stop_distance
        t1 = entry_price - (stop_distance * 1.0)  # 1R
        t2 = entry_price - (stop_distance * 2.0)  # 2R
        t3 = entry_price - (stop_distance * 3.0)  # 3R
    
    return {
        'hard_stop': hard_stop,
        't1': t1,
        't2': t2,
        't3': t3,
        'risk_units': stop_distance
    }


def simulate_trade(entry: Dict, bars: List[OHLCVBar], entry_index: int) -> Trade:
    """Simulate a complete trade from entry to exit."""
    
    # Parse entry information
    entry_time = datetime.fromisoformat(entry['timestamp'])
    entry_price = entry['entry_price']
    direction = entry['direction']
    symbol = entry['symbol']
    qrs = entry.get('qrs_score', entry.get('qrs', 0))
    volume_spike = entry.get('volume_spike', 0)
    wick_ratio = entry.get('wick_ratio', 0)
    
    # Calculate targets and stops
    targets = calculate_targets_and_stops(entry_price, direction)
    
    # Create trade
    trade = Trade(
        entry_time=entry_time,
        entry_price=entry_price,
        direction=direction,
        symbol=symbol,
        qrs=qrs,
        volume_spike=volume_spike,
        wick_ratio=wick_ratio,
        hard_stop=targets['hard_stop'],
        t1_price=targets['t1'],
        t2_price=targets['t2'],
        t3_price=targets['t3'],
        risk_units=targets['risk_units']
    )
    
    # Simulate from entry forward
    eod_cutoff = entry_time.replace(hour=15, minute=55, second=0)
    
    for i in range(entry_index + 1, len(bars)):
        bar = bars[i]
        trade.bars_held += 1
        
        # Check EOD first
        if bar.timestamp >= eod_cutoff:
            trade.exit_time = bar.timestamp
            trade.exit_price = bar.close
            trade.exit_type = 'EOD'
            break
        
        if direction == 'LONG':
            # Check hard stop first (low)
            if bar.low <= trade.hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            # Check T3 (high)
            if bar.high >= trade.t3_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t3_price
                trade.exit_type = 'T3'
                break
            
            # Check T2
            if bar.high >= trade.t2_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t2_price
                trade.exit_type = 'T2'
                break
            
            # Check T1
            if bar.high >= trade.t1_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t1_price
                trade.exit_type = 'T1'
                break
        
        else:  # SHORT
            # Check hard stop first (high)
            if bar.high >= trade.hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            # Check T3 (low)
            if bar.low <= trade.t3_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t3_price
                trade.exit_type = 'T3'
                break
            
            # Check T2
            if bar.low <= trade.t2_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t2_price
                trade.exit_type = 'T2'
                break
            
            # Check T1
            if bar.low <= trade.t1_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t1_price
                trade.exit_type = 'T1'
                break
    
    # Calculate P&L
    if trade.exit_price:
        if direction == 'LONG':
            trade.pnl = trade.exit_price - entry_price
        else:
            trade.pnl = entry_price - trade.exit_price
    
    return trade


def run_validation() -> Dict:
    """Run complete validation simulation."""
    
    print("=" * 80)
    print("🔬 PERFORMANCE VALIDATION - IMPROVED ENTRIES")
    print("=" * 80)
    print()
    
    # Load data
    all_data = load_2024_data()
    entries = load_improved_entries()
    
    if not entries:
        print("❌ No entries to validate")
        return {}
    
    # Simulate all trades
    print("🎮 Simulating trades...")
    trades: List[Trade] = []
    
    for entry in entries:
        symbol = entry['symbol']
        
        if symbol not in all_data:
            continue
        
        bars = all_data[symbol]
        entry_time = datetime.fromisoformat(entry['timestamp'])
        
        # Find entry bar
        entry_index = None
        for i, bar in enumerate(bars):
            if bar.timestamp >= entry_time:
                entry_index = i
                break
        
        if entry_index is None:
            continue
        
        # Simulate trade
        trade = simulate_trade(entry, bars, entry_index)
        trades.append(trade)
    
    print(f"✅ Simulated {len(trades)} trades\n")
    
    # Analyze results
    print("=" * 80)
    print("📊 VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    # Overall statistics
    total_trades = len(trades)
    hard_stops = sum(1 for t in trades if t.hit_hard_stop())
    winners = sum(1 for t in trades if t.is_winner())
    
    hard_stop_rate = (hard_stops / total_trades * 100) if total_trades > 0 else 0
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum(t.pnl for t in trades)
    winning_pnl = sum(t.pnl for t in trades if t.is_winner())
    losing_pnl = abs(sum(t.pnl for t in trades if not t.is_winner()))
    
    profit_factor = (winning_pnl / losing_pnl) if losing_pnl > 0 else 0
    
    print(f"📈 **OVERALL PERFORMANCE**")
    print(f"   Total Trades: {total_trades}")
    print(f"   Hard Stop Rate: {hard_stop_rate:.1f}%")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print()
    
    # Exit breakdown
    exit_counts = {}
    for trade in trades:
        exit_type = trade.exit_type or 'UNKNOWN'
        exit_counts[exit_type] = exit_counts.get(exit_type, 0) + 1
    
    print(f"📊 **EXIT BREAKDOWN**")
    for exit_type, count in sorted(exit_counts.items()):
        pct = (count / total_trades * 100) if total_trades > 0 else 0
        print(f"   {exit_type}: {count} ({pct:.1f}%)")
    print()
    
    # Symbol breakdown
    print(f"📈 **BY SYMBOL**")
    for symbol in ['SPY', 'QQQ', 'IWM']:
        symbol_trades = [t for t in trades if t.symbol == symbol]
        if not symbol_trades:
            continue
        
        sym_total = len(symbol_trades)
        sym_winners = sum(1 for t in symbol_trades if t.is_winner())
        sym_hard_stops = sum(1 for t in symbol_trades if t.hit_hard_stop())
        sym_win_rate = (sym_winners / sym_total * 100) if sym_total > 0 else 0
        sym_hard_stop_rate = (sym_hard_stops / sym_total * 100) if sym_total > 0 else 0
        sym_pnl = sum(t.pnl for t in symbol_trades)
        
        print(f"   {symbol}:")
        print(f"      Trades: {sym_total}")
        print(f"      Win Rate: {sym_win_rate:.1f}%")
        print(f"      Hard Stop Rate: {sym_hard_stop_rate:.1f}%")
        print(f"      P&L: ${sym_pnl:.2f}")
    print()
    
    # Direction breakdown
    print(f"📊 **BY DIRECTION**")
    for direction in ['LONG', 'SHORT']:
        dir_trades = [t for t in trades if t.direction == direction]
        if not dir_trades:
            continue
        
        dir_total = len(dir_trades)
        dir_winners = sum(1 for t in dir_trades if t.is_winner())
        dir_hard_stops = sum(1 for t in dir_trades if t.hit_hard_stop())
        dir_win_rate = (dir_winners / dir_total * 100) if dir_total > 0 else 0
        dir_hard_stop_rate = (dir_hard_stops / dir_total * 100) if dir_total > 0 else 0
        dir_pnl = sum(t.pnl for t in dir_trades)
        
        print(f"   {direction}:")
        print(f"      Trades: {dir_total}")
        print(f"      Win Rate: {dir_win_rate:.1f}%")
        print(f"      Hard Stop Rate: {dir_hard_stop_rate:.1f}%")
        print(f"      P&L: ${dir_pnl:.2f}")
    print()
    
    # QRS analysis
    winning_trades = [t for t in trades if t.is_winner()]
    hard_stop_trades = [t for t in trades if t.hit_hard_stop()]
    
    if winning_trades and hard_stop_trades:
        avg_winner_qrs = statistics.mean(t.qrs for t in winning_trades)
        avg_hard_stop_qrs = statistics.mean(t.qrs for t in hard_stop_trades)
        
        print(f"🎯 **QRS ANALYSIS**")
        print(f"   Winning Trade Avg QRS: {avg_winner_qrs:.2f}")
        print(f"   Hard Stop Avg QRS: {avg_hard_stop_qrs:.2f}")
        print(f"   Difference: {avg_winner_qrs - avg_hard_stop_qrs:+.2f}")
        print()
    
    # Comparison to original
    print("=" * 80)
    print("📊 COMPARISON TO ORIGINAL BACKTEST")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<25} {'Original':<15} {'Improved':<15} {'Change':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'Trades':<25} {453:<15} {total_trades:<15} {total_trades-453:<15}")
    print(f"{'Hard Stop Rate':<25} {'85.0%':<15} {f'{hard_stop_rate:.1f}%':<15} {f'{hard_stop_rate-85.0:+.1f}%':<15}")
    print(f"{'Win Rate':<25} {'15.9%':<15} {f'{win_rate:.1f}%':<15} {f'{win_rate-15.9:+.1f}%':<15}")
    print(f"{'Profit Factor':<25} {'0.70':<15} {f'{profit_factor:.2f}':<15} {f'{profit_factor-0.70:+.2f}':<15}")
    print(f"{'P&L':<25} {'-$242.89':<15} {f'${total_pnl:.2f}':<15} {f'${total_pnl+242.89:+.2f}':<15}")
    print()
    
    # Target validation
    print("=" * 80)
    print("🎯 TARGET VALIDATION")
    print("=" * 80)
    print()
    
    targets = {
        'Hard Stop Rate': {
            'target': '<50%',
            'actual': f'{hard_stop_rate:.1f}%',
            'met': hard_stop_rate < 50
        },
        'Win Rate': {
            'target': '>40%',
            'actual': f'{win_rate:.1f}%',
            'met': win_rate > 40
        },
        'Profit Factor': {
            'target': '>1.5',
            'actual': f'{profit_factor:.2f}',
            'met': profit_factor > 1.5
        }
    }
    
    for metric, data in targets.items():
        status = "✅ MET" if data['met'] else "❌ NOT MET"
        print(f"   {metric}:")
        print(f"      Target: {data['target']}")
        print(f"      Actual: {data['actual']}")
        print(f"      Status: {status}")
        print()
    
    targets_met = sum(1 for data in targets.values() if data['met'])
    print(f"📊 Targets Met: {targets_met}/3")
    print()
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_trades': total_trades,
        'hard_stop_rate': hard_stop_rate,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'exit_breakdown': exit_counts,
        'targets_met': targets_met,
        'targets': targets,
        'trades': [
            {
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'exit_type': t.exit_type,
                'pnl': t.pnl,
                'qrs': t.qrs,
                'bars_held': t.bars_held
            }
            for t in trades
        ]
    }
    
    output_dir = Path(__file__).parent.parent / "results" / "2024" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "performance_validation_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Final assessment
    print("=" * 80)
    print("🎉 VALIDATION COMPLETE")
    print("=" * 80)
    print()
    
    if targets_met == 3:
        print("✅ **SUCCESS!** All targets met!")
        print("   The improvements have been validated.")
        print("   System is ready for paper trading.")
    elif targets_met >= 2:
        print("🎯 **PARTIAL SUCCESS** - Most targets met")
        print("   System shows significant improvement.")
        print("   Consider fine-tuning for remaining targets.")
    else:
        print("⚠️  **NEEDS IMPROVEMENT** - Targets not met")
        print("   Further optimization may be needed.")
    
    print()
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_validation()
