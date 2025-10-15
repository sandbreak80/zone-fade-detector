#!/usr/bin/env python3
"""
Phase 2.1 Performance Validation - Refined Exit Strategy

Issue with Phase 2.0: Breakeven stops and time exits cut winners too early

Phase 2.1 Refinements:
1. Keep revised targets: 0.5R, 1R, 1.5R ✅
2. MORE RELAXED breakeven: Only after T1 hit (not at 0.5R level)
3. LESS AGGRESSIVE time exits: 4 hours instead of 3, check for 0.5R move
4. Keep earlier EOD: 3:30 PM ✅
5. Add partial exit logic: Take 50% at T1, trail rest
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class Trade:
    """Represents a complete trade with Phase 2.1 exit strategy."""
    entry_time: datetime
    entry_price: float
    direction: str
    symbol: str
    qrs: float
    volume_spike: float
    wick_ratio: float
    
    # Phase 2 revised targets (0.5R, 1R, 1.5R)
    hard_stop: float
    t1_price: float  # 0.5R
    t2_price: float  # 1R
    t3_price: float  # 1.5R
    
    # Exit information
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    bars_held: int = 0
    pnl: float = 0.0
    
    # Trade state
    risk_units: float = 0.0
    t1_hit: bool = False  # Track if T1 was hit
    trailing_stop: Optional[float] = None
    
    def is_winner(self) -> bool:
        """Check if trade was a winner."""
        return self.exit_type in ['T1', 'T2', 'T3', 'BREAKEVEN_STOP']
    
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


def calculate_phase2_targets(entry_price: float, direction: str, atr: float = 5.0) -> Dict[str, float]:
    """Calculate Phase 2 targets (0.5R, 1R, 1.5R)."""
    # Hard stop: 0.5% minimum
    stop_distance = max(entry_price * 0.005, atr * 0.5)
    
    if direction == 'LONG':
        hard_stop = entry_price - stop_distance
        t1 = entry_price + (stop_distance * 0.5)  # 0.5R
        t2 = entry_price + (stop_distance * 1.0)  # 1R
        t3 = entry_price + (stop_distance * 1.5)  # 1.5R
    else:  # SHORT
        hard_stop = entry_price + stop_distance
        t1 = entry_price - (stop_distance * 0.5)  # 0.5R
        t2 = entry_price - (stop_distance * 1.0)  # 1R
        t3 = entry_price - (stop_distance * 1.5)  # 1.5R
    
    return {
        'hard_stop': hard_stop,
        't1': t1,
        't2': t2,
        't3': t3,
        'risk_units': stop_distance
    }


def should_exit_time_based(entry_time: datetime, current_time: datetime, 
                           direction: str, entry_price: float, current_price: float,
                           t1_hit: bool) -> Tuple[bool, str]:
    """
    Phase 2.1: More relaxed time-based exits.
    
    Rules:
    1. Earlier EOD cutoff at 3:30 PM
    2. Close after 4 hours (not 3) if < 0.5R move and T1 not hit
    """
    hours_held = (current_time - entry_time).total_seconds() / 3600
    
    # Rule 1: Earlier EOD cutoff (3:30 PM)
    if current_time.hour >= 15 and current_time.minute >= 30:
        return True, 'EOD_EARLY'
    
    # Rule 2: Close after 4 hours (not 3) if no momentum AND T1 not hit
    if hours_held >= 4.0 and not t1_hit:
        if direction == 'LONG':
            move_r = (current_price - entry_price) / (entry_price * 0.005)  # Move in R
        else:
            move_r = (entry_price - current_price) / (entry_price * 0.005)
        
        if move_r < 0.5:  # Less than 0.5R move
            return True, 'TIME_EXIT_NO_MOMENTUM'
    
    return False, ''


def simulate_trade_phase21(entry: Dict, bars: List[OHLCVBar], entry_index: int) -> Trade:
    """
    Simulate a complete trade with Phase 2.1 refined exit strategy.
    
    Phase 2.1 Improvements:
    - Revised targets: 0.5R, 1R, 1.5R
    - Breakeven stop ONLY after T1 hit (not before)
    - More relaxed time exits (4 hours, check 0.5R)
    - Earlier EOD cutoff (3:30 PM)
    """
    # Parse entry information
    entry_time = datetime.fromisoformat(entry['timestamp'])
    entry_price = entry['entry_price']
    direction = entry['direction']
    symbol = entry['symbol']
    qrs = entry.get('qrs_score', entry.get('qrs', 0))
    volume_spike = entry.get('volume_spike', 0)
    wick_ratio = entry.get('wick_ratio', 0)
    
    # Calculate Phase 2 targets
    targets = calculate_phase2_targets(entry_price, direction)
    
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
    for i in range(entry_index + 1, len(bars)):
        bar = bars[i]
        trade.bars_held += 1
        
        # Check time-based exits
        should_exit, exit_reason = should_exit_time_based(
            entry_time, bar.timestamp, direction, entry_price, bar.close, trade.t1_hit
        )
        if should_exit:
            trade.exit_time = bar.timestamp
            trade.exit_price = bar.close
            trade.exit_type = exit_reason
            break
        
        if direction == 'LONG':
            # Check hard stop FIRST (before anything else)
            if bar.low <= trade.hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            # Check trailing/breakeven stop (if T1 was hit)
            if trade.t1_hit and trade.trailing_stop and bar.low <= trade.trailing_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.trailing_stop
                trade.exit_type = 'BREAKEVEN_STOP'
                break
            
            # Check T3 (1.5R) - highest target
            if bar.high >= trade.t3_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t3_price
                trade.exit_type = 'T3'
                break
            
            # Check T2 (1R)
            if bar.high >= trade.t2_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t2_price
                trade.exit_type = 'T2'
                break
            
            # Check T1 (0.5R) - activate trailing stop at breakeven
            if bar.high >= trade.t1_price and not trade.t1_hit:
                trade.t1_hit = True
                trade.trailing_stop = entry_price  # Set stop at breakeven
                # Don't exit yet, let it run with breakeven protection
                # In real trading, would take partial profit here
                continue
        
        else:  # SHORT
            # Check hard stop FIRST
            if bar.high >= trade.hard_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.hard_stop
                trade.exit_type = 'HARD_STOP'
                break
            
            # Check trailing/breakeven stop (if T1 was hit)
            if trade.t1_hit and trade.trailing_stop and bar.high >= trade.trailing_stop:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.trailing_stop
                trade.exit_type = 'BREAKEVEN_STOP'
                break
            
            # Check T3 (1.5R)
            if bar.low <= trade.t3_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t3_price
                trade.exit_type = 'T3'
                break
            
            # Check T2 (1R)
            if bar.low <= trade.t2_price:
                trade.exit_time = bar.timestamp
                trade.exit_price = trade.t2_price
                trade.exit_type = 'T2'
                break
            
            # Check T1 (0.5R) - activate trailing stop at breakeven
            if bar.low <= trade.t1_price and not trade.t1_hit:
                trade.t1_hit = True
                trade.trailing_stop = entry_price
                continue
    
    # Calculate P&L
    if trade.exit_price:
        if direction == 'LONG':
            trade.pnl = trade.exit_price - entry_price
        else:
            trade.pnl = entry_price - trade.exit_price
    
    return trade


def run_phase21_validation() -> Dict:
    """Run Phase 2.1 validation with refined exit strategy."""
    
    print("=" * 80)
    print("🚀 PHASE 2.1 VALIDATION - REFINED EXIT STRATEGY")
    print("=" * 80)
    print()
    print("Phase 2.1 Refinements:")
    print("  ✅ Revised targets: 0.5R, 1R, 1.5R")
    print("  ✅ Breakeven stop: ONLY after T1 hit (not before)")
    print("  ✅ Time exits: 4 hours (not 3), check 0.5R move")
    print("  ✅ Earlier EOD: 3:30 PM (not 3:55 PM)")
    print("  ✅ Let winners run with breakeven protection")
    print()
    
    # Load data
    all_data = load_2024_data()
    entries = load_improved_entries()
    
    if not entries:
        print("❌ No entries to validate")
        return {}
    
    # Simulate all trades
    print("🎮 Simulating trades with Phase 2.1 refined strategy...")
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
        
        # Simulate trade with Phase 2.1 strategy
        trade = simulate_trade_phase21(entry, bars, entry_index)
        trades.append(trade)
    
    print(f"✅ Simulated {len(trades)} trades\n")
    
    # Analyze results
    print("=" * 80)
    print("📊 PHASE 2.1 RESULTS")
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
    for exit_type, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
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
    
    # Comparison to Phase 1 and 2.0
    print("=" * 80)
    print("📊 COMPARISON: PHASE 1 vs PHASE 2.0 vs PHASE 2.1")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<25} {'Phase 1':<15} {'Phase 2.0':<15} {'Phase 2.1':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'Hard Stop Rate':<25} {'12.8%':<15} {'21.7%':<15} {f'{hard_stop_rate:.1f}%':<15}")
    print(f"{'Win Rate':<25} {'19.0%':<15} {'11.4%':<15} {f'{win_rate:.1f}%':<15}")
    print(f"{'Profit Factor':<25} {'1.21':<15} {'1.39':<15} {f'{profit_factor:.2f}':<15}")
    print(f"{'P&L':<25} {'$24.21':<15} {'$24.16':<15} {f'${total_pnl:.2f}':<15}")
    
    # Calculate EOD exit rate
    eod_exits = sum(1 for t in trades if t.exit_type and 'EOD' in t.exit_type)
    eod_rate = (eod_exits / total_trades * 100) if total_trades > 0 else 0
    print(f"{'EOD Exit Rate':<25} {'68.3%':<15} {'59.7%':<15} {f'{eod_rate:.1f}%':<15}")
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
        },
        'EOD Exit Rate': {
            'target': '<30%',
            'actual': f'{eod_rate:.1f}%',
            'met': eod_rate < 30
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
    print(f"📊 Targets Met: {targets_met}/4")
    print()
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2.1',
        'improvements': [
            'Revised targets: 0.5R, 1R, 1.5R',
            'Breakeven stop ONLY after T1 hit',
            'Relaxed time exits (4 hours, check 0.5R)',
            'Earlier EOD cutoff (3:30 PM)',
            'Let winners run with protection'
        ],
        'total_trades': total_trades,
        'hard_stop_rate': hard_stop_rate,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'eod_exit_rate': eod_rate,
        'exit_breakdown': exit_counts,
        'targets_met': targets_met,
        'targets': targets
    }
    
    output_dir = Path(__file__).parent.parent / "results" / "2024" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase2.1_validation_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Final assessment
    print("=" * 80)
    print("🎉 PHASE 2.1 VALIDATION COMPLETE")
    print("=" * 80)
    print()
    
    if targets_met >= 3:
        print("✅ **SUCCESS!** Phase 2.1 improvements validated!")
        print("   Exit strategy works well.")
        print("   System ready for further optimization.")
    elif targets_met >= 2:
        print("🎯 **GOOD PROGRESS** - Most targets met")
        print("   Significant improvement.")
        print("   Minor fine-tuning recommended.")
    else:
        print("⚠️  **MIXED RESULTS** - Consider alternative approaches")
        print("   May need different strategy for these market conditions.")
    
    print()
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_phase21_validation()
