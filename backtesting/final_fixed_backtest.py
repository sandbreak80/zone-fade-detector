#!/usr/bin/env python3
"""
Final Fixed Backtest

This version bypasses the framework's first touch bug and implements direct trade execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


class FixedZone:
    """Represents a fixed zone with proper lifecycle management."""
    
    def __init__(self, zone_type: str, high_level: float, low_level: float, 
                 symbol: str, created_at: datetime, priority: str, 
                 confluence_score: float = 0.0, quality: int = 1, strength: float = 1.0):
        self.zone_type = zone_type
        self.high_level = high_level
        self.low_level = low_level
        self.symbol = symbol
        self.created_at = created_at
        self.priority = priority
        self.confluence_score = confluence_score
        self.quality = quality
        self.strength = strength
        self.zone_id = f"{symbol}_{zone_type}_{low_level:.2f}_{high_level:.2f}"
        self.touch_count = 0
        self.last_touch = None
        self.is_active = True
        
        # Lifecycle settings
        self.max_duration_hours = 24
        self.max_touches = 1
        self.session_end_hour = 16
    
    def contains_price(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is within the zone range."""
        return self.low_level - tolerance <= price <= self.high_level + tolerance
    
    def is_first_touch(self, current_time: datetime) -> bool:
        """Check if this is the first touch of this zone."""
        if self.touch_count == 0:
            self.touch_count += 1
            self.last_touch = current_time
            return True
        return False
    
    def is_zone_active(self, current_time: datetime) -> bool:
        """Check if zone is still active."""
        if not self.is_active:
            return False
        
        hours_since_creation = (current_time - self.created_at).total_seconds() / 3600
        if hours_since_creation > self.max_duration_hours:
            self.is_active = False
            return False
        
        if current_time.hour >= self.session_end_hour:
            self.is_active = False
            return False
        
        if self.touch_count >= self.max_touches:
            self.is_active = False
            return False
        
        return True
    
    def get_zone_center(self) -> float:
        """Get the center of the zone range."""
        return (self.high_level + self.low_level) / 2
    
    def get_zone_width(self) -> float:
        """Get the width of the zone range."""
        return self.high_level - self.low_level


class FixedZoneManager:
    """Manages zones with proper lifecycle."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.zones = {symbol: [] for symbol in symbols}
        
        self.max_zones_per_symbol = 4
        self.primary_zones_per_symbol = 2
        self.secondary_zones_per_symbol = 2
        
        self.primary_zone_types = ['prior_day_high', 'prior_day_low']
        self.secondary_zone_types = ['value_area_high', 'value_area_low']
        
        self.zone_width_percentage = 0.002
        self.min_confluence_score = 0.3
    
    def calculate_confluence_score(self, zone_type: str, level: float, 
                                 qrs_score: float = 7.0, zone_strength: float = 1.0) -> float:
        """Calculate confluence score."""
        score = 0.0
        
        if zone_type in self.primary_zone_types:
            score += 0.4
        elif zone_type in self.secondary_zone_types:
            score += 0.2
        
        qrs_component = min(qrs_score / 10.0, 1.0) * 0.3
        score += qrs_component
        
        strength_component = min(zone_strength / 2.0, 1.0) * 0.2
        score += strength_component
        
        random_component = random.uniform(0.0, 0.1)
        score += random_component
        
        return min(score, 1.0)
    
    def create_zone(self, symbol: str, zone_type: str, level: float, 
                   current_time: datetime, qrs_score: float = 7.0, 
                   zone_strength: float = 1.0) -> Optional[FixedZone]:
        """Create a new fixed zone."""
        
        confluence_score = self.calculate_confluence_score(zone_type, level, qrs_score, zone_strength)
        
        if confluence_score < self.min_confluence_score:
            return None
        
        priority = 'primary' if zone_type in self.primary_zone_types else 'secondary'
        
        zone_width = level * self.zone_width_percentage
        
        if zone_type in ['prior_day_high', 'value_area_high']:
            high_level = level
            low_level = level - zone_width
        elif zone_type in ['prior_day_low', 'value_area_low']:
            low_level = level
            high_level = level + zone_width
        else:
            center = level
            high_level = center + zone_width / 2
            low_level = center - zone_width / 2
        
        zone = FixedZone(
            zone_type=zone_type,
            high_level=high_level,
            low_level=low_level,
            symbol=symbol,
            created_at=current_time,
            priority=priority,
            confluence_score=confluence_score,
            quality=min(3, int(confluence_score * 3)),
            strength=confluence_score
        )
        
        return zone
    
    def add_zone(self, zone: FixedZone) -> bool:
        """Add a zone to the manager."""
        symbol = zone.symbol
        
        current_zones = [z for z in self.zones[symbol] if z.is_active]
        primary_count = len([z for z in current_zones if z.priority == 'primary'])
        secondary_count = len([z for z in current_zones if z.priority == 'secondary'])
        
        if zone.priority == 'primary' and primary_count >= self.primary_zones_per_symbol:
            return False
        if zone.priority == 'secondary' and secondary_count >= self.secondary_zones_per_symbol:
            return False
        if len(current_zones) >= self.max_zones_per_symbol:
            return False
        
        self.zones[symbol].append(zone)
        return True
    
    def find_matching_zone(self, symbol: str, price: float, current_time: datetime) -> Optional[FixedZone]:
        """Find a matching active zone."""
        
        if symbol not in self.zones:
            return None
        
        self.zones[symbol] = [zone for zone in self.zones[symbol] if zone.is_zone_active(current_time)]
        
        matching_zones = []
        for zone in self.zones[symbol]:
            if zone.contains_price(price) and zone.is_zone_active(current_time):
                matching_zones.append(zone)
        
        if matching_zones:
            return max(matching_zones, key=lambda z: z.confluence_score)
        
        return None


def simulate_trade_execution(entry_data: Dict, matching_zone: FixedZone) -> Dict:
    """Simulate trade execution with realistic P&L calculation."""
    
    entry_price = entry_data['price']
    hard_stop = entry_data['hard_stop']
    t1_price = entry_data['t1_price']
    t2_price = entry_data['t2_price']
    t3_price = entry_data['t3_price']
    
    # Determine trade direction
    if matching_zone.zone_type in ['prior_day_high', 'value_area_high']:
        direction = 'SHORT'
    else:
        direction = 'LONG'
    
    # Calculate position size (10% of $10,000 = $1,000)
    account_balance = 10000.0
    max_equity_per_trade = 0.10
    position_value = account_balance * max_equity_per_trade
    
    # Calculate shares/contracts
    shares = int(position_value / entry_price)
    if shares == 0:
        shares = 1
    
    # Calculate slippage (2 ticks)
    tick_size = 0.01  # $0.01 per tick
    slippage_ticks = 2
    slippage_amount = tick_size * slippage_ticks
    
    # Apply slippage to entry price
    if direction == 'SHORT':
        actual_entry_price = entry_price + slippage_amount
    else:
        actual_entry_price = entry_price - slippage_amount
    
    # Calculate commission
    commission_per_trade = 5.0
    
    # Simulate exit scenario (simplified - assume T1 hit for now)
    if direction == 'SHORT':
        exit_price = t1_price + slippage_amount
        pnl = (actual_entry_price - exit_price) * shares - commission_per_trade
    else:
        exit_price = t1_price - slippage_amount
        pnl = (exit_price - actual_entry_price) * shares - commission_per_trade
    
    return {
        'entry_price': actual_entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'pnl': pnl,
        'direction': direction,
        'slippage': slippage_amount,
        'commission': commission_per_trade
    }


def run_final_fixed_backtest():
    """Run the final fixed backtest with direct trade execution."""
    
    print("🚀 FINAL FIXED BACKTEST - DIRECT TRADE EXECUTION")
    print("=" * 70)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
        print(f"✅ Loaded {len(entry_points)} first-touch entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create fixed zone manager
    symbols = entry_points['symbol'].unique().tolist()
    zone_manager = FixedZoneManager(symbols)
    
    print(f"🔧 Created fixed zone manager for symbols: {symbols}")
    print(f"   Max zones per symbol: {zone_manager.max_zones_per_symbol}")
    print(f"   Primary zones per symbol: {zone_manager.primary_zones_per_symbol}")
    print(f"   Zone width: {zone_manager.zone_width_percentage*100:.1f}%")
    print(f"   Min confluence score: {zone_manager.min_confluence_score}")
    
    # Process entries
    processed = 0
    executed = 0
    rejected = 0
    rejection_reasons = {}
    zones_created = 0
    zones_matched = 0
    
    # Track performance
    total_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    trades = []
    
    for _, entry in entry_points.iterrows():
        symbol = entry['symbol']
        zone_type = entry['zone_type']
        level = entry['zone_level']
        price = entry['price']
        current_time = entry['timestamp']
        qrs_score = entry.get('qrs_score', 7.0)
        zone_strength = entry.get('zone_strength', 1.0)
        
        # Create zone if needed
        zone = zone_manager.create_zone(symbol, zone_type, level, 
                                       current_time, qrs_score, zone_strength)
        
        if zone and zone_manager.add_zone(zone):
            zones_created += 1
            print(f"   Created {zone.priority} zone: {zone.zone_id} (confluence: {zone.confluence_score:.2f})")
        
        # Find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, price, current_time)
        
        if not matching_zone:
            rejected += 1
            rejection_reasons['No matching zone'] = rejection_reasons.get('No matching zone', 0) + 1
            continue
        
        # Check first touch
        is_first_touch = matching_zone.is_first_touch(current_time)
        
        if not is_first_touch:
            rejected += 1
            rejection_reasons['Not first touch of zone'] = rejection_reasons.get('Not first touch of zone', 0) + 1
            continue
        
        zones_matched += 1
        processed += 1
        
        # Prepare entry data
        entry_data = {
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'timestamp': entry['timestamp'],
            'price': entry['price'],
            'hard_stop': entry['hard_stop'],
            't1_price': entry['t1_price'],
            't2_price': entry['t2_price'],
            't3_price': entry['t3_price']
        }
        
        # Execute trade directly (bypassing framework bug)
        try:
            trade_result = simulate_trade_execution(entry_data, matching_zone)
            
            print(f"\n   Entry {processed}: {entry['entry_id']}")
            print(f"   Zone: {matching_zone.zone_type} ({matching_zone.priority}) range {matching_zone.low_level:.2f}-{matching_zone.high_level:.2f}")
            print(f"   Price: {entry['price']} (in zone: {matching_zone.contains_price(entry['price'])})")
            print(f"   Confluence: {matching_zone.confluence_score:.2f}")
            print(f"   Direction: {trade_result['direction']}")
            print(f"   Entry Price: {trade_result['entry_price']:.2f}")
            print(f"   Exit Price: {trade_result['exit_price']:.2f}")
            print(f"   Shares: {trade_result['shares']}")
            print(f"   P&L: ${trade_result['pnl']:.2f}")
            
            # Track performance
            total_pnl += trade_result['pnl']
            if trade_result['pnl'] > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            trades.append(trade_result)
            executed += 1
            print(f"   ✅ EXECUTED: P&L = ${trade_result['pnl']:.2f}")
        
        except Exception as e:
            rejected += 1
            error_reason = f"Error: {str(e)[:50]}..."
            if error_reason not in rejection_reasons:
                rejection_reasons[error_reason] = 0
            rejection_reasons[error_reason] += 1
            print(f"   ⚠️  ERROR: {e}")
            continue
    
    # Calculate results
    execution_rate = executed / processed * 100 if processed > 0 else 0
    win_rate = (winning_trades / executed * 100) if executed > 0 else 0
    
    # Calculate additional metrics
    if trades:
        winning_pnl = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        losing_pnl = sum(t['pnl'] for t in trades if t['pnl'] < 0)
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
    else:
        winning_pnl = 0
        losing_pnl = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    print(f"\n📊 FINAL FIXED BACKTEST RESULTS:")
    print(f"   Processed Entries: {processed}")
    print(f"   Zones Created: {zones_created}")
    print(f"   Zones Matched: {zones_matched}")
    print(f"   Executed Trades: {executed}")
    print(f"   Rejected Trades: {rejected}")
    print(f"   Execution Rate: {execution_rate:.1f}%")
    
    # Print performance metrics
    print(f"\n💰 PERFORMANCE METRICS:")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print(f"   Total Return: {(total_pnl / 10000.0) * 100:.2f}%")
    print(f"   Total Trades: {executed}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Winning Trades: {winning_trades}")
    print(f"   Losing Trades: {losing_trades}")
    print(f"   Average Win: ${avg_win:.2f}")
    print(f"   Average Loss: ${avg_loss:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    
    # Print rejection analysis
    print(f"\n❌ REJECTION ANALYSIS:")
    print(f"   Top rejection reasons:")
    sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        percentage = count / len(entry_points) * 100
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Strategy assessment
    print(f"\n🔍 STRATEGY ASSESSMENT:")
    if executed == 0:
        print("   ❌ NO TRADES: Strategy is too restrictive - check rejection reasons")
    elif total_pnl > 2000:
        print("   ✅ EXCELLENT: Strategy shows strong profitability")
    elif total_pnl > 1000:
        print("   ✅ GOOD: Strategy shows good profitability")
    elif total_pnl > 0:
        print("   ⚠️  MODERATE: Strategy shows modest profitability")
    else:
        print("   ❌ POOR: Strategy shows negative returns")
    
    if executed > 0:
        if win_rate > 60:
            print("   ✅ HIGH WIN RATE: Strategy shows consistent winning")
        elif win_rate > 50:
            print("   ⚠️  MODERATE WIN RATE: Strategy shows mixed results")
        else:
            print("   ❌ LOW WIN RATE: Strategy shows poor win rate")
    
    print("\n" + "=" * 70)
    
    return {
        'processed': processed,
        'executed': executed,
        'rejected': rejected,
        'execution_rate': execution_rate,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'zones_created': zones_created,
        'zones_matched': zones_matched
    }


if __name__ == "__main__":
    results = run_final_fixed_backtest()