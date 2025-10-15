#!/usr/bin/env python3
"""
Intraday Zone Strategy

Implements proper intraday zone detection with appropriate persistence and quantity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


class IntradayZone:
    """Represents an intraday zone with proper persistence management."""
    
    def __init__(self, zone_type: str, high_level: float, low_level: float, 
                 symbol: str, created_at: datetime, quality: int = 1, strength: float = 1.0):
        self.zone_type = zone_type
        self.high_level = high_level
        self.low_level = low_level
        self.symbol = symbol
        self.created_at = created_at
        self.quality = quality
        self.strength = strength
        self.zone_id = f"{symbol}_{zone_type}_{low_level:.2f}_{high_level:.2f}"
        self.touch_count = 0
        self.last_touch = None
        self.is_active = True
        
        # Persistence settings
        self.max_duration_hours = 6  # Zones expire after 6 hours
        self.max_touches = 3  # Zones expire after 3 touches
        self.session_end_hour = 16  # 4 PM ET (end of regular session)
    
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
        """Check if zone is still active based on persistence rules."""
        if not self.is_active:
            return False
        
        # Check time-based expiration
        hours_since_creation = (current_time - self.created_at).total_seconds() / 3600
        if hours_since_creation > self.max_duration_hours:
            self.is_active = False
            return False
        
        # Check session-based expiration
        if current_time.hour >= self.session_end_hour:
            self.is_active = False
            return False
        
        # Check touch-based expiration
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


class IntradayZoneManager:
    """Manages intraday zones with proper persistence and quantity control."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.zones = {symbol: [] for symbol in symbols}
        self.zone_creation_threshold = 0.002  # 0.2% price movement
        self.zone_width_percentage = 0.002  # 0.2% zone width
        self.max_zones_per_symbol = 20  # Maximum zones per symbol
    
    def create_zone(self, symbol: str, zone_type: str, level: float, 
                   current_time: datetime, quality: int = 1, strength: float = 1.0) -> IntradayZone:
        """Create a new intraday zone."""
        
        # Calculate zone range
        zone_width = level * self.zone_width_percentage
        
        if zone_type == 'prior_day_high':
            high_level = level
            low_level = level - zone_width
        elif zone_type == 'prior_day_low':
            low_level = level
            high_level = level + zone_width
        else:
            center = level
            high_level = center + zone_width / 2
            low_level = center - zone_width / 2
        
        zone = IntradayZone(
            zone_type=zone_type,
            high_level=high_level,
            low_level=low_level,
            symbol=symbol,
            created_at=current_time,
            quality=quality,
            strength=strength
        )
        
        return zone
    
    def add_zone(self, zone: IntradayZone):
        """Add a zone to the manager."""
        symbol = zone.symbol
        self.zones[symbol].append(zone)
        
        # Limit zones per symbol
        if len(self.zones[symbol]) > self.max_zones_per_symbol:
            # Remove oldest zone
            self.zones[symbol] = sorted(self.zones[symbol], key=lambda z: z.created_at)
            self.zones[symbol].pop(0)
    
    def find_matching_zone(self, symbol: str, price: float, current_time: datetime) -> Optional[IntradayZone]:
        """Find a matching active zone for the given price."""
        
        if symbol not in self.zones:
            return None
        
        # Clean up expired zones
        self.zones[symbol] = [zone for zone in self.zones[symbol] if zone.is_zone_active(current_time)]
        
        # Find matching zone
        for zone in self.zones[symbol]:
            if zone.contains_price(price) and zone.is_zone_active(current_time):
                return zone
        
        return None
    
    def should_create_zone(self, symbol: str, zone_type: str, level: float, 
                          current_time: datetime) -> bool:
        """Check if a new zone should be created."""
        
        if symbol not in self.zones:
            return True
        
        # Check if we already have a similar zone
        for zone in self.zones[symbol]:
            if zone.zone_type == zone_type and zone.is_zone_active(current_time):
                # Check if level is too close to existing zone
                distance = abs(zone.get_zone_center() - level) / level
                if distance < self.zone_creation_threshold:
                    return False
        
        return True
    
    def get_zone_stats(self) -> Dict:
        """Get statistics about current zones."""
        stats = {}
        for symbol in self.symbols:
            active_zones = [zone for zone in self.zones[symbol] if zone.is_active]
            stats[symbol] = {
                'total_zones': len(self.zones[symbol]),
                'active_zones': len(active_zones),
                'avg_touches': np.mean([zone.touch_count for zone in active_zones]) if active_zones else 0
            }
        return stats


def test_intraday_zone_strategy():
    """Test the intraday zone strategy."""
    
    print("🚀 TESTING INTRADAY ZONE STRATEGY")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
        print(f"✅ Loaded {len(entry_points)} first-touch entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create zone manager
    symbols = entry_points['symbol'].unique().tolist()
    zone_manager = IntradayZoneManager(symbols)
    
    print(f"🔧 Created zone manager for symbols: {symbols}")
    print(f"   Max zones per symbol: {zone_manager.max_zones_per_symbol}")
    print(f"   Zone width: {zone_manager.zone_width_percentage*100:.1f}%")
    print(f"   Zone creation threshold: {zone_manager.zone_creation_threshold*100:.1f}%")
    
    # Process entries and create zones
    processed = 0
    zones_created = 0
    zones_matched = 0
    
    for _, entry in entry_points.iterrows():
        symbol = entry['symbol']
        zone_type = entry['zone_type']
        level = entry['zone_level']
        price = entry['price']
        current_time = entry['timestamp']
        
        processed += 1
        
        # Check if we should create a new zone
        if zone_manager.should_create_zone(symbol, zone_type, level, current_time):
            zone = zone_manager.create_zone(symbol, zone_type, level, current_time)
            zone_manager.add_zone(zone)
            zones_created += 1
            print(f"   Created zone: {zone.zone_id}")
        
        # Try to find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, price, current_time)
        if matching_zone:
            zones_matched += 1
            is_first_touch = matching_zone.is_first_touch(current_time)
            print(f"   Matched zone: {matching_zone.zone_id} (first touch: {is_first_touch})")
    
    # Get zone statistics
    stats = zone_manager.get_zone_stats()
    
    print(f"\n📊 ZONE STRATEGY RESULTS:")
    print(f"   Processed Entries: {processed}")
    print(f"   Zones Created: {zones_created}")
    print(f"   Zones Matched: {zones_matched}")
    print(f"   Match Rate: {zones_matched/processed*100:.1f}%")
    
    print(f"\n📈 ZONE STATISTICS BY SYMBOL:")
    for symbol, stat in stats.items():
        print(f"   {symbol}:")
        print(f"     Total Zones: {stat['total_zones']}")
        print(f"     Active Zones: {stat['active_zones']}")
        print(f"     Avg Touches: {stat['avg_touches']:.1f}")
    
    return zone_manager


def run_intraday_framework_backtest(zone_manager: IntradayZoneManager):
    """Run backtest with intraday zone strategy."""
    
    print(f"\n🚀 RUNNING INTRADAY FRAMEWORK BACKTEST")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    entry_points = pd.read_csv(csv_file)
    entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
    
    # Create adequate bars data
    bars_data = {}
    for symbol in entry_points['symbol'].unique():
        symbol_entries = entry_points[entry_points['symbol'] == symbol].sort_values('bar_index')
        max_bar_index = symbol_entries['bar_index'].max()
        
        bars = []
        base_price = 100.0
        total_bars = max_bar_index + 100
        
        for i in range(total_bars):
            trend = i * 0.001
            volatility = np.random.normal(0, 0.2)
            price = base_price + trend + volatility
            
            bar = type('Bar', (), {
                'open': price - 0.05,
                'high': price + 0.1,
                'low': price - 0.1,
                'close': price,
                'volume': 1000 + np.random.randint(-100, 100),
                'timestamp': datetime.now() + timedelta(minutes=i)
            })()
            bars.append(bar)
        
        bars_data[symbol] = bars
    
    # Create unified framework
    framework = create_unified_framework(
        initial_balance=10000.0,
        max_equity_per_trade=0.10,
        slippage_ticks=2,
        commission_per_trade=5.0,
        min_confidence=0.5
    )
    
    print(f"🔄 Processing entries with intraday zones...")
    
    # Process entries
    processed = 0
    executed = 0
    rejected = 0
    rejection_reasons = {}
    
    for _, entry in entry_points.iterrows():
        symbol = entry['symbol']
        if symbol not in bars_data:
            continue
        
        bars = bars_data[symbol]
        bar_index = entry['bar_index']
        
        if bar_index >= len(bars):
            continue
        
        # Find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, entry['price'], entry['timestamp'])
        
        if not matching_zone:
            rejected += 1
            rejection_reasons['No matching zone'] = rejection_reasons.get('No matching zone', 0) + 1
            continue
        
        # Check first touch
        is_first_touch = matching_zone.is_first_touch(entry['timestamp'])
        
        if not is_first_touch:
            rejected += 1
            rejection_reasons['Not first touch of zone'] = rejection_reasons.get('Not first touch of zone', 0) + 1
            continue
        
        # Prepare zone data for framework
        zone_data = {
            'zone_type': matching_zone.zone_type,
            'zone_level': matching_zone.get_zone_center(),
            'zone_high': matching_zone.high_level,
            'zone_low': matching_zone.low_level,
            'zone_width': matching_zone.get_zone_width()
        }
        
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
        
        try:
            # Evaluate trade opportunity
            trade_decision, choch_signal = framework.evaluate_trade_opportunity(
                bars, bar_index, zone_data, entry['price']
            )
            
            processed += 1
            
            # Track rejection reasons
            reason = trade_decision.reason
            if reason not in rejection_reasons:
                rejection_reasons[reason] = 0
            rejection_reasons[reason] += 1
            
            print(f"\n   Entry {processed}: {entry['entry_id']}")
            print(f"   Zone: {matching_zone.zone_type} range {matching_zone.low_level:.2f}-{matching_zone.high_level:.2f}")
            print(f"   Price: {entry['price']} (in zone: {matching_zone.contains_price(entry['price'])})")
            print(f"   Decision: {trade_decision.trade_type.value}")
            print(f"   Reason: {trade_decision.reason}")
            print(f"   Confidence: {trade_decision.confidence:.2f}")
            print(f"   Is First Touch: {is_first_touch}")
            print(f"   Should Execute: {framework.should_execute_trade(trade_decision)}")
            
            # Check if trade should be executed
            if framework.should_execute_trade(trade_decision):
                # Execute trade
                trade_execution = framework.execute_trade(
                    trade_decision, choch_signal, entry_data, bars, bar_index, simulation_mode=True
                )
                executed += 1
                print(f"   ✅ EXECUTED: P&L = ${trade_execution.pnl:.2f}")
            else:
                rejected += 1
                print(f"   ❌ REJECTED: {trade_decision.reason}")
        
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
    metrics = framework.get_performance_metrics()
    
    print(f"\n📊 INTRADAY FRAMEWORK RESULTS:")
    print(f"   Processed Entries: {processed}")
    print(f"   Executed Trades: {executed}")
    print(f"   Rejected Trades: {rejected}")
    print(f"   Execution Rate: {execution_rate:.1f}%")
    
    # Print performance metrics
    print(f"\n💰 PERFORMANCE METRICS:")
    print(f"   Final Balance: ${metrics['final_balance']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.2f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Max Drawdown: ${metrics['max_drawdown']:,.2f}")
    
    # Print rejection analysis
    print(f"\n❌ REJECTION ANALYSIS:")
    print(f"   Top rejection reasons:")
    sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        percentage = count / processed * 100 if processed > 0 else 0
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    return {
        'metrics': metrics,
        'processed': processed,
        'executed': executed,
        'rejected': rejected,
        'execution_rate': execution_rate,
        'rejection_reasons': rejection_reasons
    }


if __name__ == "__main__":
    # Test intraday zone strategy
    zone_manager = test_intraday_zone_strategy()
    
    # Run intraday framework backtest
    results = run_intraday_framework_backtest(zone_manager)