#!/usr/bin/env python3
"""
Corrected Zone Implementation

This implements proper zones as price ranges (not single levels) according to the BattleCard framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


class CorrectedZone:
    """Represents a proper zone as a price range, not a single level."""
    
    def __init__(self, zone_type: str, high_level: float, low_level: float, 
                 quality: int = 1, strength: float = 1.0):
        self.zone_type = zone_type
        self.high_level = high_level
        self.low_level = low_level
        self.quality = quality
        self.strength = strength
        self.zone_id = f"{zone_type}_{low_level:.2f}_{high_level:.2f}"
    
    def contains_price(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is within the zone range."""
        return self.low_level - tolerance <= price <= self.high_level + tolerance
    
    def get_zone_center(self) -> float:
        """Get the center of the zone range."""
        return (self.high_level + self.low_level) / 2
    
    def get_zone_width(self) -> float:
        """Get the width of the zone range."""
        return self.high_level - self.low_level
    
    def get_zone_width_percentage(self, reference_price: float) -> float:
        """Get zone width as percentage of reference price."""
        return (self.get_zone_width() / reference_price) * 100


def create_corrected_zones_from_entries(entry_points: pd.DataFrame) -> List[CorrectedZone]:
    """Create proper zones (ranges) from the entry points data."""
    
    zones = []
    
    # Group by zone type and create ranges
    for zone_type in entry_points['zone_type'].unique():
        type_entries = entry_points[entry_points['zone_type'] == zone_type]
        
        # Get all unique levels for this zone type
        levels = sorted(type_entries['zone_level'].unique())
        
        # Create zones as ranges around each level
        for level in levels:
            # Define zone range based on zone type and level
            if zone_type == 'prior_day_high':
                # Resistance zone: range below the high
                zone_width = level * 0.002  # 0.2% of price
                high_level = level
                low_level = level - zone_width
            elif zone_type == 'prior_day_low':
                # Support zone: range above the low
                zone_width = level * 0.002  # 0.2% of price
                low_level = level
                high_level = level + zone_width
            else:
                # Default: symmetric range
                zone_width = level * 0.002
                center = level
                high_level = center + zone_width / 2
                low_level = center - zone_width / 2
            
            # Calculate quality and strength
            level_entries = type_entries[type_entries['zone_level'] == level]
            quality = min(2, len(level_entries) // 5)  # Quality based on touches
            strength = min(2.0, level_entries['zone_strength'].mean() if 'zone_strength' in level_entries.columns else 1.0)
            
            zone = CorrectedZone(
                zone_type=zone_type,
                high_level=high_level,
                low_level=low_level,
                quality=quality,
                strength=strength
            )
            zones.append(zone)
    
    return zones


def test_corrected_zone_implementation():
    """Test the corrected zone implementation."""
    
    print("🎯 TESTING CORRECTED ZONE IMPLEMENTATION")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(entry_points)} first-touch entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create corrected zones
    print("\n🔧 Creating corrected zones (ranges)...")
    zones = create_corrected_zones_from_entries(entry_points)
    
    print(f"✅ Created {len(zones)} corrected zones")
    
    # Display zone information
    print(f"\n📊 ZONE ANALYSIS:")
    for i, zone in enumerate(zones[:5]):  # Show first 5 zones
        print(f"   Zone {i+1}: {zone.zone_type}")
        print(f"     Range: {zone.low_level:.2f} - {zone.high_level:.2f}")
        print(f"     Width: {zone.get_zone_width():.4f} ({zone.get_zone_width_percentage(zone.get_zone_center()):.2f}%)")
        print(f"     Quality: {zone.quality}, Strength: {zone.strength:.2f}")
        print(f"     Zone ID: {zone.zone_id}")
        print()
    
    # Test zone matching
    print(f"🔍 TESTING ZONE MATCHING:")
    test_prices = [220.47, 240.08, 485.24, 512.19, 528.37]
    
    for price in test_prices:
        matching_zones = [zone for zone in zones if zone.contains_price(price)]
        print(f"   Price {price}: {len(matching_zones)} matching zones")
        for zone in matching_zones:
            print(f"     - {zone.zone_type}: {zone.low_level:.2f} - {zone.high_level:.2f}")
    
    # Compare with original single-level approach
    print(f"\n📈 COMPARISON: Ranges vs Single Levels")
    print(f"   Original Approach: {len(entry_points)} single levels")
    print(f"   Corrected Approach: {len(zones)} zone ranges")
    print(f"   Average Zone Width: {np.mean([zone.get_zone_width() for zone in zones]):.4f}")
    print(f"   Average Zone Width %: {np.mean([zone.get_zone_width_percentage(zone.get_zone_center()) for zone in zones]):.2f}%")
    
    # Test zone ID generation
    print(f"\n🔑 ZONE ID GENERATION:")
    for zone in zones[:3]:
        print(f"   {zone.zone_type}: {zone.zone_id}")
    
    return zones


def run_corrected_framework_backtest(zones: List[CorrectedZone]):
    """Run backtest with corrected zone implementation."""
    
    print(f"\n🚀 RUNNING CORRECTED FRAMEWORK BACKTEST")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    entry_points = pd.read_csv(csv_file)
    
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
    
    print(f"🔄 Processing entries with corrected zones...")
    
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
        
        # Reset zone touch history
        framework.bias_detector.zone_touch_history = {}
        
        # Find matching zone
        matching_zone = None
        for zone in zones:
            if zone.contains_price(entry['price']):
                matching_zone = zone
                break
        
        if not matching_zone:
            rejected += 1
            rejection_reasons['No matching zone'] = rejection_reasons.get('No matching zone', 0) + 1
            continue
        
        # Prepare zone data for framework
        zone_data = {
            'zone_type': matching_zone.zone_type,
            'zone_level': matching_zone.get_zone_center(),  # Use center as reference
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
            print(f"   Is First Touch: {trade_decision.is_first_touch}")
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
    
    print(f"\n📊 CORRECTED FRAMEWORK RESULTS:")
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
    # Test corrected zone implementation
    zones = test_corrected_zone_implementation()
    
    # Run corrected framework backtest
    results = run_corrected_framework_backtest(zones)