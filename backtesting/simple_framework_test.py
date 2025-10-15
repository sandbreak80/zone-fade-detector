#!/usr/bin/env python3
"""
Simple Framework Test

Test the unified framework with a simplified approach using actual entry data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


def create_simple_bars_data(entry_points: pd.DataFrame) -> Dict[str, List]:
    """Create simple bars data based on entry points."""
    bars_data = {}
    
    for symbol in entry_points['symbol'].unique():
        symbol_entries = entry_points[entry_points['symbol'] == symbol].sort_values('bar_index')
        
        bars = []
        base_price = 100.0
        
        # Create bars up to the maximum bar index
        max_bar_index = symbol_entries['bar_index'].max()
        
        for i in range(max_bar_index + 100):  # Add some buffer
            # Create realistic price movement
            trend = i * 0.001  # Slight upward trend
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
        print(f"   Created {len(bars)} bars for {symbol} (max index: {max_bar_index})")
    
    return bars_data


def test_unified_framework():
    """Test the unified framework with actual entry data."""
    
    print("🧪 Testing Unified Trading Framework")
    print("=" * 50)
    
    # Load entry points
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(entry_points)} entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create bars data
    print("📊 Creating bars data...")
    bars_data = create_simple_bars_data(entry_points)
    
    # Create framework
    framework = create_unified_framework(
        initial_balance=10000.0,
        max_equity_per_trade=0.10,
        slippage_ticks=2,
        commission_per_trade=5.0,
        min_confidence=0.6
    )
    
    print(f"\n🔄 Processing entry points...")
    
    # Process first 10 entry points as a test
    test_entries = entry_points.head(10)
    processed = 0
    executed = 0
    rejected = 0
    
    for _, entry in test_entries.iterrows():
        symbol = entry['symbol']
        if symbol not in bars_data:
            continue
        
        bars = bars_data[symbol]
        bar_index = entry['bar_index']
        
        # Ensure we have enough bars
        if bar_index >= len(bars):
            print(f"   Skipping {entry['entry_id']} - insufficient bars")
            continue
        
        # Prepare zone data
        zone_data = {
            'zone_type': entry['zone_type'],
            'zone_level': entry['zone_level']
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
            
            print(f"\n   Entry {processed}: {entry['entry_id']}")
            print(f"   Zone: {entry['zone_type']} at {entry['zone_level']}")
            print(f"   Price: {entry['price']}")
            print(f"   Decision: {trade_decision.trade_type.value}")
            print(f"   Reason: {trade_decision.reason}")
            print(f"   Confidence: {trade_decision.confidence:.2f}")
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
            print(f"   ⚠️  ERROR processing {entry['entry_id']}: {e}")
            continue
    
    print(f"\n📊 Test Results:")
    print(f"   Processed: {processed}")
    print(f"   Executed: {executed}")
    print(f"   Rejected: {rejected}")
    print(f"   Execution Rate: {executed/processed*100:.1f}%" if processed > 0 else "   Execution Rate: 0%")
    
    # Print framework performance
    if executed > 0:
        print(f"\n🎯 Framework Performance:")
        framework.print_performance_summary()
    else:
        print(f"\n⚠️  No trades executed - cannot show performance metrics")


if __name__ == "__main__":
    test_unified_framework()