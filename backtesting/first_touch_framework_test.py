#!/usr/bin/env python3
"""
First Touch Framework Test

Test the unified framework with first-touch-only data to verify it works correctly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


def create_bars_data_from_entries(entry_points: pd.DataFrame) -> Dict[str, List]:
    """Create bars data based on entry points with proper indexing."""
    bars_data = {}
    
    for symbol in entry_points['symbol'].unique():
        symbol_entries = entry_points[entry_points['symbol'] == symbol].sort_values('bar_index')
        
        bars = []
        base_price = 100.0
        
        # Create bars up to the maximum bar index + buffer
        max_bar_index = symbol_entries['bar_index'].max()
        total_bars = max_bar_index + 200  # Add buffer
        
        for i in range(total_bars):
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


def test_first_touch_framework():
    """Test the unified framework with first-touch-only data."""
    
    print("🎯 TESTING FRAMEWORK WITH FIRST TOUCH DATA")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(entry_points)} first-touch entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create bars data
    print("📊 Creating bars data...")
    bars_data = create_bars_data_from_entries(entry_points)
    
    # Create unified framework
    framework = create_unified_framework(
        initial_balance=10000.0,
        max_equity_per_trade=0.10,
        slippage_ticks=2,
        commission_per_trade=5.0,
        min_confidence=0.5  # Lower confidence for testing
    )
    
    print(f"\n🔄 Processing first-touch entries...")
    print(f"   Initial Balance: ${framework.initial_balance:,.2f}")
    print(f"   Max Equity per Trade: {framework.max_equity_per_trade*100:.1f}%")
    print(f"   Min Confidence: {framework.min_confidence:.1f}")
    
    # Process all first-touch entries
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
        
        # Ensure we have enough bars
        if bar_index >= len(bars):
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
            # Evaluate trade opportunity using unified framework
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
            print(f"   Zone: {entry['zone_type']} at {entry['zone_level']}")
            print(f"   Price: {entry['price']}")
            print(f"   Decision: {trade_decision.trade_type.value}")
            print(f"   Reason: {trade_decision.reason}")
            print(f"   Confidence: {trade_decision.confidence:.2f}")
            print(f"   Bias: {trade_decision.bias_analysis.bias.value}")
            print(f"   Session: {trade_decision.bias_analysis.session_type.value}")
            print(f"   CHoCH: {choch_signal is not None}")
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
    
    print(f"\n📊 FIRST TOUCH TEST RESULTS:")
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
    
    # Print framework compliance
    if metrics['total_trades'] > 0:
        print(f"\n🎯 FRAMEWORK COMPLIANCE:")
        framework_compliance = metrics['framework_compliant_trades'] / metrics['total_trades'] * 100
        print(f"   Framework Compliant: {metrics['framework_compliant_trades']}/{metrics['total_trades']} ({framework_compliance:.1f}%)")
        
        if metrics['choch_required_trades'] > 0:
            choch_compliance = metrics['choch_aligned_trades'] / metrics['choch_required_trades'] * 100
            print(f"   CHoCH Compliance: {metrics['choch_aligned_trades']}/{metrics['choch_required_trades']} ({choch_compliance:.1f}%)")
        
        print(f"\n📊 TRADE BREAKDOWN:")
        print(f"   Fade Trades: {metrics['fade_trades']} ({metrics['fade_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Continuation Trades: {metrics['continuation_trades']} ({metrics['continuation_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Bullish Bias: {metrics['bullish_trades']} ({metrics['bullish_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Bearish Bias: {metrics['bearish_trades']} ({metrics['bearish_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Trend Day: {metrics['trend_day_trades']} ({metrics['trend_day_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Balanced Day: {metrics['balanced_day_trades']} ({metrics['balanced_day_trades']/metrics['total_trades']*100:.1f}%)")
    
    # Print rejection analysis
    print(f"\n❌ REJECTION ANALYSIS:")
    print(f"   Top rejection reasons:")
    sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        percentage = count / processed * 100 if processed > 0 else 0
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Strategy assessment
    print(f"\n🔍 STRATEGY ASSESSMENT:")
    if metrics['total_trades'] == 0:
        print("   ❌ NO TRADES: Strategy is too restrictive - check rejection reasons")
    elif metrics['total_return'] > 20:
        print("   ✅ EXCELLENT: Strategy shows strong profitability")
    elif metrics['total_return'] > 10:
        print("   ✅ GOOD: Strategy shows good profitability")
    elif metrics['total_return'] > 0:
        print("   ⚠️  MODERATE: Strategy shows modest profitability")
    else:
        print("   ❌ POOR: Strategy shows negative returns")
    
    if metrics['total_trades'] > 0:
        if metrics['win_rate'] > 60:
            print("   ✅ HIGH WIN RATE: Strategy shows consistent winning")
        elif metrics['win_rate'] > 50:
            print("   ⚠️  MODERATE WIN RATE: Strategy shows mixed results")
        else:
            print("   ❌ LOW WIN RATE: Strategy shows poor win rate")
        
        if metrics['max_drawdown'] > -1000:
            print("   ✅ LOW RISK: Strategy shows controlled drawdowns")
        elif metrics['max_drawdown'] > -5000:
            print("   ⚠️  MODERATE RISK: Strategy shows acceptable drawdowns")
        else:
            print("   ❌ HIGH RISK: Strategy shows concerning drawdowns")
    
    print("\n" + "=" * 60)
    
    return {
        'metrics': metrics,
        'processed': processed,
        'executed': executed,
        'rejected': rejected,
        'execution_rate': execution_rate,
        'rejection_reasons': rejection_reasons
    }


if __name__ == "__main__":
    test_first_touch_framework()