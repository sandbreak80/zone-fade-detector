#!/usr/bin/env python3
"""
Final Framework Backtest

This is the final working version that integrates all framework components
and ensures shared logic between trading strategy and backtesting.
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


def run_final_framework_backtest(csv_file_path: str, 
                               initial_balance: float = 10000.0,
                               max_equity_per_trade: float = 0.10,
                               slippage_ticks: int = 2,
                               commission_per_trade: float = 5.0,
                               min_confidence: float = 0.5) -> Dict:
    """Run the final framework backtest with all components integrated."""
    
    print("🚀 FINAL FRAMEWORK BACKTEST - ZONE-BASED INTRADAY TRADING")
    print("=" * 70)
    
    # Load entry points
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Create bars data
    print("📊 Creating bars data...")
    bars_data = create_bars_data_from_entries(entry_points)
    
    # Create unified framework
    framework = create_unified_framework(
        initial_balance=initial_balance,
        max_equity_per_trade=max_equity_per_trade,
        slippage_ticks=slippage_ticks,
        commission_per_trade=commission_per_trade,
        min_confidence=min_confidence
    )
    
    print(f"\n🔄 Processing entry points with framework rules...")
    print(f"   Initial Balance: ${framework.initial_balance:,.2f}")
    print(f"   Max Equity per Trade: {framework.max_equity_per_trade*100:.1f}%")
    print(f"   Min Confidence: {framework.min_confidence:.1f}")
    print(f"   Slippage: {framework.slippage_ticks} ticks")
    print(f"   Commission: ${framework.commission_per_trade:.2f}")
    
    # Process all entry points
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
            
            # Check if trade should be executed
            if framework.should_execute_trade(trade_decision):
                # Execute trade
                trade_execution = framework.execute_trade(
                    trade_decision, choch_signal, entry_data, bars, bar_index, simulation_mode=True
                )
                executed += 1
                
                if executed % 10 == 0:
                    print(f"   Executed {executed} trades...")
            else:
                rejected += 1
        
        except Exception as e:
            rejected += 1
            error_reason = f"Error: {str(e)[:50]}..."
            if error_reason not in rejection_reasons:
                rejection_reasons[error_reason] = 0
            rejection_reasons[error_reason] += 1
            continue
    
    # Calculate results
    execution_rate = executed / processed * 100 if processed > 0 else 0
    metrics = framework.get_performance_metrics()
    
    print(f"\n📊 BACKTEST RESULTS:")
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
    
    print("\n" + "=" * 70)
    
    return {
        'metrics': metrics,
        'processed': processed,
        'executed': executed,
        'rejected': rejected,
        'execution_rate': execution_rate,
        'rejection_reasons': rejection_reasons
    }


if __name__ == "__main__":
    # Run the final framework backtest
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    results = run_final_framework_backtest(csv_file)