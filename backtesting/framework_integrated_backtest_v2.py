#!/usr/bin/env python3
"""
Framework Integrated Backtest V2

This backtest uses the unified trading framework to ensure consistency
between live trading and backtesting logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


class FrameworkIntegratedBacktestV2:
    """Backtest using the unified trading framework."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 max_equity_per_trade: float = 0.10,
                 slippage_ticks: int = 2,
                 commission_per_trade: float = 5.0,
                 min_confidence: float = 0.6):
        """Initialize the backtest with unified framework."""
        self.framework = create_unified_framework(
            initial_balance=initial_balance,
            max_equity_per_trade=max_equity_per_trade,
            slippage_ticks=slippage_ticks,
            commission_per_trade=commission_per_trade,
            min_confidence=min_confidence
        )
    
    def run_backtest(self, entry_points: pd.DataFrame, bars_data: Dict[str, List]) -> Dict:
        """Run the backtest using the unified framework."""
        
        print("🔄 Running Framework Integrated Backtest V2...")
        print(f"   Entry Points: {len(entry_points)}")
        print(f"   Initial Balance: ${self.framework.initial_balance:,.2f}")
        print(f"   Max Equity per Trade: {self.framework.max_equity_per_trade*100:.1f}%")
        print(f"   Min Confidence: {self.framework.min_confidence:.1f}")
        
        # Process each entry point
        processed_entries = 0
        executed_trades = 0
        rejected_trades = 0
        
        for _, entry in entry_points.iterrows():
            symbol = entry['symbol']
            if symbol not in bars_data:
                continue
            
            bars = bars_data[symbol]
            bar_index = entry['bar_index']
            
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
            
            # Evaluate trade opportunity
            trade_decision, choch_signal = self.framework.evaluate_trade_opportunity(
                bars, bar_index, zone_data, entry['price']
            )
            
            processed_entries += 1
            
            # Check if trade should be executed
            if self.framework.should_execute_trade(trade_decision):
                # Execute trade
                trade_execution = self.framework.execute_trade(
                    trade_decision, choch_signal, entry_data, bars, bar_index, simulation_mode=True
                )
                executed_trades += 1
                
                if processed_entries % 50 == 0:
                    print(f"   Processed: {processed_entries}, Executed: {executed_trades}")
            else:
                rejected_trades += 1
        
        print(f"\n📊 Backtest Complete:")
        print(f"   Processed Entries: {processed_entries}")
        print(f"   Executed Trades: {executed_trades}")
        print(f"   Rejected Trades: {rejected_trades}")
        print(f"   Execution Rate: {executed_trades/processed_entries*100:.1f}%")
        
        # Get performance metrics
        metrics = self.framework.get_performance_metrics()
        
        return {
            'framework_metrics': metrics,
            'processed_entries': processed_entries,
            'executed_trades': executed_trades,
            'rejected_trades': rejected_trades,
            'execution_rate': executed_trades/processed_entries*100 if processed_entries > 0 else 0
        }
    
    def print_detailed_results(self, results: Dict):
        """Print detailed results with framework analysis."""
        
        print("\n" + "="*80)
        print("FRAMEWORK INTEGRATED BACKTEST V2 - DETAILED RESULTS")
        print("="*80)
        
        # Execution Summary
        print(f"\n📈 EXECUTION SUMMARY")
        print(f"   Processed Entries: {results['processed_entries']}")
        print(f"   Executed Trades: {results['executed_trades']}")
        print(f"   Rejected Trades: {results['rejected_trades']}")
        print(f"   Execution Rate: {results['execution_rate']:.1f}%")
        
        # Framework Performance
        metrics = results['framework_metrics']
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"   Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"   Total Return: {metrics['total_return']:.2f}%")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Max Drawdown: ${metrics['max_drawdown']:,.2f}")
        
        # Framework Compliance Analysis
        print(f"\n🎯 FRAMEWORK COMPLIANCE ANALYSIS")
        if metrics['total_trades'] > 0:
            framework_compliance = metrics['framework_compliant_trades'] / metrics['total_trades'] * 100
            print(f"   Framework Compliant Trades: {metrics['framework_compliant_trades']} ({framework_compliance:.1f}%)")
            
            if metrics['choch_required_trades'] > 0:
                choch_compliance = metrics['choch_aligned_trades'] / metrics['choch_required_trades'] * 100
                print(f"   CHoCH Compliance: {metrics['choch_aligned_trades']}/{metrics['choch_required_trades']} ({choch_compliance:.1f}%)")
            else:
                print(f"   CHoCH Compliance: No continuation trades requiring CHoCH")
        else:
            print(f"   No trades executed - cannot analyze compliance")
        
        # Trade Type Analysis
        print(f"\n📊 TRADE TYPE BREAKDOWN")
        if metrics['total_trades'] > 0:
            fade_pct = metrics['fade_trades'] / metrics['total_trades'] * 100
            continuation_pct = metrics['continuation_trades'] / metrics['total_trades'] * 100
            print(f"   Fade Trades: {metrics['fade_trades']} ({fade_pct:.1f}%)")
            print(f"   Continuation Trades: {metrics['continuation_trades']} ({continuation_pct:.1f}%)")
        else:
            print(f"   No trades executed")
        
        # Bias Analysis
        print(f"\n🎯 BIAS ANALYSIS")
        if metrics['total_trades'] > 0:
            bullish_pct = metrics['bullish_trades'] / metrics['total_trades'] * 100
            bearish_pct = metrics['bearish_trades'] / metrics['total_trades'] * 100
            neutral_pct = metrics['neutral_trades'] / metrics['total_trades'] * 100
            print(f"   Bullish Bias: {metrics['bullish_trades']} ({bullish_pct:.1f}%)")
            print(f"   Bearish Bias: {metrics['bearish_trades']} ({bearish_pct:.1f}%)")
            print(f"   Neutral Bias: {metrics['neutral_trades']} ({neutral_pct:.1f}%)")
        else:
            print(f"   No trades executed")
        
        # Session Type Analysis
        print(f"\n📈 SESSION TYPE ANALYSIS")
        if metrics['total_trades'] > 0:
            trend_pct = metrics['trend_day_trades'] / metrics['total_trades'] * 100
            balanced_pct = metrics['balanced_day_trades'] / metrics['total_trades'] * 100
            choppy_pct = metrics['choppy_day_trades'] / metrics['total_trades'] * 100
            print(f"   Trend Day Trades: {metrics['trend_day_trades']} ({trend_pct:.1f}%)")
            print(f"   Balanced Day Trades: {metrics['balanced_day_trades']} ({balanced_pct:.1f}%)")
            print(f"   Choppy Day Trades: {metrics['choppy_day_trades']} ({choppy_pct:.1f}%)")
        else:
            print(f"   No trades executed")
        
        # Strategy Assessment
        print(f"\n🔍 STRATEGY ASSESSMENT")
        if metrics['total_trades'] == 0:
            print("   ❌ NO TRADES: Strategy is too restrictive or data issues")
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
        
        print("\n" + "="*80)


def run_framework_backtest_v2(csv_file_path: str, 
                            initial_balance: float = 10000.0,
                            max_equity_per_trade: float = 0.10,
                            slippage_ticks: int = 2,
                            commission_per_trade: float = 5.0,
                            min_confidence: float = 0.6) -> Dict:
    """Run the framework integrated backtest V2."""
    
    # Load entry points
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Create dummy bars data for testing (in real implementation, load from actual data files)
    bars_data = {}
    symbols = entry_points['symbol'].unique()
    
    print("📊 Creating sample bars data for testing...")
    for symbol in symbols:
        bars_data[symbol] = []
        base_price = 100.0
        
        # Create realistic price movement
        for i in range(2000):  # 2000 bars
            # Add some trend and volatility
            trend = i * 0.01  # Slight upward trend
            volatility = np.random.normal(0, 0.5)
            price = base_price + trend + volatility
            
            bar = type('Bar', (), {
                'open': price - 0.1,
                'high': price + 0.3,
                'low': price - 0.3,
                'close': price,
                'volume': 1000 + np.random.randint(-200, 200),
                'timestamp': datetime.now() + timedelta(minutes=i)
            })()
            bars_data[symbol].append(bar)
        
        print(f"   Created {len(bars_data[symbol])} bars for {symbol}")
    
    # Initialize and run backtest
    backtest = FrameworkIntegratedBacktestV2(
        initial_balance=initial_balance,
        max_equity_per_trade=max_equity_per_trade,
        slippage_ticks=slippage_ticks,
        commission_per_trade=commission_per_trade,
        min_confidence=min_confidence
    )
    
    # Run backtest
    results = backtest.run_backtest(entry_points, bars_data)
    
    # Print detailed results
    backtest.print_detailed_results(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    results = run_framework_backtest_v2(csv_file)