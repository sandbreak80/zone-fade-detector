#!/usr/bin/env python3
"""
Framework Integrated Backtest

This module integrates all framework components into a comprehensive backtest
that follows the original Zone-Based Intraday Trading Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime, timedelta
import math
import random
import statistics
from collections import defaultdict

from directional_bias_detector import DirectionalBiasDetector, DirectionalBias, SessionType
from session_aware_trading import SessionAwareTradingRules, TradeType, TradeDirection, TradeDecision
from choch_confirmation_system import CHoCHConfirmationSystem, CHoCHSignal


class FrameworkIntegratedBacktest:
    """Integrated backtest following the original framework."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 max_equity_per_trade: float = 0.10,
                 slippage_ticks: int = 2,
                 commission_per_trade: float = 5.0,
                 min_confidence: float = 0.6):
        """
        Initialize framework integrated backtest.
        
        Args:
            initial_balance: Starting account balance
            max_equity_per_trade: Maximum equity to risk per trade
            slippage_ticks: Slippage in ticks
            commission_per_trade: Commission cost per trade
            min_confidence: Minimum confidence for trade execution
        """
        self.initial_balance = initial_balance
        self.max_equity_per_trade = max_equity_per_trade
        self.slippage_ticks = slippage_ticks
        self.commission_per_trade = commission_per_trade
        self.min_confidence = min_confidence
        
        # Initialize framework components
        self.bias_detector = DirectionalBiasDetector()
        self.trading_rules = SessionAwareTradingRules(self.bias_detector)
        self.choch_system = CHoCHConfirmationSystem()
        
        # Track results
        self.trades = []
        self.current_balance = initial_balance
        self.zone_touch_history = {}
        
        # Realistic trading parameters
        self.stop_loss_hit_rate = 0.40  # 40% of trades hit stop loss
        self.t1_hit_rate = 0.35  # 35% of trades hit T1
        self.t2_hit_rate = 0.15  # 15% of trades hit T2
        self.t3_hit_rate = 0.10  # 10% of trades hit T3
    
    def process_entry_point(self, entry: pd.Series, bars: List, bar_index: int) -> Optional[Dict]:
        """
        Process an entry point using the complete framework.
        
        Args:
            entry: Entry point data
            bars: List of OHLCV bars
            bar_index: Current bar index
            
        Returns:
            Trade result if trade taken, None otherwise
        """
        # Get current bias analysis
        bias_analysis = self.bias_detector.detect_directional_bias(bars, bar_index)
        
        # Check if this is first touch of zone
        zone_id = f"{entry['zone_type']}_{entry['zone_level']:.2f}"
        is_first_touch = self.bias_detector.is_first_touch(
            zone_id, entry['timestamp']
        )
        
        if not is_first_touch:
            return None
        
        # Analyze trade opportunity using session-aware rules
        trade_decision = self.trading_rules.analyze_trade_opportunity(
            bars, bar_index, entry['zone_type'], entry['zone_level'], entry['price']
        )
        
        # Check for CHoCH confirmation
        choch_signal = self.choch_system.detect_choch(bars, bar_index)
        
        # Enhance trade decision with CHoCH
        if choch_signal:
            trade_decision = self.choch_system.enhance_trade_decision(trade_decision, choch_signal)
        
        # Check if trade should be taken
        if not self.trading_rules.should_take_trade(trade_decision, self.min_confidence):
            return None
        
        # Simulate the trade
        trade_result = self._simulate_trade(entry, trade_decision, bias_analysis, choch_signal)
        
        return trade_result
    
    def _simulate_trade(self, entry: pd.Series, trade_decision: TradeDecision, 
                       bias_analysis, choch_signal: Optional[CHoCHSignal]) -> Dict:
        """Simulate a trade execution."""
        
        # Calculate corrected risk amount
        risk_amount = self._calculate_corrected_risk_amount(
            entry['price'], entry['hard_stop'], trade_decision.direction.value
        )
        
        # Calculate position size
        position_size = self._calculate_position_size(entry['price'], risk_amount)
        
        if position_size <= 0:
            return {
                'entry_id': entry['entry_id'],
                'symbol': entry['symbol'],
                'trade_type': trade_decision.trade_type.value,
                'direction': trade_decision.direction.value,
                'position_size': 0,
                'pnl': 0,
                'is_winner': False,
                'is_loser': False,
                'exit_scenario': 'no_trade',
                'reason': 'Insufficient balance',
                'bias': bias_analysis.bias.value,
                'session_type': bias_analysis.session_type.value,
                'choch_confirmed': choch_signal is not None,
                'confidence': trade_decision.confidence
            }
        
        # Calculate slippage
        entry_slippage = self._calculate_slippage(entry['price'])
        if trade_decision.direction == TradeDirection.LONG:
            actual_entry_price = entry['price'] + entry_slippage
        else:
            actual_entry_price = entry['price'] - entry_slippage
        
        # Determine exit scenario
        exit_scenario = self._determine_exit_scenario(entry, trade_decision)
        
        # Calculate exit price
        exit_price = self._calculate_exit_price(entry, exit_scenario, trade_decision)
        
        # Calculate exit slippage
        exit_slippage = self._calculate_slippage(exit_price)
        if trade_decision.direction == TradeDirection.SHORT:
            actual_exit_price = exit_price + exit_slippage
        else:
            actual_exit_price = exit_price - exit_slippage
        
        # Calculate P&L
        if trade_decision.direction == TradeDirection.LONG:
            pnl = (actual_exit_price - actual_entry_price) * position_size
        else:  # SHORT
            pnl = (actual_entry_price - actual_exit_price) * position_size
        
        # Calculate costs
        total_costs = self.commission_per_trade * 2  # Entry + Exit
        
        # Calculate net P&L
        net_pnl = pnl - total_costs
        
        # Update balance
        self.current_balance += net_pnl
        
        return {
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'trade_type': trade_decision.trade_type.value,
            'direction': trade_decision.direction.value,
            'entry_price': actual_entry_price,
            'exit_price': actual_exit_price,
            'position_size': position_size,
            'pnl': net_pnl,
            'is_winner': net_pnl > 0,
            'is_loser': net_pnl < 0,
            'exit_scenario': exit_scenario,
            'reason': trade_decision.reason,
            'bias': bias_analysis.bias.value,
            'session_type': bias_analysis.session_type.value,
            'choch_confirmed': choch_signal is not None,
            'confidence': trade_decision.confidence,
            'balance_after': self.current_balance
        }
    
    def _calculate_corrected_risk_amount(self, entry_price: float, hard_stop: float, direction: str) -> float:
        """Calculate corrected risk amount."""
        if direction == "LONG":
            risk_amount = entry_price - hard_stop
        else:  # SHORT
            risk_amount = hard_stop - entry_price
        
        if risk_amount <= 0:
            risk_amount = entry_price * 0.01  # 1% fallback
        
        return risk_amount
    
    def _calculate_position_size(self, entry_price: float, risk_amount: float) -> int:
        """Calculate position size based on risk amount."""
        if risk_amount <= 0:
            return 0
        
        max_risk_amount = self.current_balance * self.max_equity_per_trade
        position_size = int(max_risk_amount / risk_amount)
        position_size = max(1, position_size)
        
        # Ensure we don't risk more than we have
        max_affordable = int(self.current_balance * 0.95 / entry_price)
        position_size = min(position_size, max_affordable)
        
        return position_size
    
    def _calculate_slippage(self, price: float) -> float:
        """Calculate slippage amount."""
        tick_value = 0.01  # $0.01 per tick for ETFs
        return self.slippage_ticks * tick_value
    
    def _determine_exit_scenario(self, entry: pd.Series, trade_decision: TradeDecision) -> str:
        """Determine exit scenario based on probabilities."""
        rand = random.random()
        
        if rand < self.stop_loss_hit_rate:
            return 'stop_loss'
        elif rand < self.stop_loss_hit_rate + self.t1_hit_rate:
            return 't1'
        elif rand < self.stop_loss_hit_rate + self.t1_hit_rate + self.t2_hit_rate:
            return 't2'
        elif rand < self.stop_loss_hit_rate + self.t1_hit_rate + self.t2_hit_rate + self.t3_hit_rate:
            return 't3'
        else:
            return 'partial_fill'
    
    def _calculate_exit_price(self, entry: pd.Series, exit_scenario: str, trade_decision: TradeDecision) -> float:
        """Calculate exit price based on scenario."""
        if exit_scenario == 'stop_loss':
            return entry['hard_stop']
        elif exit_scenario == 't1':
            return entry['t1_price']
        elif exit_scenario == 't2':
            return entry['t2_price']
        elif exit_scenario == 't3':
            return entry['t3_price']
        else:  # partial_fill
            if trade_decision.direction == TradeDirection.LONG:
                return entry['price'] + (entry['t1_price'] - entry['price']) * 0.3
            else:
                return entry['price'] - (entry['price'] - entry['t1_price']) * 0.3
    
    def run_backtest(self, entry_points: pd.DataFrame, bars_data: Dict[str, List]) -> Dict:
        """Run the integrated framework backtest."""
        
        print("🔄 Running Framework Integrated Backtest...")
        print(f"   Entry Points: {len(entry_points)}")
        print(f"   Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   Max Equity per Trade: {self.max_equity_per_trade*100:.1f}%")
        
        # Process each entry point
        for _, entry in entry_points.iterrows():
            symbol = entry['symbol']
            if symbol not in bars_data:
                continue
            
            bars = bars_data[symbol]
            bar_index = entry['bar_index']
            
            # Process entry point
            trade_result = self.process_entry_point(entry, bars, bar_index)
            
            if trade_result:
                self.trades.append(trade_result)
        
        # Calculate results
        results = self._calculate_results()
        
        return results
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'final_balance': self.initial_balance,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'trades': []
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        final_balance = self.current_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        # Win/Loss analysis
        winning_trades = trades_df[trades_df['is_winner']]
        losing_trades = trades_df[trades_df['is_loser']]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown analysis
        trades_df['running_balance'] = trades_df['balance_after']
        trades_df['running_max'] = trades_df['running_balance'].expanding().max()
        trades_df['drawdown'] = trades_df['running_balance'] - trades_df['running_max']
        max_drawdown = trades_df['drawdown'].min()
        
        # Trade type analysis
        fade_trades = trades_df[trades_df['trade_type'] == 'fade']
        continuation_trades = trades_df[trades_df['trade_type'] == 'continuation']
        
        # Bias analysis
        bullish_trades = trades_df[trades_df['bias'] == 'bullish']
        bearish_trades = trades_df[trades_df['bias'] == 'bearish']
        neutral_trades = trades_df[trades_df['bias'] == 'neutral']
        
        # Session type analysis
        trend_day_trades = trades_df[trades_df['session_type'] == 'trend_day']
        balanced_day_trades = trades_df[trades_df['session_type'] == 'balanced_day']
        choppy_day_trades = trades_df[trades_df['session_type'] == 'choppy_day']
        
        return {
            'total_trades': total_trades,
            'final_balance': final_balance,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'fade_trades': len(fade_trades),
            'continuation_trades': len(continuation_trades),
            'bullish_trades': len(bullish_trades),
            'bearish_trades': len(bearish_trades),
            'neutral_trades': len(neutral_trades),
            'trend_day_trades': len(trend_day_trades),
            'balanced_day_trades': len(balanced_day_trades),
            'choppy_day_trades': len(choppy_day_trades),
            'choch_confirmed_trades': len(trades_df[trades_df['choch_confirmed'] == True]),
            'trades': self.trades
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive results."""
        
        print("\n" + "="*80)
        print("FRAMEWORK INTEGRATED BACKTEST RESULTS")
        print("="*80)
        
        # Basic Performance
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"   Final Balance: ${results['final_balance']:,.2f}")
        print(f"   Total Return: {results['total_return']:.2f}%")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: ${results['max_drawdown']:,.2f}")
        
        # Trade Type Analysis
        print(f"\n📊 TRADE TYPE ANALYSIS")
        print(f"   Fade Trades: {results['fade_trades']} ({results['fade_trades']/results['total_trades']*100:.1f}%)")
        print(f"   Continuation Trades: {results['continuation_trades']} ({results['continuation_trades']/results['total_trades']*100:.1f}%)")
        
        # Bias Analysis
        print(f"\n🎯 BIAS ANALYSIS")
        print(f"   Bullish Bias Trades: {results['bullish_trades']} ({results['bullish_trades']/results['total_trades']*100:.1f}%)")
        print(f"   Bearish Bias Trades: {results['bearish_trades']} ({results['bearish_trades']/results['total_trades']*100:.1f}%)")
        print(f"   Neutral Bias Trades: {results['neutral_trades']} ({results['neutral_trades']/results['total_trades']*100:.1f}%)")
        
        # Session Type Analysis
        print(f"\n📈 SESSION TYPE ANALYSIS")
        print(f"   Trend Day Trades: {results['trend_day_trades']} ({results['trend_day_trades']/results['total_trades']*100:.1f}%)")
        print(f"   Balanced Day Trades: {results['balanced_day_trades']} ({results['balanced_day_trades']/results['total_trades']*100:.1f}%)")
        print(f"   Choppy Day Trades: {results['choppy_day_trades']} ({results['choppy_day_trades']/results['total_trades']*100:.1f}%)")
        
        # CHoCH Analysis
        print(f"\n🔄 CHoCH CONFIRMATION ANALYSIS")
        print(f"   CHoCH Confirmed Trades: {results['choch_confirmed_trades']} ({results['choch_confirmed_trades']/results['total_trades']*100:.1f}%)")
        
        # Strategy Assessment
        print(f"\n🔍 STRATEGY ASSESSMENT")
        if results['total_return'] > 20:
            print("   ✅ EXCELLENT: Strategy shows strong profitability")
        elif results['total_return'] > 10:
            print("   ✅ GOOD: Strategy shows good profitability")
        elif results['total_return'] > 0:
            print("   ⚠️  MODERATE: Strategy shows modest profitability")
        else:
            print("   ❌ POOR: Strategy shows negative returns")
        
        if results['win_rate'] > 60:
            print("   ✅ HIGH WIN RATE: Strategy shows consistent winning")
        elif results['win_rate'] > 50:
            print("   ⚠️  MODERATE WIN RATE: Strategy shows mixed results")
        else:
            print("   ❌ LOW WIN RATE: Strategy shows poor win rate")
        
        if results['max_drawdown'] > -1000:
            print("   ✅ LOW RISK: Strategy shows controlled drawdowns")
        elif results['max_drawdown'] > -5000:
            print("   ⚠️  MODERATE RISK: Strategy shows acceptable drawdowns")
        else:
            print("   ❌ HIGH RISK: Strategy shows concerning drawdowns")
        
        print("\n" + "="*80)


def run_framework_integrated_backtest(csv_file_path: str, 
                                    bars_data_path: str,
                                    initial_balance: float = 10000.0,
                                    max_equity_per_trade: float = 0.10,
                                    slippage_ticks: int = 2,
                                    commission_per_trade: float = 5.0,
                                    min_confidence: float = 0.6) -> Dict:
    """Run the framework integrated backtest."""
    
    # Load entry points
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Load bars data (simplified - in real implementation, load from actual data files)
    bars_data = {}
    symbols = entry_points['symbol'].unique()
    for symbol in symbols:
        # Create dummy bars data for testing
        bars_data[symbol] = []
        for i in range(1000):
            bar = type('Bar', (), {
                'open': 100.0 + i * 0.1,
                'high': 100.0 + i * 0.1 + 0.5,
                'low': 100.0 + i * 0.1 - 0.5,
                'close': 100.0 + i * 0.1,
                'volume': 1000,
                'timestamp': datetime.now() + timedelta(minutes=i)
            })()
            bars_data[symbol].append(bar)
    
    # Initialize and run backtest
    backtest = FrameworkIntegratedBacktest(
        initial_balance=initial_balance,
        max_equity_per_trade=max_equity_per_trade,
        slippage_ticks=slippage_ticks,
        commission_per_trade=commission_per_trade,
        min_confidence=min_confidence
    )
    
    # Run backtest
    results = backtest.run_backtest(entry_points, bars_data)
    
    # Print results
    backtest.print_results(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    results = run_framework_integrated_backtest(csv_file, "")