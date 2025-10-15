#!/usr/bin/env python3
"""
Corrected Trading Simulator with Realistic Parameters

This module uses realistic trading parameters for accurate performance assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import random


class CorrectedTradingSimulator:
    """Corrected trading simulation with realistic parameters."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 max_equity_per_trade: float = 0.02,  # 2% max equity per trade (realistic)
                 slippage_pct: float = 0.0005,  # 0.05% slippage (realistic for ETFs)
                 commission_per_trade: float = 1.0):  # $1 commission per trade (realistic)
        """
        Initialize corrected trading simulator with realistic parameters.
        
        Args:
            initial_balance: Starting account balance
            max_equity_per_trade: Maximum equity to risk per trade (as decimal)
            slippage_pct: Slippage percentage (as decimal)
            commission_per_trade: Commission cost per trade
        """
        self.initial_balance = initial_balance
        self.max_equity_per_trade = max_equity_per_trade
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        
        # More realistic trading parameters
        self.stop_loss_hit_rate = 0.30  # 30% of trades hit stop loss
        self.t1_hit_rate = 0.50  # 50% of trades hit T1
        self.t2_hit_rate = 0.15  # 15% of trades hit T2
        self.t3_hit_rate = 0.05  # 5% of trades hit T3
        
    def calculate_position_size(self, entry_price: float, stop_price: float, 
                              current_balance: float) -> int:
        """Calculate position size based on realistic risk management."""
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            return 0
        
        # Calculate maximum risk amount (2% of equity)
        max_risk_amount = current_balance * self.max_equity_per_trade
        
        # Calculate position size based on risk
        position_size = int(max_risk_amount / risk_per_share)
        
        # Ensure minimum position size
        position_size = max(1, position_size)
        
        # Ensure we don't risk more than we have
        max_affordable = int(current_balance * 0.95 / entry_price)  # Leave 5% buffer
        position_size = min(position_size, max_affordable)
        
        return position_size
    
    def simulate_trade(self, entry: pd.Series, current_balance: float) -> Dict:
        """Simulate a single trade with realistic execution."""
        # Calculate position size
        position_size = self.calculate_position_size(
            entry['price'], entry['hard_stop'], current_balance
        )
        
        if position_size <= 0:
            return {
                'entry_id': entry['entry_id'],
                'symbol': entry['symbol'],
                'entry_time': entry['timestamp'],
                'entry_price': entry['price'],
                'exit_price': entry['price'],
                'direction': entry['direction'],
                'position_size': 0,
                'pnl': 0,
                'pnl_pct': 0,
                'exit_reason': 'Insufficient Balance',
                'exit_scenario': 'no_trade',
                'qrs_score': entry['qrs_score'],
                'qrs_grade': entry['qrs_grade'],
                'risk_amount': 0,
                'is_winner': False,
                'is_loser': False,
                'is_breakeven': True,
                'slippage_entry': 0,
                'slippage_exit': 0,
                'commission_entry': 0,
                'commission_exit': 0,
                'total_costs': 0,
                'balance_before': current_balance,
                'balance_after': current_balance
            }
        
        # Calculate slippage (realistic)
        entry_slippage = entry['price'] * self.slippage_pct
        if entry['direction'] == 'LONG':
            actual_entry_price = entry['price'] + entry_slippage
        else:
            actual_entry_price = entry['price'] - entry_slippage
        
        # Determine exit scenario
        exit_scenario = self._determine_exit_scenario(entry)
        
        # Calculate exit price
        if exit_scenario == 'stop_loss':
            exit_price = entry['hard_stop']
            exit_reason = 'Stop Loss Hit'
        elif exit_scenario == 't1':
            exit_price = entry['t1_price']
            exit_reason = 'T1 Target Hit'
        elif exit_scenario == 't2':
            exit_price = entry['t2_price']
            exit_reason = 'T2 Target Hit'
        elif exit_scenario == 't3':
            exit_price = entry['t3_price']
            exit_reason = 'T3 Target Hit'
        else:  # partial_fill
            if entry['direction'] == 'LONG':
                exit_price = actual_entry_price + (entry['t1_price'] - actual_entry_price) * 0.3
            else:
                exit_price = actual_entry_price - (actual_entry_price - entry['t1_price']) * 0.3
            exit_reason = 'Partial Fill'
        
        # Calculate exit slippage
        exit_slippage = exit_price * self.slippage_pct
        if entry['direction'] == 'SHORT':
            actual_exit_price = exit_price + exit_slippage
        else:
            actual_exit_price = exit_price - exit_slippage
        
        # Calculate P&L
        if entry['direction'] == 'LONG':
            pnl = (actual_exit_price - actual_entry_price) * position_size
        else:  # SHORT
            pnl = (actual_entry_price - actual_exit_price) * position_size
        
        # Calculate costs
        commission_entry = self.commission_per_trade
        commission_exit = self.commission_per_trade
        total_costs = commission_entry + commission_exit
        
        # Calculate net P&L
        net_pnl = pnl - total_costs
        
        # Calculate new balance
        new_balance = current_balance + net_pnl
        
        return {
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'entry_time': entry['timestamp'],
            'entry_price': actual_entry_price,
            'exit_price': actual_exit_price,
            'direction': entry['direction'],
            'position_size': position_size,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / (actual_entry_price * position_size) * 100 if position_size > 0 else 0,
            'exit_reason': exit_reason,
            'exit_scenario': exit_scenario,
            'qrs_score': entry['qrs_score'],
            'qrs_grade': entry['qrs_grade'],
            'risk_amount': abs(actual_entry_price - entry['hard_stop']) * position_size,
            'is_winner': net_pnl > 0,
            'is_loser': net_pnl < 0,
            'is_breakeven': net_pnl == 0,
            'slippage_entry': entry_slippage * position_size,
            'slippage_exit': exit_slippage * position_size,
            'commission_entry': commission_entry,
            'commission_exit': commission_exit,
            'total_costs': total_costs,
            'balance_before': current_balance,
            'balance_after': new_balance
        }
    
    def _determine_exit_scenario(self, entry: pd.Series) -> str:
        """Determine realistic exit scenario based on probabilities."""
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
    
    def simulate_trading_sequence(self, entry_points: pd.DataFrame) -> pd.DataFrame:
        """Simulate complete trading sequence with account balance tracking."""
        trades = []
        current_balance = self.initial_balance
        
        # Sort by entry time to simulate chronological trading
        entry_points_sorted = entry_points.sort_values('timestamp').copy()
        
        for _, entry in entry_points_sorted.iterrows():
            trade = self.simulate_trade(entry, current_balance)
            trades.append(trade)
            current_balance = trade['balance_after']
        
        return pd.DataFrame(trades)
    
    def calculate_drawdown_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(trades) == 0:
            return {}
        
        # Calculate running balance
        trades_sorted = trades.sort_values('entry_time').copy()
        trades_sorted['running_balance'] = trades_sorted['balance_after']
        
        # Calculate running maximum
        trades_sorted['running_max'] = trades_sorted['running_balance'].expanding().max()
        
        # Calculate drawdown
        trades_sorted['drawdown'] = trades_sorted['running_balance'] - trades_sorted['running_max']
        trades_sorted['drawdown_pct'] = (trades_sorted['drawdown'] / trades_sorted['running_max']) * 100
        
        # Find maximum drawdown
        max_drawdown = trades_sorted['drawdown'].min()
        max_drawdown_pct = trades_sorted['drawdown_pct'].min()
        
        # Find drawdown duration
        in_drawdown = trades_sorted['drawdown'] < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'total_drawdown_periods': len(drawdown_periods),
            'final_balance': trades_sorted['balance_after'].iloc[-1],
            'total_return': ((trades_sorted['balance_after'].iloc[-1] - self.initial_balance) / self.initial_balance) * 100
        }
    
    def calculate_performance_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(trades) == 0:
            return {}
        
        # Basic P&L metrics
        total_pnl = trades['pnl'].sum()
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        # Win/Loss analysis
        winning_trades = trades[trades['is_winner']]
        losing_trades = trades[trades['is_loser']]
        total_trades = len(trades)
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)
        
        # Cost analysis
        total_commission = trades['total_costs'].sum()
        total_slippage = trades['slippage_entry'].sum() + trades['slippage_exit'].sum()
        
        # Risk metrics
        returns = trades['pnl_pct'].values
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Drawdown metrics
        drawdown_metrics = self.calculate_drawdown_metrics(trades)
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'volatility': volatility,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_costs': total_commission + total_slippage,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'breakeven_trades': len(trades[trades['is_breakeven']]),
            **drawdown_metrics
        }
    
    def generate_trading_report(self, trades: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive trading report."""
        if len(trades) == 0:
            return {"error": "No trades to analyze"}
        
        report = {
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'initial_balance': self.initial_balance,
                'max_equity_per_trade': self.max_equity_per_trade,
                'slippage_pct': self.slippage_pct,
                'commission_per_trade': self.commission_per_trade,
                'total_trades': len(trades),
                'analysis_period': {
                    'start': trades['entry_time'].min(),
                    'end': trades['entry_time'].max()
                }
            },
            'performance': self.calculate_performance_metrics(trades),
            'trades': trades
        }
        
        return report
    
    def print_trading_summary(self, report: Dict[str, any]):
        """Print a formatted trading summary."""
        print("\n" + "="*100)
        print("CORRECTED ZONE FADE TRADING SIMULATION (REALISTIC PARAMETERS)")
        print("="*100)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 SIMULATION PARAMETERS")
        print(f"   Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"   Max Equity per Trade: {summary['max_equity_per_trade']*100:.1f}%")
        print(f"   Slippage: {summary['slippage_pct']*100:.3f}%")
        print(f"   Commission per Trade: ${summary['commission_per_trade']:.2f}")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Period: {summary['analysis_period']['start']} to {summary['analysis_period']['end']}")
        
        # Performance
        perf = report['performance']
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"   Final Balance: ${perf['final_balance']:,.2f}")
        print(f"   Total Return: {perf['total_return']:.2f}%")
        print(f"   Net Profit: ${perf['net_profit']:,.2f}")
        print(f"   Win Rate: {perf['win_rate']:.1f}%")
        print(f"   Average Win: ${perf['avg_win']:.2f}")
        print(f"   Average Loss: ${perf['avg_loss']:.2f}")
        print(f"   Win/Loss Ratio: {perf['win_loss_ratio']:.2f}")
        print(f"   Profit Factor: {perf['profit_factor']:.2f}")
        print(f"   Expectancy: ${perf['expectancy']:.2f}")
        
        # Risk Metrics
        print(f"\n⚠️  RISK METRICS")
        print(f"   Max Drawdown: ${perf['max_drawdown']:,.2f} ({perf['max_drawdown_pct']:.2f}%)")
        print(f"   Max Drawdown Duration: {perf['max_drawdown_duration']:.0f} trades")
        print(f"   Avg Drawdown Duration: {perf['avg_drawdown_duration']:.1f} trades")
        print(f"   Volatility: {perf['volatility']:.2f}%")
        
        # Cost Analysis
        print(f"\n💸 COST ANALYSIS")
        print(f"   Total Commission: ${perf['total_commission']:,.2f}")
        print(f"   Total Slippage: ${perf['total_slippage']:,.2f}")
        print(f"   Total Costs: ${perf['total_costs']:,.2f}")
        if perf['net_profit'] > 0:
            print(f"   Cost as % of Profit: {(perf['total_costs']/perf['net_profit']*100):.1f}%")
        else:
            print(f"   Cost as % of Loss: {(perf['total_costs']/abs(perf['net_profit'])*100):.1f}%")
        
        # Trade Statistics
        print(f"\n📈 TRADE STATISTICS")
        print(f"   Winning Trades: {perf['winning_trades']}")
        print(f"   Losing Trades: {perf['losing_trades']}")
        print(f"   Breakeven Trades: {perf['breakeven_trades']}")
        
        print("\n" + "="*100)


def simulate_corrected_trading(csv_file_path: str, 
                             initial_balance: float = 10000.0,
                             max_equity_per_trade: float = 0.02,
                             slippage_pct: float = 0.0005,
                             commission_per_trade: float = 1.0) -> Dict[str, any]:
    """
    Simulate corrected trading with realistic parameters.
    
    Args:
        csv_file_path: Path to the entry points CSV file
        initial_balance: Starting account balance
        max_equity_per_trade: Maximum equity to risk per trade
        slippage_pct: Slippage percentage
        commission_per_trade: Commission cost per trade
    
    Returns:
        Comprehensive trading simulation report
    """
    # Load entry points data
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Initialize simulator
    simulator = CorrectedTradingSimulator(
        initial_balance=initial_balance,
        max_equity_per_trade=max_equity_per_trade,
        slippage_pct=slippage_pct,
        commission_per_trade=commission_per_trade
    )
    
    # Simulate trading sequence
    print("🔄 Simulating corrected trading sequence with realistic parameters...")
    trades = simulator.simulate_trading_sequence(entry_points)
    
    # Generate report
    print("📊 Calculating comprehensive performance metrics...")
    report = simulator.generate_trading_report(trades)
    
    # Print summary
    simulator.print_trading_summary(report)
    
    return report


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    report = simulate_corrected_trading(csv_file)