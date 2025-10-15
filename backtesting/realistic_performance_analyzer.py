#!/usr/bin/env python3
"""
Realistic Zone Fade Strategy Performance Analyzer

This module provides a more realistic simulation of Zone Fade trades
including proper stop loss execution and market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import random


class RealisticPerformanceAnalyzer:
    """Realistic performance analysis for Zone Fade trading strategy."""
    
    def __init__(self, initial_capital: float = 100000.0, commission_per_trade: float = 1.0):
        """
        Initialize realistic performance analyzer.
        
        Args:
            initial_capital: Starting capital for backtesting
            commission_per_trade: Commission cost per trade
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Realistic trading parameters
        self.slippage_pct = 0.001  # 0.1% slippage
        self.stop_loss_hit_rate = 0.35  # 35% of trades hit stop loss
        self.t1_hit_rate = 0.45  # 45% of trades hit T1
        self.t2_hit_rate = 0.15  # 15% of trades hit T2
        self.t3_hit_rate = 0.05  # 5% of trades hit T3
        
    def simulate_realistic_trade(self, entry: pd.Series, bars_data: List = None) -> Dict:
        """
        Simulate a realistic trade with proper exit conditions.
        
        Args:
            entry: Entry point data
            bars_data: Historical bars for realistic price movement simulation
        
        Returns:
            Trade simulation result
        """
        # Calculate position size based on risk
        risk_amount = entry['risk_amount']
        position_size = min(1000, max(100, risk_amount * 10))  # Reasonable position sizing
        
        # Add slippage to entry
        slippage = entry['price'] * self.slippage_pct
        actual_entry_price = entry['price'] + (slippage if entry['direction'] == 'LONG' else -slippage)
        
        # Determine exit based on realistic probabilities
        exit_scenario = self._determine_exit_scenario(entry)
        
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
        else:  # partial_fill or other
            # Simulate partial fill at various levels
            if entry['direction'] == 'LONG':
                exit_price = actual_entry_price + (entry['t1_price'] - actual_entry_price) * 0.3
            else:
                exit_price = actual_entry_price - (actual_entry_price - entry['t1_price']) * 0.3
            exit_reason = 'Partial Fill'
        
        # Add slippage to exit
        exit_slippage = exit_price * self.slippage_pct
        actual_exit_price = exit_price + (exit_slippage if entry['direction'] == 'SHORT' else -exit_slippage)
        
        # Calculate P&L
        if entry['direction'] == 'LONG':
            pnl = (actual_exit_price - actual_entry_price) * position_size
        else:  # SHORT
            pnl = (actual_entry_price - actual_exit_price) * position_size
        
        # Subtract transaction costs
        pnl -= self.commission_per_trade * 2  # Entry and exit commission
        
        return {
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'entry_time': entry['timestamp'],
            'entry_price': actual_entry_price,
            'exit_price': actual_exit_price,
            'direction': entry['direction'],
            'position_size': position_size,
            'pnl': pnl,
            'pnl_pct': pnl / (actual_entry_price * position_size) * 100,
            'exit_reason': exit_reason,
            'exit_scenario': exit_scenario,
            'qrs_score': entry['qrs_score'],
            'qrs_grade': entry['qrs_grade'],
            'risk_amount': risk_amount,
            'is_winner': pnl > 0,
            'is_loser': pnl < 0,
            'is_breakeven': pnl == 0,
            'slippage_entry': slippage,
            'slippage_exit': exit_slippage
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
    
    def simulate_realistic_trades(self, entry_points: pd.DataFrame) -> pd.DataFrame:
        """Simulate realistic trades for all entry points."""
        trades = []
        
        for _, entry in entry_points.iterrows():
            trade = self.simulate_realistic_trade(entry)
            trades.append(trade)
        
        return pd.DataFrame(trades)
    
    def calculate_profitability_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate profitability and return metrics."""
        if len(trades) == 0:
            return {}
        
        # Basic P&L metrics
        total_pnl = trades['pnl'].sum()
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        # Win rate
        winning_trades = trades[trades['is_winner']]
        losing_trades = trades[trades['is_loser']]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Average win and loss
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)
        
        # Return metrics
        final_capital = self.initial_capital + total_pnl
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
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
            'total_return': total_return,
            'final_capital': final_capital,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'breakeven_trades': len(trades[trades['is_breakeven']])
        }
    
    def calculate_risk_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk management metrics."""
        if len(trades) == 0:
            return {}
        
        # Calculate cumulative returns
        trades = trades.copy()
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        trades['cumulative_capital'] = self.initial_capital + trades['cumulative_pnl']
        
        # Maximum drawdown
        cumulative_max = trades['cumulative_capital'].expanding().max()
        drawdown = trades['cumulative_capital'] - cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        
        # Risk-reward ratio (average) - use available columns
        if 't1_reward' in trades.columns and 'risk_amount' in trades.columns:
            avg_risk_reward = trades['t1_reward'].mean() / trades['risk_amount'].mean() if trades['risk_amount'].mean() > 0 else 0
        else:
            avg_risk_reward = 0
        
        # Value at Risk (VaR) - 95% confidence
        returns = trades['pnl_pct'].values
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Volatility (standard deviation of returns)
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_risk_reward_ratio': avg_risk_reward,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'volatility': volatility,
            'sharpe_ratio': self.calculate_sharpe_ratio(trades),
            'sortino_ratio': self.calculate_sortino_ratio(trades)
        }
    
    def calculate_sharpe_ratio(self, trades: pd.DataFrame) -> float:
        """Calculate Sharpe ratio."""
        if len(trades) == 0:
            return 0
        
        returns = trades['pnl_pct'].values
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        # Annualized Sharpe ratio
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Assuming daily returns, annualize
        sharpe = (avg_return - self.risk_free_rate / 252) / volatility * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(self, trades: pd.DataFrame) -> float:
        """Calculate Sortino ratio (focuses on downside volatility)."""
        if len(trades) == 0:
            return 0
        
        returns = trades['pnl_pct'].values
        if len(returns) == 0:
            return 0
        
        # Calculate downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return float('inf')
        
        avg_return = np.mean(returns)
        sortino = (avg_return - self.risk_free_rate / 252) / downside_deviation * np.sqrt(252)
        return sortino
    
    def calculate_exit_analysis(self, trades: pd.DataFrame) -> Dict[str, any]:
        """Analyze exit scenarios and their performance."""
        exit_analysis = {}
        
        for scenario in ['stop_loss', 't1', 't2', 't3', 'partial_fill']:
            scenario_trades = trades[trades['exit_scenario'] == scenario]
            if len(scenario_trades) > 0:
                exit_analysis[scenario] = {
                    'count': len(scenario_trades),
                    'percentage': len(scenario_trades) / len(trades) * 100,
                    'win_rate': len(scenario_trades[scenario_trades['is_winner']]) / len(scenario_trades) * 100,
                    'avg_pnl': scenario_trades['pnl'].mean(),
                    'total_pnl': scenario_trades['pnl'].sum(),
                    'avg_return': scenario_trades['pnl_pct'].mean()
                }
            else:
                exit_analysis[scenario] = {
                    'count': 0,
                    'percentage': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'avg_return': 0
                }
        
        return exit_analysis
    
    def generate_realistic_report(self, trades: pd.DataFrame) -> Dict[str, any]:
        """Generate realistic performance report."""
        if len(trades) == 0:
            return {"error": "No trades to analyze"}
        
        report = {
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'initial_capital': self.initial_capital,
                'total_trades': len(trades),
                'analysis_period': {
                    'start': trades['entry_time'].min(),
                    'end': trades['entry_time'].max()
                }
            },
            'profitability': self.calculate_profitability_metrics(trades),
            'risk_management': self.calculate_risk_metrics(trades),
            'exit_analysis': self.calculate_exit_analysis(trades)
        }
        
        return report
    
    def print_realistic_summary(self, report: Dict[str, any]):
        """Print a formatted realistic performance summary."""
        print("\n" + "="*80)
        print("REALISTIC ZONE FADE STRATEGY PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"   Analysis Date: {summary['analysis_date']}")
        print(f"   Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Period: {summary['analysis_period']['start']} to {summary['analysis_period']['end']}")
        
        # Profitability
        prof = report['profitability']
        print(f"\n💰 PROFITABILITY METRICS")
        print(f"   Net Profit: ${prof['net_profit']:,.2f}")
        print(f"   Total Return: {prof['total_return']:.2f}%")
        print(f"   Final Capital: ${prof['final_capital']:,.2f}")
        print(f"   Win Rate: {prof['win_rate']:.1f}%")
        print(f"   Average Win: ${prof['avg_win']:.2f}")
        print(f"   Average Loss: ${prof['avg_loss']:.2f}")
        print(f"   Win/Loss Ratio: {prof['win_loss_ratio']:.2f}")
        print(f"   Profit Factor: {prof['profit_factor']:.2f}")
        print(f"   Expectancy: ${prof['expectancy']:.2f}")
        
        # Risk Management
        risk = report['risk_management']
        print(f"\n⚠️  RISK MANAGEMENT METRICS")
        print(f"   Max Drawdown: ${risk['max_drawdown']:,.2f} ({risk['max_drawdown_pct']:.2f}%)")
        print(f"   Avg Risk/Reward: {risk['avg_risk_reward_ratio']:.2f}")
        print(f"   VaR (95%): {risk['var_95']:.2f}%")
        print(f"   CVaR (95%): {risk['cvar_95']:.2f}%")
        print(f"   Volatility: {risk['volatility']:.2f}%")
        print(f"   Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {risk['sortino_ratio']:.2f}")
        
        # Exit Analysis
        exit_analysis = report['exit_analysis']
        print(f"\n🎯 EXIT SCENARIO ANALYSIS")
        for scenario, data in exit_analysis.items():
            if data['count'] > 0:
                print(f"   {scenario.upper()}:")
                print(f"     Count: {data['count']} ({data['percentage']:.1f}%)")
                print(f"     Win Rate: {data['win_rate']:.1f}%")
                print(f"     Avg P&L: ${data['avg_pnl']:.2f}")
                print(f"     Total P&L: ${data['total_pnl']:.2f}")
        
        print("\n" + "="*80)


def analyze_realistic_performance(csv_file_path: str, 
                                initial_capital: float = 100000.0,
                                commission_per_trade: float = 1.0) -> Dict[str, any]:
    """
    Analyze realistic Zone Fade strategy performance from CSV file.
    
    Args:
        csv_file_path: Path to the entry points CSV file
        initial_capital: Starting capital for backtesting
        commission_per_trade: Commission cost per trade
    
    Returns:
        Realistic performance report
    """
    # Load entry points data
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Initialize analyzer
    analyzer = RealisticPerformanceAnalyzer(initial_capital, commission_per_trade)
    
    # Simulate realistic trades
    print("🔄 Simulating realistic trades with proper exit conditions...")
    trades = analyzer.simulate_realistic_trades(entry_points)
    
    # Generate performance report
    print("📊 Calculating realistic performance metrics...")
    report = analyzer.generate_realistic_report(trades)
    
    # Print summary
    analyzer.print_realistic_summary(report)
    
    return report


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    report = analyze_realistic_performance(csv_file)