#!/usr/bin/env python3
"""
Zone Fade Strategy Performance Analyzer

This module calculates comprehensive P&L and key trading metrics to evaluate
profitability, risk management, and overall performance of the Zone Fade strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math


class PerformanceAnalyzer:
    """Comprehensive performance analysis for Zone Fade trading strategy."""
    
    def __init__(self, initial_capital: float = 100000.0, commission_per_trade: float = 1.0):
        """
        Initialize performance analyzer.
        
        Args:
            initial_capital: Starting capital for backtesting
            commission_per_trade: Commission cost per trade
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_trade_pnl(self, entry_price: float, exit_price: float, 
                           direction: str, position_size: float) -> float:
        """Calculate P&L for a single trade."""
        if direction == "LONG":
            pnl = (exit_price - entry_price) * position_size
        else:  # SHORT
            pnl = (entry_price - exit_price) * position_size
        
        return pnl - self.commission_per_trade
    
    def simulate_trades(self, entry_points: pd.DataFrame, 
                       position_sizing_method: str = "fixed",
                       risk_per_trade: float = 0.02) -> pd.DataFrame:
        """
        Simulate trades based on entry points and exit strategy.
        
        Args:
            entry_points: DataFrame with entry point data
            position_sizing_method: "fixed", "risk_based", or "kelly"
            risk_per_trade: Risk per trade as percentage of capital
        """
        trades = []
        current_capital = self.initial_capital
        
        for _, entry in entry_points.iterrows():
            # Calculate position size
            if position_sizing_method == "fixed":
                position_size = 100  # Fixed 100 shares
            elif position_sizing_method == "risk_based":
                risk_amount = current_capital * risk_per_trade
                position_size = risk_amount / entry['risk_amount'] if entry['risk_amount'] > 0 else 100
            else:  # kelly
                # Simplified Kelly criterion (would need win rate and avg win/loss)
                position_size = 100
            
            # Simulate exit at T1 target (simplified for now)
            exit_price = entry['t1_price']
            
            # Calculate P&L
            pnl = self.calculate_trade_pnl(
                entry['price'], exit_price, entry['direction'], position_size
            )
            
            # Update capital
            current_capital += pnl
            
            trade = {
                'entry_id': entry['entry_id'],
                'symbol': entry['symbol'],
                'entry_time': entry['timestamp'],
                'entry_price': entry['price'],
                'exit_price': exit_price,
                'direction': entry['direction'],
                'position_size': position_size,
                'pnl': pnl,
                'pnl_pct': pnl / (entry['price'] * position_size) * 100,
                'capital_after': current_capital,
                'qrs_score': entry['qrs_score'],
                'qrs_grade': entry['qrs_grade'],
                'risk_amount': entry['risk_amount'],
                't1_reward': entry['t1_reward'],
                't2_reward': entry['t2_reward'],
                't3_reward': entry['t3_reward'],
                'is_winner': pnl > 0,
                'is_loser': pnl < 0,
                'is_breakeven': pnl == 0
            }
            
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
        final_capital = trades['capital_after'].iloc[-1] if len(trades) > 0 else self.initial_capital
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
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        trades['cumulative_capital'] = self.initial_capital + trades['cumulative_pnl']
        
        # Maximum drawdown
        cumulative_max = trades['cumulative_capital'].expanding().max()
        drawdown = trades['cumulative_capital'] - cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        
        # Risk-reward ratio (average)
        avg_risk_reward = trades['t1_reward'].mean() / trades['risk_amount'].mean() if trades['risk_amount'].mean() > 0 else 0
        
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
    
    def calculate_operational_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate operational and execution metrics."""
        if len(trades) == 0:
            return {}
        
        # Trade frequency
        if len(trades) > 1:
            start_date = pd.to_datetime(trades['entry_time'].min())
            end_date = pd.to_datetime(trades['entry_time'].max())
            days = (end_date - start_date).days
            trades_per_day = len(trades) / days if days > 0 else 0
        else:
            trades_per_day = 0
        
        # Transaction costs
        total_commission = len(trades) * self.commission_per_trade
        
        # Slippage (simplified - assume 0.01% per trade)
        total_slippage = trades['position_size'].sum() * 0.0001
        
        return {
            'trades_per_day': trades_per_day,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'avg_commission_per_trade': self.commission_per_trade,
            'total_transaction_costs': total_commission + total_slippage
        }
    
    def calculate_qrs_performance(self, trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by QRS grade."""
        qrs_performance = {}
        
        for grade in ['A', 'B', 'C']:
            grade_trades = trades[trades['qrs_grade'] == grade]
            if len(grade_trades) > 0:
                qrs_performance[grade] = {
                    'count': len(grade_trades),
                    'win_rate': len(grade_trades[grade_trades['is_winner']]) / len(grade_trades) * 100,
                    'avg_pnl': grade_trades['pnl'].mean(),
                    'total_pnl': grade_trades['pnl'].sum(),
                    'avg_return': grade_trades['pnl_pct'].mean(),
                    'max_win': grade_trades['pnl'].max(),
                    'max_loss': grade_trades['pnl'].min()
                }
            else:
                qrs_performance[grade] = {
                    'count': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'avg_return': 0,
                    'max_win': 0,
                    'max_loss': 0
                }
        
        return qrs_performance
    
    def generate_performance_report(self, trades: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive performance report."""
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
            'operational': self.calculate_operational_metrics(trades),
            'qrs_performance': self.calculate_qrs_performance(trades)
        }
        
        return report
    
    def print_performance_summary(self, report: Dict[str, any]):
        """Print a formatted performance summary."""
        print("\n" + "="*80)
        print("ZONE FADE STRATEGY PERFORMANCE ANALYSIS")
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
        
        # Operational
        ops = report['operational']
        print(f"\n⚙️  OPERATIONAL METRICS")
        print(f"   Trades per Day: {ops['trades_per_day']:.2f}")
        print(f"   Total Commission: ${ops['total_commission']:.2f}")
        print(f"   Total Transaction Costs: ${ops['total_transaction_costs']:.2f}")
        
        # QRS Performance
        qrs = report['qrs_performance']
        print(f"\n🎯 QRS GRADE PERFORMANCE")
        for grade in ['A', 'B', 'C']:
            grade_data = qrs[grade]
            print(f"   Grade {grade}:")
            print(f"     Trades: {grade_data['count']}")
            print(f"     Win Rate: {grade_data['win_rate']:.1f}%")
            print(f"     Avg P&L: ${grade_data['avg_pnl']:.2f}")
            print(f"     Total P&L: ${grade_data['total_pnl']:.2f}")
        
        print("\n" + "="*80)


def analyze_zone_fade_performance(csv_file_path: str, 
                                initial_capital: float = 100000.0,
                                commission_per_trade: float = 1.0) -> Dict[str, any]:
    """
    Analyze Zone Fade strategy performance from CSV file.
    
    Args:
        csv_file_path: Path to the entry points CSV file
        initial_capital: Starting capital for backtesting
        commission_per_trade: Commission cost per trade
    
    Returns:
        Comprehensive performance report
    """
    # Load entry points data
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(initial_capital, commission_per_trade)
    
    # Simulate trades
    print("🔄 Simulating trades...")
    trades = analyzer.simulate_trades(entry_points, position_sizing_method="risk_based")
    
    # Generate performance report
    print("📊 Calculating performance metrics...")
    report = analyzer.generate_performance_report(trades)
    
    # Print summary
    analyzer.print_performance_summary(report)
    
    return report


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    report = analyze_zone_fade_performance(csv_file)