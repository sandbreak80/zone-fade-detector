#!/usr/bin/env python3
"""
Asset-Specific Performance Analyzer for Zone Fade Strategy

This module provides detailed performance analysis by asset (SPY, QQQ, IWM)
including risk/reward, win rates, and strategic recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import random


class AssetPerformanceAnalyzer:
    """Asset-specific performance analysis for Zone Fade trading strategy."""
    
    def __init__(self, initial_capital: float = 100000.0, commission_per_trade: float = 1.0):
        """
        Initialize asset performance analyzer.
        
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
        
    def simulate_realistic_trade(self, entry: pd.Series) -> Dict:
        """Simulate a realistic trade with proper exit conditions."""
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
            't1_reward': entry.get('t1_reward', 0),
            't2_reward': entry.get('t2_reward', 0),
            't3_reward': entry.get('t3_reward', 0),
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
    
    def calculate_asset_metrics(self, trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for each asset."""
        asset_metrics = {}
        
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            
            if len(symbol_trades) == 0:
                continue
            
            # Basic P&L metrics
            total_pnl = symbol_trades['pnl'].sum()
            gross_profit = symbol_trades[symbol_trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(symbol_trades[symbol_trades['pnl'] < 0]['pnl'].sum())
            
            # Win/Loss analysis
            winning_trades = symbol_trades[symbol_trades['is_winner']]
            losing_trades = symbol_trades[symbol_trades['is_loser']]
            total_trades = len(symbol_trades)
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            # Risk/Reward metrics
            avg_risk_amount = symbol_trades['risk_amount'].mean()
            avg_t1_reward = symbol_trades['t1_reward'].mean()
            avg_t2_reward = symbol_trades['t2_reward'].mean()
            avg_t3_reward = symbol_trades['t3_reward'].mean()
            
            # Risk/Reward ratios
            risk_reward_t1 = avg_t1_reward / avg_risk_amount if avg_risk_amount > 0 else 0
            risk_reward_t2 = avg_t2_reward / avg_risk_amount if avg_risk_amount > 0 else 0
            risk_reward_t3 = avg_t3_reward / avg_risk_amount if avg_risk_amount > 0 else 0
            
            # Win/Loss ratio
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Profit factor
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy
            loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0
            expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)
            
            # Return metrics
            total_return = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            
            # Risk metrics
            returns = symbol_trades['pnl_pct'].values
            volatility = np.std(returns) if len(returns) > 0 else 0
            max_win = symbol_trades['pnl'].max()
            max_loss = symbol_trades['pnl'].min()
            
            # Drawdown analysis
            symbol_trades_sorted = symbol_trades.sort_values('entry_time').copy()
            symbol_trades_sorted['cumulative_pnl'] = symbol_trades_sorted['pnl'].cumsum()
            symbol_trades_sorted['cumulative_capital'] = self.initial_capital + symbol_trades_sorted['cumulative_pnl']
            
            cumulative_max = symbol_trades_sorted['cumulative_capital'].expanding().max()
            drawdown = symbol_trades_sorted['cumulative_capital'] - cumulative_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            
            # Exit scenario analysis
            exit_scenarios = symbol_trades['exit_scenario'].value_counts()
            stop_loss_rate = exit_scenarios.get('stop_loss', 0) / total_trades * 100
            t1_rate = exit_scenarios.get('t1', 0) / total_trades * 100
            t2_rate = exit_scenarios.get('t2', 0) / total_trades * 100
            t3_rate = exit_scenarios.get('t3', 0) / total_trades * 100
            
            asset_metrics[symbol] = {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'net_profit': total_pnl,
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_risk_amount': avg_risk_amount,
                'avg_t1_reward': avg_t1_reward,
                'avg_t2_reward': avg_t2_reward,
                'avg_t3_reward': avg_t3_reward,
                'risk_reward_t1': risk_reward_t1,
                'risk_reward_t2': risk_reward_t2,
                'risk_reward_t3': risk_reward_t3,
                'volatility': volatility,
                'max_win': max_win,
                'max_loss': max_loss,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'stop_loss_rate': stop_loss_rate,
                't1_rate': t1_rate,
                't2_rate': t2_rate,
                't3_rate': t3_rate,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'breakeven_trades': len(symbol_trades[symbol_trades['is_breakeven']])
            }
        
        return asset_metrics
    
    def calculate_overall_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall strategy metrics."""
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
        
        # Risk/Reward metrics
        avg_risk_amount = trades['risk_amount'].mean()
        avg_t1_reward = trades['t1_reward'].mean()
        avg_t2_reward = trades['t2_reward'].mean()
        avg_t3_reward = trades['t3_reward'].mean()
        
        # Risk/Reward ratios
        risk_reward_t1 = avg_t1_reward / avg_risk_amount if avg_risk_amount > 0 else 0
        risk_reward_t2 = avg_t2_reward / avg_risk_amount if avg_risk_amount > 0 else 0
        risk_reward_t3 = avg_t3_reward / avg_risk_amount if avg_risk_amount > 0 else 0
        
        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)
        
        # Return metrics
        total_return = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Risk metrics
        returns = trades['pnl_pct'].values
        volatility = np.std(returns) if len(returns) > 0 else 0
        max_win = trades['pnl'].max()
        max_loss = trades['pnl'].min()
        
        # Drawdown analysis
        trades_sorted = trades.sort_values('entry_time').copy()
        trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()
        trades_sorted['cumulative_capital'] = self.initial_capital + trades_sorted['cumulative_pnl']
        
        cumulative_max = trades_sorted['cumulative_capital'].expanding().max()
        drawdown = trades_sorted['cumulative_capital'] - cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': total_pnl,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_risk_amount': avg_risk_amount,
            'avg_t1_reward': avg_t1_reward,
            'avg_t2_reward': avg_t2_reward,
            'avg_t3_reward': avg_t3_reward,
            'risk_reward_t1': risk_reward_t1,
            'risk_reward_t2': risk_reward_t2,
            'risk_reward_t3': risk_reward_t3,
            'volatility': volatility,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct
        }
    
    def generate_asset_report(self, trades: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive asset-specific report."""
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
            'overall': self.calculate_overall_metrics(trades),
            'by_asset': self.calculate_asset_metrics(trades)
        }
        
        return report
    
    def print_asset_summary(self, report: Dict[str, any]):
        """Print a formatted asset-specific performance summary."""
        print("\n" + "="*100)
        print("ZONE FADE STRATEGY - ASSET-SPECIFIC PERFORMANCE ANALYSIS")
        print("="*100)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"   Analysis Date: {summary['analysis_date']}")
        print(f"   Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Period: {summary['analysis_period']['start']} to {summary['analysis_period']['end']}")
        
        # Overall metrics
        overall = report['overall']
        print(f"\n🎯 OVERALL STRATEGY PERFORMANCE")
        print(f"   Net Profit: ${overall['net_profit']:,.2f} ({overall['total_return']:.2f}%)")
        print(f"   Win Rate: {overall['win_rate']:.1f}%")
        print(f"   Average Win: ${overall['avg_win']:.2f}")
        print(f"   Average Loss: ${overall['avg_loss']:.2f}")
        print(f"   Win/Loss Ratio: {overall['win_loss_ratio']:.2f}")
        print(f"   Profit Factor: {overall['profit_factor']:.2f}")
        print(f"   Expectancy: ${overall['expectancy']:.2f}")
        print(f"   Max Drawdown: ${overall['max_drawdown']:,.2f} ({overall['max_drawdown_pct']:.2f}%)")
        
        # Asset-specific metrics
        print(f"\n📈 ASSET-SPECIFIC PERFORMANCE")
        print(f"{'Asset':<6} {'Trades':<7} {'P&L':<12} {'Return%':<8} {'Win%':<6} {'Avg Win':<10} {'Avg Loss':<10} {'W/L Ratio':<10} {'R/R T1':<8} {'R/R T2':<8} {'R/R T3':<8} {'DD%':<6}")
        print("-" * 100)
        
        for symbol, metrics in report['by_asset'].items():
            print(f"{symbol:<6} {metrics['total_trades']:<7} ${metrics['net_profit']:<11,.0f} {metrics['total_return']:<7.1f}% {metrics['win_rate']:<5.1f}% ${metrics['avg_win']:<9.0f} ${metrics['avg_loss']:<9.0f} {metrics['win_loss_ratio']:<9.1f} {metrics['risk_reward_t1']:<7.2f} {metrics['risk_reward_t2']:<7.2f} {metrics['risk_reward_t3']:<7.2f} {metrics['max_drawdown_pct']:<5.1f}%")
        
        # Detailed asset analysis
        print(f"\n🔍 DETAILED ASSET ANALYSIS")
        for symbol, metrics in report['by_asset'].items():
            print(f"\n   {symbol} DETAILED METRICS:")
            print(f"     Total Trades: {metrics['total_trades']}")
            print(f"     Net Profit: ${metrics['net_profit']:,.2f} ({metrics['total_return']:.2f}%)")
            print(f"     Win Rate: {metrics['win_rate']:.1f}% ({metrics['winning_trades']} wins, {metrics['losing_trades']} losses)")
            print(f"     Average Win: ${metrics['avg_win']:.2f}")
            print(f"     Average Loss: ${metrics['avg_loss']:.2f}")
            print(f"     Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
            print(f"     Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"     Expectancy: ${metrics['expectancy']:.2f}")
            print(f"     Risk/Reward T1: {metrics['risk_reward_t1']:.2f}")
            print(f"     Risk/Reward T2: {metrics['risk_reward_t2']:.2f}")
            print(f"     Risk/Reward T3: {metrics['risk_reward_t3']:.2f}")
            print(f"     Max Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
            print(f"     Volatility: {metrics['volatility']:.2f}%")
            print(f"     Stop Loss Rate: {metrics['stop_loss_rate']:.1f}%")
            print(f"     T1 Hit Rate: {metrics['t1_rate']:.1f}%")
            print(f"     T2 Hit Rate: {metrics['t2_rate']:.1f}%")
            print(f"     T3 Hit Rate: {metrics['t3_rate']:.1f}%")
        
        print("\n" + "="*100)
    
    def generate_improvement_recommendations(self, report: Dict[str, any]) -> List[str]:
        """Generate strategic improvement recommendations."""
        recommendations = []
        overall = report['overall']
        by_asset = report['by_asset']
        
        # Win rate analysis
        if overall['win_rate'] < 60:
            recommendations.append("🔴 LOW WIN RATE: Consider improving entry criteria or stop loss placement")
        elif overall['win_rate'] < 70:
            recommendations.append("🟡 MODERATE WIN RATE: Fine-tune entry timing and risk management")
        else:
            recommendations.append("🟢 GOOD WIN RATE: Maintain current approach")
        
        # Risk/Reward analysis
        if overall['risk_reward_t1'] < 0.5:
            recommendations.append("🔴 POOR RISK/REWARD: T1 targets too close to entry, consider wider targets")
        elif overall['risk_reward_t1'] < 1.0:
            recommendations.append("🟡 MODERATE RISK/REWARD: T1 targets could be improved")
        else:
            recommendations.append("🟢 GOOD RISK/REWARD: T1 targets are well-positioned")
        
        # Drawdown analysis
        if overall['max_drawdown_pct'] > 5:
            recommendations.append("🔴 HIGH DRAWDOWN: Implement better risk management or position sizing")
        elif overall['max_drawdown_pct'] > 2:
            recommendations.append("🟡 MODERATE DRAWDOWN: Consider reducing position sizes during losing streaks")
        else:
            recommendations.append("🟢 LOW DRAWDOWN: Excellent risk management")
        
        # Asset-specific recommendations
        for symbol, metrics in by_asset.items():
            if metrics['win_rate'] < 50:
                recommendations.append(f"🔴 {symbol}: Very low win rate ({metrics['win_rate']:.1f}%) - consider avoiding or improving")
            elif metrics['win_rate'] < 60:
                recommendations.append(f"🟡 {symbol}: Low win rate ({metrics['win_rate']:.1f}%) - needs improvement")
            
            if metrics['risk_reward_t1'] < 0.3:
                recommendations.append(f"🔴 {symbol}: Poor T1 risk/reward ({metrics['risk_reward_t1']:.2f}) - adjust targets")
            
            if metrics['max_drawdown_pct'] > 3:
                recommendations.append(f"🔴 {symbol}: High drawdown ({metrics['max_drawdown_pct']:.1f}%) - reduce position size")
        
        # Additional strategic recommendations
        recommendations.extend([
            "📊 CONSIDER: Add market regime filters (trend vs range-bound)",
            "📊 CONSIDER: Implement dynamic position sizing based on volatility",
            "📊 CONSIDER: Add time-of-day filters for better entry timing",
            "📊 CONSIDER: Implement correlation analysis between assets",
            "📊 CONSIDER: Add economic calendar filters for major events",
            "📊 CONSIDER: Implement trailing stops for T2/T3 targets",
            "📊 CONSIDER: Add volume profile analysis for better zone identification",
            "📊 CONSIDER: Implement multi-timeframe analysis for better context"
        ])
        
        return recommendations


def analyze_asset_performance(csv_file_path: str, 
                            initial_capital: float = 100000.0,
                            commission_per_trade: float = 1.0) -> Dict[str, any]:
    """
    Analyze asset-specific Zone Fade strategy performance from CSV file.
    
    Args:
        csv_file_path: Path to the entry points CSV file
        initial_capital: Starting capital for backtesting
        commission_per_trade: Commission cost per trade
    
    Returns:
        Comprehensive asset-specific performance report
    """
    # Load entry points data
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Initialize analyzer
    analyzer = AssetPerformanceAnalyzer(initial_capital, commission_per_trade)
    
    # Simulate realistic trades
    print("🔄 Simulating realistic trades for asset analysis...")
    trades = []
    for _, entry in entry_points.iterrows():
        trade = analyzer.simulate_realistic_trade(entry)
        trades.append(trade)
    trades_df = pd.DataFrame(trades)
    
    # Generate performance report
    print("📊 Calculating asset-specific performance metrics...")
    report = analyzer.generate_asset_report(trades_df)
    
    # Print summary
    analyzer.print_asset_summary(report)
    
    # Generate recommendations
    recommendations = analyzer.generate_improvement_recommendations(report)
    print(f"\n🎯 STRATEGIC IMPROVEMENT RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"   {rec}")
    
    return report


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    report = analyze_asset_performance(csv_file)