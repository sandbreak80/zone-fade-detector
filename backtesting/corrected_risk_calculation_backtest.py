#!/usr/bin/env python3
"""
Corrected Risk Calculation Backtest

This script fixes the critical risk calculation error that was causing
unrealistic 100% success rates in our Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import random
import statistics
from collections import defaultdict


class CorrectedRiskSimulator:
    """Simulator with corrected risk calculation."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 max_equity_per_trade: float = 0.10,
                 slippage_ticks: int = 2,
                 commission_per_trade: float = 5.0,
                 num_simulations: int = 100):
        """Initialize corrected simulator."""
        self.initial_balance = initial_balance
        self.max_equity_per_trade = max_equity_per_trade
        self.slippage_ticks = slippage_ticks
        self.commission_per_trade = commission_per_trade
        self.num_simulations = num_simulations
        
        # More realistic trading parameters
        self.stop_loss_hit_rate = 0.40  # 40% of trades hit stop loss
        self.t1_hit_rate = 0.35  # 35% of trades hit T1
        self.t2_hit_rate = 0.15  # 15% of trades hit T2
        self.t3_hit_rate = 0.10  # 10% of trades hit T3
        
        # Track results
        self.simulation_results = []
        self.asset_results = defaultdict(list)
    
    def calculate_corrected_risk_amount(self, entry_price: float, hard_stop: float, direction: str) -> float:
        """Calculate risk amount with CORRECTED logic."""
        if direction == "LONG":
            # For LONG trades: risk = entry_price - hard_stop (entry above stop)
            risk_amount = entry_price - hard_stop
        else:  # SHORT
            # For SHORT trades: risk = hard_stop - entry_price (stop above entry)
            risk_amount = hard_stop - entry_price
        
        # Ensure positive risk amount
        if risk_amount <= 0:
            risk_amount = entry_price * 0.01  # 1% fallback
        
        return risk_amount
    
    def calculate_tick_value(self, price: float) -> float:
        """Calculate tick value based on price level."""
        return 0.01  # $0.01 per tick for ETFs
    
    def calculate_slippage_amount(self, price: float) -> float:
        """Calculate slippage amount based on tick value."""
        tick_value = self.calculate_tick_value(price)
        return self.slippage_ticks * tick_value
    
    def calculate_position_size(self, entry_price: float, risk_amount: float, 
                              current_balance: float) -> int:
        """Calculate position size based on corrected risk amount."""
        if risk_amount <= 0:
            return 0
        
        max_risk_amount = current_balance * self.max_equity_per_trade
        position_size = int(max_risk_amount / risk_amount)
        position_size = max(1, position_size)
        
        # Ensure we don't risk more than we have
        max_affordable = int(current_balance * 0.95 / entry_price)
        position_size = min(position_size, max_affordable)
        
        return position_size
    
    def simulate_trade(self, entry: pd.Series, current_balance: float, 
                      simulation_id: int) -> Dict:
        """Simulate a single trade with corrected risk calculation."""
        # Calculate CORRECTED risk amount
        risk_amount = self.calculate_corrected_risk_amount(
            entry['price'], entry['hard_stop'], entry['direction']
        )
        
        # Calculate position size
        position_size = self.calculate_position_size(
            entry['price'], risk_amount, current_balance
        )
        
        if position_size <= 0:
            return {
                'simulation_id': simulation_id,
                'entry_id': entry['entry_id'],
                'symbol': entry['symbol'],
                'entry_price': entry['price'],
                'exit_price': entry['price'],
                'position_size': 0,
                'pnl': 0,
                'is_winner': False,
                'is_loser': False,
                'exit_scenario': 'no_trade',
                'balance_after': current_balance,
                'corrected_risk_amount': risk_amount
            }
        
        # Calculate slippage
        entry_slippage = self.calculate_slippage_amount(entry['price'])
        if entry['direction'] == 'LONG':
            actual_entry_price = entry['price'] + entry_slippage
        else:
            actual_entry_price = entry['price'] - entry_slippage
        
        # Determine exit scenario
        exit_scenario = self._determine_exit_scenario(entry, simulation_id)
        
        # Calculate exit price
        if exit_scenario == 'stop_loss':
            exit_price = entry['hard_stop']
        elif exit_scenario == 't1':
            exit_price = entry['t1_price']
        elif exit_scenario == 't2':
            exit_price = entry['t2_price']
        elif exit_scenario == 't3':
            exit_price = entry['t3_price']
        else:  # partial_fill
            if entry['direction'] == 'LONG':
                exit_price = actual_entry_price + (entry['t1_price'] - actual_entry_price) * 0.3
            else:
                exit_price = actual_entry_price - (actual_entry_price - entry['t1_price']) * 0.3
        
        # Calculate exit slippage
        exit_slippage = self.calculate_slippage_amount(exit_price)
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
        total_costs = self.commission_per_trade * 2  # Entry + Exit
        
        # Calculate net P&L
        net_pnl = pnl - total_costs
        
        # Calculate new balance
        new_balance = current_balance + net_pnl
        
        return {
            'simulation_id': simulation_id,
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'entry_price': actual_entry_price,
            'exit_price': actual_exit_price,
            'position_size': position_size,
            'pnl': net_pnl,
            'is_winner': net_pnl > 0,
            'is_loser': net_pnl < 0,
            'exit_scenario': exit_scenario,
            'balance_after': new_balance,
            'corrected_risk_amount': risk_amount,
            'original_risk_amount': entry['risk_amount']
        }
    
    def _determine_exit_scenario(self, entry: pd.Series, simulation_id: int) -> str:
        """Determine exit scenario based on probabilities."""
        random.seed(simulation_id + hash(entry['entry_id']) % 1000)
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
    
    def run_single_simulation(self, entry_points: pd.DataFrame, simulation_id: int) -> Dict:
        """Run a single Monte Carlo simulation."""
        trades = []
        current_balance = self.initial_balance
        
        # Sort by entry time
        entry_points_sorted = entry_points.sort_values('timestamp').copy()
        
        for _, entry in entry_points_sorted.iterrows():
            trade = self.simulate_trade(entry, current_balance, simulation_id)
            trades.append(trade)
            current_balance = trade['balance_after']
        
        # Calculate metrics for this simulation
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) == 0:
            return {
                'simulation_id': simulation_id,
                'final_balance': self.initial_balance,
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
        
        # Calculate performance metrics
        total_pnl = trades_df['pnl'].sum()
        final_balance = self.initial_balance + total_pnl
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        winning_trades = trades_df[trades_df['is_winner']]
        losing_trades = trades_df[trades_df['is_loser']]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)
        
        # Calculate max drawdown
        trades_df['running_balance'] = trades_df['balance_after']
        trades_df['running_max'] = trades_df['running_balance'].expanding().max()
        trades_df['drawdown'] = trades_df['running_balance'] - trades_df['running_max']
        max_drawdown = trades_df['drawdown'].min()
        
        return {
            'simulation_id': simulation_id,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def run_monte_carlo_simulation(self, entry_points: pd.DataFrame) -> Dict[str, any]:
        """Run complete Monte Carlo simulation."""
        print(f"🔄 Running {self.num_simulations} corrected Monte Carlo simulations...")
        
        # Group by symbol for asset-specific analysis
        symbols = entry_points['symbol'].unique()
        
        for simulation_id in range(self.num_simulations):
            if simulation_id % 10 == 0:
                print(f"   Simulation {simulation_id + 1}/{self.num_simulations}")
            
            # Run simulation for all symbols combined
            result = self.run_single_simulation(entry_points, simulation_id)
            self.simulation_results.append(result)
            
            # Run simulation for each symbol individually
            for symbol in symbols:
                symbol_entries = entry_points[entry_points['symbol'] == symbol]
                symbol_result = self.run_single_simulation(symbol_entries, simulation_id)
                symbol_result['symbol'] = symbol
                self.asset_results[symbol].append(symbol_result)
        
        # Calculate statistics
        return self._calculate_monte_carlo_statistics()
    
    def _calculate_monte_carlo_statistics(self) -> Dict[str, any]:
        """Calculate comprehensive Monte Carlo statistics."""
        if not self.simulation_results:
            return {"error": "No simulation results to analyze"}
        
        # Overall statistics
        overall_stats = self._calculate_statistics_for_results(self.simulation_results, "Overall")
        
        # Asset-specific statistics
        asset_stats = {}
        for symbol, results in self.asset_results.items():
            asset_stats[symbol] = self._calculate_statistics_for_results(results, symbol)
        
        return {
            'overall': overall_stats,
            'assets': asset_stats,
            'simulation_parameters': {
                'num_simulations': self.num_simulations,
                'initial_balance': self.initial_balance,
                'max_equity_per_trade': self.max_equity_per_trade,
                'slippage_ticks': self.slippage_ticks,
                'commission_per_trade': self.commission_per_trade
            }
        }
    
    def _calculate_statistics_for_results(self, results: List[Dict], name: str) -> Dict[str, any]:
        """Calculate statistics for a set of simulation results."""
        if not results:
            return {}
        
        # Extract metrics
        final_balances = [r['final_balance'] for r in results]
        total_returns = [r['total_return'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        profit_factors = [r['profit_factor'] for r in results if r['profit_factor'] != float('inf')]
        expectancies = [r['expectancy'] for r in results]
        
        # Calculate statistics
        stats = {
            'name': name,
            'num_simulations': len(results),
            'final_balance': {
                'mean': statistics.mean(final_balances),
                'median': statistics.median(final_balances),
                'std': statistics.stdev(final_balances) if len(final_balances) > 1 else 0,
                'min': min(final_balances),
                'max': max(final_balances),
                'percentile_25': np.percentile(final_balances, 25),
                'percentile_75': np.percentile(final_balances, 75)
            },
            'total_return': {
                'mean': statistics.mean(total_returns),
                'median': statistics.median(total_returns),
                'std': statistics.stdev(total_returns) if len(total_returns) > 1 else 0,
                'min': min(total_returns),
                'max': max(total_returns),
                'percentile_25': np.percentile(total_returns, 25),
                'percentile_75': np.percentile(total_returns, 75)
            },
            'win_rate': {
                'mean': statistics.mean(win_rates),
                'median': statistics.median(win_rates),
                'std': statistics.stdev(win_rates) if len(win_rates) > 1 else 0,
                'min': min(win_rates),
                'max': max(win_rates)
            },
            'max_drawdown': {
                'mean': statistics.mean(max_drawdowns),
                'median': statistics.median(max_drawdowns),
                'std': statistics.stdev(max_drawdowns) if len(max_drawdowns) > 1 else 0,
                'min': min(max_drawdowns),
                'max': max(max_drawdowns)
            },
            'profit_factor': {
                'mean': statistics.mean(profit_factors) if profit_factors else 0,
                'median': statistics.median(profit_factors) if profit_factors else 0,
                'std': statistics.stdev(profit_factors) if len(profit_factors) > 1 else 0,
                'min': min(profit_factors) if profit_factors else 0,
                'max': max(profit_factors) if profit_factors else 0
            },
            'expectancy': {
                'mean': statistics.mean(expectancies),
                'median': statistics.median(expectancies),
                'std': statistics.stdev(expectancies) if len(expectancies) > 1 else 0,
                'min': min(expectancies),
                'max': max(expectancies)
            }
        }
        
        # Calculate success rate (profitable simulations)
        profitable_simulations = sum(1 for r in results if r['total_return'] > 0)
        stats['success_rate'] = profitable_simulations / len(results) * 100
        
        return stats
    
    def print_monte_carlo_summary(self, stats: Dict[str, any]):
        """Print comprehensive Monte Carlo summary."""
        print("\n" + "="*120)
        print("CORRECTED RISK CALCULATION - MONTE CARLO SIMULATION RESULTS")
        print("="*120)
        
        # Simulation parameters
        params = stats['simulation_parameters']
        print(f"\n📊 SIMULATION PARAMETERS")
        print(f"   Number of Simulations: {params['num_simulations']}")
        print(f"   Initial Balance: ${params['initial_balance']:,.2f}")
        print(f"   Max Equity per Trade: {params['max_equity_per_trade']*100:.1f}%")
        print(f"   Slippage: {params['slippage_ticks']} ticks")
        print(f"   Commission per Trade: ${params['commission_per_trade']:.2f}")
        
        # Overall results
        overall = stats['overall']
        print(f"\n🎯 OVERALL STRATEGY PERFORMANCE")
        print(f"   Success Rate: {overall['success_rate']:.1f}% of simulations profitable")
        print(f"   Final Balance: ${overall['final_balance']['mean']:,.2f} ± ${overall['final_balance']['std']:,.2f}")
        print(f"   Total Return: {overall['total_return']['mean']:.2f}% ± {overall['total_return']['std']:.2f}%")
        print(f"   Win Rate: {overall['win_rate']['mean']:.1f}% ± {overall['win_rate']['std']:.1f}%")
        print(f"   Max Drawdown: ${overall['max_drawdown']['mean']:,.2f} ± ${overall['max_drawdown']['std']:,.2f}")
        print(f"   Profit Factor: {overall['profit_factor']['mean']:.2f} ± {overall['profit_factor']['std']:.2f}")
        print(f"   Expectancy: ${overall['expectancy']['mean']:.2f} ± ${overall['expectancy']['std']:.2f}")
        
        # Risk analysis
        print(f"\n⚠️  RISK ANALYSIS")
        print(f"   Worst Case Return: {overall['total_return']['min']:.2f}%")
        print(f"   Best Case Return: {overall['total_return']['max']:.2f}%")
        print(f"   25th Percentile: {overall['total_return']['percentile_25']:.2f}%")
        print(f"   75th Percentile: {overall['total_return']['percentile_75']:.2f}%")
        print(f"   Worst Drawdown: ${overall['max_drawdown']['min']:,.2f}")
        print(f"   Best Drawdown: ${overall['max_drawdown']['max']:,.2f}")
        
        # Strategy assessment
        print(f"\n🔍 STRATEGY ASSESSMENT")
        if overall['success_rate'] > 80:
            print("   ✅ HIGH CONFIDENCE: Strategy shows consistent profitability")
        elif overall['success_rate'] > 60:
            print("   ⚠️  MODERATE CONFIDENCE: Strategy shows good profitability with some risk")
        elif overall['success_rate'] > 40:
            print("   ⚠️  LOW CONFIDENCE: Strategy shows mixed results")
        else:
            print("   ❌ LOW CONFIDENCE: Strategy shows poor profitability")
        
        if overall['total_return']['mean'] > 50:
            print("   ✅ HIGH RETURNS: Strategy generates strong returns on average")
        elif overall['total_return']['mean'] > 10:
            print("   ⚠️  MODERATE RETURNS: Strategy generates decent returns")
        else:
            print("   ❌ LOW RETURNS: Strategy generates weak returns")
        
        if overall['max_drawdown']['mean'] > -1000:
            print("   ✅ LOW RISK: Strategy shows controlled drawdowns")
        elif overall['max_drawdown']['mean'] > -5000:
            print("   ⚠️  MODERATE RISK: Strategy shows acceptable drawdowns")
        else:
            print("   ❌ HIGH RISK: Strategy shows concerning drawdowns")
        
        print("\n" + "="*120)


def run_corrected_risk_analysis(csv_file_path: str, 
                              initial_balance: float = 10000.0,
                              max_equity_per_trade: float = 0.10,
                              slippage_ticks: int = 2,
                              commission_per_trade: float = 5.0,
                              num_simulations: int = 100) -> Dict[str, any]:
    """Run corrected risk calculation analysis."""
    # Load entry points data
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Initialize corrected simulator
    simulator = CorrectedRiskSimulator(
        initial_balance=initial_balance,
        max_equity_per_trade=max_equity_per_trade,
        slippage_ticks=slippage_ticks,
        commission_per_trade=commission_per_trade,
        num_simulations=num_simulations
    )
    
    # Run Monte Carlo simulation
    results = simulator.run_monte_carlo_simulation(entry_points)
    
    # Print summary
    simulator.print_monte_carlo_summary(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    results = run_corrected_risk_analysis(csv_file, num_simulations=100)