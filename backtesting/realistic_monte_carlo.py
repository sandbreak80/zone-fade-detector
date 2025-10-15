#!/usr/bin/env python3
"""
Realistic Monte Carlo Simulation for Zone Fade Strategy

This module addresses potential issues in our previous simulation:
1. More realistic exit probabilities
2. Market condition variations
3. Position sizing errors
4. Calculation validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import random
import statistics
from collections import defaultdict


class RealisticMonteCarloSimulator:
    """More realistic Monte Carlo simulation with market variations."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 max_equity_per_trade: float = 0.10,
                 slippage_ticks: int = 2,
                 commission_per_trade: float = 5.0,
                 num_simulations: int = 100):
        """Initialize realistic Monte Carlo simulator."""
        self.initial_balance = initial_balance
        self.max_equity_per_trade = max_equity_per_trade
        self.slippage_ticks = slippage_ticks
        self.commission_per_trade = commission_per_trade
        self.num_simulations = num_simulations
        
        # More realistic trading parameters with market variations
        self.base_stop_loss_rate = 0.40  # 40% base stop loss rate
        self.base_t1_rate = 0.35  # 35% base T1 rate
        self.base_t2_rate = 0.15  # 15% base T2 rate
        self.base_t3_rate = 0.10  # 10% base T3 rate
        
        # Market condition variations
        self.market_conditions = ['bull', 'bear', 'sideways', 'volatile']
        self.condition_probabilities = [0.3, 0.2, 0.3, 0.2]
        
        # Track results
        self.simulation_results = []
        self.asset_results = defaultdict(list)
        self.calculation_errors = []
    
    def get_market_condition(self, simulation_id: int, trade_index: int) -> str:
        """Determine market condition for this trade."""
        random.seed(simulation_id * 1000 + trade_index)
        rand = random.random()
        
        cumulative = 0
        for condition, prob in zip(self.market_conditions, self.condition_probabilities):
            cumulative += prob
            if rand <= cumulative:
                return condition
        return 'sideways'
    
    def get_exit_probabilities(self, market_condition: str) -> Dict[str, float]:
        """Get exit probabilities based on market condition."""
        if market_condition == 'bull':
            return {
                'stop_loss': 0.25,  # Lower stop loss in bull market
                't1': 0.40,
                't2': 0.20,
                't3': 0.15
            }
        elif market_condition == 'bear':
            return {
                'stop_loss': 0.55,  # Higher stop loss in bear market
                't1': 0.25,
                't2': 0.15,
                't3': 0.05
            }
        elif market_condition == 'volatile':
            return {
                'stop_loss': 0.45,  # Higher stop loss in volatile market
                't1': 0.30,
                't2': 0.15,
                't3': 0.10
            }
        else:  # sideways
            return {
                'stop_loss': 0.40,
                't1': 0.35,
                't2': 0.15,
                't3': 0.10
            }
    
    def calculate_tick_value(self, price: float) -> float:
        """Calculate tick value based on price level."""
        return 0.01  # $0.01 per tick for ETFs
    
    def calculate_slippage_amount(self, price: float, market_condition: str) -> float:
        """Calculate slippage amount with market condition variation."""
        base_tick_value = self.calculate_tick_value(price)
        
        # Adjust slippage based on market condition
        if market_condition == 'volatile':
            multiplier = 1.5
        elif market_condition == 'bear':
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return self.slippage_ticks * base_tick_value * multiplier
    
    def calculate_position_size(self, entry_price: float, stop_price: float, 
                              current_balance: float, market_condition: str) -> int:
        """Calculate position size with market condition adjustments."""
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            return 0
        
        # Adjust position size based on market condition
        if market_condition == 'volatile':
            max_equity = self.max_equity_per_trade * 0.7  # Reduce size in volatile markets
        elif market_condition == 'bear':
            max_equity = self.max_equity_per_trade * 0.8  # Reduce size in bear markets
        else:
            max_equity = self.max_equity_per_trade
        
        max_risk_amount = current_balance * max_equity
        position_size = int(max_risk_amount / risk_per_share)
        position_size = max(1, position_size)
        
        # Ensure we don't risk more than we have
        max_affordable = int(current_balance * 0.95 / entry_price)
        position_size = min(position_size, max_affordable)
        
        return position_size
    
    def validate_entry_data(self, entry: pd.Series) -> List[str]:
        """Validate entry data for calculation errors."""
        errors = []
        
        # Check for missing required fields
        required_fields = ['price', 'hard_stop', 't1_price', 't2_price', 't3_price']
        for field in required_fields:
            if field not in entry or pd.isna(entry[field]):
                errors.append(f"Missing {field}")
        
        # Check for logical errors
        if 'price' in entry and 'hard_stop' in entry:
            if entry['price'] <= 0 or entry['hard_stop'] <= 0:
                errors.append("Invalid price or stop price")
            
            if entry['direction'] == 'LONG' and entry['hard_stop'] >= entry['price']:
                errors.append("Long trade: stop should be below entry price")
            elif entry['direction'] == 'SHORT' and entry['hard_stop'] <= entry['price']:
                errors.append("Short trade: stop should be above entry price")
        
        # Check target prices
        if all(field in entry for field in ['price', 't1_price', 't2_price', 't3_price']):
            if entry['direction'] == 'LONG':
                if entry['t1_price'] <= entry['price']:
                    errors.append("Long trade: T1 should be above entry price")
                if entry['t2_price'] <= entry['t1_price']:
                    errors.append("T2 should be above T1")
                if entry['t3_price'] <= entry['t2_price']:
                    errors.append("T3 should be above T2")
            else:  # SHORT
                if entry['t1_price'] >= entry['price']:
                    errors.append("Short trade: T1 should be below entry price")
                if entry['t2_price'] >= entry['t1_price']:
                    errors.append("T2 should be below T1")
                if entry['t3_price'] >= entry['t2_price']:
                    errors.append("T3 should be below T2")
        
        return errors
    
    def simulate_trade(self, entry: pd.Series, current_balance: float, 
                      simulation_id: int, trade_index: int) -> Dict:
        """Simulate a single trade with realistic market conditions."""
        # Validate entry data
        validation_errors = self.validate_entry_data(entry)
        if validation_errors:
            self.calculation_errors.extend(validation_errors)
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
                'exit_scenario': 'validation_error',
                'balance_after': current_balance,
                'validation_errors': validation_errors
            }
        
        # Determine market condition
        market_condition = self.get_market_condition(simulation_id, trade_index)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            entry['price'], entry['hard_stop'], current_balance, market_condition
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
                'market_condition': market_condition
            }
        
        # Calculate slippage
        entry_slippage = self.calculate_slippage_amount(entry['price'], market_condition)
        if entry['direction'] == 'LONG':
            actual_entry_price = entry['price'] + entry_slippage
        else:
            actual_entry_price = entry['price'] - entry_slippage
        
        # Determine exit scenario
        exit_scenario = self._determine_exit_scenario(entry, simulation_id, trade_index, market_condition)
        
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
        exit_slippage = self.calculate_slippage_amount(exit_price, market_condition)
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
            'market_condition': market_condition,
            'entry_slippage': entry_slippage,
            'exit_slippage': exit_slippage
        }
    
    def _determine_exit_scenario(self, entry: pd.Series, simulation_id: int, 
                               trade_index: int, market_condition: str) -> str:
        """Determine exit scenario based on market condition."""
        # Use simulation_id and trade_index as seed for consistency
        random.seed(simulation_id * 1000 + trade_index)
        rand = random.random()
        
        # Get exit probabilities for this market condition
        probs = self.get_exit_probabilities(market_condition)
        
        if rand < probs['stop_loss']:
            return 'stop_loss'
        elif rand < probs['stop_loss'] + probs['t1']:
            return 't1'
        elif rand < probs['stop_loss'] + probs['t1'] + probs['t2']:
            return 't2'
        elif rand < probs['stop_loss'] + probs['t1'] + probs['t2'] + probs['t3']:
            return 't3'
        else:
            return 'partial_fill'
    
    def run_single_simulation(self, entry_points: pd.DataFrame, simulation_id: int) -> Dict:
        """Run a single Monte Carlo simulation."""
        trades = []
        current_balance = self.initial_balance
        
        # Sort by entry time
        entry_points_sorted = entry_points.sort_values('timestamp').copy()
        
        for trade_index, (_, entry) in enumerate(entry_points_sorted.iterrows()):
            trade = self.simulate_trade(entry, current_balance, simulation_id, trade_index)
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
                'expectancy': 0.0,
                'validation_errors': len(self.calculation_errors)
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
            'avg_loss': avg_loss,
            'validation_errors': len(self.calculation_errors)
        }
    
    def run_monte_carlo_simulation(self, entry_points: pd.DataFrame) -> Dict[str, any]:
        """Run complete Monte Carlo simulation."""
        print(f"🔄 Running {self.num_simulations} realistic Monte Carlo simulations...")
        
        # Group by symbol for asset-specific analysis
        symbols = entry_points['symbol'].unique()
        
        for simulation_id in range(self.num_simulations):
            if simulation_id % 10 == 0:
                print(f"   Simulation {simulation_id + 1}/{self.num_simulations}")
            
            # Reset calculation errors for each simulation
            self.calculation_errors = []
            
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
        print("REALISTIC MONTE CARLO SIMULATION RESULTS - ZONE FADE STRATEGY")
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
        
        # Asset-specific results
        print(f"\n📈 ASSET-SPECIFIC PERFORMANCE")
        for symbol, asset_stats in stats['assets'].items():
            print(f"\n   {symbol}:")
            print(f"     Success Rate: {asset_stats['success_rate']:.1f}%")
            print(f"     Avg Return: {asset_stats['total_return']['mean']:.2f}% ± {asset_stats['total_return']['std']:.2f}%")
            print(f"     Avg Win Rate: {asset_stats['win_rate']['mean']:.1f}% ± {asset_stats['win_rate']['std']:.1f}%")
            print(f"     Avg Drawdown: ${asset_stats['max_drawdown']['mean']:,.2f} ± ${asset_stats['max_drawdown']['std']:,.2f}")
            print(f"     Avg Profit Factor: {asset_stats['profit_factor']['mean']:.2f} ± {asset_stats['profit_factor']['std']:.2f}")
        
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


def run_realistic_monte_carlo_analysis(csv_file_path: str, 
                                     initial_balance: float = 10000.0,
                                     max_equity_per_trade: float = 0.10,
                                     slippage_ticks: int = 2,
                                     commission_per_trade: float = 5.0,
                                     num_simulations: int = 100) -> Dict[str, any]:
    """Run comprehensive realistic Monte Carlo analysis."""
    # Load entry points data
    try:
        entry_points = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(entry_points)} entry points from {csv_file_path}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {"error": f"Failed to load CSV file: {e}"}
    
    # Initialize realistic Monte Carlo simulator
    simulator = RealisticMonteCarloSimulator(
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
    results = run_realistic_monte_carlo_analysis(csv_file, num_simulations=100)