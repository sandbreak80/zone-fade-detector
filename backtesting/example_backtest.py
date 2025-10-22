#!/usr/bin/env python3
"""
Example Bitcoin Zone Fade Backtesting Script.

This script demonstrates how to use the backtesting system with different
configurations and market conditions.
"""

import asyncio
import yaml
from pathlib import Path
from backtest_bitcoin import BitcoinBacktester

async def run_bull_market_backtest():
    """Test the strategy during a bull market period."""
    print("🐂 Running Bull Market Backtest (2024 Q1)")
    print("=" * 50)
    
    # Bull market configuration
    config = {
        'backtesting': {
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 10000,
            'position_size': 0.1,
            'commission': 0.001
        },
        'strategy': {
            'min_qrs_score': 6,      # Lower threshold for bull market
            'volume_threshold': 1.5,  # Moderate volume requirement
            'rejection_threshold': 0.3,
            'zone_proximity': 0.8
        }
    }
    
    backtester = BitcoinBacktester(config)
    results = await backtester.run_backtest(
        start_date="2024-01-01",
        end_date="2024-03-31",
        symbols=["bitcoin"]
    )
    
    print(f"📊 Bull Market Results:")
    print(f"   Total Return: {results.total_return:.2%}")
    print(f"   Win Rate: {results.win_rate:.2%}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {results.max_drawdown:.2%}")
    print()

async def run_bear_market_backtest():
    """Test the strategy during a bear market period."""
    print("🐻 Running Bear Market Backtest (2022)")
    print("=" * 50)
    
    # Bear market configuration
    config = {
        'backtesting': {
            'start_date': '2022-01-01',
            'end_date': '2022-12-31',
            'initial_capital': 10000,
            'position_size': 0.05,  # Smaller position size
            'commission': 0.001
        },
        'strategy': {
            'min_qrs_score': 7,      # Higher threshold for bear market
            'volume_threshold': 2.0,  # Higher volume requirement
            'rejection_threshold': 0.4,
            'zone_proximity': 1.0
        }
    }
    
    backtester = BitcoinBacktester(config)
    results = await backtester.run_backtest(
        start_date="2022-01-01",
        end_date="2022-12-31",
        symbols=["bitcoin"]
    )
    
    print(f"📊 Bear Market Results:")
    print(f"   Total Return: {results.total_return:.2%}")
    print(f"   Win Rate: {results.win_rate:.2%}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {results.max_drawdown:.2%}")
    print()

async def run_high_volatility_backtest():
    """Test the strategy during high volatility periods."""
    print("⚡ Running High Volatility Backtest (2024 Q2)")
    print("=" * 50)
    
    # High volatility configuration
    config = {
        'backtesting': {
            'start_date': '2024-04-01',
            'end_date': '2024-06-30',
            'initial_capital': 10000,
            'position_size': 0.08,  # Smaller position size
            'commission': 0.001
        },
        'strategy': {
            'min_qrs_score': 7,      # Higher quality required
            'volume_threshold': 2.5,  # Much higher volume
            'rejection_threshold': 0.4,
            'zone_proximity': 1.2    # Wider tolerance
        }
    }
    
    backtester = BitcoinBacktester(config)
    results = await backtester.run_backtest(
        start_date="2024-04-01",
        end_date="2024-06-30",
        symbols=["bitcoin"]
    )
    
    print(f"📊 High Volatility Results:")
    print(f"   Total Return: {results.total_return:.2%}")
    print(f"   Win Rate: {results.win_rate:.2%}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {results.max_drawdown:.2%}")
    print()

async def run_multi_symbol_backtest():
    """Test the strategy on multiple cryptocurrencies."""
    print("🪙 Running Multi-Symbol Backtest")
    print("=" * 50)
    
    # Multi-symbol configuration
    config = {
        'backtesting': {
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 10000,
            'position_size': 0.05,  # Smaller per symbol
            'commission': 0.001
        },
        'strategy': {
            'min_qrs_score': 6,
            'volume_threshold': 1.5,
            'rejection_threshold': 0.3,
            'zone_proximity': 0.8
        }
    }
    
    backtester = BitcoinBacktester(config)
    results = await backtester.run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        symbols=["bitcoin", "ethereum"]
    )
    
    print(f"📊 Multi-Symbol Results:")
    print(f"   Total Return: {results.total_return:.2%}")
    print(f"   Win Rate: {results.win_rate:.2%}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {results.max_drawdown:.2%}")
    print()

async def run_parameter_optimization():
    """Demonstrate parameter optimization."""
    print("🔧 Running Parameter Optimization")
    print("=" * 50)
    
    # Test different QRS thresholds
    qrs_scores = [5, 6, 7, 8]
    best_score = None
    best_return = -float('inf')
    
    for qrs_score in qrs_scores:
        config = {
            'backtesting': {
                'start_date': '2024-01-01',
                'end_date': '2024-03-31',
                'initial_capital': 10000,
                'position_size': 0.1,
                'commission': 0.001
            },
            'strategy': {
                'min_qrs_score': qrs_score,
                'volume_threshold': 1.5,
                'rejection_threshold': 0.3,
                'zone_proximity': 0.8
            }
        }
        
        backtester = BitcoinBacktester(config)
        results = await backtester.run_backtest(
            start_date="2024-01-01",
            end_date="2024-03-31",
            symbols=["bitcoin"]
        )
        
        print(f"   QRS Score {qrs_score}: Return {results.total_return:.2%}, Win Rate {results.win_rate:.2%}")
        
        if results.total_return > best_return:
            best_return = results.total_return
            best_score = qrs_score
    
    print(f"\n🏆 Best QRS Score: {best_score} (Return: {best_return:.2%})")
    print()

async def main():
    """Run all backtest examples."""
    print("🚀 Bitcoin Zone Fade Detector - Backtesting Examples")
    print("=" * 60)
    print()
    
    # Run different market condition tests
    await run_bull_market_backtest()
    await run_bear_market_backtest()
    await run_high_volatility_backtest()
    await run_multi_symbol_backtest()
    await run_parameter_optimization()
    
    print("✅ All backtest examples completed!")
    print("\n📚 Next Steps:")
    print("1. Review the results above")
    print("2. Modify parameters in config/bitcoin_backtest.yaml")
    print("3. Run your own custom backtests")
    print("4. Analyze performance metrics")
    print("5. Optimize for your risk tolerance")

if __name__ == "__main__":
    asyncio.run(main())