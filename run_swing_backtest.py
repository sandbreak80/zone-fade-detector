#!/usr/bin/env python3
"""
Simple Swing Trading Backtest Runner

This script runs the swing trading strategies using free yfinance data.
Perfect for 15-minute delayed data - no real-time feeds needed!
"""

import sys
import logging
from swing_trading_strategies import (
    SwingTradingBacktester,
    OversoldBounceStrategy,
    BreakoutContinuationStrategy,
    VolatilityExpansionStrategy,
    SectorRotationStrategy,
    EarningsMomentumStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run swing trading strategy backtests."""
    logger.info("🚀 Starting Swing Trading Strategy Backtest")
    logger.info("📊 Using FREE yfinance data - perfect for 15-minute delayed trading!")
    
    # Initialize backtester
    backtester = SwingTradingBacktester(initial_capital=10000)
    
    # Test parameters
    symbols = ['SPY', 'QQQ', 'IWM']  # Start with major ETFs
    start_date = '2023-01-01'  # 2 years of daily data
    end_date = '2024-12-31'
    
    # Initialize strategies
    strategies = [
        OversoldBounceStrategy(),
        BreakoutContinuationStrategy(),
        VolatilityExpansionStrategy(),
        SectorRotationStrategy(),
        EarningsMomentumStrategy()
    ]
    
    results = []
    
    logger.info(f"🧪 Testing {len(strategies)} strategies on {len(symbols)} symbols")
    logger.info(f"📅 Date range: {start_date} to {end_date}")
    
    # Run backtests
    for strategy in strategies:
        logger.info(f"\n📈 Testing {strategy.name}...")
        
        for symbol in symbols:
            try:
                logger.info(f"   Testing {symbol}...")
                result = backtester.backtest_strategy(strategy, symbol, start_date, end_date)
                
                if result and result.total_trades > 0:
                    results.append(result)
                    logger.info(f"   ✅ {symbol}: {result.total_return:.1%} return, "
                              f"{result.total_trades} trades, {result.win_rate:.1%} win rate")
                else:
                    logger.info(f"   ⚠️ {symbol}: No trades generated")
                    
            except Exception as e:
                logger.error(f"   ❌ Error testing {symbol}: {e}")
    
    # Generate summary
    if results:
        generate_summary(results)
    else:
        logger.warning("⚠️ No successful backtests completed")
    
    logger.info("\n🎉 Backtest Complete!")
    logger.info("💡 These strategies are designed for 15-minute delayed data")
    logger.info("📊 Perfect for swing trading with realistic expectations!")

def generate_summary(results):
    """Generate summary of results."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 SWING TRADING BACKTEST SUMMARY")
    logger.info("=" * 60)
    
    # Group by strategy
    strategy_groups = {}
    for result in results:
        if result.strategy_name not in strategy_groups:
            strategy_groups[result.strategy_name] = []
        strategy_groups[result.strategy_name].append(result)
    
    # Calculate averages
    for strategy_name, strategy_results in strategy_groups.items():
        avg_return = sum(r.total_return for r in strategy_results) / len(strategy_results)
        avg_sharpe = sum(r.sharpe_ratio for r in strategy_results) / len(strategy_results)
        avg_win_rate = sum(r.win_rate for r in strategy_results) / len(strategy_results)
        total_trades = sum(r.total_trades for r in strategy_results)
        avg_excess = sum(r.excess_return for r in strategy_results) / len(strategy_results)
        
        logger.info(f"\n📈 {strategy_name}:")
        logger.info(f"   Average Return: {avg_return:.1%}")
        logger.info(f"   Average Sharpe: {avg_sharpe:.2f}")
        logger.info(f"   Average Win Rate: {avg_win_rate:.1%}")
        logger.info(f"   Total Trades: {total_trades}")
        logger.info(f"   Average Excess Return: {avg_excess:.1%}")
        
        # Show individual results
        for result in strategy_results:
            logger.info(f"     {result.symbol}: {result.total_return:.1%} "
                       f"({result.total_trades} trades, {result.win_rate:.1%} win rate)")
    
    # Overall stats
    all_returns = [r.total_return for r in results]
    all_excess = [r.excess_return for r in results]
    
    logger.info(f"\n🎯 OVERALL PERFORMANCE:")
    logger.info(f"   Average Strategy Return: {sum(all_returns)/len(all_returns):.1%}")
    logger.info(f"   Average Excess Return: {sum(all_excess)/len(all_excess):.1%}")
    logger.info(f"   Total Tests: {len(results)}")
    
    # Best performers
    best_results = sorted(results, key=lambda x: x.total_return, reverse=True)[:3]
    logger.info(f"\n🏆 TOP PERFORMING TESTS:")
    for i, result in enumerate(best_results, 1):
        logger.info(f"   {i}. {result.strategy_name} on {result.symbol}: {result.total_return:.1%}")

if __name__ == "__main__":
    main()
