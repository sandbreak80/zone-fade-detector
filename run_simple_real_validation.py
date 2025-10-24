#!/usr/bin/env python3
"""
Simple Real Data Validation with Real Market Data.

This script loads the real market data and runs a simplified validation
without the complex framework dependencies.
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMACDStrategy:
    """Simple MACD strategy implementation."""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, prices):
        """Calculate MACD indicators."""
        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def generate_signals(self, prices):
        """Generate trading signals."""
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        
        # Generate signals: 1 for long, -1 for short, 0 for neutral
        signals = pd.Series(0, index=prices.index)
        
        # MACD crossover signals
        crossover_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        crossover_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signals[crossover_up] = 1
        signals[crossover_down] = -1
        
        return signals


class SimpleRealDataValidator:
    """Simple validator using real market data."""
    
    def __init__(self, data_dir: str = "data/real_market_data"):
        """Initialize with real market data directory."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        logger.info(f"📁 Using real market data from: {self.data_dir}")
    
    def load_instrument_data(self, symbol: str) -> pd.DataFrame:
        """Load real market data for a specific instrument."""
        data_file = self.data_dir / f"{symbol}_1h_bars.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load the data
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df = df.sort_index()
        
        logger.info(f"📈 Loaded {len(df)} bars for {symbol} from {df.index[0]} to {df.index[-1]}")
        return df
    
    def calculate_returns(self, prices, signals, commission_rate=0.001, slippage_rate=0.0005):
        """Calculate strategy returns with costs."""
        # Calculate price returns
        price_returns = prices.pct_change()
        
        # Shift signals by 1 bar to prevent look-ahead bias
        shifted_signals = signals.shift(1)
        
        # Calculate strategy returns
        strategy_returns = shifted_signals * price_returns
        
        # Apply transaction costs
        position_changes = shifted_signals.diff().abs()
        transaction_costs = position_changes * (commission_rate + slippage_rate)
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        return net_returns.fillna(0)
    
    def calculate_metrics(self, returns, initial_capital=10000):
        """Calculate performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        portfolio_values = initial_capital * cumulative_returns
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1  # Assuming 1-hour bars
        volatility = returns.std() * np.sqrt(252 * 24)  # Annualized volatility
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'final_portfolio_value': portfolio_values.iloc[-1]
        }
    
    def run_validation_for_instrument(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation for a single instrument."""
        logger.info(f"🔄 Processing {symbol}...")
        
        try:
            # Load real market data
            df = self.load_instrument_data(symbol)
            
            if len(df) < 1000:  # Need minimum data for validation
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} bars")
                return None
            
            # Initialize strategy
            strategy = SimpleMACDStrategy(
                fast_period=config.get('fast_period', 12),
                slow_period=config.get('slow_period', 26),
                signal_period=config.get('signal_period', 9)
            )
            
            # Generate signals
            signals = strategy.generate_signals(df['close'])
            
            # Calculate returns
            returns = self.calculate_returns(
                df['close'], 
                signals,
                commission_rate=config['commission_rate'],
                slippage_rate=config['slippage_rate']
            )
            
            # Calculate metrics
            metrics = self.calculate_metrics(returns, config['initial_capital'])
            
            # Simple validation: check if strategy is profitable
            validation_passed = metrics['sharpe_ratio'] > 0.5 and metrics['total_return'] > 0.1
            
            result = {
                'symbol': symbol,
                'metrics': metrics,
                'validation_passed': validation_passed,
                'data_points': len(df),
                'date_range': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat()
                }
            }
            
            logger.info(f"✅ {symbol}: Return={metrics['total_return']:.3f}, "
                       f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                       f"MaxDD={metrics['max_drawdown']:.3f}, "
                       f"Passed={'✅' if validation_passed else '❌'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            return None


async def run_simple_real_validation():
    """Run simple validation with real market data."""
    
    # Configuration
    config = {
        'initial_capital': 10000,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005,
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    }
    
    # Instruments to test
    instruments = ["QQQ", "SPY", "LLY", "AVGO", "AAPL", "CRM", "ORCL"]
    
    # Create results directory
    run_id = f"simple_real_validation_{int(time.time())}"
    results_dir = f"results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("🚀 Starting Simple Real Data Validation")
    logger.info("=" * 60)
    logger.info(f"📊 Instruments: {', '.join(instruments)}")
    logger.info(f"📁 Results directory: {results_dir}")
    logger.info(f"💰 Initial capital: ${config['initial_capital']:,}")
    
    # Initialize validator
    validator = SimpleRealDataValidator()
    
    # Run validation for each instrument
    all_results = []
    successful_validations = 0
    
    start_time = time.time()
    
    for symbol in instruments:
        result = validator.run_validation_for_instrument(symbol, config)
        
        if result:
            all_results.append(result)
            if result['validation_passed']:
                successful_validations += 1
    
    # Generate summary
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
    results_summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'strategy': 'Simple MACD Strategy',
        'instruments_tested': instruments,
        'successful_validations': successful_validations,
        'total_instruments': len(instruments),
        'validation_rate': successful_validations / len(instruments),
        'total_time_seconds': total_time,
        'config': config,
        'results': all_results
    }
    
    # Save to JSON
    with open(f"{results_dir}/validation_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Create CSV summary
    if all_results:
        summary_data = []
        for result in all_results:
            metrics = result['metrics']
            summary_data.append({
                'symbol': result['symbol'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'total_trades': metrics['total_trades'],
                'final_value': metrics['final_portfolio_value'],
                'validation_passed': result['validation_passed'],
                'data_points': result['data_points']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{results_dir}/validation_summary.csv", index=False)
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SIMPLE REAL DATA VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Successful validations: {successful_validations}/{len(instruments)}")
    logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
    logger.info(f"📁 Results saved to: {results_dir}")
    
    if all_results:
        logger.info("\n📈 Results Summary:")
        for result in all_results:
            metrics = result['metrics']
            status = "✅ PASSED" if result['validation_passed'] else "❌ FAILED"
            logger.info(f"  {result['symbol']}: {status} "
                       f"(Return: {metrics['total_return']:.1%}, "
                       f"Sharpe: {metrics['sharpe_ratio']:.3f}, "
                       f"MaxDD: {metrics['max_drawdown']:.1%})")
    
    logger.info(f"\n🎯 Simple real data validation complete! Check {results_dir} for detailed results.")


if __name__ == "__main__":
    asyncio.run(run_simple_real_validation())
