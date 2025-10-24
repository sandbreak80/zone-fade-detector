#!/usr/bin/env python3
"""
Run Phase 3 Validation with Real Market Data.

This script loads the real market data we just downloaded and runs
the complete 4-step validation battery with authentic market data.
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

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our framework components
from zone_fade_detector.strategies import get_strategy_class
from zone_fade_detector.validation.validation_orchestrator import ValidationOrchestrator
from zone_fade_detector.utils.returns_engine import BarReturnsEngine


class RealDataValidator:
    """
    Validator that uses real market data for testing.
    """
    
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
    
    def create_ohlcv_bars(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to OHLCV bar format."""
        bars = []
        for timestamp, row in df.iterrows():
            bars.append({
                'timestamp': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        return bars
    
    async def run_validation_for_instrument(self, 
                                          symbol: str, 
                                          strategy_instance,
                                          results_dir: str,
                                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation for a single instrument."""
        logger.info(f"🔄 Processing {symbol}...")
        
        try:
            # Load real market data
            df = self.load_instrument_data(symbol)
            bars = self.create_ohlcv_bars(df)
            
            if len(bars) < 1000:  # Need minimum data for validation
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(bars)} bars")
                return None
            
            # Initialize Validation Orchestrator
            orchestrator = ValidationOrchestrator(
                strategy=strategy_instance,
                bars=bars,
                initial_capital=config['initial_capital'],
                max_position_size_ratio=config['max_position_size_ratio'],
                commission_rate=config['commission_rate'],
                slippage_rate=config['slippage_rate'],
                train_window_size=config['train_window_size'],
                retrain_frequency=config['retrain_frequency'],
                imcpt_n_permutations=config['imcpt_n_permutations'],
                wfpt_m_permutations=config['wfpt_m_permutations'],
                random_seed=config['permutation_seed'],
                results_dir=results_dir,
                ticker=symbol
            )
            
            # Run validation battery
            logger.info(f"🧪 Running 4-step validation for {symbol}")
            is_metrics, oos_metrics, imcpt_p_value, wfpt_p_value, validation_passed = await orchestrator.run_validation_battery()
            
            result = {
                'symbol': symbol,
                'is_metrics': is_metrics,
                'oos_metrics': oos_metrics,
                'imcpt_p_value': imcpt_p_value,
                'wfpt_p_value': wfpt_p_value,
                'validation_passed': validation_passed,
                'data_points': len(bars),
                'date_range': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat()
                }
            }
            
            logger.info(f"✅ {symbol}: IS Sharpe={is_metrics.get('sharpe_ratio', 0):.3f}, "
                       f"OOS Score={oos_metrics.get('total_oos_score', 0):.3f}, "
                       f"IMCPT p={imcpt_p_value:.3f}, WFPT p={wfpt_p_value:.3f}, "
                       f"Passed={'✅' if validation_passed else '❌'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            return None


async def run_real_data_validation():
    """Run complete validation with real market data."""
    
    # Configuration
    config = {
        'initial_capital': 10000,
        'max_position_size_ratio': 0.20,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005,
        'train_window_size': 2000,
        'retrain_frequency': 100,
        'imcpt_n_permutations': 1000,
        'wfpt_m_permutations': 200,
        'permutation_seed': 4242
    }
    
    # Instruments to test
    instruments = ["QQQ", "SPY", "LLY", "AVGO", "AAPL", "CRM", "ORCL"]
    
    # Create results directory
    run_id = f"real_data_validation_{int(time.time())}"
    results_dir = f"results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("🚀 Starting Real Data Validation")
    logger.info("=" * 60)
    logger.info(f"📊 Instruments: {', '.join(instruments)}")
    logger.info(f"📁 Results directory: {results_dir}")
    logger.info(f"💰 Initial capital: ${config['initial_capital']:,}")
    logger.info(f"📈 Max position size: {config['max_position_size_ratio']*100:.1f}%")
    
    # Initialize validator
    validator = RealDataValidator()
    
    # Initialize strategy
    strategy_name = "MACD Crossover Strategy"
    StrategyClass = get_strategy_class(strategy_name)
    strategy_instance = StrategyClass()
    logger.info(f"🎯 Strategy: {strategy_instance.get_name()}")
    
    # Run validation for each instrument
    all_results = []
    successful_validations = 0
    
    start_time = time.time()
    
    for symbol in instruments:
        result = await validator.run_validation_for_instrument(
            symbol=symbol,
            strategy_instance=strategy_instance,
            results_dir=results_dir,
            config=config
        )
        
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
        'strategy': strategy_name,
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
            summary_data.append({
                'symbol': result['symbol'],
                'is_sharpe': result['is_metrics'].get('sharpe_ratio', 0),
                'is_return': result['is_metrics'].get('total_return', 0),
                'oos_score': result['oos_metrics'].get('total_oos_score', 0),
                'imcpt_p_value': result['imcpt_p_value'],
                'wfpt_p_value': result['wfpt_p_value'],
                'validation_passed': result['validation_passed'],
                'data_points': result['data_points']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{results_dir}/validation_summary.csv", index=False)
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 REAL DATA VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Successful validations: {successful_validations}/{len(instruments)}")
    logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
    logger.info(f"📁 Results saved to: {results_dir}")
    
    if all_results:
        logger.info("\n📈 Results Summary:")
        for result in all_results:
            status = "✅ PASSED" if result['validation_passed'] else "❌ FAILED"
            logger.info(f"  {result['symbol']}: {status} "
                       f"(IS Sharpe: {result['is_metrics'].get('sharpe_ratio', 0):.3f}, "
                       f"IMCPT p: {result['imcpt_p_value']:.3f}, "
                       f"WFPT p: {result['wfpt_p_value']:.3f})")
    
    logger.info(f"\n🎯 Real data validation complete! Check {results_dir} for detailed results.")


if __name__ == "__main__":
    asyncio.run(run_real_data_validation())
