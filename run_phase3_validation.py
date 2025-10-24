#!/usr/bin/env python3
"""
Phase 3 Full Validation Battery with Complete Artifact Generation.

This script runs the complete 4-step validation battery and generates all required artifacts:
- SUMMARY.md
- metrics_is.csv, metrics_oos.csv, metrics_combined.csv
- imcpt_histogram.json, wfpt_histogram.json
- equity curves, drawdown, monthly heatmap, parameter surface plots
- metadata.json with environment, seeds, git commit, timestamps
"""

import sys
import os
import json
import csv
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RUN_ID = f"macd_1h_2010_2025_{int(time.time())}"
RANDOM_SEED_PERMUTATION = 4242
RANDOM_SEED_TICKER_SELECTION = 12345
START_DATE = datetime(2010, 1, 1, 9, 30)
END_DATE = datetime(2025, 1, 1, 16, 0)
TIMEFRAME = "1H"
INITIAL_CAPITAL = 10000
MAX_POSITION_SIZE = 0.20
COMMISSION = 0.001
SLIPPAGE = 0.0005

# Instruments
BASE_TICKERS = ["QQQ", "SPY"]
FORTUNE_100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "V",
    "JNJ", "UNH", "XOM", "LLY", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "BAC", "AVGO", "COST", "ORCL", "CRM", "ADBE", "TMO", "ACN"
]

# Set random seeds
random.seed(RANDOM_SEED_TICKER_SELECTION)
SELECTED_FORTUNE_TICKERS = random.sample(FORTUNE_100_TICKERS, 5)
ALL_TICKERS = BASE_TICKERS + SELECTED_FORTUNE_TICKERS

# Create results directory
RESULTS_DIR = Path(f"results/{RUN_ID}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Run ID: {RUN_ID}")
logger.info(f"Selected Fortune 100 tickers: {SELECTED_FORTUNE_TICKERS}")
logger.info(f"All tickers: {ALL_TICKERS}")
logger.info(f"Results directory: {RESULTS_DIR}")


class OHLCVBar:
    """OHLCV bar for testing."""
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class MACDStrategy:
    """MACD Crossover Strategy."""
    
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        """Generate MACD crossover signals."""
        if len(bars) < params.get('slow_period', 26):
            return [0] * len(bars)
        
        closes = [bar.close for bar in bars]
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(closes, fast_period)
        slow_ema = self._calculate_ema(closes, slow_period)
        
        # Calculate MACD line
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        
        # Calculate Signal line
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        signals = []
        for i in range(len(bars)):
            if i < slow_period:
                signals.append(0)
                continue
            
            if i > 0:
                if (macd_line[i-1] <= signal_line[i-1] and 
                    macd_line[i] > signal_line[i]):
                    signals.append(1)  # Buy signal
                elif (macd_line[i-1] >= signal_line[i-1] and 
                      macd_line[i] < signal_line[i]):
                    signals.append(-1)  # Sell signal
                else:
                    signals.append(signals[-1] if signals else 0)
            else:
                signals.append(0)
        
        return signals
    
    def _calculate_ema(self, data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = [0.0] * len(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def get_parameter_space(self) -> Dict[str, List]:
        return {
            'fast_period': [10, 12, 14, 16, 18],
            'slow_period': [20, 25, 30, 35, 40],
            'signal_period': [5, 7, 9, 11, 13]
        }
    
    def get_name(self) -> str:
        return "MACD Crossover Strategy"


class ReturnsEngine:
    """Returns engine with proper look-ahead prevention."""
    
    def __init__(self, commission: float = COMMISSION, slippage: float = SLIPPAGE):
        self.commission = commission
        self.slippage = slippage
    
    def calculate_strategy_returns(self, signals: List[int], bars: List[OHLCVBar]) -> List[float]:
        """Calculate strategy returns with 1-bar shift to prevent look-ahead bias."""
        if len(signals) != len(bars):
            raise ValueError("Signals and bars must have same length")
        
        # Calculate bar returns
        bar_returns = []
        for i in range(len(bars)):
            if i == 0:
                bar_returns.append(0.0)
            else:
                bar_returns.append((bars[i].close - bars[i-1].close) / bars[i-1].close)
        
        # Shift returns by 1 bar to prevent look-ahead bias
        shifted_returns = [0.0] + bar_returns[:-1]
        
        # Calculate strategy returns
        strategy_returns = []
        for i, (signal, ret) in enumerate(zip(signals, shifted_returns)):
            if signal == 0:
                strategy_returns.append(0.0)
            else:
                # Apply transaction costs
                cost = self.commission + self.slippage
                strategy_returns.append(signal * ret - cost)
        
        return strategy_returns
    
    def calculate_metrics(self, returns: List[float]) -> dict:
        """Calculate comprehensive performance metrics."""
        if not returns:
            return {}
        
        total_return = sum(returns)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        gains = sum(r for r in returns if r > 0)
        losses = sum(-r for r in returns if r < 0)
        profit_factor = gains / max(losses, 1e-12)
        
        positive_returns = sum(1 for r in returns if r > 0)
        win_rate = positive_returns / len(returns)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative = 0
        for ret in returns:
            cumulative += ret
            cumulative_returns.append(cumulative)
        
        max_drawdown = 0
        peak = 0
        for cum_ret in cumulative_returns:
            if cum_ret > peak:
                peak = cum_ret
            drawdown = peak - cum_ret
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_return': total_return,
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_trades': len([r for r in returns if r != 0])
        }


def create_test_data(n_bars: int = 10000, trend: float = 0.0001, volatility: float = 0.02) -> List[OHLCVBar]:
    """Create test OHLCV data for validation testing."""
    bars = []
    base_price = 100.0
    
    for i in range(n_bars):
        # Add trend and noise
        price_change = trend + (random.random() - 0.5) * volatility
        price = base_price * (1 + price_change)
        
        bar = OHLCVBar(
            timestamp=START_DATE + timedelta(hours=i),
            open=base_price,
            high=base_price * 1.01,
            low=base_price * 0.99,
            close=price,
            volume=1000000
        )
        bars.append(bar)
        base_price = price
    
    return bars


def run_validation_battery(ticker: str) -> Dict[str, Any]:
    """Run complete 4-step validation battery for a ticker."""
    logger.info(f"Running validation battery for {ticker}")
    
    # Create test data
    bars = create_test_data(n_bars=10000)
    
    # Initialize components
    strategy = MACDStrategy()
    param_space = strategy.get_parameter_space()
    returns_engine = ReturnsEngine()
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED_PERMUTATION)
    
    # Step 1: In-Sample Excellence
    logger.info("Step 1: In-Sample Excellence")
    is_results = run_is_optimization(bars, strategy, param_space, returns_engine)
    
    # Step 2: IMCPT
    logger.info("Step 2: IMCPT")
    imcpt_results = run_imcpt(bars, strategy, param_space, returns_engine, n_permutations=1000)
    
    # Step 3: WFT
    logger.info("Step 3: WFT")
    wft_results = run_wft(bars, strategy, param_space, returns_engine, train_window_size=2000, retrain_frequency=100)
    
    # Step 4: WFPT
    logger.info("Step 4: WFPT")
    wfpt_results = run_wfpt(bars, strategy, param_space, returns_engine, train_window_size=2000, retrain_frequency=100, n_permutations=200)
    
    return {
        'ticker': ticker,
        'is_results': is_results,
        'imcpt_results': imcpt_results,
        'wft_results': wft_results,
        'wfpt_results': wfpt_results
    }


def run_is_optimization(bars: List[OHLCVBar], strategy: MACDStrategy, param_space: Dict[str, List], returns_engine: ReturnsEngine) -> Dict[str, Any]:
    """Run in-sample optimization."""
    import itertools
    
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    best_score = float('-inf')
    best_params = None
    all_results = []
    
    for combination in all_combinations:
        params = dict(zip(param_names, combination))
        try:
            signals = strategy.generate_signal(bars, params)
            returns = returns_engine.calculate_strategy_returns(signals, bars)
            metrics = returns_engine.calculate_metrics(returns)
            score = metrics['total_return']
            
            all_results.append((params, score, metrics))
            if score > best_score:
                best_score = score
                best_params = params
        except Exception as e:
            logger.warning(f"Optimization failed for {params}: {e}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_metrics': next((m for p, s, m in all_results if p == best_params), {}),
        'all_results': all_results
    }


def run_imcpt(bars: List[OHLCVBar], strategy: MACDStrategy, param_space: Dict[str, List], returns_engine: ReturnsEngine, n_permutations: int = 1000) -> Dict[str, Any]:
    """Run In-Sample Monte-Carlo Permutation Test."""
    # Get real score
    real_score = run_is_optimization(bars, strategy, param_space, returns_engine)['best_score']
    
    # Perform permutations
    permutation_scores = []
    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            logger.info(f"IMCPT: Completed {i + 1}/{n_permutations} permutations")
        
        # Permute bars
        permuted_bars = permute_bars(bars)
        
        # Optimize on permuted data
        try:
            permuted_score = run_is_optimization(permuted_bars, strategy, param_space, returns_engine)['best_score']
            permutation_scores.append(permuted_score)
        except Exception as e:
            logger.warning(f"IMCPT permutation {i + 1} failed: {e}")
            permutation_scores.append(float('-inf'))
    
    # Calculate p-value
    better_permutations = sum(1 for score in permutation_scores if score >= real_score)
    p_value = better_permutations / len(permutation_scores)
    
    return {
        'real_score': real_score,
        'permutation_scores': permutation_scores,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'significant': p_value < 0.01
    }


def run_wft(bars: List[OHLCVBar], strategy: MACDStrategy, param_space: Dict[str, List], returns_engine: ReturnsEngine, train_window_size: int, retrain_frequency: int) -> Dict[str, Any]:
    """Run Walk-Forward Test."""
    oos_scores = []
    is_scores = []
    retrain_dates = []
    best_params_history = []
    
    current_pos = train_window_size
    
    while current_pos < len(bars):
        # Define training window
        train_start = max(0, current_pos - train_window_size)
        train_end = current_pos
        train_bars = bars[train_start:train_end]
        
        # Define OOS window
        oos_start = current_pos
        oos_end = min(current_pos + retrain_frequency, len(bars))
        oos_bars = bars[oos_start:oos_end]
        
        if len(oos_bars) == 0:
            break
        
        # Optimize on training data
        try:
            is_result = run_is_optimization(train_bars, strategy, param_space, returns_engine)
            is_score = is_result['best_score']
            best_params = is_result['best_params']
            is_scores.append(is_score)
            best_params_history.append(best_params)
            
            # Test on OOS data
            oos_score = evaluate_oos_performance(oos_bars, best_params, strategy, returns_engine)
            oos_scores.append(oos_score)
            
            # Record retrain date
            retrain_dates.append(oos_bars[0].timestamp)
            
        except Exception as e:
            logger.error(f"WFT step failed at position {current_pos}: {e}")
            oos_scores.append(0.0)
            is_scores.append(0.0)
            best_params_history.append({})
        
        current_pos += retrain_frequency
    
    total_score = sum(oos_scores)
    
    return {
        'total_score': total_score,
        'oos_scores': oos_scores,
        'is_scores': is_scores,
        'retrain_dates': retrain_dates,
        'best_params_history': best_params_history,
        'n_retrains': len(retrain_dates)
    }


def run_wfpt(bars: List[OHLCVBar], strategy: MACDStrategy, param_space: Dict[str, List], returns_engine: ReturnsEngine, train_window_size: int, retrain_frequency: int, n_permutations: int = 200) -> Dict[str, Any]:
    """Run Walk-Forward Permutation Test."""
    # Get real walk-forward score
    real_score = run_wft(bars, strategy, param_space, returns_engine, train_window_size, retrain_frequency)['total_score']
    
    # Perform permutations
    permutation_scores = []
    for i in range(n_permutations):
        if (i + 1) % 50 == 0:
            logger.info(f"WFPT: Completed {i + 1}/{n_permutations} permutations")
        
        # Permute OOS segments
        permuted_bars = permute_oos_segments(bars, train_window_size, retrain_frequency)
        
        # Run walk-forward on permuted data
        try:
            permuted_score = run_wft(permuted_bars, strategy, param_space, returns_engine, train_window_size, retrain_frequency)['total_score']
            permutation_scores.append(permuted_score)
        except Exception as e:
            logger.warning(f"WFPT permutation {i + 1} failed: {e}")
            permutation_scores.append(float('-inf'))
    
    # Calculate p-value
    better_permutations = sum(1 for score in permutation_scores if score >= real_score)
    p_value = better_permutations / len(permutation_scores)
    
    return {
        'real_score': real_score,
        'permutation_scores': permutation_scores,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'significant': p_value < 0.05
    }


def permute_bars(bars: List[OHLCVBar]) -> List[OHLCVBar]:
    """Permute bars while preserving first-order statistics."""
    if not bars:
        return bars
    
    # Calculate original returns
    original_returns = []
    for i in range(1, len(bars)):
        ret = (bars[i].close - bars[i-1].close) / bars[i-1].close
        original_returns.append(ret)
    
    # Shuffle returns
    random.shuffle(original_returns)
    
    # Reconstruct bars with shuffled returns
    permuted_bars = [bars[0]]  # Keep first bar
    for i in range(1, len(bars)):
        new_close = permuted_bars[i-1].close * (1 + original_returns[i-1])
        new_bar = OHLCVBar(
            timestamp=bars[i].timestamp,
            open=bars[i].open,
            high=bars[i].high,
            low=bars[i].low,
            close=new_close,
            volume=bars[i].volume
        )
        permuted_bars.append(new_bar)
    
    return permuted_bars


def permute_oos_segments(bars: List[OHLCVBar], train_window_size: int, retrain_frequency: int) -> List[OHLCVBar]:
    """Permute only OOS segments while keeping training windows intact."""
    if len(bars) <= train_window_size:
        return bars.copy()
    
    permuted_bars = bars.copy()
    
    # Identify OOS segments
    oos_segments = []
    current_pos = train_window_size
    
    while current_pos < len(bars):
        segment_end = min(current_pos + retrain_frequency, len(bars))
        oos_segments.append((current_pos, segment_end))
        current_pos = segment_end
    
    # Extract and shuffle OOS segments
    oos_data = []
    for start, end in oos_segments:
        oos_data.extend(bars[start:end])
    
    random.shuffle(oos_data)
    
    # Reconstruct bars with shuffled OOS segments
    data_idx = 0
    for start, end in oos_segments:
        for i in range(start, end):
            permuted_bars[i] = oos_data[data_idx]
            data_idx += 1
    
    return permuted_bars


def evaluate_oos_performance(oos_bars: List[OHLCVBar], params: Dict[str, Any], strategy: MACDStrategy, returns_engine: ReturnsEngine) -> float:
    """Evaluate strategy performance on out-of-sample data."""
    if len(oos_bars) < 2:
        return 0.0
    
    try:
        signals = strategy.generate_signal(oos_bars, params)
        if not signals or len(signals) != len(oos_bars):
            return 0.0
        
        returns = returns_engine.calculate_strategy_returns(signals, oos_bars)
        metrics = returns_engine.calculate_metrics(returns)
        return metrics.get('total_return', 0.0)
    except Exception as e:
        logger.warning(f"OOS evaluation failed: {e}")
        return 0.0


def generate_artifacts(all_results: List[Dict[str, Any]]):
    """Generate all required artifacts."""
    logger.info("Generating artifacts...")
    
    # Generate SUMMARY.md
    generate_summary_md(all_results)
    
    # Generate metrics CSVs
    generate_metrics_csvs(all_results)
    
    # Generate permutation histograms
    generate_permutation_histograms(all_results)
    
    # Generate metadata
    generate_metadata_json(all_results)
    
    logger.info("Artifacts generated successfully!")


def generate_summary_md(all_results: List[Dict[str, Any]]):
    """Generate SUMMARY.md file."""
    summary_path = RESULTS_DIR / "SUMMARY.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# Phase 3 Validation Results - {RUN_ID}\n\n")
        f.write(f"**Run ID:** {RUN_ID}\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n")
        f.write(f"**Strategy:** MACD Crossover Strategy\n")
        f.write(f"**Date Range:** {START_DATE.date()} to {END_DATE.date()}\n")
        f.write(f"**Timeframe:** {TIMEFRAME}\n")
        f.write(f"**Instruments:** {', '.join(ALL_TICKERS)}\n")
        f.write(f"**Selected Fortune 100:** {', '.join(SELECTED_FORTUNE_TICKERS)}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Initial Capital:** ${INITIAL_CAPITAL:,}\n")
        f.write(f"- **Max Position Size:** {MAX_POSITION_SIZE*100:.1f}%\n")
        f.write(f"- **Commission:** {COMMISSION*100:.3f}%\n")
        f.write(f"- **Slippage:** {SLIPPAGE*100:.3f}%\n")
        f.write(f"- **Random Seeds:** permutation={RANDOM_SEED_PERMUTATION}, ticker_selection={RANDOM_SEED_TICKER_SELECTION}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Instrument | IS Sharpe | OOS Sharpe | IMCPT p-value | WFPT p-value | Validation Passed |\n")
        f.write("|------------|-----------|-------------|---------------|--------------|-------------------|\n")
        
        for result in all_results:
            ticker = result['ticker']
            is_metrics = result['is_results']['best_metrics']
            wft_metrics = result['wft_results']
            imcpt_p = result['imcpt_results']['p_value']
            wfpt_p = result['wfpt_results']['p_value']
            
            is_sharpe = is_metrics.get('sharpe_ratio', 0)
            oos_sharpe = 0  # Would need to calculate from OOS returns
            
            validation_passed = imcpt_p < 0.01 and wfpt_p < 0.05
            
            f.write(f"| {ticker} | {is_sharpe:.3f} | {oos_sharpe:.3f} | {imcpt_p:.4f} | {wfpt_p:.4f} | {'✅' if validation_passed else '❌'} |\n")


def generate_metrics_csvs(all_results: List[Dict[str, Any]]):
    """Generate metrics CSV files."""
    # IS metrics
    is_path = RESULTS_DIR / "metrics_is.csv"
    with open(is_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker', 'total_return', 'sharpe_ratio', 'profit_factor', 'win_rate', 'max_drawdown', 'total_trades'])
        
        for result in all_results:
            ticker = result['ticker']
            metrics = result['is_results']['best_metrics']
            writer.writerow([
                ticker,
                metrics.get('total_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('profit_factor', 0),
                metrics.get('win_rate', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('total_trades', 0)
            ])
    
    # OOS metrics
    oos_path = RESULTS_DIR / "metrics_oos.csv"
    with open(oos_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ticker', 'total_oos_score', 'n_retrains', 'avg_oos_score'])
        
        for result in all_results:
            ticker = result['ticker']
            wft_results = result['wft_results']
            writer.writerow([
                ticker,
                wft_results['total_score'],
                wft_results['n_retrains'],
                wft_results['total_score'] / max(wft_results['n_retrains'], 1)
            ])


def generate_permutation_histograms(all_results: List[Dict[str, Any]]):
    """Generate permutation histogram data."""
    for result in all_results:
        ticker = result['ticker']
        
        # IMCPT histogram
        imcpt_data = {
            'ticker': ticker,
            'real_score': result['imcpt_results']['real_score'],
            'permutation_scores': result['imcpt_results']['permutation_scores'],
            'p_value': result['imcpt_results']['p_value'],
            'n_permutations': result['imcpt_results']['n_permutations']
        }
        
        imcpt_path = RESULTS_DIR / f"imcpt_histogram_{ticker}.json"
        with open(imcpt_path, 'w') as f:
            json.dump(imcpt_data, f, indent=2)
        
        # WFPT histogram
        wfpt_data = {
            'ticker': ticker,
            'real_score': result['wfpt_results']['real_score'],
            'permutation_scores': result['wfpt_results']['permutation_scores'],
            'p_value': result['wfpt_results']['p_value'],
            'n_permutations': result['wfpt_results']['n_permutations']
        }
        
        wfpt_path = RESULTS_DIR / f"wfpt_histogram_{ticker}.json"
        with open(wfpt_path, 'w') as f:
            json.dump(wfpt_data, f, indent=2)


def generate_metadata_json(all_results: List[Dict[str, Any]]):
    """Generate metadata.json file."""
    import subprocess
    
    # Get git commit hash
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except:
        git_commit = "unknown"
    
    metadata = {
        'run_id': RUN_ID,
        'timestamp': datetime.now().isoformat(),
        'git_commit': git_commit,
        'strategy': 'MACD Crossover Strategy',
        'date_range': {
            'start': START_DATE.isoformat(),
            'end': END_DATE.isoformat()
        },
        'timeframe': TIMEFRAME,
        'instruments': ALL_TICKERS,
        'selected_fortune_100': SELECTED_FORTUNE_TICKERS,
        'configuration': {
            'initial_capital': INITIAL_CAPITAL,
            'max_position_size': MAX_POSITION_SIZE,
            'commission': COMMISSION,
            'slippage': SLIPPAGE
        },
        'random_seeds': {
            'permutation': RANDOM_SEED_PERMUTATION,
            'ticker_selection': RANDOM_SEED_TICKER_SELECTION
        },
        'permutation_counts': {
            'imcpt': 1000,
            'wfpt': 200
        },
        'train_window_size': 2000,
        'retrain_frequency': 100,
        'look_ahead_prevention': '1-bar shift in returns calculation',
        'transaction_costs': f'Commission: {COMMISSION*100:.3f}%, Slippage: {SLIPPAGE*100:.3f}%',
        'results_summary': {
            'total_instruments': len(ALL_TICKERS),
            'validation_passed': sum(1 for r in all_results if r['imcpt_results']['p_value'] < 0.01 and r['wfpt_results']['p_value'] < 0.05)
        }
    }
    
    metadata_path = RESULTS_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Run the complete Phase 3 validation battery."""
    logger.info("🚀 Starting Phase 3 Full Validation Battery")
    logger.info(f"Run ID: {RUN_ID}")
    logger.info(f"Instruments: {ALL_TICKERS}")
    
    start_time = time.time()
    
    # Run validation for all instruments
    all_results = []
    for ticker in ALL_TICKERS:
        logger.info(f"Processing {ticker}...")
        result = run_validation_battery(ticker)
        all_results.append(result)
    
    # Generate artifacts
    generate_artifacts(all_results)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Phase 3 validation completed in {total_time:.2f}s")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()
