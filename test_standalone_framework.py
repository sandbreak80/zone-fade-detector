#!/usr/bin/env python3
"""
Standalone Framework Test for Trading Strategy Testing Framework.

This script tests the core framework components without dependencies
on the existing Zone Fade Detector codebase.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Standalone implementations for testing
class OHLCVBar:
    """Standalone OHLCV bar for testing."""
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class BaseStrategy(ABC):
    """Standalone BaseStrategy for testing."""
    
    @abstractmethod
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class MACDStrategy(BaseStrategy):
    """Standalone MACD strategy for testing."""
    
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        if len(bars) < params['slow_period']:
            return [0] * len(bars)
        
        closes = [bar.close for bar in bars]
        macd_line, signal_line = self._calculate_macd(closes, params)
        
        signals = []
        for i in range(len(bars)):
            if i < params['slow_period']:
                signals.append(0)
                continue
            
            if i > 0:
                if (macd_line[i-1] <= signal_line[i-1] and 
                    macd_line[i] > signal_line[i]):
                    signals.append(1)
                elif (macd_line[i-1] >= signal_line[i-1] and 
                      macd_line[i] < signal_line[i]):
                    signals.append(-1)
                else:
                    signals.append(signals[-1] if signals else 0)
            else:
                signals.append(0)
        
        return signals
    
    def _calculate_macd(self, closes: List[float], params: Dict[str, Any]) -> tuple:
        fast_ema = self._calculate_ema(closes, params['fast_period'])
        slow_ema = self._calculate_ema(closes, params['slow_period'])
        macd_line = fast_ema - slow_ema
        signal_line = self._calculate_ema(macd_line, params['signal_period'])
        return macd_line, signal_line
    
    def _calculate_ema(self, data: List[float], period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def get_parameter_space(self) -> Dict[str, List]:
        return {
            'fast_period': [10, 12, 14, 16, 18, 20],
            'slow_period': [20, 25, 30, 35, 40],
            'signal_period': [5, 7, 9, 11, 13]
        }
    
    def get_name(self) -> str:
        return "MACD Crossover Strategy"


class ReturnsEngine:
    """Standalone returns engine for testing."""
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        self.commission = commission
        self.slippage = slippage
    
    def calculate_strategy_returns(self, signals: List[int], 
                                  bars: List[OHLCVBar]) -> List[float]:
        if len(signals) != len(bars):
            raise ValueError("Signals and bars must have same length")
        
        bar_returns = []
        for i in range(len(bars)):
            if i == 0:
                bar_returns.append(0.0)
            else:
                bar_returns.append((bars[i].close - bars[i-1].close) / bars[i-1].close)
        
        shifted_returns = [0.0] + bar_returns[:-1]
        
        strategy_returns = []
        for i, (signal, ret) in enumerate(zip(signals, shifted_returns)):
            if signal == 0:
                strategy_returns.append(0.0)
            else:
                cost = self.commission + self.slippage
                strategy_returns.append(signal * ret - cost)
        
        return strategy_returns
    
    def calculate_metrics(self, returns: List[float]) -> dict:
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
        
        return {
            'total_return': total_return,
            'mean_return': mean_return,
            'volatility': volatility,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_trades': len([r for r in returns if r != 0])
        }


class Fortune100Client:
    """Standalone Fortune 100 client for testing."""
    
    def __init__(self, year: int = 2024):
        self.year = year
        self.fortune_100_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ORCL', 'CRM', 'ADBE', 'INTC',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CB',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'PSX', 'VLO', 'MPC', 'OXY',
            'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
            'VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'DISH', 'SIRI', 'LUMN',
            'TSLA', 'F', 'GM', 'TM', 'HMC', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID',
            'BRK.B', 'V', 'MA', 'PYPL', 'SQ', 'ROKU', 'ZM', 'PTON', 'SNOW', 'PLTR'
        ]
    
    def get_fortune_100_tickers(self) -> List[str]:
        return self.fortune_100_tickers
    
    def select_random_tickers(self, n: int = 5, seed: int = None) -> List[str]:
        if seed is not None:
            random.seed(seed)
        
        return random.sample(self.fortune_100_tickers, min(n, len(self.fortune_100_tickers)))


def create_test_data() -> List[OHLCVBar]:
    """Create test OHLCV data for framework testing."""
    bars = []
    base_price = 100.0
    
    for i in range(1000):
        price_change = (i % 2 - 0.5) * 0.01
        price = base_price + price_change * i
        
        bar = OHLCVBar(
            timestamp=datetime.now() + timedelta(minutes=i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1000000
        )
        bars.append(bar)
        base_price = price
    
    return bars


def test_strategy_interface():
    """Test the strategy interface and MACD strategy."""
    logger.info("Testing strategy interface...")
    
    try:
        macd_strategy = MACDStrategy()
        logger.info(f"MACD strategy name: {macd_strategy.get_name()}")
        
        param_space = macd_strategy.get_parameter_space()
        logger.info(f"MACD parameter space: {param_space}")
        
        valid_params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        signals = macd_strategy.generate_signal(create_test_data(), valid_params)
        
        assert len(signals) > 0, "Should generate signals"
        assert all(s in [-1, 0, 1] for s in signals), "All signals should be -1, 0, or 1"
        
        logger.info("✅ Strategy interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy interface test failed: {e}")
        return False


def test_returns_engine():
    """Test the returns engine."""
    logger.info("Testing returns engine...")
    
    try:
        bars = create_test_data()
        signals = [1 if i % 10 == 0 else 0 for i in range(len(bars))]
        
        returns_engine = ReturnsEngine(commission=0.001, slippage=0.0005)
        strategy_returns = returns_engine.calculate_strategy_returns(signals, bars)
        
        assert len(strategy_returns) == len(bars), "Returns length should match bars length"
        
        metrics = returns_engine.calculate_metrics(strategy_returns)
        assert 'total_return' in metrics, "Metrics should include total_return"
        
        logger.info(f"Calculated metrics: {metrics}")
        logger.info("✅ Returns engine test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Returns engine test failed: {e}")
        return False


def test_fortune_100_client():
    """Test Fortune 100 client."""
    logger.info("Testing Fortune 100 client...")
    
    try:
        fortune_client = Fortune100Client()
        
        tickers = fortune_client.get_fortune_100_tickers()
        assert len(tickers) > 0, "Should have Fortune 100 tickers"
        
        selected = fortune_client.select_random_tickers(n=5, seed=42)
        assert len(selected) == 5, "Should select 5 tickers"
        
        selected2 = fortune_client.select_random_tickers(n=5, seed=42)
        assert selected == selected2, "Same seed should produce same selection"
        
        logger.info(f"Selected tickers: {selected}")
        logger.info("✅ Fortune 100 client test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fortune 100 client test failed: {e}")
        return False


def test_integration():
    """Test integration of all components."""
    logger.info("Testing component integration...")
    
    try:
        bars = create_test_data()
        
        macd_strategy = MACDStrategy()
        params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        
        signals = macd_strategy.generate_signal(bars, params)
        returns_engine = ReturnsEngine()
        returns = returns_engine.calculate_strategy_returns(signals, bars)
        metrics = returns_engine.calculate_metrics(returns)
        
        assert len(signals) == len(bars), "Signals should match bars length"
        assert len(returns) == len(bars), "Returns should match bars length"
        assert 'total_return' in metrics, "Should have total return metric"
        
        logger.info(f"Integration test metrics: {metrics}")
        logger.info("✅ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run standalone framework tests."""
    logger.info("🚀 Starting Standalone Framework Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Strategy Interface", test_strategy_interface),
        ("Returns Engine", test_returns_engine),
        ("Fortune 100 Client", test_fortune_100_client),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 STANDALONE FRAMEWORK TEST RESULTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Framework foundation is solid.")
        return True
    else:
        logger.error("💥 Some tests failed. Framework needs fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
