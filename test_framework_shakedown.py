#!/usr/bin/env python3
"""
Framework Shakedown Test for Trading Strategy Testing Framework.

This script tests the core framework components to ensure they work correctly.
It serves as a validation that the framework foundation is solid before
implementing the full 4-step validation battery.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.strategies import get_strategy, list_strategies
from zone_fade_detector.utils.returns_engine import ReturnsEngine
from zone_fade_detector.data.fortune_100_client import Fortune100Client
from zone_fade_detector.core.models import OHLCVBar

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data() -> List[OHLCVBar]:
    """
    Create test OHLCV data for framework testing.
    
    Returns:
        List of OHLCVBar objects
    """
    bars = []
    base_price = 100.0
    
    for i in range(1000):  # 1000 bars of test data
        # Simple random walk for price
        price_change = (i % 2 - 0.5) * 0.01  # Alternating small changes
        price = base_price + price_change * i
        
        # Create OHLCV bar
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
        # Test strategy listing
        strategies = list_strategies()
        logger.info(f"Available strategies: {strategies}")
        
        # Test MACD strategy
        macd_strategy = get_strategy('macd')
        logger.info(f"MACD strategy name: {macd_strategy.get_name()}")
        
        # Test parameter space
        param_space = macd_strategy.get_parameter_space()
        logger.info(f"MACD parameter space: {param_space}")
        
        # Test parameter validation
        valid_params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        invalid_params = {'fast_period': 12, 'slow_period': 26}  # Missing signal_period
        
        assert macd_strategy.validate_parameters(valid_params), "Valid parameters should pass validation"
        assert not macd_strategy.validate_parameters(invalid_params), "Invalid parameters should fail validation"
        
        logger.info("✅ Strategy interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy interface test failed: {e}")
        return False


def test_macd_strategy():
    """Test MACD strategy signal generation."""
    logger.info("Testing MACD strategy...")
    
    try:
        # Create test data
        bars = create_test_data()
        
        # Get MACD strategy
        macd_strategy = get_strategy('macd')
        
        # Test with default parameters
        default_params = macd_strategy.get_default_parameters()
        signals = macd_strategy.generate_signal(bars, default_params)
        
        # Validate signals
        assert len(signals) == len(bars), "Signals length should match bars length"
        assert all(s in [-1, 0, 1] for s in signals), "All signals should be -1, 0, or 1"
        
        # Check that we have some signals (not all zeros)
        non_zero_signals = sum(1 for s in signals if s != 0)
        assert non_zero_signals > 0, "Should have some non-zero signals"
        
        logger.info(f"Generated {len(signals)} signals with {non_zero_signals} non-zero signals")
        logger.info("✅ MACD strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ MACD strategy test failed: {e}")
        return False


def test_returns_engine():
    """Test the returns engine."""
    logger.info("Testing returns engine...")
    
    try:
        # Create test data
        bars = create_test_data()
        signals = [1 if i % 10 == 0 else 0 for i in range(len(bars))]  # Simple signal pattern
        
        # Test returns engine
        returns_engine = ReturnsEngine(commission=0.001, slippage=0.0005)
        strategy_returns = returns_engine.calculate_strategy_returns(signals, bars)
        
        # Validate returns
        assert len(strategy_returns) == len(bars), "Returns length should match bars length"
        
        # Test metrics calculation
        metrics = returns_engine.calculate_metrics(strategy_returns)
        assert 'total_return' in metrics, "Metrics should include total_return"
        assert 'sharpe_ratio' in metrics, "Metrics should include sharpe_ratio"
        
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
        # Test Fortune 100 client
        fortune_client = Fortune100Client()
        
        # Test ticker fetching
        tickers = fortune_client.get_fortune_100_tickers()
        assert len(tickers) > 0, "Should have Fortune 100 tickers"
        
        # Test random selection
        selected = fortune_client.select_random_tickers(n=5, seed=42)
        assert len(selected) == 5, "Should select 5 tickers"
        
        # Test reproducibility
        selected2 = fortune_client.select_random_tickers(n=5, seed=42)
        assert selected == selected2, "Same seed should produce same selection"
        
        # Test portfolio creation
        portfolio = fortune_client.create_test_portfolio(n=3, seed=123)
        assert len(portfolio['tickers']) == 3, "Portfolio should have 3 tickers"
        
        logger.info(f"Selected tickers: {selected}")
        logger.info(f"Portfolio: {portfolio}")
        logger.info("✅ Fortune 100 client test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fortune 100 client test failed: {e}")
        return False


def test_integration():
    """Test integration of all components."""
    logger.info("Testing component integration...")
    
    try:
        # Create test data
        bars = create_test_data()
        
        # Get strategy
        macd_strategy = get_strategy('macd')
        params = macd_strategy.get_default_parameters()
        
        # Generate signals
        signals = macd_strategy.generate_signal(bars, params)
        
        # Calculate returns
        returns_engine = ReturnsEngine()
        returns = returns_engine.calculate_strategy_returns(signals, bars)
        
        # Calculate metrics
        metrics = returns_engine.calculate_metrics(returns)
        
        # Validate integration
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
    """Run framework shakedown tests."""
    logger.info("🚀 Starting Framework Shakedown Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Strategy Interface", test_strategy_interface),
        ("MACD Strategy", test_macd_strategy),
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
    logger.info("📊 FRAMEWORK SHAKEDOWN RESULTS")
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
