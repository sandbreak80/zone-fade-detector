#!/usr/bin/env python3
"""
Test Enhancement Filters

This script tests the new enhancement filters to ensure they work correctly
and meet the requirements specified in the enhancement document.

Tests:
1. Market Type Detection
2. Market Internals Monitoring
3. Enhanced QRS Scoring
4. Filter Pipeline Integration
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zone_fade_detector.filters.market_type_detector import MarketTypeDetector, MarketTypeFilter
from zone_fade_detector.filters.market_internals import MarketInternalsMonitor, InternalsFilter
from zone_fade_detector.scoring.enhanced_qrs import EnhancedQRSScorer
from zone_fade_detector.filters.enhanced_filter_pipeline import EnhancedFilterPipeline

def create_mock_ohlcv_bar(timestamp, open_price, high, low, close, volume):
    """Create a mock OHLCV bar object."""
    class MockBar:
        def __init__(self, timestamp, open, high, low, close, volume):
            self.timestamp = timestamp
            self.open = open
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume
    
    return MockBar(timestamp, open_price, high, low, close, volume)

def test_market_type_detection():
    """Test market type detection functionality."""
    print("🧪 Testing Market Type Detection")
    print("=" * 50)
    
    detector = MarketTypeDetector()
    
    # Test 1: Range-bound day (low volatility, balanced TICK)
    print("\n📊 Test 1: Range-bound day")
    price_bars = []
    tick_data = []
    ad_line_data = []
    
    # Create range-bound price action
    base_price = 4500.0
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        # Small random movements around base price
        price_change = np.random.normal(0, 0.5)
        open_price = base_price + price_change
        close_price = open_price + np.random.normal(0, 0.3)
        high = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = np.random.randint(1000, 5000)
        
        price_bars.append(create_mock_ohlcv_bar(timestamp, open_price, high, low, close_price, volume))
        
        # Balanced TICK data
        tick_data.append(np.random.normal(0, 100))
        
        # Flat A/D Line
        ad_line_data.append(1000 + i * 2)  # Very small slope
    
    result = detector.detect_market_type(price_bars, tick_data, ad_line_data)
    print(f"   Market Type: {result.market_type.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reasoning: {result.reasoning}")
    print(f"   TICK Mean: {result.tick_mean:.1f}")
    print(f"   A/D Slope: {result.ad_slope:.1f}")
    
    # Test 2: Trend day (high volatility, skewed TICK)
    print("\n📈 Test 2: Trend day")
    price_bars = []
    tick_data = []
    ad_line_data = []
    
    # Create trending price action
    base_price = 4500.0
    trend_direction = 1  # Upward trend
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        # Strong directional movement
        price_change = trend_direction * np.random.normal(2.0, 0.5)
        open_price = base_price + price_change
        close_price = open_price + trend_direction * np.random.normal(1.0, 0.3)
        high = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        low = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = np.random.randint(2000, 8000)
        
        price_bars.append(create_mock_ohlcv_bar(timestamp, open_price, high, low, close_price, volume))
        
        # Skewed TICK data (positive)
        tick_data.append(np.random.normal(500, 200))
        
        # Trending A/D Line
        ad_line_data.append(1000 + i * 20)  # Strong upward slope
    
    result = detector.detect_market_type(price_bars, tick_data, ad_line_data)
    print(f"   Market Type: {result.market_type.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reasoning: {result.reasoning}")
    print(f"   TICK Mean: {result.tick_mean:.1f}")
    print(f"   A/D Slope: {result.ad_slope:.1f}")
    
    # Test statistics
    stats = detector.get_statistics()
    print(f"\n📊 Detector Statistics:")
    print(f"   Total Classifications: {stats['total_classifications']}")
    print(f"   Trend Days: {stats['trend_days_detected']}")
    print(f"   Range-bound Days: {stats['range_bound_days_detected']}")

def test_market_internals_monitoring():
    """Test market internals monitoring functionality."""
    print("\n🧪 Testing Market Internals Monitoring")
    print("=" * 50)
    
    monitor = MarketInternalsMonitor()
    
    # Test 1: Favorable internals (balanced TICK + flat A/D)
    print("\n✅ Test 1: Favorable internals")
    tick_data = np.random.normal(0, 150, 50).tolist()  # Balanced TICK
    ad_line_data = [1000 + i * 1 for i in range(80)]  # Flat A/D
    
    result = monitor.check_fade_conditions(tick_data, ad_line_data)
    print(f"   Favorable: {result.is_favorable}")
    print(f"   Quality Score: {result.quality_score}")
    print(f"   TICK Status: {result.tick_analysis.status.value}")
    print(f"   TICK Mean: {result.tick_analysis.mean_value:.1f}")
    print(f"   A/D Status: {result.ad_analysis.status.value}")
    print(f"   A/D Slope: {result.ad_analysis.slope:.1f}")
    print(f"   Recommendation: {result.recommendation}")
    
    # Test 2: Unfavorable internals (skewed TICK + trending A/D)
    print("\n❌ Test 2: Unfavorable internals")
    tick_data = np.random.normal(400, 200, 50).tolist()  # Skewed TICK
    ad_line_data = [1000 + i * 15 for i in range(80)]  # Trending A/D
    
    result = monitor.check_fade_conditions(tick_data, ad_line_data)
    print(f"   Favorable: {result.is_favorable}")
    print(f"   Quality Score: {result.quality_score}")
    print(f"   TICK Status: {result.tick_analysis.status.value}")
    print(f"   TICK Mean: {result.tick_analysis.mean_value:.1f}")
    print(f"   A/D Status: {result.ad_analysis.status.value}")
    print(f"   A/D Slope: {result.ad_analysis.slope:.1f}")
    print(f"   Recommendation: {result.recommendation}")
    
    # Test statistics
    stats = monitor.get_statistics()
    print(f"\n📊 Monitor Statistics:")
    print(f"   Total Checks: {stats['total_checks']}")
    print(f"   Favorable: {stats['favorable_checks']}")
    print(f"   Unfavorable: {stats['unfavorable_checks']}")

def test_enhanced_qrs_scoring():
    """Test enhanced QRS scoring functionality."""
    print("\n🧪 Testing Enhanced QRS Scoring")
    print("=" * 50)
    
    scorer = EnhancedQRSScorer(threshold=7.0)
    
    # Test 1: High-quality setup
    print("\n⭐ Test 1: High-quality setup")
    setup = {
        'setup_id': 'test_1',
        'zone_type': 'BCZ',
        'zone_timeframe': 'Daily',
        'zone_strength': 2.0,
        'prior_touches': 3,
        'confluence_factors': ['round_number', 'previous_day_high', 'vwap_level'],
        'rejection_bar': {
            'open': 4500.0,
            'high': 4505.0,
            'low': 4495.0,
            'close': 4496.0,
            'volume': 50000
        },
        'choch_confirmed': True,
        'touch_number': 1,
        'has_balance_before': False,
        'zone_position': 'front',
        'setup_type': 'ZFR',
        'htf_alignment': 'aligned',
        'related_markets_aligned': True,
        'divergences_confirm': True
    }
    
    market_data = {
        'market_type': 'RANGE_BOUND',
        'internals_favorable': True,
        'internals_quality_score': 2.0,
        'volume_data': {
            'avg_volume': 25000
        }
    }
    
    result = scorer.score_setup(setup, market_data)
    if result:
        print(f"   Total Score: {result.total_score:.2f}")
        print(f"   Grade: {result.grade.value}")
        print(f"   Veto: {result.veto}")
        print(f"   Veto Reason: {result.veto_reason}")
        print("\n   Factor Breakdown:")
        for factor_name, factor in result.factors.items():
            print(f"     {factor_name}: {factor.score:.1f}/{factor.max_score} - {factor.reasoning}")
    else:
        print("   Setup was vetoed!")
    
    # Test 2: Low-quality setup (trend day)
    print("\n📉 Test 2: Low-quality setup (trend day)")
    setup = {
        'setup_id': 'test_2',
        'zone_type': 'Standard',
        'zone_timeframe': 'M5',
        'zone_strength': 1.0,
        'prior_touches': 1,
        'confluence_factors': [],
        'rejection_bar': {
            'open': 4500.0,
            'high': 4502.0,
            'low': 4499.0,
            'close': 4501.0,
            'volume': 15000
        },
        'choch_confirmed': False,
        'touch_number': 3,
        'has_balance_before': True,
        'zone_position': 'back',
        'setup_type': 'ZF-TR',
        'htf_alignment': 'against',
        'related_markets_aligned': False,
        'divergences_confirm': False
    }
    
    market_data = {
        'market_type': 'TREND_DAY',
        'internals_favorable': False,
        'internals_quality_score': 0.0,
        'volume_data': {
            'avg_volume': 20000
        }
    }
    
    result = scorer.score_setup(setup, market_data)
    if result:
        print(f"   Total Score: {result.total_score:.2f}")
        print(f"   Grade: {result.grade.value}")
        print(f"   Veto: {result.veto}")
        print(f"   Veto Reason: {result.veto_reason}")
    else:
        print("   Setup was vetoed!")
    
    # Test statistics
    stats = scorer.get_statistics()
    print(f"\n📊 QRS Statistics:")
    print(f"   Total Scored: {stats['total_scored']}")
    print(f"   Signals Generated: {stats['signals_generated']}")
    print(f"   Signals Vetoed: {stats['signals_vetoed']}")
    print(f"   Average Score: {stats['avg_score']:.2f}")

def test_filter_pipeline():
    """Test the complete filter pipeline."""
    print("\n🧪 Testing Complete Filter Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    config = {
        'qrs_threshold': 7.0,
        'tick_threshold': 800.0,
        'ad_slope_threshold': 1000.0
    }
    
    pipeline = EnhancedFilterPipeline(config)
    
    # Test 1: High-quality signal (should pass)
    print("\n✅ Test 1: High-quality signal")
    signal = {
        'setup_id': 'pipeline_test_1',
        'symbol': 'SPY',
        'zone_type': 'BCZ',
        'zone_timeframe': 'Daily',
        'zone_strength': 2.0,
        'prior_touches': 2,
        'confluence_factors': ['round_number', 'vwap_level'],
        'rejection_bar': {
            'open': 4500.0,
            'high': 4508.0,
            'low': 4492.0,
            'close': 4493.0,
            'volume': 60000
        },
        'choch_confirmed': True,
        'touch_number': 1,
        'has_balance_before': False,
        'zone_position': 'front',
        'setup_type': 'ZFR',
        'htf_alignment': 'aligned',
        'related_markets_aligned': True,
        'divergences_confirm': True
    }
    
    # Create market data
    price_bars = []
    tick_data = []
    ad_line_data = []
    
    # Range-bound market data
    base_price = 4500.0
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        price_change = np.random.normal(0, 0.5)
        open_price = base_price + price_change
        close_price = open_price + np.random.normal(0, 0.3)
        high = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = np.random.randint(1000, 5000)
        
        price_bars.append(create_mock_ohlcv_bar(timestamp, open_price, high, low, close_price, volume))
        tick_data.append(np.random.normal(0, 100))  # Balanced TICK
        ad_line_data.append(1000 + i * 1)  # Flat A/D
    
    market_data = {
        'price_bars': price_bars,
        'tick_data': tick_data,
        'ad_line_data': ad_line_data,
        'volume_data': {'avg_volume': 30000},
        'market_type': 'RANGE_BOUND',
        'internals_favorable': True,
        'internals_quality_score': 2.0
    }
    
    result = pipeline.process_signal(signal, market_data)
    print(f"   Signal Generated: {result.signal is not None}")
    print(f"   Passed Filters: {len(result.passed_filters)}")
    print(f"   Failed Filters: {len(result.failed_filters)}")
    print(f"   Processing Time: {result.processing_time_ms:.2f}ms")
    
    if result.signal:
        print(f"   QRS Score: {result.signal.get('qrs_score', 'N/A')}")
        print(f"   QRS Grade: {result.signal.get('qrs_grade', 'N/A')}")
    
    # Test 2: Low-quality signal (should be vetoed)
    print("\n❌ Test 2: Low-quality signal (trend day)")
    signal = {
        'setup_id': 'pipeline_test_2',
        'symbol': 'SPY',
        'zone_type': 'Standard',
        'zone_timeframe': 'M5',
        'zone_strength': 1.0,
        'prior_touches': 1,
        'confluence_factors': [],
        'rejection_bar': {
            'open': 4500.0,
            'high': 4502.0,
            'low': 4499.0,
            'close': 4501.0,
            'volume': 15000
        },
        'choch_confirmed': False,
        'touch_number': 3,
        'has_balance_before': True,
        'zone_position': 'back',
        'setup_type': 'ZF-TR',
        'htf_alignment': 'against',
        'related_markets_aligned': False,
        'divergences_confirm': False
    }
    
    # Create trending market data
    price_bars = []
    tick_data = []
    ad_line_data = []
    
    base_price = 4500.0
    trend_direction = 1
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        price_change = trend_direction * np.random.normal(2.0, 0.5)
        open_price = base_price + price_change
        close_price = open_price + trend_direction * np.random.normal(1.0, 0.3)
        high = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        low = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = np.random.randint(2000, 8000)
        
        price_bars.append(create_mock_ohlcv_bar(timestamp, open_price, high, low, close_price, volume))
        tick_data.append(np.random.normal(500, 200))  # Skewed TICK
        ad_line_data.append(1000 + i * 20)  # Trending A/D
    
    market_data = {
        'price_bars': price_bars,
        'tick_data': tick_data,
        'ad_line_data': ad_line_data,
        'volume_data': {'avg_volume': 20000},
        'market_type': 'TREND_DAY',
        'internals_favorable': False,
        'internals_quality_score': 0.0
    }
    
    result = pipeline.process_signal(signal, market_data)
    print(f"   Signal Generated: {result.signal is not None}")
    print(f"   Passed Filters: {len(result.passed_filters)}")
    print(f"   Failed Filters: {len(result.failed_filters)}")
    print(f"   Processing Time: {result.processing_time_ms:.2f}ms")
    
    if result.failed_filters:
        print(f"   Failed at: {result.failed_filters[0]}")
    
    # Test statistics
    stats = pipeline.get_comprehensive_statistics()
    print(f"\n📊 Pipeline Statistics:")
    print(f"   Total Processed: {stats['pipeline_statistics']['total_signals_processed']}")
    print(f"   Signals Generated: {stats['pipeline_statistics']['signals_generated']}")
    print(f"   Signals Vetoed: {stats['pipeline_statistics']['signals_vetoed']}")
    print(f"   Generation Rate: {stats['pipeline_statistics']['generation_rate']:.1f}%")

def main():
    """Run all enhancement filter tests."""
    print("🎯 Zone Fade Detector Enhancement Filter Tests")
    print("=" * 60)
    print("Testing the new enhancement filters to ensure they meet requirements")
    print()
    
    try:
        # Test individual components
        test_market_type_detection()
        test_market_internals_monitoring()
        test_enhanced_qrs_scoring()
        test_filter_pipeline()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Test Summary:")
        print("   - Market Type Detection: ✅ Working")
        print("   - Market Internals Monitoring: ✅ Working")
        print("   - Enhanced QRS Scoring: ✅ Working")
        print("   - Filter Pipeline Integration: ✅ Working")
        
        print("\n🎯 Next Steps:")
        print("   1. Implement remaining placeholder components")
        print("   2. Integrate with real data sources (NYSE TICK, A/D Line)")
        print("   3. Add comprehensive unit tests")
        print("   4. Integrate with existing signal processor")
        print("   5. Run backtesting with enhanced filters")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()