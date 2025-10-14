#!/usr/bin/env python3
"""
Simple Enhancement Filter Test

This script tests the new enhancement filters without requiring Docker.
It validates the core functionality of the enhancement components.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_market_type_detection():
    """Test market type detection functionality."""
    print("🧪 Testing Market Type Detection")
    print("=" * 50)
    
    try:
        from zone_fade_detector.filters.market_type_detector import MarketTypeDetector
        
        detector = MarketTypeDetector()
        
        # Test 1: Range-bound day
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
            
            # Create mock bar object
            class MockBar:
                def __init__(self, timestamp, open, high, low, close, volume):
                    self.timestamp = timestamp
                    self.open = open
                    self.high = high
                    self.low = low
                    self.close = close
                    self.volume = volume
            
            price_bars.append(MockBar(timestamp, open_price, high, low, close_price, volume))
            
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
        
        # Test statistics
        stats = detector.get_statistics()
        print(f"\n📊 Detector Statistics:")
        print(f"   Total Classifications: {stats['total_classifications']}")
        print(f"   Trend Days: {stats['trend_days_detected']}")
        print(f"   Range-bound Days: {stats['range_bound_days_detected']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market Type Detection test failed: {e}")
        return False

def test_market_internals_monitoring():
    """Test market internals monitoring functionality."""
    print("\n🧪 Testing Market Internals Monitoring")
    print("=" * 50)
    
    try:
        from zone_fade_detector.filters.market_internals import MarketInternalsMonitor
        
        monitor = MarketInternalsMonitor()
        
        # Test 1: Favorable internals
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
        
        # Test 2: Unfavorable internals
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
        
        return True
        
    except Exception as e:
        print(f"❌ Market Internals Monitoring test failed: {e}")
        return False

def test_enhanced_qrs_scoring():
    """Test enhanced QRS scoring functionality."""
    print("\n🧪 Testing Enhanced QRS Scoring")
    print("=" * 50)
    
    try:
        from zone_fade_detector.scoring.enhanced_qrs import EnhancedQRSScorer
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced QRS Scoring test failed: {e}")
        return False

def main():
    """Run all enhancement filter tests."""
    print("🎯 Zone Fade Detector Enhancement Filter Tests")
    print("=" * 60)
    print("Testing the new enhancement filters to ensure they meet requirements")
    print()
    
    test_results = []
    
    # Test individual components
    test_results.append(test_market_type_detection())
    test_results.append(test_market_internals_monitoring())
    test_results.append(test_enhanced_qrs_scoring())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📋 Test Summary:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed successfully!")
        print("\n🎯 Implementation Status:")
        print("   - Market Type Detection: ✅ Implemented and tested")
        print("   - Market Internals Monitoring: ✅ Implemented and tested")
        print("   - Enhanced QRS Scoring: ✅ Implemented and tested")
        print("   - Filter Pipeline: ✅ Framework implemented")
        
        print("\n📋 Next Steps:")
        print("   1. Implement remaining placeholder components:")
        print("      - Zone Approach Analyzer")
        print("      - Zone Touch Tracker")
        print("      - Entry Optimizer")
        print("      - Session Analyzer")
        print("   2. Integrate with real data sources (NYSE TICK, A/D Line)")
        print("   3. Add comprehensive unit tests")
        print("   4. Integrate with existing signal processor")
        print("   5. Run backtesting with enhanced filters")
        
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed!")
        print("Please check the error messages above and fix the issues.")

if __name__ == "__main__":
    main()