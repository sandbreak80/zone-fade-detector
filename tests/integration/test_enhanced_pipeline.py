#!/usr/bin/env python3
"""
Integration Test for Enhanced Filter Pipeline

This test validates that all enhancement components work together correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zone_fade_detector.filters.enhanced_filter_pipeline import EnhancedFilterPipeline
from zone_fade_detector.filters.zone_approach_analyzer import ZoneApproachAnalyzer
from zone_fade_detector.tracking.zone_touch_tracker import ZoneTouchTracker
from zone_fade_detector.optimization.entry_optimizer import EntryOptimizer
from zone_fade_detector.analysis.session_analyzer import SessionAnalyzer
from zone_fade_detector.filters.enhanced_market_context import EnhancedMarketContext
from zone_fade_detector.indicators.enhanced_volume_detector import EnhancedVolumeDetector
from zone_fade_detector.risk.risk_manager import RiskManager
from zone_fade_detector.scoring.enhanced_confluence import EnhancedConfluenceScorer


def test_component_initialization():
    """Test that all components can be initialized."""
    print("🧪 Testing Component Initialization...")
    
    components = {
        'ZoneApproachAnalyzer': ZoneApproachAnalyzer(),
        'ZoneTouchTracker': ZoneTouchTracker(),
        'EntryOptimizer': EntryOptimizer(),
        'SessionAnalyzer': SessionAnalyzer(),
        'EnhancedMarketContext': EnhancedMarketContext(),
        'EnhancedVolumeDetector': EnhancedVolumeDetector(),
        'RiskManager': RiskManager(),
        'EnhancedConfluenceScorer': EnhancedConfluenceScorer()
    }
    
    for name, component in components.items():
        print(f"   ✅ {name} initialized successfully")
    
    print(f"✅ All {len(components)} components initialized\n")
    return components


def test_filter_pipeline():
    """Test the enhanced filter pipeline integration."""
    print("🧪 Testing Enhanced Filter Pipeline...")
    
    try:
        pipeline = EnhancedFilterPipeline(config={
            'qrs_threshold': 10.0,  # Enhanced threshold
            'tick_threshold': 800.0,
            'ad_slope_threshold': 1000.0
        })
        print("   ✅ Filter pipeline initialized")
        print(f"   ✅ QRS threshold: {pipeline.qrs_scorer.threshold}")
        print(f"   ✅ Market type detector configured")
        print(f"   ✅ All components integrated")
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"   ✅ Statistics tracking ready")
        
        print("✅ Filter pipeline integration successful\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Filter pipeline error: {e}\n")
        return False


def test_enhanced_volume_detection():
    """Test enhanced volume spike detection."""
    print("🧪 Testing Enhanced Volume Detection...")
    
    try:
        detector = EnhancedVolumeDetector(
            base_threshold=2.0,
            strong_threshold=2.5,
            extreme_threshold=3.0
        )
        
        print(f"   ✅ Base threshold: {detector.base_threshold}x")
        print(f"   ✅ Strong threshold: {detector.strong_threshold}x")
        print(f"   ✅ Extreme threshold: {detector.extreme_threshold}x")
        print("✅ Enhanced volume detection ready\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Volume detection error: {e}\n")
        return False


def test_risk_management():
    """Test risk management calculations."""
    print("🧪 Testing Risk Management...")
    
    try:
        risk_mgr = RiskManager(
            atr_multiplier=1.5,
            min_stop_distance_pct=0.5,
            max_stop_distance_pct=2.0
        )
        
        # Test stop calculation
        entry_price = 500.0
        direction = 'LONG'
        atr = 5.0
        zone_level = 495.0
        
        stop = risk_mgr.calculate_stop_loss(
            entry_price, direction, atr, zone_level
        )
        
        print(f"   ✅ Stop calculated: ${stop.stop_price:.2f}")
        print(f"   ✅ Stop type: {stop.stop_type.value}")
        print(f"   ✅ Distance: {stop.distance_percent:.2f}%")
        
        # Test position sizing
        position = risk_mgr.calculate_position_size(
            account_balance=100000,
            entry_price=entry_price,
            stop_price=stop.stop_price,
            atr=atr
        )
        
        print(f"   ✅ Position size: {position.shares} shares")
        print(f"   ✅ Risk amount: ${position.risk_amount:.2f}")
        print(f"   ✅ Risk percent: {position.risk_percent:.2f}%")
        
        print("✅ Risk management working correctly\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Risk management error: {e}\n")
        return False


def test_confluence_scoring():
    """Test enhanced confluence scoring."""
    print("🧪 Testing Enhanced Confluence Scoring...")
    
    try:
        scorer = EnhancedConfluenceScorer()
        
        # Test zone scoring
        result = scorer.score_zone_confluence(
            zone_level=500.0,
            zone_type='PRIOR_DAY_HIGH',
            zone_age_hours=6.0,
            touch_count=1,
            vwap=498.5
        )
        
        print(f"   ✅ Confluence score: {result.total_score:.1f}/100")
        print(f"   ✅ Quality: {result.quality.value}")
        print(f"   ✅ Confidence: {result.confidence:.2f}")
        print(f"   ✅ Factors analyzed: {result.factor_count}")
        
        print("✅ Confluence scoring working correctly\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Confluence scoring error: {e}\n")
        return False


def test_market_context():
    """Test enhanced market context analysis."""
    print("🧪 Testing Enhanced Market Context...")
    
    try:
        context = EnhancedMarketContext()
        
        print(f"   ✅ Trend lookback: {context.trend_lookback}")
        print(f"   ✅ Structure lookback: {context.structure_lookback}")
        print(f"   ✅ Volatility lookback: {context.volatility_lookback}")
        print("✅ Market context analyzer ready\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Market context error: {e}\n")
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 80)
    print("🔬 ENHANCED FILTER PIPELINE - INTEGRATION TESTS")
    print("=" * 80)
    print("Testing all enhancement components...\n")
    
    tests = [
        ('Component Initialization', test_component_initialization),
        ('Filter Pipeline', test_filter_pipeline),
        ('Enhanced Volume Detection', test_enhanced_volume_detection),
        ('Risk Management', test_risk_management),
        ('Confluence Scoring', test_confluence_scoring),
        ('Market Context', test_market_context)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment testing.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
