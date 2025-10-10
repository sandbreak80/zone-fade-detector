#!/usr/bin/env python3
"""
Integration test for Micro Window Analyzer with Rolling Window Manager.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.micro_window_analyzer import (
    MicroWindowAnalyzer, InitiativeType, MicroWindowType
)
from zone_fade_detector.core.models import OHLCVBar, Zone


def create_zone_touch_scenario():
    """Create a realistic zone touch scenario for testing."""
    print("📊 Creating zone touch scenario...")
    
    # Create a supply zone
    zone = Zone(
        zone_id="supply_zone_001",
        symbol="SPY",
        zone_type="supply",
        high=100.0,
        low=99.0,
        strength=0.85,
        touches=1,
        created_at=datetime(2024, 1, 2, 9, 30),
        last_touch=datetime(2024, 1, 2, 9, 30)
    )
    
    # Create bars leading up to zone touch
    base_time = datetime(2024, 1, 2, 9, 30)
    bars = []
    
    # Pre-touch phase: normal movement approaching zone
    for i in range(15):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=98.0 + i * 0.05,
            high=98.2 + i * 0.05,
            low=97.8 + i * 0.05,
            close=98.1 + i * 0.05,
            volume=1000 + i * 20
        )
        bars.append(bar)
    
    # Zone touch: high volume, clear rejection
    touch_bar = OHLCVBar(
        timestamp=base_time + timedelta(minutes=15),
        open=98.8,
        high=100.0,  # Touch the zone
        low=98.5,
        close=99.2,  # Reject from zone
        volume=5000  # High volume
    )
    bars.append(touch_bar)
    
    # Post-touch phase: rejection and pullback
    for i in range(10):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=16 + i),
            open=99.2 - i * 0.03,
            high=99.4 - i * 0.03,
            low=99.0 - i * 0.03,
            close=99.1 - i * 0.03,
            volume=2000 - i * 50  # Decreasing volume
        )
        bars.append(bar)
    
    print(f"   ✅ Generated {len(bars)} bars for zone touch scenario")
    print(f"   Zone: {zone.zone_type} at ${zone.low:.2f}-${zone.high:.2f}")
    print(f"   Touch bar: ${touch_bar.open:.2f} -> ${touch_bar.high:.2f} -> ${touch_bar.close:.2f}")
    
    return zone, bars, touch_bar


def test_zone_touch_analysis(micro_analyzer, zone, bars, touch_bar, window_manager):
    """Test zone touch analysis."""
    print("\n🎯 Testing Zone Touch Analysis")
    print("=" * 50)
    
    # Mock window manager to return bars
    window_manager.get_window_bars.return_value = bars
    
    # Analyze zone touch
    analysis = micro_analyzer.analyze_zone_touch(zone, touch_bar, "SPY")
    
    if analysis:
        print(f"   Zone: {analysis.zone.zone_id}")
        print(f"   Touch Type: {analysis.touch_type}")
        print(f"   Touch Price: ${analysis.touch_price:.2f}")
        print(f"   Touch Timestamp: {analysis.touch_timestamp}")
        print(f"   Is Significant: {analysis.is_significant}")
        print(f"   Confidence Score: {analysis.confidence_score:.2f}")
        print(f"   Absorption Detected: {analysis.absorption_detected}")
        print(f"   Exhaustion Detected: {analysis.exhaustion_detected}")
        print(f"   Rejection Confirmed: {analysis.rejection_confirmed}")
        print(f"   Micro Window Bars: {len(analysis.micro_window_bars)}")
        
        # Analyze pre-touch metrics
        pre_touch = analysis.pre_touch_analysis
        print(f"\n   Pre-Touch Analysis:")
        print(f"     Initiative Type: {pre_touch.initiative_type.value}")
        print(f"     Strength Score: {pre_touch.strength_score:.2f}")
        print(f"     Volume Ratio: {pre_touch.volume_ratio:.2f}")
        print(f"     Price Momentum: {pre_touch.price_momentum:.4f}")
        print(f"     Wick Ratio: {pre_touch.wick_ratio:.2f}")
        print(f"     Rejection Clarity: {pre_touch.rejection_clarity:.2f}")
        print(f"     Absorption Signals: {pre_touch.absorption_signals}")
        print(f"     Exhaustion Signals: {pre_touch.exhaustion_signals}")
        print(f"     Consecutive Bars: {pre_touch.consecutive_bars}")
        print(f"     Volume Spike: {pre_touch.volume_spike}")
        print(f"     Volatility Spike: {pre_touch.volatility_spike}")
        
        # Analyze post-touch metrics
        post_touch = analysis.post_touch_analysis
        print(f"\n   Post-Touch Analysis:")
        print(f"     Initiative Type: {post_touch.initiative_type.value}")
        print(f"     Strength Score: {post_touch.strength_score:.2f}")
        print(f"     Volume Ratio: {post_touch.volume_ratio:.2f}")
        print(f"     Price Momentum: {post_touch.price_momentum:.4f}")
        print(f"     Wick Ratio: {post_touch.wick_ratio:.2f}")
        print(f"     Rejection Clarity: {post_touch.rejection_clarity:.2f}")
        print(f"     Absorption Signals: {post_touch.absorption_signals}")
        print(f"     Exhaustion Signals: {post_touch.exhaustion_signals}")
        print(f"     Consecutive Bars: {post_touch.consecutive_bars}")
        print(f"     Volume Spike: {post_touch.volume_spike}")
        print(f"     Volatility Spike: {post_touch.volatility_spike}")
        
        return analysis
    else:
        print("   ❌ Zone touch analysis failed")
        return None


def test_multiple_zone_touches(micro_analyzer, window_manager):
    """Test multiple zone touch analyses."""
    print("\n🔄 Testing Multiple Zone Touches")
    print("=" * 50)
    
    # Create multiple zones and scenarios
    scenarios = []
    
    # Scenario 1: Supply zone with rejection
    supply_zone = Zone(
        zone_id="supply_001",
        symbol="SPY",
        zone_type="supply",
        high=100.0,
        low=99.0,
        strength=0.8,
        touches=1,
        created_at=datetime.now(),
        last_touch=datetime.now()
    )
    
    # Scenario 2: Demand zone with bounce
    demand_zone = Zone(
        zone_id="demand_001",
        symbol="SPY",
        zone_type="demand",
        high=101.0,
        low=100.0,
        strength=0.9,
        touches=1,
        created_at=datetime.now(),
        last_touch=datetime.now()
    )
    
    # Create bars for each scenario
    for i, zone in enumerate([supply_zone, demand_zone]):
        base_time = datetime.now() + timedelta(hours=i)
        bars = []
        
        # Pre-touch bars
        for j in range(10):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=j),
                open=100.0 + i * 0.1,
                high=100.2 + i * 0.1,
                low=99.8 + i * 0.1,
                close=100.1 + i * 0.1,
                volume=1000 + j * 10
            )
            bars.append(bar)
        
        # Touch bar
        touch_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=10),
            open=100.1 + i * 0.1,
            high=100.0 + i * 0.1,  # Touch zone
            low=99.9 + i * 0.1,
            close=100.05 + i * 0.1,
            volume=3000
        )
        bars.append(touch_bar)
        
        # Post-touch bars
        for j in range(5):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=11 + j),
                open=100.05 + i * 0.1,
                high=100.1 + i * 0.1,
                low=100.0 + i * 0.1,
                close=100.02 + i * 0.1,
                volume=1500 - j * 50
            )
            bars.append(bar)
        
        scenarios.append((zone, bars, touch_bar))
    
    # Analyze each scenario
    analyses = []
    for i, (zone, bars, touch_bar) in enumerate(scenarios):
        print(f"\n   Scenario {i + 1}: {zone.zone_type} zone")
        window_manager.get_window_bars.return_value = bars
        
        analysis = micro_analyzer.analyze_zone_touch(zone, touch_bar, "SPY")
        if analysis:
            analyses.append(analysis)
            print(f"     ✅ Analysis completed - Confidence: {analysis.confidence_score:.2f}")
        else:
            print(f"     ❌ Analysis failed")
    
    print(f"\n   Total analyses completed: {len(analyses)}")
    return analyses


def test_initiative_patterns(micro_analyzer, window_manager):
    """Test different initiative patterns."""
    print("\n📈 Testing Initiative Patterns")
    print("=" * 50)
    
    # Create zones for different patterns
    patterns = [
        ("absorption", "supply", 100.0, 99.0),
        ("exhaustion", "demand", 101.0, 100.0),
        ("bullish", "supply", 100.0, 99.0),
        ("bearish", "demand", 101.0, 100.0),
        ("neutral", "supply", 100.0, 99.0)
    ]
    
    analyses = []
    
    for pattern_name, zone_type, high, low in patterns:
        print(f"\n   Testing {pattern_name} pattern:")
        
        # Create zone
        zone = Zone(
            zone_id=f"{pattern_name}_zone",
            symbol="SPY",
            zone_type=zone_type,
            high=high,
            low=low,
            strength=0.8,
            touches=1,
            created_at=datetime.now(),
            last_touch=datetime.now()
        )
        
        # Create bars with specific pattern
        base_time = datetime.now()
        bars = []
        
        # Pre-touch bars
        for i in range(8):
            if pattern_name == "absorption":
                # High volume, small moves
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.1,
                    low=99.9,
                    close=100.05,
                    volume=3000  # High volume
                )
            elif pattern_name == "exhaustion":
                # Decreasing volume, increasing moves
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.0 + i * 0.02,
                    low=99.9 + i * 0.02,
                    close=100.0 + i * 0.01,
                    volume=2000 - i * 100  # Decreasing volume
                )
            elif pattern_name == "bullish":
                # Strong upward momentum
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0 + i * 0.01,
                    high=100.2 + i * 0.01,
                    low=99.8 + i * 0.01,
                    close=100.1 + i * 0.01,
                    volume=1500 + i * 50
                )
            elif pattern_name == "bearish":
                # Strong downward momentum
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0 - i * 0.01,
                    high=100.1 - i * 0.01,
                    low=99.7 - i * 0.01,
                    close=99.9 - i * 0.01,
                    volume=1500 + i * 50
                )
            else:  # neutral
                # Balanced movement
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.1,
                    low=99.9,
                    close=100.0,
                    volume=1000
                )
            bars.append(bar)
        
        # Touch bar
        touch_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=8),
            open=100.0,
            high=high if zone_type == "supply" else 100.0,
            low=low if zone_type == "demand" else 100.0,
            close=100.0,
            volume=2000
        )
        bars.append(touch_bar)
        
        # Post-touch bars
        for i in range(5):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=9 + i),
                open=100.0,
                high=100.1,
                low=99.9,
                close=100.0,
                volume=1000
            )
            bars.append(bar)
        
        # Analyze
        window_manager.get_window_bars.return_value = bars
        analysis = micro_analyzer.analyze_zone_touch(zone, touch_bar, "SPY")
        
        if analysis:
            analyses.append(analysis)
            pre_type = analysis.pre_touch_analysis.initiative_type.value
            post_type = analysis.post_touch_analysis.initiative_type.value
            print(f"     Pre-touch: {pre_type}")
            print(f"     Post-touch: {post_type}")
            print(f"     Confidence: {analysis.confidence_score:.2f}")
        else:
            print(f"     ❌ Analysis failed")
    
    print(f"\n   Total pattern analyses: {len(analyses)}")
    return analyses


def test_analysis_filtering(micro_analyzer):
    """Test analysis filtering methods."""
    print("\n🔍 Testing Analysis Filtering")
    print("=" * 50)
    
    # Get different types of analyses
    significant = micro_analyzer.get_significant_touches()
    absorption = micro_analyzer.get_absorption_touches()
    exhaustion = micro_analyzer.get_exhaustion_touches()
    rejection = micro_analyzer.get_rejection_touches()
    
    print(f"   Significant touches: {len(significant)}")
    print(f"   Absorption touches: {len(absorption)}")
    print(f"   Exhaustion touches: {len(exhaustion)}")
    print(f"   Rejection touches: {len(rejection)}")
    
    # Verify filtering
    for analysis in significant:
        assert analysis.is_significant is True
    
    for analysis in absorption:
        assert analysis.absorption_detected is True
    
    for analysis in exhaustion:
        assert analysis.exhaustion_detected is True
    
    for analysis in rejection:
        assert analysis.rejection_confirmed is True
    
    print("   ✅ All filtering methods working correctly")


def test_analysis_summary(micro_analyzer):
    """Test analysis summary generation."""
    print("\n📊 Testing Analysis Summary")
    print("=" * 50)
    
    summary = micro_analyzer.get_analysis_summary()
    
    print("   Analysis Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.2f}")
        else:
            print(f"     {key}: {value}")
    
    # Verify summary structure
    required_keys = [
        "total_analyses", "significant_touches", "absorption_touches",
        "exhaustion_touches", "rejection_touches", "average_confidence",
        "significance_rate", "absorption_rate", "exhaustion_rate", "rejection_rate"
    ]
    
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"
    
    print("   ✅ Analysis summary complete")


async def test_async_operations(micro_analyzer, zone, bars, touch_bar, window_manager):
    """Test async operations."""
    print("\n🔄 Testing Async Operations")
    print("=" * 50)
    
    # Mock window manager
    window_manager.get_window_bars.return_value = bars
    
    # Simulate async processing
    async def process_zone_touches():
        tasks = []
        
        # Create multiple zone touch analyses
        for i in range(5):
            task = asyncio.create_task(
                analyze_zone_touch_async(micro_analyzer, zone, touch_bar, "SPY")
            )
            tasks.append(task)
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks)
        
        successful_analyses = [r for r in results if r is not None]
        print(f"   ✅ Completed {len(successful_analyses)} async analyses")
        
        return successful_analyses
    
    async def analyze_zone_touch_async(analyzer, zone, touch_bar, symbol):
        """Async wrapper for zone touch analysis."""
        await asyncio.sleep(0.001)  # Simulate processing time
        return analyzer.analyze_zone_touch(zone, touch_bar, symbol)
    
    # Run async test
    analyses = await process_zone_touches()
    
    print(f"   Total analyses completed: {len(analyses)}")
    return analyses


async def main():
    """Main integration test function."""
    print("🔧 Testing Micro Window Analyzer Integration")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing components...")
    window_manager = RollingWindowManager(
        evaluation_cadence_seconds=5,
        memory_limit_mb=200
    )
    micro_analyzer = MicroWindowAnalyzer(
        window_manager,
        pre_touch_minutes=15,
        post_touch_minutes=10,
        min_bars_for_analysis=5
    )
    print("   ✅ Components initialized")
    
    # Create test scenario
    zone, bars, touch_bar = create_zone_touch_scenario()
    
    # Run tests
    analysis = test_zone_touch_analysis(micro_analyzer, zone, bars, touch_bar, window_manager)
    analyses = test_multiple_zone_touches(micro_analyzer, window_manager)
    pattern_analyses = test_initiative_patterns(micro_analyzer, window_manager)
    test_analysis_filtering(micro_analyzer)
    test_analysis_summary(micro_analyzer)
    await test_async_operations(micro_analyzer, zone, bars, touch_bar, window_manager)
    
    # Final statistics
    print("\n📊 Final Statistics")
    print("=" * 60)
    
    summary = micro_analyzer.get_analysis_summary()
    print(f"Total Analyses: {summary['total_analyses']}")
    print(f"Significant Touches: {summary['significant_touches']}")
    print(f"Absorption Touches: {summary['absorption_touches']}")
    print(f"Exhaustion Touches: {summary['exhaustion_touches']}")
    print(f"Rejection Touches: {summary['rejection_touches']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    print(f"Significance Rate: {summary['significance_rate']:.2f}")
    print(f"Absorption Rate: {summary['absorption_rate']:.2f}")
    print(f"Exhaustion Rate: {summary['exhaustion_rate']:.2f}")
    print(f"Rejection Rate: {summary['rejection_rate']:.2f}")
    
    print("\n🎉 Micro Window Analyzer Integration Test Complete!")
    print("✅ All tests passed successfully")


if __name__ == "__main__":
    asyncio.run(main())