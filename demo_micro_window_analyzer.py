#!/usr/bin/env python3
"""
Demonstration of Micro Window Analyzer for Zone Fade Strategy.

This script demonstrates how the Micro Window Analyzer provides pre/post zone touch
analysis for initiative/lack-of-initiative patterns, enabling precise detection of
absorption and exhaustion around key levels.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.micro_window_analyzer import (
    MicroWindowAnalyzer, InitiativeType, MicroWindowType
)
from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType


def create_zone_touch_scenarios():
    """Create multiple zone touch scenarios for demonstration."""
    print("📊 Creating Zone Touch Scenarios...")
    
    scenarios = []
    
    # Scenario 1: Supply Zone with Clear Rejection
    supply_zone = Zone(
        level=100.0,
        zone_type=ZoneType.PRIOR_DAY_HIGH,
        quality=2,
        strength=0.85,
        touches=1,
        created_at=datetime(2024, 1, 2, 9, 30),
        last_touch=datetime(2024, 1, 2, 9, 30)
    )
    
    # Create bars for supply zone rejection
    base_time = datetime(2024, 1, 2, 9, 30)
    supply_bars = []
    
    # Pre-touch: approaching zone
    for i in range(15):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=98.0 + i * 0.05,
            high=98.2 + i * 0.05,
            low=97.8 + i * 0.05,
            close=98.1 + i * 0.05,
            volume=1000 + i * 20
        )
        supply_bars.append(bar)
    
    # Touch: high volume rejection
    touch_bar = OHLCVBar(
        timestamp=base_time + timedelta(minutes=15),
        open=98.8,
        high=100.0,  # Touch zone
        low=98.5,
        close=99.2,  # Reject
        volume=5000  # High volume
    )
    supply_bars.append(touch_bar)
    
    # Post-touch: pullback
    for i in range(10):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=16 + i),
            open=99.2 - i * 0.03,
            high=99.4 - i * 0.03,
            low=99.0 - i * 0.03,
            close=99.1 - i * 0.03,
            volume=2000 - i * 50
        )
        supply_bars.append(bar)
    
    scenarios.append(("Supply Zone Rejection", supply_zone, supply_bars, touch_bar))
    
    # Scenario 2: Demand Zone with Absorption
    demand_zone = Zone(
        level=100.0,
        zone_type=ZoneType.PRIOR_DAY_LOW,
        quality=2,
        strength=0.90,
        touches=1,
        created_at=datetime(2024, 1, 2, 10, 0),
        last_touch=datetime(2024, 1, 2, 10, 0)
    )
    
    # Create bars for demand zone absorption
    base_time = datetime(2024, 1, 2, 10, 0)
    demand_bars = []
    
    # Pre-touch: approaching zone
    for i in range(12):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=101.5 - i * 0.08,
            high=101.7 - i * 0.08,
            low=101.3 - i * 0.08,
            close=101.6 - i * 0.08,
            volume=1200 + i * 30
        )
        demand_bars.append(bar)
    
    # Touch: absorption pattern
    touch_bar = OHLCVBar(
        timestamp=base_time + timedelta(minutes=12),
        open=100.5,
        high=101.0,  # Touch zone
        low=100.2,
        close=100.8,  # Small move
        volume=4000  # High volume
    )
    demand_bars.append(touch_bar)
    
    # Post-touch: absorption continues
    for i in range(8):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=13 + i),
            open=100.8 + i * 0.01,
            high=100.9 + i * 0.01,
            low=100.7 + i * 0.01,
            close=100.85 + i * 0.01,
            volume=3000 - i * 100  # Decreasing volume
        )
        demand_bars.append(bar)
    
    scenarios.append(("Demand Zone Absorption", demand_zone, demand_bars, touch_bar))
    
    # Scenario 3: Exhaustion Pattern
    exhaustion_zone = Zone(
        level=102.0,
        zone_type=ZoneType.WEEKLY_HIGH,
        quality=1,
        strength=0.75,
        touches=1,
        created_at=datetime(2024, 1, 2, 11, 0),
        last_touch=datetime(2024, 1, 2, 11, 0)
    )
    
    # Create bars for exhaustion pattern
    base_time = datetime(2024, 1, 2, 11, 0)
    exhaustion_bars = []
    
    # Pre-touch: strong momentum
    for i in range(10):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 0.08,
            high=100.3 + i * 0.08,
            low=99.7 + i * 0.08,
            close=100.2 + i * 0.08,
            volume=1500 + i * 50
        )
        exhaustion_bars.append(bar)
    
    # Touch: exhaustion
    touch_bar = OHLCVBar(
        timestamp=base_time + timedelta(minutes=10),
        open=100.8,
        high=102.0,  # Touch zone
        low=100.5,
        close=101.2,  # Large move
        volume=2000  # Lower volume
    )
    exhaustion_bars.append(touch_bar)
    
    # Post-touch: momentum loss
    for i in range(6):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=11 + i),
            open=101.2 - i * 0.02,
            high=101.4 - i * 0.02,
            low=101.0 - i * 0.02,
            close=101.1 - i * 0.02,
            volume=1000 - i * 50  # Decreasing volume
        )
        exhaustion_bars.append(bar)
    
    scenarios.append(("Exhaustion Pattern", exhaustion_zone, exhaustion_bars, touch_bar))
    
    print(f"   ✅ Created {len(scenarios)} zone touch scenarios")
    return scenarios


def demonstrate_zone_touch_analysis(micro_analyzer, scenarios, window_manager):
    """Demonstrate zone touch analysis for different scenarios."""
    print("\n🎯 Demonstrating Zone Touch Analysis")
    print("=" * 60)
    
    analyses = []
    
    for i, (scenario_name, zone, bars, touch_bar) in enumerate(scenarios):
        print(f"\n📊 Scenario {i + 1}: {scenario_name}")
        print("-" * 40)
        
        # Add bars to rolling windows
        for bar in bars:
            window_manager.add_bar(WindowType.SESSION_CONTEXT, bar, "SPY")
            window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
            if len(window_manager.get_window_bars(WindowType.OPENING_RANGE, "SPY")) < 30:
                window_manager.add_bar(WindowType.OPENING_RANGE, bar, "SPY")
        
        # Analyze zone touch
        analysis = micro_analyzer.analyze_zone_touch(zone, touch_bar, "SPY")
        
        if analysis:
            analyses.append(analysis)
            
            print(f"   Zone: {zone.zone_type.value} at ${zone.level:.2f}")
            print(f"   Touch Type: {analysis.touch_type}")
            print(f"   Touch Price: ${analysis.touch_price:.2f}")
            print(f"   Is Significant: {analysis.is_significant}")
            print(f"   Confidence Score: {analysis.confidence_score:.2f}")
            print(f"   Absorption Detected: {analysis.absorption_detected}")
            print(f"   Exhaustion Detected: {analysis.exhaustion_detected}")
            print(f"   Rejection Confirmed: {analysis.rejection_confirmed}")
            
            # Pre-touch analysis
            pre_touch = analysis.pre_touch_analysis
            print(f"\n   Pre-Touch Initiative:")
            print(f"     Type: {pre_touch.initiative_type.value}")
            print(f"     Strength: {pre_touch.strength_score:.2f}")
            print(f"     Volume Ratio: {pre_touch.volume_ratio:.2f}")
            print(f"     Price Momentum: {pre_touch.price_momentum:.4f}")
            print(f"     Rejection Clarity: {pre_touch.rejection_clarity:.2f}")
            print(f"     Absorption Signals: {pre_touch.absorption_signals}")
            print(f"     Exhaustion Signals: {pre_touch.exhaustion_signals}")
            print(f"     Volume Spike: {pre_touch.volume_spike}")
            print(f"     Volatility Spike: {pre_touch.volatility_spike}")
            
            # Post-touch analysis
            post_touch = analysis.post_touch_analysis
            print(f"\n   Post-Touch Initiative:")
            print(f"     Type: {post_touch.initiative_type.value}")
            print(f"     Strength: {post_touch.strength_score:.2f}")
            print(f"     Volume Ratio: {post_touch.volume_ratio:.2f}")
            print(f"     Price Momentum: {post_touch.price_momentum:.4f}")
            print(f"     Rejection Clarity: {post_touch.rejection_clarity:.2f}")
            print(f"     Absorption Signals: {post_touch.absorption_signals}")
            print(f"     Exhaustion Signals: {post_touch.exhaustion_signals}")
            print(f"     Volume Spike: {post_touch.volume_spike}")
            print(f"     Volatility Spike: {post_touch.volatility_spike}")
            
            # Pattern interpretation
            print(f"\n   Pattern Interpretation:")
            if analysis.absorption_detected:
                print(f"     🔄 Absorption detected - buyers/sellers absorbing at zone")
            if analysis.exhaustion_detected:
                print(f"     ⚡ Exhaustion detected - momentum losing steam")
            if analysis.rejection_confirmed:
                print(f"     🚫 Rejection confirmed - clear reversal signal")
            
            if pre_touch.initiative_type == InitiativeType.BULLISH and post_touch.initiative_type == InitiativeType.BEARISH:
                print(f"     📈➡️📉 Bullish to bearish shift - potential reversal")
            elif pre_touch.initiative_type == InitiativeType.BEARISH and post_touch.initiative_type == InitiativeType.BULLISH:
                print(f"     📉➡️📈 Bearish to bullish shift - potential reversal")
            elif pre_touch.initiative_type == InitiativeType.ABSORPTION:
                print(f"     🔄 Pre-touch absorption - zone acting as support/resistance")
            elif post_touch.initiative_type == InitiativeType.EXHAUSTION:
                print(f"     ⚡ Post-touch exhaustion - momentum fading")
        else:
            print(f"   ❌ Analysis failed for {scenario_name}")
    
    return analyses


def demonstrate_initiative_patterns(micro_analyzer, window_manager):
    """Demonstrate different initiative patterns."""
    print("\n📈 Demonstrating Initiative Patterns")
    print("=" * 60)
    
    patterns = [
        ("Absorption Pattern", "supply", 100.0, 99.0, "absorption"),
        ("Exhaustion Pattern", "demand", 101.0, 100.0, "exhaustion"),
        ("Bullish Momentum", "supply", 100.0, 99.0, "bullish"),
        ("Bearish Momentum", "demand", 101.0, 100.0, "bearish"),
        ("Neutral/Indecisive", "supply", 100.0, 99.0, "neutral")
    ]
    
    analyses = []
    
    for pattern_name, zone_type, high, low, pattern_type in patterns:
        print(f"\n🔍 {pattern_name}")
        print("-" * 30)
        
        # Create zone
        zone_type_enum = ZoneType.PRIOR_DAY_HIGH if zone_type == "supply" else ZoneType.PRIOR_DAY_LOW
        zone = Zone(
            level=high if zone_type == "supply" else low,
            zone_type=zone_type_enum,
            quality=2,
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
            if pattern_type == "absorption":
                # High volume, small moves
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.1,
                    low=99.9,
                    close=100.05,
                    volume=3000  # High volume
                )
            elif pattern_type == "exhaustion":
                # Decreasing volume, increasing moves
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.0 + i * 0.02,
                    low=99.9,
                    close=100.0 + i * 0.01,
                    volume=2000 - i * 100  # Decreasing volume
                )
            elif pattern_type == "bullish":
                # Strong upward momentum
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0 + i * 0.01,
                    high=100.2 + i * 0.01,
                    low=99.8 + i * 0.01,
                    close=100.1 + i * 0.01,
                    volume=1500 + i * 50
                )
            elif pattern_type == "bearish":
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
        
        # Add bars to rolling windows
        for bar in bars:
            window_manager.add_bar(WindowType.SESSION_CONTEXT, bar, "SPY")
            window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
            if len(window_manager.get_window_bars(WindowType.OPENING_RANGE, "SPY")) < 30:
                window_manager.add_bar(WindowType.OPENING_RANGE, bar, "SPY")
        
        # Analyze
        analysis = micro_analyzer.analyze_zone_touch(zone, touch_bar, "SPY")
        
        if analysis:
            analyses.append(analysis)
            pre_type = analysis.pre_touch_analysis.initiative_type.value
            post_type = analysis.post_touch_analysis.initiative_type.value
            
            print(f"   Pre-touch Initiative: {pre_type}")
            print(f"   Post-touch Initiative: {post_type}")
            print(f"   Confidence Score: {analysis.confidence_score:.2f}")
            print(f"   Absorption Detected: {analysis.absorption_detected}")
            print(f"   Exhaustion Detected: {analysis.exhaustion_detected}")
            print(f"   Rejection Confirmed: {analysis.rejection_confirmed}")
            
            # Pattern analysis
            if pre_type == "absorption":
                print(f"   🔄 Pre-touch absorption - high volume, small moves")
            elif pre_type == "exhaustion":
                print(f"   ⚡ Pre-touch exhaustion - decreasing volume, increasing moves")
            elif pre_type == "bullish":
                print(f"   📈 Pre-touch bullish momentum - strong upward pressure")
            elif pre_type == "bearish":
                print(f"   📉 Pre-touch bearish momentum - strong downward pressure")
            else:
                print(f"   ⚖️ Pre-touch neutral - balanced/indecisive")
        else:
            print(f"   ❌ Analysis failed")
    
    return analyses


def demonstrate_analysis_filtering(micro_analyzer):
    """Demonstrate analysis filtering methods."""
    print("\n🔍 Demonstrating Analysis Filtering")
    print("=" * 60)
    
    # Get different types of analyses
    significant = micro_analyzer.get_significant_touches()
    absorption = micro_analyzer.get_absorption_touches()
    exhaustion = micro_analyzer.get_exhaustion_touches()
    rejection = micro_analyzer.get_rejection_touches()
    recent = micro_analyzer.get_recent_analyses(limit=5)
    
    print(f"   📊 Analysis Counts:")
    print(f"     Total Analyses: {len(micro_analyzer.completed_analyses)}")
    print(f"     Significant Touches: {len(significant)}")
    print(f"     Absorption Touches: {len(absorption)}")
    print(f"     Exhaustion Touches: {len(exhaustion)}")
    print(f"     Rejection Touches: {len(rejection)}")
    print(f"     Recent Analyses: {len(recent)}")
    
    # Show significant touches
    if significant:
        print(f"\n   🎯 Significant Touches:")
        for i, analysis in enumerate(significant[:3]):  # Show first 3
            print(f"     {i + 1}. {analysis.zone.zone_type.value} - Confidence: {analysis.confidence_score:.2f}")
            print(f"        Pre: {analysis.pre_touch_analysis.initiative_type.value}")
            print(f"        Post: {analysis.post_touch_analysis.initiative_type.value}")
            print(f"        Absorption: {analysis.absorption_detected}")
            print(f"        Exhaustion: {analysis.exhaustion_detected}")
            print(f"        Rejection: {analysis.rejection_confirmed}")
    
    # Show absorption touches
    if absorption:
        print(f"\n   🔄 Absorption Touches:")
        for i, analysis in enumerate(absorption[:3]):  # Show first 3
            print(f"     {i + 1}. {analysis.zone.zone_type.value} - {analysis.touch_type}")
            print(f"        Confidence: {analysis.confidence_score:.2f}")
            print(f"        Pre-touch: {analysis.pre_touch_analysis.absorption_signals} signals")
            print(f"        Post-touch: {analysis.post_touch_analysis.absorption_signals} signals")
    
    # Show exhaustion touches
    if exhaustion:
        print(f"\n   ⚡ Exhaustion Touches:")
        for i, analysis in enumerate(exhaustion[:3]):  # Show first 3
            print(f"     {i + 1}. {analysis.zone.zone_type.value} - {analysis.touch_type}")
            print(f"        Confidence: {analysis.confidence_score:.2f}")
            print(f"        Pre-touch: {analysis.pre_touch_analysis.exhaustion_signals} signals")
            print(f"        Post-touch: {analysis.post_touch_analysis.exhaustion_signals} signals")
    
    # Show rejection touches
    if rejection:
        print(f"\n   🚫 Rejection Touches:")
        for i, analysis in enumerate(rejection[:3]):  # Show first 3
            print(f"     {i + 1}. {analysis.zone.zone_type.value} - {analysis.touch_type}")
            print(f"        Confidence: {analysis.confidence_score:.2f}")
            print(f"        Rejection Clarity: {analysis.post_touch_analysis.rejection_clarity:.2f}")


def demonstrate_analysis_summary(micro_analyzer):
    """Demonstrate analysis summary generation."""
    print("\n📊 Demonstrating Analysis Summary")
    print("=" * 60)
    
    summary = micro_analyzer.get_analysis_summary()
    
    print("   📈 Analysis Summary:")
    print(f"     Total Analyses: {summary['total_analyses']}")
    print(f"     Significant Touches: {summary['significant_touches']}")
    print(f"     Absorption Touches: {summary['absorption_touches']}")
    print(f"     Exhaustion Touches: {summary['exhaustion_touches']}")
    print(f"     Rejection Touches: {summary['rejection_touches']}")
    print(f"     Average Confidence: {summary['average_confidence']:.2f}")
    print(f"     Significance Rate: {summary['significance_rate']:.2f}")
    print(f"     Absorption Rate: {summary['absorption_rate']:.2f}")
    print(f"     Exhaustion Rate: {summary['exhaustion_rate']:.2f}")
    print(f"     Rejection Rate: {summary['rejection_rate']:.2f}")
    
    # Interpretation
    print(f"\n   📊 Pattern Analysis:")
    if summary['significance_rate'] > 0.7:
        print(f"     🎯 High significance rate - many meaningful zone touches")
    elif summary['significance_rate'] > 0.4:
        print(f"     ⚖️ Moderate significance rate - some meaningful zone touches")
    else:
        print(f"     📉 Low significance rate - few meaningful zone touches")
    
    if summary['absorption_rate'] > 0.3:
        print(f"     🔄 High absorption rate - zones acting as support/resistance")
    else:
        print(f"     📈 Low absorption rate - zones not acting as support/resistance")
    
    if summary['exhaustion_rate'] > 0.3:
        print(f"     ⚡ High exhaustion rate - momentum often fading at zones")
    else:
        print(f"     🚀 Low exhaustion rate - momentum often continuing through zones")
    
    if summary['rejection_rate'] > 0.5:
        print(f"     🚫 High rejection rate - zones often rejecting price")
    else:
        print(f"     📈 Low rejection rate - zones often allowing price through")


async def demonstrate_async_operations(micro_analyzer, scenarios, window_manager):
    """Demonstrate async operations."""
    print("\n🔄 Demonstrating Async Operations")
    print("=" * 60)
    
    # Simulate async processing of multiple zone touches
    async def process_zone_touches_async():
        tasks = []
        
        # Create async tasks for each scenario
        for i, (scenario_name, zone, bars, touch_bar) in enumerate(scenarios):
            task = asyncio.create_task(
                analyze_zone_touch_async(micro_analyzer, zone, touch_bar, "SPY", scenario_name)
            )
            tasks.append(task)
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks)
        
        successful_analyses = [r for r in results if r is not None]
        print(f"   ✅ Completed {len(successful_analyses)} async analyses")
        
        return successful_analyses
    
    async def analyze_zone_touch_async(analyzer, zone, touch_bar, symbol, scenario_name):
        """Async wrapper for zone touch analysis."""
        await asyncio.sleep(0.01)  # Simulate processing time
        # Use existing data from window manager
        return analyzer.analyze_zone_touch(zone, touch_bar, symbol)
    
    # Run async test
    analyses = await process_zone_touches_async()
    
    print(f"   Total analyses completed: {len(analyses)}")
    return analyses


def main():
    """Main demonstration function."""
    print("🚀 Micro Window Analyzer Demonstration")
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
    
    # Create test scenarios
    scenarios = create_zone_touch_scenarios()
    
    # Run demonstrations
    analyses = demonstrate_zone_touch_analysis(micro_analyzer, scenarios, window_manager)
    pattern_analyses = demonstrate_initiative_patterns(micro_analyzer, window_manager)
    demonstrate_analysis_filtering(micro_analyzer)
    demonstrate_analysis_summary(micro_analyzer)
    
    # Run async demonstration
    print("\n🔄 Running async demonstrations...")
    asyncio.run(demonstrate_async_operations(micro_analyzer, scenarios, window_manager))
    
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
    
    print("\n🎉 Micro Window Analyzer Demonstration Complete!")
    print("✅ All demonstrations completed successfully")


if __name__ == "__main__":
    main()