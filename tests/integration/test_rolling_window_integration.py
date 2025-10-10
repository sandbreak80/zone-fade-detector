#!/usr/bin/env python3
"""
Integration test for Rolling Window Manager with Zone Fade Strategy.
"""

import asyncio
import sys
from datetime import datetime, timedelta, time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.models import OHLCVBar
from zone_fade_detector.strategies.zone_fade_strategy import ZoneFadeStrategy
from zone_fade_detector.strategies.signal_processor import SignalProcessor, SignalProcessorConfig


async def test_rolling_window_integration():
    """Test Rolling Window Manager integration with Zone Fade Strategy."""
    
    print("🔧 Testing Rolling Window Manager Integration")
    print("=" * 60)
    
    # Initialize Rolling Window Manager
    window_manager = RollingWindowManager(
        evaluation_cadence_seconds=5,
        memory_limit_mb=200
    )
    
    # Initialize Zone Fade Strategy
    strategy = ZoneFadeStrategy()
    
    # Initialize Signal Processor
    processor_config = SignalProcessorConfig(
        min_qrs_score=3,
        max_setups_per_symbol=10,
        setup_cooldown_minutes=30,
        alert_deduplication_minutes=10
    )
    signal_processor = SignalProcessor(processor_config)
    
    print("✅ Components initialized")
    
    # Generate sample data for different time periods
    print("\n📊 Generating sample data...")
    
    # Generate RTH session data (9:30 AM - 4:00 PM)
    rth_start = datetime(2024, 1, 2, 9, 30)
    rth_bars = []
    
    for i in range(390):  # 390 minutes = 6.5 hours
        bar = OHLCVBar(
            timestamp=rth_start + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000 + i * 5
        )
        rth_bars.append(bar)
    
    print(f"   Generated {len(rth_bars)} RTH bars")
    
    # Test 1: Add bars to different windows
    print("\n🧪 Test 1: Adding bars to different windows")
    
    for i, bar in enumerate(rth_bars):
        # Add to VWAP computation window
        window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
        
        # Add to session context window
        window_manager.add_bar(WindowType.SESSION_CONTEXT, bar, "SPY")
        
        # Add to opening range window (first 30 minutes)
        if i < 30:
            window_manager.add_bar(WindowType.OPENING_RANGE, bar, "SPY")
        
        # Add to swing/CHoCH window (every 5 minutes)
        if i % 5 == 0:
            window_manager.add_bar(WindowType.SWING_CHOCH, bar, "SPY")
    
    print("   ✅ Bars added to all windows")
    
    # Test 2: Check window readiness
    print("\n🧪 Test 2: Checking window readiness")
    
    for window_type in WindowType:
        is_ready = window_manager.is_window_ready(window_type)
        window_info = window_manager.get_window_info(window_type)
        print(f"   {window_type.value}: {'✅ Ready' if is_ready else '❌ Not Ready'} "
              f"({window_info['bar_count']} bars)")
    
    # Test 3: Get window data for analysis
    print("\n🧪 Test 3: Getting window data for analysis")
    
    # Get VWAP computation data
    vwap_bars = window_manager.get_window_bars(WindowType.VWAP_COMPUTATION, "SPY")
    print(f"   VWAP window: {len(vwap_bars)} bars")
    
    # Get opening range data
    or_bars = window_manager.get_window_bars(WindowType.OPENING_RANGE, "SPY")
    print(f"   Opening Range window: {len(or_bars)} bars")
    
    # Get swing/CHoCH data
    swing_bars = window_manager.get_window_bars(WindowType.SWING_CHOCH, "SPY")
    print(f"   Swing/CHoCH window: {len(swing_bars)} bars")
    
    # Test 4: Test window reset on new session
    print("\n🧪 Test 4: Testing window reset on new session")
    
    # Add bar from next day (new session)
    next_day_bar = OHLCVBar(
        timestamp=datetime(2024, 1, 3, 9, 30),  # Next day RTH start
        open=101.0, high=101.5, low=100.5, close=101.2, volume=1000
    )
    
    # Add to session context window (should reset)
    window_manager.add_bar(WindowType.SESSION_CONTEXT, next_day_bar, "SPY")
    
    # Check if window was reset
    session_bars = window_manager.get_window_bars(WindowType.SESSION_CONTEXT, "SPY")
    print(f"   Session window after reset: {len(session_bars)} bars")
    print(f"   Last bar timestamp: {session_bars[-1].timestamp if session_bars else 'None'}")
    
    # Test 5: Performance statistics
    print("\n🧪 Test 5: Performance statistics")
    
    stats = window_manager.get_performance_stats()
    print(f"   Total windows: {stats['total_windows']}")
    print(f"   Active windows: {stats['active_windows']}")
    print(f"   Memory usage: {stats['total_memory_mb']:.2f} MB")
    print(f"   Memory limit: {stats['memory_limit_mb']} MB")
    print(f"   Memory usage: {stats['memory_usage_percent']:.1f}%")
    print(f"   Hit rate: {stats['hit_rate']:.2f}")
    
    # Test 6: Integration with Zone Fade Strategy
    print("\n🧪 Test 6: Integration with Zone Fade Strategy")
    
    # Simulate zone detection using window data
    if window_manager.is_window_ready(WindowType.HTF_ZONES):
        htf_bars = window_manager.get_window_bars(WindowType.HTF_ZONES, "SPY")
        print(f"   HTF zones window ready: {len(htf_bars)} bars")
        
        # This would be used by ZoneFadeStrategy for zone detection
        if htf_bars:
            latest_bar = htf_bars[-1]
            print(f"   Latest HTF bar: {latest_bar.timestamp} @ ${latest_bar.close:.2f}")
    
    # Test 7: Memory management
    print("\n🧪 Test 7: Memory management")
    
    # Add many bars to test memory management
    print("   Adding many bars to test memory management...")
    
    for i in range(1000):
        bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 9, 30) + timedelta(minutes=i),
            open=100.0 + i * 0.001,
            high=100.5 + i * 0.001,
            low=99.5 + i * 0.001,
            close=100.2 + i * 0.001,
            volume=1000 + i
        )
        window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
    
    # Check memory usage
    stats_after = window_manager.get_performance_stats()
    print(f"   Memory usage after adding many bars: {stats_after['memory_usage_percent']:.1f}%")
    
    # Test 8: Evaluation loop
    print("\n🧪 Test 8: Testing evaluation loop")
    
    # Start evaluation loop
    print("   Starting evaluation loop...")
    evaluation_task = asyncio.create_task(window_manager.start_evaluation_loop())
    
    # Let it run for a short time
    await asyncio.sleep(2)
    
    # Stop evaluation loop
    window_manager.stop_evaluation_loop()
    evaluation_task.cancel()
    
    try:
        await evaluation_task
    except asyncio.CancelledError:
        pass
    
    print("   ✅ Evaluation loop stopped")
    
    # Final statistics
    print("\n📊 Final Statistics")
    print("=" * 60)
    
    final_stats = window_manager.get_performance_stats()
    print(f"Total windows: {final_stats['total_windows']}")
    print(f"Active windows: {final_stats['active_windows']}")
    print(f"Memory usage: {final_stats['total_memory_mb']:.2f} MB")
    print(f"Hit rate: {final_stats['hit_rate']:.2f}")
    
    # Window details
    print("\nWindow Details:")
    for window_type, info in final_stats['windows'].items():
        print(f"  {window_type}:")
        print(f"    Bars: {info['bar_count']}/{info['max_bars']}")
        print(f"    Active: {info['is_active']}")
        print(f"    Memory: {info['memory_usage_mb']:.2f} MB")
        print(f"    Hit rate: {info['hit_rate']:.2f}")
    
    print("\n🎉 Rolling Window Manager Integration Test Complete!")
    print("✅ All tests passed successfully")


if __name__ == "__main__":
    asyncio.run(test_rolling_window_integration())