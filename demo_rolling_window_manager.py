#!/usr/bin/env python3
"""
Demonstration of Rolling Window Manager for Zone Fade Strategy.

This script demonstrates how the Rolling Window Manager works with different
window types and how it integrates with the Zone Fade strategy.
"""

import asyncio
import sys
from datetime import datetime, timedelta, time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.models import OHLCVBar


def create_sample_data():
    """Create sample market data for demonstration."""
    print("📊 Creating sample market data...")
    
    # Create RTH session data (9:30 AM - 4:00 PM ET)
    rth_start = datetime(2024, 1, 2, 9, 30)
    bars = []
    
    # Generate 390 minutes of data (6.5 hours)
    for i in range(390):
        # Simulate price movement with some volatility
        base_price = 100.0 + i * 0.01
        volatility = 0.5 * (i % 10) / 10  # Some volatility
        
        bar = OHLCVBar(
            timestamp=rth_start + timedelta(minutes=i),
            open=base_price + volatility,
            high=base_price + volatility + 0.3,
            low=base_price + volatility - 0.2,
            close=base_price + volatility + 0.1,
            volume=1000 + i * 5 + (i % 20) * 10  # Volume with some spikes
        )
        bars.append(bar)
    
    print(f"   ✅ Generated {len(bars)} bars from {bars[0].timestamp} to {bars[-1].timestamp}")
    return bars


def demonstrate_window_types(manager, bars):
    """Demonstrate different window types."""
    print("\n🪟 Demonstrating Window Types")
    print("=" * 50)
    
    # Add bars to different windows
    for i, bar in enumerate(bars):
        # VWAP computation (full session)
        manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
        
        # Session context (rolling RTH)
        manager.add_bar(WindowType.SESSION_CONTEXT, bar, "SPY")
        
        # Opening range (first 30 minutes)
        if i < 30:
            manager.add_bar(WindowType.OPENING_RANGE, bar, "SPY")
        
        # Swing/CHoCH (every 5 minutes for structure)
        if i % 5 == 0:
            manager.add_bar(WindowType.SWING_CHOCH, bar, "SPY")
        
        # Initiative analysis (around potential zone touches)
        if i % 15 == 0:  # Every 15 minutes
            manager.add_bar(WindowType.INITIATIVE_ANALYSIS, bar, "SPY")
    
    # Show window status
    for window_type in WindowType:
        info = manager.get_window_info(window_type)
        status = "✅ Ready" if manager.is_window_ready(window_type) else "⏳ Building"
        print(f"   {window_type.value:20} {status:10} ({info['bar_count']:3d} bars)")


def demonstrate_window_data_access(manager):
    """Demonstrate accessing window data."""
    print("\n📈 Demonstrating Window Data Access")
    print("=" * 50)
    
    # Get data from different windows
    windows_to_check = [
        WindowType.VWAP_COMPUTATION,
        WindowType.OPENING_RANGE,
        WindowType.SWING_CHOCH,
        WindowType.INITIATIVE_ANALYSIS
    ]
    
    for window_type in windows_to_check:
        bars = manager.get_window_bars(window_type, "SPY")
        if bars:
            print(f"\n   {window_type.value}:")
            print(f"     Total bars: {len(bars)}")
            print(f"     Time range: {bars[0].timestamp} to {bars[-1].timestamp}")
            print(f"     Price range: ${min(b.low for b in bars):.2f} - ${max(b.high for b in bars):.2f}")
            print(f"     Volume range: {min(b.volume for b in bars):,} - {max(b.volume for b in bars):,}")
        else:
            print(f"   {window_type.value}: No data")


def demonstrate_timestamp_based_access(manager, bars):
    """Demonstrate timestamp-based window access."""
    print("\n⏰ Demonstrating Timestamp-Based Access")
    print("=" * 50)
    
    # Test different timestamps
    test_timestamps = [
        bars[30].timestamp,   # 30 minutes in
        bars[120].timestamp,  # 2 hours in
        bars[240].timestamp,  # 4 hours in
    ]
    
    for timestamp in test_timestamps:
        print(f"\n   Window data at {timestamp}:")
        
        # Get swing/CHoCH window for this timestamp
        swing_bars = manager.get_window_for_timestamp(WindowType.SWING_CHOCH, timestamp)
        print(f"     Swing/CHoCH: {len(swing_bars)} bars")
        
        # Get initiative analysis window
        initiative_bars = manager.get_window_for_timestamp(WindowType.INITIATIVE_ANALYSIS, timestamp)
        print(f"     Initiative: {len(initiative_bars)} bars")


def demonstrate_memory_management(manager):
    """Demonstrate memory management features."""
    print("\n💾 Demonstrating Memory Management")
    print("=" * 50)
    
    # Get initial stats
    stats = manager.get_performance_stats()
    print(f"   Initial memory usage: {stats['total_memory_mb']:.2f} MB")
    print(f"   Memory limit: {stats['memory_limit_mb']} MB")
    print(f"   Usage: {stats['memory_usage_percent']:.1f}%")
    
    # Add many bars to test memory management
    print("\n   Adding many bars to test memory management...")
    
    for i in range(500):
        bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 9, 30) + timedelta(minutes=i),
            open=100.0 + i * 0.001,
            high=100.5 + i * 0.001,
            low=99.5 + i * 0.001,
            close=100.2 + i * 0.001,
            volume=1000 + i
        )
        manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
    
    # Check memory after adding many bars
    stats_after = manager.get_performance_stats()
    print(f"   Memory usage after: {stats_after['total_memory_mb']:.2f} MB")
    print(f"   Usage: {stats_after['memory_usage_percent']:.1f}%")
    
    # Show window details
    print("\n   Window details:")
    for window_type, info in stats_after['windows'].items():
        if info['bar_count'] > 0:
            print(f"     {window_type}: {info['bar_count']} bars, {info['memory_usage_mb']:.2f} MB")


def demonstrate_session_reset(manager):
    """Demonstrate session reset functionality."""
    print("\n🔄 Demonstrating Session Reset")
    print("=" * 50)
    
    # Add bar from next day (new session)
    next_day_bar = OHLCVBar(
        timestamp=datetime(2024, 1, 3, 9, 30),  # Next day RTH start
        open=101.0, high=101.5, low=100.5, close=101.2, volume=1000
    )
    
    print(f"   Adding bar from next session: {next_day_bar.timestamp}")
    
    # Add to session context window (should reset)
    manager.add_bar(WindowType.SESSION_CONTEXT, next_day_bar, "SPY")
    
    # Check if window was reset
    session_bars = manager.get_window_bars(WindowType.SESSION_CONTEXT, "SPY")
    print(f"   Session window bars after reset: {len(session_bars)}")
    
    if session_bars:
        print(f"   Last bar timestamp: {session_bars[-1].timestamp}")
        print(f"   Last bar price: ${session_bars[-1].close:.2f}")


async def demonstrate_evaluation_loop(manager):
    """Demonstrate the evaluation loop."""
    print("\n🔄 Demonstrating Evaluation Loop")
    print("=" * 50)
    
    print("   Starting evaluation loop...")
    
    # Start evaluation loop
    evaluation_task = asyncio.create_task(manager.start_evaluation_loop())
    
    # Let it run for a few seconds
    await asyncio.sleep(3)
    
    # Stop evaluation loop
    manager.stop_evaluation_loop()
    evaluation_task.cancel()
    
    try:
        await evaluation_task
    except asyncio.CancelledError:
        pass
    
    print("   ✅ Evaluation loop stopped")


def main():
    """Main demonstration function."""
    print("🚀 Rolling Window Manager Demonstration")
    print("=" * 60)
    
    # Initialize Rolling Window Manager
    print("🔧 Initializing Rolling Window Manager...")
    manager = RollingWindowManager(
        evaluation_cadence_seconds=2,
        memory_limit_mb=100
    )
    print("   ✅ Manager initialized")
    
    # Create sample data
    bars = create_sample_data()
    
    # Demonstrate different aspects
    demonstrate_window_types(manager, bars)
    demonstrate_window_data_access(manager)
    demonstrate_timestamp_based_access(manager, bars)
    demonstrate_memory_management(manager)
    demonstrate_session_reset(manager)
    
    # Run async demonstration
    print("\n🔄 Running async demonstrations...")
    asyncio.run(demonstrate_evaluation_loop(manager))
    
    # Final statistics
    print("\n📊 Final Statistics")
    print("=" * 60)
    
    final_stats = manager.get_performance_stats()
    print(f"Total windows: {final_stats['total_windows']}")
    print(f"Active windows: {final_stats['active_windows']}")
    print(f"Memory usage: {final_stats['total_memory_mb']:.2f} MB")
    print(f"Memory limit: {final_stats['memory_limit_mb']} MB")
    print(f"Memory usage: {final_stats['memory_usage_percent']:.1f}%")
    print(f"Hit rate: {final_stats['hit_rate']:.2f}")
    
    # Window details
    print("\nWindow Details:")
    for window_type, info in final_stats['windows'].items():
        if info['bar_count'] > 0:
            print(f"  {window_type}:")
            print(f"    Bars: {info['bar_count']}/{info['max_bars']}")
            print(f"    Active: {info['is_active']}")
            print(f"    Memory: {info['memory_usage_mb']:.2f} MB")
            print(f"    Hit rate: {info['hit_rate']:.2f}")
    
    print("\n🎉 Rolling Window Manager Demonstration Complete!")
    print("✅ All demonstrations completed successfully")


if __name__ == "__main__":
    main()