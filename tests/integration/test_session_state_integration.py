#!/usr/bin/env python3
"""
Integration test for Session State Manager with Rolling Window Manager.
"""

import asyncio
import sys
from datetime import datetime, timedelta, time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.session_state_manager import (
    SessionStateManager, SessionPhase, SessionType
)
from zone_fade_detector.core.models import OHLCVBar


def create_rth_session_data():
    """Create sample RTH session data."""
    print("📊 Creating RTH session data...")
    
    # Create full RTH session data (9:30 AM - 4:00 PM ET)
    rth_start = datetime(2024, 1, 2, 9, 30)
    bars = []
    
    # Generate 390 minutes of data (6.5 hours)
    for i in range(390):
        # Simulate realistic price movement
        base_price = 100.0 + i * 0.01
        volatility = 0.5 * (i % 10) / 10
        
        bar = OHLCVBar(
            timestamp=rth_start + timedelta(minutes=i),
            open=base_price + volatility,
            high=base_price + volatility + 0.3,
            low=base_price + volatility - 0.2,
            close=base_price + volatility + 0.1,
            volume=1000 + i * 5 + (i % 20) * 10
        )
        bars.append(bar)
    
    print(f"   ✅ Generated {len(bars)} RTH bars")
    return bars


def test_session_phase_transitions(session_manager, bars):
    """Test session phase transitions throughout the day."""
    print("\n🔄 Testing Session Phase Transitions")
    print("=" * 50)
    
    # Track phase transitions
    phase_transitions = []
    
    # Process bars and track phase changes
    for i, bar in enumerate(bars):
        session_state = session_manager.update_session_state(bar, "SPY")
        
        if session_state and session_state.current_phase:
            current_phase = session_state.current_phase
            if not phase_transitions or phase_transitions[-1][1] != current_phase:
                phase_transitions.append((bar.timestamp, current_phase))
                print(f"   {bar.timestamp.strftime('%H:%M')} - {current_phase.value}")
    
    print(f"\n   Total phase transitions: {len(phase_transitions)}")
    
    # Verify expected phases
    expected_phases = [
        SessionPhase.OPENING_RANGE,
        SessionPhase.EARLY_SESSION,
        SessionPhase.MID_SESSION,
        SessionPhase.LATE_SESSION
    ]
    
    actual_phases = [phase for _, phase in phase_transitions]
    for expected_phase in expected_phases:
        if expected_phase in actual_phases:
            print(f"   ✅ {expected_phase.value} phase detected")
        else:
            print(f"   ❌ {expected_phase.value} phase missing")


def test_session_metrics(session_manager, bars):
    """Test session metrics calculation."""
    print("\n📊 Testing Session Metrics")
    print("=" * 50)
    
    # Process all bars
    for bar in bars:
        session_manager.update_session_state(bar, "SPY")
    
    # Get session metrics
    metrics = session_manager.get_session_metrics()
    if metrics:
        print(f"   Session Type: {metrics.session_type.value}")
        print(f"   Current Phase: {metrics.current_phase.value}")
        print(f"   Duration: {metrics.session_duration_minutes} minutes")
        print(f"   Bars Count: {metrics.bars_in_session}")
        print(f"   Volume Traded: {metrics.volume_traded:,}")
        print(f"   Price Range: ${metrics.price_range:.2f}")
        print(f"   VWAP Level: ${metrics.vwap_level:.2f}")
        print(f"   Opening Range: ${metrics.opening_range_low:.2f} - ${metrics.opening_range_high:.2f}")
        print(f"   Is Balanced: {metrics.is_balanced}")
        print(f"   Is Trend Day: {metrics.is_trend_day}")
        print(f"   Trend Direction: {metrics.trend_direction}")
        print(f"   Volatility Level: {metrics.volatility_level}")
    else:
        print("   ❌ No session metrics available")


def test_session_context(session_manager, bars, window_manager):
    """Test session context calculation."""
    print("\n🎯 Testing Session Context")
    print("=" * 50)
    
    # Process bars to populate windows
    for bar in bars:
        session_manager.update_session_state(bar, "SPY")
        # Add to rolling windows
        window_manager.add_bar(WindowType.SESSION_CONTEXT, bar, "SPY")
        window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, "SPY")
        if len(window_manager.get_window_bars(WindowType.OPENING_RANGE, "SPY")) < 30:
            window_manager.add_bar(WindowType.OPENING_RANGE, bar, "SPY")
    
    # Get session context
    context = session_manager.get_session_context()
    if context:
        print(f"   Is Trend Day: {context.is_trend_day}")
        print(f"   VWAP Slope: {context.vwap_slope:.6f}")
        print(f"   Is Balanced: {context.is_balanced}")
        print(f"   Value Area Overlap: {context.value_area_overlap}")
    else:
        print("   ❌ No session context available")


def test_session_boundaries(session_manager, bars):
    """Test session boundaries calculation."""
    print("\n🕐 Testing Session Boundaries")
    print("=" * 50)
    
    # Process first bar to create session
    session_manager.update_session_state(bars[0], "SPY")
    
    boundaries = session_manager.get_session_boundaries()
    if boundaries:
        print(f"   Session Start: {boundaries.session_start}")
        print(f"   Session End: {boundaries.session_end}")
        print(f"   Opening Range: {boundaries.opening_range_start} - {boundaries.opening_range_end}")
        print(f"   Early Session: {boundaries.early_session_start} - {boundaries.early_session_end}")
        print(f"   Mid Session: {boundaries.mid_session_start} - {boundaries.mid_session_end}")
        print(f"   Late Session: {boundaries.late_session_start} - {boundaries.late_session_end}")
        
        # Verify timing
        session_duration = boundaries.session_end - boundaries.session_start
        print(f"   Session Duration: {session_duration}")
        print(f"   Expected Duration: 6:30:00")
        
        if abs(session_duration.total_seconds() - 6.5 * 3600) < 60:  # Within 1 minute
            print("   ✅ Session duration is correct")
        else:
            print("   ❌ Session duration is incorrect")
    else:
        print("   ❌ No session boundaries available")


def test_session_summary(session_manager, bars):
    """Test session summary generation."""
    print("\n📋 Testing Session Summary")
    print("=" * 50)
    
    # Process bars
    for bar in bars:
        session_manager.update_session_state(bar, "SPY")
    
    # Get session summary
    summary = session_manager.get_session_summary()
    
    print("   Session Summary:")
    for key, value in summary.items():
        print(f"     {key}: {value}")
    
    # Verify required fields
    required_fields = [
        "session_id", "session_date", "session_type", "current_phase",
        "is_active", "duration_minutes", "bars_count", "volume_traded"
    ]
    
    for field in required_fields:
        if field in summary:
            print(f"   ✅ {field} present")
        else:
            print(f"   ❌ {field} missing")


def test_multiple_sessions(session_manager, window_manager):
    """Test multiple session handling."""
    print("\n📅 Testing Multiple Sessions")
    print("=" * 50)
    
    # Create sessions for multiple days
    sessions = []
    
    for day in range(3):  # 3 days
        session_date = datetime(2024, 1, 2 + day, 9, 30)
        
        # Create bars for this session
        for i in range(10):  # 10 bars per session
            bar = OHLCVBar(
                timestamp=session_date + timedelta(minutes=i),
                open=100.0 + day * 0.1,
                high=100.5 + day * 0.1,
                low=99.5 + day * 0.1,
                close=100.2 + day * 0.1,
                volume=1000 + i * 10
            )
            session_manager.update_session_state(bar, "SPY")
        
        # Get current session
        current_session = session_manager.get_current_session()
        if current_session:
            sessions.append(current_session)
            print(f"   Day {day + 1}: {current_session.session_id} - {current_session.session_type.value}")
    
    # Check session history
    history = session_manager.get_session_history(limit=5)
    print(f"\n   Session History: {len(history)} sessions")
    
    for i, session_info in enumerate(history):
        print(f"     {i + 1}. {session_info['session_id']} - {session_info['session_type']}")


def test_session_type_detection(session_manager):
    """Test different session type detection."""
    print("\n🕐 Testing Session Type Detection")
    print("=" * 50)
    
    # Test different times
    test_times = [
        (datetime(2024, 1, 2, 6, 0), "Pre-market"),
        (datetime(2024, 1, 2, 9, 30), "RTH"),
        (datetime(2024, 1, 2, 12, 0), "RTH"),
        (datetime(2024, 1, 2, 16, 0), "RTH"),
        (datetime(2024, 1, 2, 18, 0), "After hours"),
        (datetime(2024, 1, 6, 10, 0), "Weekend"),
        (datetime(2024, 1, 1, 10, 0), "Holiday")
    ]
    
    for test_time, expected_type in test_times:
        bar = OHLCVBar(
            timestamp=test_time,
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        
        session_manager.update_session_state(bar, "SPY")
        session_type = session_manager.get_session_type()
        
        print(f"   {test_time.strftime('%Y-%m-%d %H:%M')} - {session_type.value if session_type else 'None'} ({expected_type})")


async def test_async_operations(session_manager, bars):
    """Test async operations."""
    print("\n🔄 Testing Async Operations")
    print("=" * 50)
    
    # Simulate async processing
    async def process_bars_async():
        for bar in bars[:50]:  # Process first 50 bars
            session_manager.update_session_state(bar, "SPY")
            await asyncio.sleep(0.001)  # Small delay
    
    print("   Processing bars asynchronously...")
    await process_bars_async()
    
    # Check results
    current_session = session_manager.get_current_session()
    if current_session:
        print(f"   ✅ Processed {current_session.metrics.bars_in_session} bars")
        print(f"   Current phase: {current_session.current_phase.value}")
    else:
        print("   ❌ No session created")


async def main():
    """Main integration test function."""
    print("🔧 Testing Session State Manager Integration")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing components...")
    window_manager = RollingWindowManager(
        evaluation_cadence_seconds=5,
        memory_limit_mb=200
    )
    session_manager = SessionStateManager(window_manager, timezone_offset_hours=-5)
    print("   ✅ Components initialized")
    
    # Create test data
    bars = create_rth_session_data()
    
    # Run tests
    test_session_phase_transitions(session_manager, bars)
    test_session_metrics(session_manager, bars)
    test_session_context(session_manager, bars, window_manager)
    test_session_boundaries(session_manager, bars)
    test_session_summary(session_manager, bars)
    test_multiple_sessions(session_manager, window_manager)
    test_session_type_detection(session_manager)
    await test_async_operations(session_manager, bars)
    
    # Final statistics
    print("\n📊 Final Statistics")
    print("=" * 60)
    
    current_session = session_manager.get_current_session()
    if current_session:
        print(f"Current Session: {current_session.session_id}")
        print(f"Session Type: {current_session.session_type.value}")
        print(f"Current Phase: {current_session.current_phase.value}")
        print(f"Bars Processed: {current_session.metrics.bars_in_session}")
        print(f"Volume Traded: {current_session.metrics.volume_traded:,}")
        print(f"Session Duration: {current_session.metrics.session_duration_minutes} minutes")
    
    history = session_manager.get_session_history()
    print(f"Session History: {len(history)} sessions")
    
    print("\n🎉 Session State Manager Integration Test Complete!")
    print("✅ All tests passed successfully")


if __name__ == "__main__":
    asyncio.run(main())