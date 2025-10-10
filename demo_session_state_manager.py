#!/usr/bin/env python3
"""
Demonstration of Session State Manager for Zone Fade Strategy.

This script demonstrates how the Session State Manager tracks RTH sessions,
phase transitions, and provides session-specific context for the Zone Fade strategy.
"""

import asyncio
import sys
from datetime import datetime, timedelta, time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.session_state_manager import (
    SessionStateManager, SessionPhase, SessionType
)
from zone_fade_detector.core.models import OHLCVBar


def create_full_trading_day_data():
    """Create data for a full trading day including pre-market, RTH, and after-hours."""
    print("📊 Creating full trading day data...")
    
    bars = []
    base_date = datetime(2024, 1, 2)
    
    # Pre-market (4:00 AM - 9:30 AM)
    pre_market_start = base_date.replace(hour=4, minute=0)
    for i in range(330):  # 5.5 hours
        bar = OHLCVBar(
            timestamp=pre_market_start + timedelta(minutes=i),
            open=99.0 + i * 0.001,
            high=99.3 + i * 0.001,
            low=98.7 + i * 0.001,
            close=99.1 + i * 0.001,
            volume=500 + i * 2
        )
        bars.append(bar)
    
    # RTH (9:30 AM - 4:00 PM)
    rth_start = base_date.replace(hour=9, minute=30)
    for i in range(390):  # 6.5 hours
        bar = OHLCVBar(
            timestamp=rth_start + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000 + i * 5
        )
        bars.append(bar)
    
    # After-hours (4:00 PM - 8:00 PM)
    after_hours_start = base_date.replace(hour=16, minute=0)
    for i in range(240):  # 4 hours
        bar = OHLCVBar(
            timestamp=after_hours_start + timedelta(minutes=i),
            open=103.9 + i * 0.001,
            high=104.2 + i * 0.001,
            low=103.6 + i * 0.001,
            close=104.0 + i * 0.001,
            volume=300 + i * 2
        )
        bars.append(bar)
    
    print(f"   ✅ Generated {len(bars)} bars")
    print(f"   Pre-market: {len(bars[:330])} bars")
    print(f"   RTH: {len(bars[330:720])} bars")
    print(f"   After-hours: {len(bars[720:])} bars")
    
    return bars


def demonstrate_session_phases(session_manager, bars):
    """Demonstrate session phase detection throughout the day."""
    print("\n🔄 Demonstrating Session Phases")
    print("=" * 50)
    
    phase_transitions = []
    current_phase = None
    
    # Process bars and track phase changes
    for i, bar in enumerate(bars):
        session_state = session_manager.update_session_state(bar, "SPY")
        
        if session_state and session_state.current_phase:
            new_phase = session_state.current_phase
            if new_phase != current_phase:
                phase_transitions.append((bar.timestamp, new_phase, session_state.session_type))
                current_phase = new_phase
                
                # Print phase transition
                phase_name = new_phase.value.replace('_', ' ').title()
                session_name = session_state.session_type.value.replace('_', ' ').title()
                print(f"   {bar.timestamp.strftime('%H:%M')} - {phase_name} ({session_name})")
    
    print(f"\n   Total phase transitions: {len(phase_transitions)}")
    
    # Group by session type
    rth_transitions = [t for t in phase_transitions if t[2] == SessionType.RTH]
    print(f"   RTH phase transitions: {len(rth_transitions)}")
    
    return phase_transitions


def demonstrate_session_metrics(session_manager, bars):
    """Demonstrate session metrics calculation."""
    print("\n📊 Demonstrating Session Metrics")
    print("=" * 50)
    
    # Process all bars
    for bar in bars:
        session_manager.update_session_state(bar, "SPY")
    
    # Get current session
    current_session = session_manager.get_current_session()
    if not current_session:
        print("   ❌ No active session")
        return
    
    metrics = current_session.metrics
    
    print(f"   Session ID: {current_session.session_id}")
    print(f"   Session Date: {current_session.session_date.strftime('%Y-%m-%d')}")
    print(f"   Session Type: {metrics.session_type.value}")
    print(f"   Current Phase: {metrics.current_phase.value}")
    print(f"   Duration: {metrics.session_duration_minutes} minutes")
    print(f"   Bars Processed: {metrics.bars_in_session}")
    print(f"   Volume Traded: {metrics.volume_traded:,}")
    print(f"   Price Range: ${metrics.price_range:.2f}")
    print(f"   VWAP Level: ${metrics.vwap_level:.2f}")
    print(f"   Opening Range: ${metrics.opening_range_low:.2f} - ${metrics.opening_range_high:.2f}")
    print(f"   Opening Range Volume: {metrics.opening_range_volume:,}")
    print(f"   Is Balanced: {metrics.is_balanced}")
    print(f"   Is Trend Day: {metrics.is_trend_day}")
    print(f"   Trend Direction: {metrics.trend_direction}")
    print(f"   Volatility Level: {metrics.volatility_level}")


def demonstrate_session_context(session_manager, bars, window_manager):
    """Demonstrate session context calculation."""
    print("\n🎯 Demonstrating Session Context")
    print("=" * 50)
    
    # Process RTH bars and populate windows
    for bar in bars[330:720]:  # RTH bars
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
        
        # Interpret context
        if context.is_trend_day:
            trend_direction = "Bullish" if context.vwap_slope > 0 else "Bearish"
            print(f"   Market Analysis: {trend_direction} trend day")
        elif context.is_balanced:
            print(f"   Market Analysis: Balanced market")
        else:
            print(f"   Market Analysis: Neutral market")
    else:
        print("   ❌ No session context available")


def demonstrate_session_boundaries(session_manager, bars):
    """Demonstrate session boundaries calculation."""
    print("\n🕐 Demonstrating Session Boundaries")
    print("=" * 50)
    
    # Process RTH bars to create session
    for bar in bars[330:360]:  # First 30 minutes of RTH
        session_manager.update_session_state(bar, "SPY")
    
    boundaries = session_manager.get_session_boundaries()
    if boundaries:
        print("   Session Boundaries:")
        print(f"     Session: {boundaries.session_start.strftime('%H:%M')} - {boundaries.session_end.strftime('%H:%M')}")
        print(f"     Opening Range: {boundaries.opening_range_start.strftime('%H:%M')} - {boundaries.opening_range_end.strftime('%H:%M')}")
        print(f"     Early Session: {boundaries.early_session_start.strftime('%H:%M')} - {boundaries.early_session_end.strftime('%H:%M')}")
        print(f"     Mid Session: {boundaries.mid_session_start.strftime('%H:%M')} - {boundaries.mid_session_end.strftime('%H:%M')}")
        print(f"     Late Session: {boundaries.late_session_start.strftime('%H:%M')} - {boundaries.late_session_end.strftime('%H:%M')}")
        
        # Calculate durations
        session_duration = boundaries.session_end - boundaries.session_start
        or_duration = boundaries.opening_range_end - boundaries.opening_range_start
        
        print(f"\n   Durations:")
        print(f"     Total Session: {session_duration}")
        print(f"     Opening Range: {or_duration}")
        print(f"     Early Session: {boundaries.early_session_end - boundaries.early_session_start}")
        print(f"     Mid Session: {boundaries.mid_session_end - boundaries.mid_session_start}")
        print(f"     Late Session: {boundaries.late_session_end - boundaries.late_session_start}")
    else:
        print("   ❌ No session boundaries available")


def demonstrate_session_summary(session_manager, bars):
    """Demonstrate session summary generation."""
    print("\n📋 Demonstrating Session Summary")
    print("=" * 50)
    
    # Process all bars
    for bar in bars:
        session_manager.update_session_state(bar, "SPY")
    
    # Get session summary
    summary = session_manager.get_session_summary()
    
    print("   Session Summary:")
    for key, value in summary.items():
        if isinstance(value, str) and len(value) > 50:
            print(f"     {key}: {value[:50]}...")
        else:
            print(f"     {key}: {value}")


def demonstrate_multiple_days(session_manager, window_manager):
    """Demonstrate multiple trading days."""
    print("\n📅 Demonstrating Multiple Trading Days")
    print("=" * 50)
    
    # Create sessions for multiple days
    for day in range(3):
        print(f"\n   Day {day + 1}:")
        
        # Create RTH session for this day
        session_date = datetime(2024, 1, 2 + day, 9, 30)
        
        # Add opening bar
        opening_bar = OHLCVBar(
            timestamp=session_date,
            open=100.0 + day * 0.5,
            high=100.5 + day * 0.5,
            low=99.5 + day * 0.5,
            close=100.2 + day * 0.5,
            volume=1000
        )
        session_manager.update_session_state(opening_bar, "SPY")
        
        # Add closing bar
        closing_bar = OHLCVBar(
            timestamp=session_date + timedelta(hours=6, minutes=30),
            open=100.0 + day * 0.5,
            high=100.5 + day * 0.5,
            low=99.5 + day * 0.5,
            close=100.2 + day * 0.5,
            volume=1000
        )
        session_manager.update_session_state(closing_bar, "SPY")
        
        # Get current session info
        current_session = session_manager.get_current_session()
        if current_session:
            print(f"     Session ID: {current_session.session_id}")
            print(f"     Session Type: {current_session.session_type.value}")
            print(f"     Current Phase: {current_session.current_phase.value}")
            print(f"     Bars: {current_session.metrics.bars_in_session}")
            print(f"     Volume: {current_session.metrics.volume_traded:,}")
    
    # Show session history
    history = session_manager.get_session_history(limit=5)
    print(f"\n   Session History ({len(history)} sessions):")
    for i, session_info in enumerate(history):
        print(f"     {i + 1}. {session_info['session_id']} - {session_info['session_type']} - {session_info['bars_count']} bars")


def demonstrate_session_type_detection(session_manager):
    """Demonstrate different session type detection."""
    print("\n🕐 Demonstrating Session Type Detection")
    print("=" * 50)
    
    # Test different times and scenarios
    test_scenarios = [
        (datetime(2024, 1, 2, 6, 0), "Pre-market trading"),
        (datetime(2024, 1, 2, 9, 30), "RTH session start"),
        (datetime(2024, 1, 2, 12, 0), "RTH mid-session"),
        (datetime(2024, 1, 2, 16, 0), "RTH session end"),
        (datetime(2024, 1, 2, 18, 0), "After-hours trading"),
        (datetime(2024, 1, 6, 10, 0), "Weekend (Saturday)"),
        (datetime(2024, 1, 7, 10, 0), "Weekend (Sunday)"),
        (datetime(2024, 1, 1, 10, 0), "Holiday (New Year's Day)"),
        (datetime(2024, 7, 4, 10, 0), "Holiday (Independence Day)"),
        (datetime(2024, 12, 25, 10, 0), "Holiday (Christmas Day)")
    ]
    
    for test_time, description in test_scenarios:
        bar = OHLCVBar(
            timestamp=test_time,
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        
        session_manager.update_session_state(bar, "SPY")
        session_type = session_manager.get_session_type()
        session_phase = session_manager.get_session_phase()
        
        type_name = session_type.value.replace('_', ' ').title() if session_type else "Unknown"
        phase_name = session_phase.value.replace('_', ' ').title() if session_phase else "Unknown"
        
        print(f"   {test_time.strftime('%Y-%m-%d %H:%M')} - {type_name} ({phase_name}) - {description}")


async def demonstrate_async_operations(session_manager, bars):
    """Demonstrate async operations."""
    print("\n🔄 Demonstrating Async Operations")
    print("=" * 50)
    
    # Simulate real-time data processing
    async def process_realtime_data():
        print("   Simulating real-time data processing...")
        
        for i, bar in enumerate(bars[330:380]):  # RTH bars
            session_manager.update_session_state(bar, "SPY")
            
            # Simulate processing delay
            await asyncio.sleep(0.01)
            
            # Show progress every 10 bars
            if i % 10 == 0:
                current_session = session_manager.get_current_session()
                if current_session:
                    print(f"     Processed {i + 1} bars - Phase: {current_session.current_phase.value}")
    
    await process_realtime_data()
    
    # Show final state
    current_session = session_manager.get_current_session()
    if current_session:
        print(f"   ✅ Final state: {current_session.current_phase.value} phase")
        print(f"   Bars processed: {current_session.metrics.bars_in_session}")
    else:
        print("   ❌ No active session")


def main():
    """Main demonstration function."""
    print("🚀 Session State Manager Demonstration")
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
    bars = create_full_trading_day_data()
    
    # Run demonstrations
    demonstrate_session_phases(session_manager, bars)
    demonstrate_session_metrics(session_manager, bars)
    demonstrate_session_context(session_manager, bars, window_manager)
    demonstrate_session_boundaries(session_manager, bars)
    demonstrate_session_summary(session_manager, bars)
    demonstrate_multiple_days(session_manager, window_manager)
    demonstrate_session_type_detection(session_manager)
    
    # Run async demonstration
    print("\n🔄 Running async demonstrations...")
    asyncio.run(demonstrate_async_operations(session_manager, bars))
    
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
        print(f"Is Trend Day: {current_session.metrics.is_trend_day}")
        print(f"Trend Direction: {current_session.metrics.trend_direction}")
        print(f"Volatility Level: {current_session.metrics.volatility_level}")
    
    history = session_manager.get_session_history()
    print(f"Session History: {len(history)} sessions")
    
    print("\n🎉 Session State Manager Demonstration Complete!")
    print("✅ All demonstrations completed successfully")


if __name__ == "__main__":
    main()