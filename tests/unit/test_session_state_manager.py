"""
Unit tests for Session State Manager.
"""

import pytest
from datetime import datetime, timedelta, time
from unittest.mock import Mock, MagicMock

from zone_fade_detector.core.session_state_manager import (
    SessionStateManager, SessionPhase, SessionType, SessionBoundaries, 
    SessionMetrics, SessionState
)
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager
from zone_fade_detector.core.models import OHLCVBar


class TestSessionStateManager:
    """Test cases for SessionStateManager."""
    
    @pytest.fixture
    def window_manager(self):
        """Create a mock window manager."""
        manager = Mock(spec=RollingWindowManager)
        manager.get_window_bars.return_value = []
        return manager
    
    @pytest.fixture
    def session_manager(self, window_manager):
        """Create a SessionStateManager instance for testing."""
        return SessionStateManager(window_manager, timezone_offset_hours=-5)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 2, 9, 30)  # RTH start
        bars = []
        
        for i in range(100):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0 + i * 0.01,
                high=100.5 + i * 0.01,
                low=99.5 + i * 0.01,
                close=100.2 + i * 0.01,
                volume=1000 + i * 10
            )
            bars.append(bar)
        
        return bars
    
    def test_initialization(self, session_manager):
        """Test SessionStateManager initialization."""
        assert session_manager.current_session is None
        assert len(session_manager.session_history) == 0
        assert session_manager.timezone_offset == -5
        assert session_manager.rth_start == time(9, 30)
        assert session_manager.rth_end == time(16, 0)
    
    def test_update_session_state_new_session(self, session_manager, sample_bars):
        """Test updating session state with new session."""
        bar = sample_bars[0]  # 9:30 AM bar
        
        session_state = session_manager.update_session_state(bar, "SPY")
        
        assert session_state is not None
        assert session_state.session_type == SessionType.RTH
        assert session_state.current_phase == SessionPhase.OPENING_RANGE
        assert session_state.is_active is True
        assert session_manager.current_session == session_state
    
    def test_session_phase_detection(self, session_manager, sample_bars):
        """Test session phase detection."""
        # Opening range (9:30-10:00)
        opening_bar = sample_bars[0]  # 9:30 AM
        session_manager.update_session_state(opening_bar, "SPY")
        assert session_manager.get_session_phase() == SessionPhase.OPENING_RANGE
        
        # Early session (10:00-12:00)
        early_bar = sample_bars[30]  # 10:00 AM
        session_manager.update_session_state(early_bar, "SPY")
        assert session_manager.get_session_phase() == SessionPhase.EARLY_SESSION
        
        # Mid session (12:00-2:00)
        mid_bar = sample_bars[150]  # 12:00 PM
        session_manager.update_session_state(mid_bar, "SPY")
        assert session_manager.get_session_phase() == SessionPhase.MID_SESSION
        
        # Late session (2:00-4:00)
        late_bar = sample_bars[270]  # 2:00 PM
        session_manager.update_session_state(late_bar, "SPY")
        assert session_manager.get_session_phase() == SessionPhase.LATE_SESSION
    
    def test_session_type_detection(self, session_manager):
        """Test session type detection."""
        # RTH session
        rth_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 14, 30),  # 2:30 PM ET
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(rth_bar, "SPY")
        assert session_manager.get_session_type() == SessionType.RTH
        
        # Pre-market session
        pre_market_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 6, 30),  # 6:30 AM ET
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(pre_market_bar, "SPY")
        assert session_manager.get_session_type() == SessionType.PRE_MARKET
        
        # After hours session
        after_hours_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 18, 30),  # 6:30 PM ET
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(after_hours_bar, "SPY")
        assert session_manager.get_session_type() == SessionType.AFTER_HOURS
    
    def test_is_rth_session(self, session_manager, sample_bars):
        """Test RTH session detection."""
        # No session
        assert not session_manager.is_rth_session()
        
        # RTH session
        rth_bar = sample_bars[0]  # 9:30 AM
        session_manager.update_session_state(rth_bar, "SPY")
        assert session_manager.is_rth_session()
        
        # Pre-market session
        pre_market_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 6, 30),  # 6:30 AM ET
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(pre_market_bar, "SPY")
        assert not session_manager.is_rth_session()
    
    def test_is_opening_range(self, session_manager, sample_bars):
        """Test opening range detection."""
        # No session
        assert not session_manager.is_opening_range()
        
        # Opening range session
        opening_bar = sample_bars[0]  # 9:30 AM
        session_manager.update_session_state(opening_bar, "SPY")
        assert session_manager.is_opening_range()
        
        # Early session
        early_bar = sample_bars[30]  # 10:00 AM
        session_manager.update_session_state(early_bar, "SPY")
        assert not session_manager.is_opening_range()
    
    def test_session_boundaries(self, session_manager, sample_bars):
        """Test session boundaries calculation."""
        bar = sample_bars[0]  # 9:30 AM
        session_manager.update_session_state(bar, "SPY")
        
        boundaries = session_manager.get_session_boundaries()
        assert boundaries is not None
        assert boundaries.session_start.hour == 9
        assert boundaries.session_start.minute == 30
        assert boundaries.session_end.hour == 16
        assert boundaries.session_end.minute == 0
        assert boundaries.opening_range_start.hour == 9
        assert boundaries.opening_range_start.minute == 30
        assert boundaries.opening_range_end.hour == 10
        assert boundaries.opening_range_end.minute == 0
    
    def test_session_metrics(self, session_manager, sample_bars):
        """Test session metrics calculation."""
        # Add multiple bars
        for i in range(10):
            session_manager.update_session_state(sample_bars[i], "SPY")
        
        metrics = session_manager.get_session_metrics()
        assert metrics is not None
        assert metrics.bars_in_session == 10
        assert metrics.volume_traded > 0
        assert metrics.session_duration_minutes >= 0
    
    def test_session_context(self, session_manager, sample_bars, window_manager):
        """Test session context calculation."""
        # Mock window data
        window_manager.get_window_bars.side_effect = [
            sample_bars[:10],  # session_context
            sample_bars[:5],   # opening_range
            sample_bars[:10]   # vwap_computation
        ]
        
        # Add bars to create session
        for i in range(10):
            session_manager.update_session_state(sample_bars[i], "SPY")
        
        context = session_manager.get_session_context()
        assert context is not None
        assert hasattr(context, 'is_trend_day')
        assert hasattr(context, 'vwap_slope')
        assert hasattr(context, 'is_balanced')
        assert hasattr(context, 'value_area_overlap')
    
    def test_session_summary(self, session_manager, sample_bars):
        """Test session summary generation."""
        # No session
        summary = session_manager.get_session_summary()
        assert summary["status"] == "no_active_session"
        
        # With session
        session_manager.update_session_state(sample_bars[0], "SPY")
        summary = session_manager.get_session_summary()
        
        assert "session_id" in summary
        assert "session_type" in summary
        assert "current_phase" in summary
        assert "is_active" in summary
        assert "duration_minutes" in summary
        assert "bars_count" in summary
    
    def test_phase_transitions(self, session_manager, sample_bars):
        """Test phase transition tracking."""
        # Start session
        session_manager.update_session_state(sample_bars[0], "SPY")
        assert len(session_manager.current_session.phase_transitions) == 0
        
        # Transition to early session
        session_manager.update_session_state(sample_bars[30], "SPY")
        assert len(session_manager.current_session.phase_transitions) == 1
        assert session_manager.current_session.phase_transitions[0][1] == SessionPhase.EARLY_SESSION
    
    def test_session_history(self, session_manager, sample_bars):
        """Test session history tracking."""
        # No history initially
        history = session_manager.get_session_history()
        assert len(history) == 0
        
        # Create a session
        session_manager.update_session_state(sample_bars[0], "SPY")
        
        # End session (simulate new day)
        next_day_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 3, 9, 30),  # Next day
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(next_day_bar, "SPY")
        
        # Check history
        history = session_manager.get_session_history()
        assert len(history) == 1
        assert history[0]["session_id"] == "session_20240102"
    
    def test_weekend_detection(self, session_manager):
        """Test weekend session detection."""
        # Saturday
        saturday_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 6, 10, 0),  # Saturday
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(saturday_bar, "SPY")
        assert session_manager.get_session_type() == SessionType.WEEKEND
        
        # Sunday
        sunday_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 7, 10, 0),  # Sunday
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(sunday_bar, "SPY")
        assert session_manager.get_session_type() == SessionType.WEEKEND
    
    def test_holiday_detection(self, session_manager):
        """Test holiday session detection."""
        # New Year's Day
        new_year_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 1, 10, 0),  # New Year's Day
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        session_manager.update_session_state(new_year_bar, "SPY")
        assert session_manager.get_session_type() == SessionType.HOLIDAY
    
    def test_trend_analysis(self, session_manager, sample_bars):
        """Test trend analysis in session metrics."""
        # Add bars with upward trend
        for i in range(20):
            bar = OHLCVBar(
                timestamp=datetime(2024, 1, 2, 9, 30) + timedelta(minutes=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.2 + i * 0.1,
                volume=1000
            )
            session_manager.update_session_state(bar, "SPY")
        
        metrics = session_manager.get_session_metrics()
        assert metrics is not None
        assert metrics.trend_direction in ["bullish", "bearish", "neutral"]
        assert isinstance(metrics.is_trend_day, bool)
    
    def test_volatility_analysis(self, session_manager):
        """Test volatility analysis in session metrics."""
        # Add bars with high volatility
        for i in range(10):
            bar = OHLCVBar(
                timestamp=datetime(2024, 1, 2, 9, 30) + timedelta(minutes=i),
                open=100.0,
                high=105.0,  # High volatility
                low=95.0,
                close=100.0,
                volume=1000
            )
            session_manager.update_session_state(bar, "SPY")
        
        metrics = session_manager.get_session_metrics()
        assert metrics is not None
        assert metrics.volatility_level in ["low", "medium", "high"]


class TestSessionPhase:
    """Test cases for SessionPhase enum."""
    
    def test_session_phase_values(self):
        """Test SessionPhase enum values."""
        assert SessionPhase.PRE_MARKET.value == "pre_market"
        assert SessionPhase.OPENING_RANGE.value == "opening_range"
        assert SessionPhase.EARLY_SESSION.value == "early_session"
        assert SessionPhase.MID_SESSION.value == "mid_session"
        assert SessionPhase.LATE_SESSION.value == "late_session"
        assert SessionPhase.POST_MARKET.value == "post_market"


class TestSessionType:
    """Test cases for SessionType enum."""
    
    def test_session_type_values(self):
        """Test SessionType enum values."""
        assert SessionType.RTH.value == "rth"
        assert SessionType.PRE_MARKET.value == "pre_market"
        assert SessionType.AFTER_HOURS.value == "after_hours"
        assert SessionType.WEEKEND.value == "weekend"
        assert SessionType.HOLIDAY.value == "holiday"


class TestSessionBoundaries:
    """Test cases for SessionBoundaries dataclass."""
    
    def test_session_boundaries_creation(self):
        """Test SessionBoundaries creation."""
        now = datetime.now()
        boundaries = SessionBoundaries(
            session_start=now,
            session_end=now + timedelta(hours=6.5),
            opening_range_start=now,
            opening_range_end=now + timedelta(minutes=30),
            early_session_start=now + timedelta(minutes=30),
            early_session_end=now + timedelta(hours=2.5),
            mid_session_start=now + timedelta(hours=2.5),
            mid_session_end=now + timedelta(hours=4.5),
            late_session_start=now + timedelta(hours=4.5),
            late_session_end=now + timedelta(hours=6.5)
        )
        
        assert boundaries.session_start == now
        assert boundaries.session_end == now + timedelta(hours=6.5)
        assert boundaries.opening_range_duration == 30


class TestSessionMetrics:
    """Test cases for SessionMetrics dataclass."""
    
    def test_session_metrics_creation(self):
        """Test SessionMetrics creation."""
        metrics = SessionMetrics(
            session_type=SessionType.RTH,
            current_phase=SessionPhase.OPENING_RANGE,
            session_duration_minutes=30,
            bars_in_session=10,
            volume_traded=10000,
            price_range=2.5,
            vwap_level=100.0,
            opening_range_high=101.0,
            opening_range_low=99.0,
            opening_range_volume=5000,
            is_balanced=True,
            is_trend_day=False,
            trend_direction="neutral",
            volatility_level="medium",
            volume_profile={}
        )
        
        assert metrics.session_type == SessionType.RTH
        assert metrics.current_phase == SessionPhase.OPENING_RANGE
        assert metrics.session_duration_minutes == 30
        assert metrics.bars_in_session == 10
        assert metrics.volume_traded == 10000
        assert metrics.price_range == 2.5
        assert metrics.vwap_level == 100.0
        assert metrics.is_balanced is True
        assert metrics.is_trend_day is False
        assert metrics.trend_direction == "neutral"
        assert metrics.volatility_level == "medium"


class TestSessionState:
    """Test cases for SessionState dataclass."""
    
    def test_session_state_creation(self):
        """Test SessionState creation."""
        now = datetime.now()
        boundaries = SessionBoundaries(
            session_start=now,
            session_end=now + timedelta(hours=6.5),
            opening_range_start=now,
            opening_range_end=now + timedelta(minutes=30),
            early_session_start=now + timedelta(minutes=30),
            early_session_end=now + timedelta(hours=2.5),
            mid_session_start=now + timedelta(hours=2.5),
            mid_session_end=now + timedelta(hours=4.5),
            late_session_start=now + timedelta(hours=4.5),
            late_session_end=now + timedelta(hours=6.5)
        )
        
        metrics = SessionMetrics(
            session_type=SessionType.RTH,
            current_phase=SessionPhase.OPENING_RANGE,
            session_duration_minutes=0,
            bars_in_session=0,
            volume_traded=0,
            price_range=0.0,
            vwap_level=0.0,
            opening_range_high=0.0,
            opening_range_low=0.0,
            opening_range_volume=0,
            is_balanced=True,
            is_trend_day=False,
            trend_direction="neutral",
            volatility_level="medium",
            volume_profile={}
        )
        
        state = SessionState(
            session_id="test_session",
            session_date=now,
            session_type=SessionType.RTH,
            current_phase=SessionPhase.OPENING_RANGE,
            boundaries=boundaries,
            metrics=metrics,
            is_active=True,
            last_update=now
        )
        
        assert state.session_id == "test_session"
        assert state.session_date == now
        assert state.session_type == SessionType.RTH
        assert state.current_phase == SessionPhase.OPENING_RANGE
        assert state.boundaries == boundaries
        assert state.metrics == metrics
        assert state.is_active is True
        assert state.last_update == now
        assert len(state.phase_transitions) == 0