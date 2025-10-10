"""
Session State Manager for Zone Fade Strategy.

This module provides comprehensive RTH session state tracking with rolling windows,
including session boundaries, market phases, and session-specific context.
"""

import logging
from datetime import datetime, timedelta, time, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque

from zone_fade_detector.core.models import OHLCVBar, MarketContext
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType


class SessionPhase(Enum):
    """Phases within an RTH session."""
    PRE_MARKET = "pre_market"           # Before 9:30 AM ET
    OPENING_RANGE = "opening_range"     # 9:30-10:00 AM ET
    EARLY_SESSION = "early_session"     # 10:00 AM-12:00 PM ET
    MID_SESSION = "mid_session"         # 12:00-2:00 PM ET
    LATE_SESSION = "late_session"       # 2:00-4:00 PM ET
    POST_MARKET = "post_market"         # After 4:00 PM ET


class SessionType(Enum):
    """Types of trading sessions."""
    RTH = "rth"                         # Regular Trading Hours
    PRE_MARKET = "pre_market"           # Pre-market hours
    AFTER_HOURS = "after_hours"         # After hours trading
    WEEKEND = "weekend"                 # Weekend (no trading)
    HOLIDAY = "holiday"                 # Market holiday


@dataclass
class SessionBoundaries:
    """Trading session boundaries."""
    session_start: datetime
    session_end: datetime
    opening_range_start: datetime
    opening_range_end: datetime
    early_session_start: datetime
    early_session_end: datetime
    mid_session_start: datetime
    mid_session_end: datetime
    late_session_start: datetime
    late_session_end: datetime


@dataclass
class SessionMetrics:
    """Metrics for the current session."""
    session_type: SessionType
    current_phase: SessionPhase
    session_duration_minutes: int
    bars_in_session: int
    volume_traded: int
    price_range: float
    vwap_level: float
    opening_range_high: float
    opening_range_low: float
    opening_range_volume: int
    is_balanced: bool
    is_trend_day: bool
    trend_direction: str  # "bullish", "bearish", "neutral"
    volatility_level: str  # "low", "medium", "high"
    volume_profile: Dict[str, Any]


@dataclass
class SessionState:
    """Current session state."""
    session_id: str
    session_date: datetime
    session_type: SessionType
    current_phase: SessionPhase
    boundaries: SessionBoundaries
    metrics: SessionMetrics
    is_active: bool
    last_update: datetime
    phase_transitions: List[Tuple[datetime, SessionPhase]] = field(default_factory=list)


class SessionStateManager:
    """
    Manages RTH session state with rolling windows.
    
    Provides comprehensive session tracking including phase detection,
    boundary management, and session-specific context for Zone Fade strategy.
    """
    
    def __init__(
        self,
        window_manager: RollingWindowManager,
        timezone_offset_hours: int = -5  # ET timezone offset
    ):
        """
        Initialize session state manager.
        
        Args:
            window_manager: Rolling window manager instance
            timezone_offset_hours: Timezone offset from UTC (ET = -5)
        """
        self.window_manager = window_manager
        self.timezone_offset = timezone_offset_hours
        self.logger = logging.getLogger(__name__)
        
        # Session state
        self.current_session: Optional[SessionState] = None
        self.session_history: List[SessionState] = []
        
        # Market hours configuration
        self.rth_start = time(9, 30)  # 9:30 AM ET
        self.rth_end = time(16, 0)    # 4:00 PM ET
        self.pre_market_start = time(4, 0)   # 4:00 AM ET
        self.after_hours_end = time(20, 0)   # 8:00 PM ET
        
        # Phase boundaries
        self.opening_range_duration = 30  # 30 minutes
        self.early_session_duration = 120  # 2 hours
        self.mid_session_duration = 120   # 2 hours
        self.late_session_duration = 120  # 2 hours
        
        self.logger.info("SessionStateManager initialized")
    
    def update_session_state(self, bar: OHLCVBar, symbol: str = "DEFAULT") -> SessionState:
        """
        Update session state with new bar.
        
        Args:
            bar: New OHLCV bar
            symbol: Symbol for the bar
            
        Returns:
            Updated session state
        """
        # Convert to ET timezone
        et_time = self._to_et_time(bar.timestamp)
        
        # Check if we need a new session
        if self._should_start_new_session(et_time):
            self._start_new_session(et_time)
        
        # Update current session
        if self.current_session:
            self._update_current_session(bar, et_time, symbol)
        
        return self.current_session
    
    def get_current_session(self) -> Optional[SessionState]:
        """Get current session state."""
        return self.current_session
    
    def get_session_phase(self) -> Optional[SessionPhase]:
        """Get current session phase."""
        return self.current_session.current_phase if self.current_session else None
    
    def get_session_type(self) -> Optional[SessionType]:
        """Get current session type."""
        return self.current_session.session_type if self.current_session else None
    
    def is_rth_session(self) -> bool:
        """Check if currently in RTH session."""
        return (self.current_session and 
                self.current_session.session_type == SessionType.RTH and
                self.current_session.is_active)
    
    def is_opening_range(self) -> bool:
        """Check if currently in opening range phase."""
        return (self.current_session and 
                self.current_session.current_phase == SessionPhase.OPENING_RANGE)
    
    def get_session_boundaries(self) -> Optional[SessionBoundaries]:
        """Get current session boundaries."""
        return self.current_session.boundaries if self.current_session else None
    
    def get_session_metrics(self) -> Optional[SessionMetrics]:
        """Get current session metrics."""
        return self.current_session.metrics if self.current_session else None
    
    def get_session_context(self) -> Optional[MarketContext]:
        """Get market context for current session."""
        if not self.current_session:
            return None
        
        # Get session data from rolling windows
        session_bars = self.window_manager.get_window_bars(WindowType.SESSION_CONTEXT)
        or_bars = self.window_manager.get_window_bars(WindowType.OPENING_RANGE)
        vwap_bars = self.window_manager.get_window_bars(WindowType.VWAP_COMPUTATION)
        
        # Calculate market context
        return self._calculate_market_context(session_bars, or_bars, vwap_bars)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        if not self.current_session:
            return {"status": "no_active_session"}
        
        return {
            "session_id": self.current_session.session_id,
            "session_date": self.current_session.session_date.isoformat(),
            "session_type": self.current_session.session_type.value,
            "current_phase": self.current_session.current_phase.value,
            "is_active": self.current_session.is_active,
            "duration_minutes": self.current_session.metrics.session_duration_minutes,
            "bars_count": self.current_session.metrics.bars_in_session,
            "volume_traded": self.current_session.metrics.volume_traded,
            "price_range": self.current_session.metrics.price_range,
            "is_balanced": self.current_session.metrics.is_balanced,
            "is_trend_day": self.current_session.metrics.is_trend_day,
            "trend_direction": self.current_session.metrics.trend_direction,
            "volatility_level": self.current_session.metrics.volatility_level,
            "last_update": self.current_session.last_update.isoformat()
        }
    
    def _should_start_new_session(self, et_time: datetime) -> bool:
        """Check if we should start a new session."""
        if not self.current_session:
            return True
        
        # Check if we've crossed to a new day
        if et_time.date() != self.current_session.session_date.date():
            return True
        
        # Check if we've crossed session boundaries
        current_time = et_time.time()
        
        # If we were in post-market and now in pre-market
        if (self.current_session.current_phase == SessionPhase.POST_MARKET and
            self.pre_market_start <= current_time < self.rth_start):
            return True
        
        # If we were in pre-market and now in RTH
        if (self.current_session.current_phase == SessionPhase.PRE_MARKET and
            self.rth_start <= current_time <= self.rth_end):
            return True
        
        return False
    
    def _start_new_session(self, et_time: datetime):
        """Start a new session."""
        session_date = et_time.date()
        session_id = f"session_{session_date.strftime('%Y%m%d')}"
        
        # Determine session type
        session_type = self._determine_session_type(et_time)
        
        # Calculate session boundaries
        boundaries = self._calculate_session_boundaries(et_time)
        
        # Determine initial phase
        current_phase = self._determine_current_phase(et_time)
        
        # Create session metrics
        metrics = SessionMetrics(
            session_type=session_type,
            current_phase=current_phase,
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
        
        # Create session state
        self.current_session = SessionState(
            session_id=session_id,
            session_date=et_time,
            session_type=session_type,
            current_phase=current_phase,
            boundaries=boundaries,
            metrics=metrics,
            is_active=True,
            last_update=et_time
        )
        
        # Archive previous session
        if self.current_session:
            self.session_history.append(self.current_session)
        
        self.logger.info(f"Started new session: {session_id} ({session_type.value}) - Phase: {current_phase.value}")
    
    def _update_current_session(self, bar: OHLCVBar, et_time: datetime, symbol: str):
        """Update current session with new bar."""
        if not self.current_session:
            return
        
        # Update phase if needed
        new_phase = self._determine_current_phase(et_time)
        if new_phase != self.current_session.current_phase:
            self._transition_phase(new_phase, et_time)
        
        # Update metrics
        self._update_session_metrics(bar, et_time)
        
        # Update last update time
        self.current_session.last_update = et_time
    
    def _determine_session_type(self, et_time: datetime) -> SessionType:
        """Determine session type based on time."""
        current_time = et_time.time()
        current_date = et_time.date()
        
        # Check if it's a weekend
        if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return SessionType.WEEKEND
        
        # Check if it's a holiday (simplified check)
        if self._is_market_holiday(current_date):
            return SessionType.HOLIDAY
        
        # Check RTH hours
        if self.rth_start <= current_time <= self.rth_end:
            return SessionType.RTH
        
        # Check pre-market hours
        if self.pre_market_start <= current_time < self.rth_start:
            return SessionType.PRE_MARKET
        
        # Check after hours
        if self.rth_end < current_time <= self.after_hours_end:
            return SessionType.AFTER_HOURS
        
        return SessionType.PRE_MARKET  # Default to pre-market
    
    def _determine_current_phase(self, et_time: datetime) -> SessionPhase:
        """Determine current session phase."""
        current_time = et_time.time()
        
        # Pre-market
        if current_time < self.rth_start:
            return SessionPhase.PRE_MARKET
        
        # Post-market
        if current_time > self.rth_end:
            return SessionPhase.POST_MARKET
        
        # RTH phases
        rth_start_time = self.rth_start
        opening_range_end = time(10, 0)  # 9:30 + 30 minutes
        early_session_end = time(12, 0)  # 10:00 + 2 hours
        mid_session_end = time(14, 0)    # 12:00 + 2 hours
        
        if rth_start_time <= current_time < opening_range_end:
            return SessionPhase.OPENING_RANGE
        elif opening_range_end <= current_time < early_session_end:
            return SessionPhase.EARLY_SESSION
        elif early_session_end <= current_time < mid_session_end:
            return SessionPhase.MID_SESSION
        else:  # mid_session_end <= current_time <= self.rth_end
            return SessionPhase.LATE_SESSION
    
    def _calculate_session_boundaries(self, et_time: datetime) -> SessionBoundaries:
        """Calculate session boundaries for the day."""
        session_date = et_time.date()
        
        # RTH session boundaries
        session_start = datetime.combine(session_date, self.rth_start)
        session_end = datetime.combine(session_date, self.rth_end)
        
        # Opening range boundaries
        opening_range_start = session_start
        opening_range_end = session_start + timedelta(minutes=self.opening_range_duration)
        
        # Early session boundaries
        early_session_start = opening_range_end
        early_session_end = early_session_start + timedelta(minutes=self.early_session_duration)
        
        # Mid session boundaries
        mid_session_start = early_session_end
        mid_session_end = mid_session_start + timedelta(minutes=self.mid_session_duration)
        
        # Late session boundaries
        late_session_start = mid_session_end
        late_session_end = session_end
        
        return SessionBoundaries(
            session_start=session_start,
            session_end=session_end,
            opening_range_start=opening_range_start,
            opening_range_end=opening_range_end,
            early_session_start=early_session_start,
            early_session_end=early_session_end,
            mid_session_start=mid_session_start,
            mid_session_end=mid_session_end,
            late_session_start=late_session_start,
            late_session_end=late_session_end
        )
    
    def _transition_phase(self, new_phase: SessionPhase, et_time: datetime):
        """Handle phase transition."""
        if not self.current_session:
            return
        
        old_phase = self.current_session.current_phase
        self.current_session.current_phase = new_phase
        self.current_session.phase_transitions.append((et_time, new_phase))
        
        self.logger.info(f"Phase transition: {old_phase.value} -> {new_phase.value} at {et_time}")
        
        # Handle phase-specific logic
        if new_phase == SessionPhase.OPENING_RANGE:
            self._handle_opening_range_start()
        elif new_phase == SessionPhase.EARLY_SESSION:
            self._handle_early_session_start()
        elif new_phase == SessionPhase.MID_SESSION:
            self._handle_mid_session_start()
        elif new_phase == SessionPhase.LATE_SESSION:
            self._handle_late_session_start()
        elif new_phase == SessionPhase.POST_MARKET:
            self._handle_session_end()
    
    def _update_session_metrics(self, bar: OHLCVBar, et_time: datetime):
        """Update session metrics with new bar."""
        if not self.current_session:
            return
        
        metrics = self.current_session.metrics
        
        # Update basic metrics
        metrics.bars_in_session += 1
        metrics.volume_traded += bar.volume
        
        # Update price range
        if metrics.bars_in_session == 1:
            metrics.price_range = 0.0
        else:
            # Calculate range from session start
            session_bars = self.window_manager.get_window_bars(WindowType.SESSION_CONTEXT)
            if session_bars:
                high_price = max(b.high for b in session_bars)
                low_price = min(b.low for b in session_bars)
                metrics.price_range = high_price - low_price
        
        # Update VWAP level
        vwap_bars = self.window_manager.get_window_bars(WindowType.VWAP_COMPUTATION)
        if vwap_bars:
            total_volume = sum(b.volume for b in vwap_bars)
            if total_volume > 0:
                vwap = sum(b.close * b.volume for b in vwap_bars) / total_volume
                metrics.vwap_level = vwap
        
        # Update opening range metrics
        if self.current_session.current_phase in [SessionPhase.OPENING_RANGE, SessionPhase.EARLY_SESSION, 
                                                 SessionPhase.MID_SESSION, SessionPhase.LATE_SESSION]:
            or_bars = self.window_manager.get_window_bars(WindowType.OPENING_RANGE)
            if or_bars:
                metrics.opening_range_high = max(b.high for b in or_bars)
                metrics.opening_range_low = min(b.low for b in or_bars)
                metrics.opening_range_volume = sum(b.volume for b in or_bars)
        
        # Update session duration
        if self.current_session.boundaries:
            duration = et_time - self.current_session.boundaries.session_start
            metrics.session_duration_minutes = int(duration.total_seconds() / 60)
        
        # Update trend and balance analysis
        self._update_trend_analysis()
        self._update_volatility_analysis()
    
    def _update_trend_analysis(self):
        """Update trend analysis for the session."""
        if not self.current_session:
            return
        
        session_bars = self.window_manager.get_window_bars(WindowType.SESSION_CONTEXT)
        if len(session_bars) < 10:  # Need minimum bars for analysis
            return
        
        metrics = self.current_session.metrics
        
        # Calculate trend direction
        first_half = session_bars[:len(session_bars)//2]
        second_half = session_bars[len(session_bars)//2:]
        
        first_avg = sum(b.close for b in first_half) / len(first_half)
        second_avg = sum(b.close for b in second_half) / len(second_half)
        
        price_change_pct = (second_avg - first_avg) / first_avg if first_avg > 0 else 0
        
        if price_change_pct > 0.01:  # 1% threshold
            metrics.trend_direction = "bullish"
            metrics.is_trend_day = True
        elif price_change_pct < -0.01:
            metrics.trend_direction = "bearish"
            metrics.is_trend_day = True
        else:
            metrics.trend_direction = "neutral"
            metrics.is_trend_day = False
        
        # Calculate balance (simplified)
        vwap_bars = self.window_manager.get_window_bars(WindowType.VWAP_COMPUTATION)
        if vwap_bars and len(vwap_bars) > 5:
            # Check if price is close to VWAP (balanced)
            latest_price = session_bars[-1].close
            vwap = metrics.vwap_level
            if vwap > 0:
                price_deviation = abs(latest_price - vwap) / vwap
                metrics.is_balanced = price_deviation < 0.005  # 0.5% threshold
    
    def _update_volatility_analysis(self):
        """Update volatility analysis for the session."""
        if not self.current_session:
            return
        
        session_bars = self.window_manager.get_window_bars(WindowType.SESSION_CONTEXT)
        if len(session_bars) < 5:
            return
        
        metrics = self.current_session.metrics
        
        # Calculate volatility based on price range
        if metrics.price_range > 0 and len(session_bars) > 0:
            avg_price = sum(b.close for b in session_bars) / len(session_bars)
            volatility_pct = metrics.price_range / avg_price if avg_price > 0 else 0
            
            if volatility_pct < 0.01:  # 1%
                metrics.volatility_level = "low"
            elif volatility_pct < 0.03:  # 3%
                metrics.volatility_level = "medium"
            else:
                metrics.volatility_level = "high"
    
    def _calculate_market_context(self, session_bars: List[OHLCVBar], 
                                 or_bars: List[OHLCVBar], 
                                 vwap_bars: List[OHLCVBar]) -> MarketContext:
        """Calculate market context from session data."""
        if not session_bars:
            return MarketContext()
        
        # Calculate VWAP slope
        vwap_slope = 0.0
        if len(vwap_bars) > 1:
            first_vwap = sum(b.close * b.volume for b in vwap_bars[:5]) / sum(b.volume for b in vwap_bars[:5])
            last_vwap = sum(b.close * b.volume for b in vwap_bars[-5:]) / sum(b.volume for b in vwap_bars[-5:])
            time_diff = (vwap_bars[-1].timestamp - vwap_bars[0].timestamp).total_seconds() / 3600  # hours
            vwap_slope = (last_vwap - first_vwap) / time_diff if time_diff > 0 else 0
        
        # Determine if it's a trend day
        is_trend_day = abs(vwap_slope) > 0.002  # 0.2% per hour threshold
        
        # Calculate value area overlap (simplified)
        value_area_overlap = False
        if or_bars and len(or_bars) > 5:
            or_high = max(b.high for b in or_bars)
            or_low = min(b.low for b in or_bars)
            current_price = session_bars[-1].close
            value_area_overlap = or_low <= current_price <= or_high
        
        return MarketContext(
            is_trend_day=is_trend_day,
            vwap_slope=vwap_slope,
            value_area_overlap=value_area_overlap,
            market_balance=0.5 if not is_trend_day and abs(vwap_slope) < 0.001 else (0.8 if vwap_slope > 0 else 0.2),
            volatility_regime="high" if abs(vwap_slope) > 0.005 else "normal",
            session_type="regular"
        )
    
    def _handle_opening_range_start(self):
        """Handle opening range phase start."""
        self.logger.info("Opening range phase started")
        # Reset opening range window
        self.window_manager.reset_window(WindowType.OPENING_RANGE)
    
    def _handle_early_session_start(self):
        """Handle early session phase start."""
        self.logger.info("Early session phase started")
    
    def _handle_mid_session_start(self):
        """Handle mid session phase start."""
        self.logger.info("Mid session phase started")
    
    def _handle_late_session_start(self):
        """Handle late session phase start."""
        self.logger.info("Late session phase started")
    
    def _handle_session_end(self):
        """Handle session end."""
        self.logger.info("Session ended")
        if self.current_session:
            self.current_session.is_active = False
    
    def _is_market_holiday(self, date) -> bool:
        """Check if date is a market holiday (simplified)."""
        # This is a simplified implementation
        # In production, you'd use a proper holiday calendar
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas Day
        ]
        
        return (date.month, date.day) in holidays
    
    def _to_et_time(self, utc_time: datetime) -> datetime:
        """Convert UTC time to ET time."""
        # Simple conversion (doesn't handle DST)
        et_offset = timedelta(hours=self.timezone_offset)
        return utc_time + et_offset
    
    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session history."""
        return [
            {
                "session_id": session.session_id,
                "session_date": session.session_date.isoformat(),
                "session_type": session.session_type.value,
                "duration_minutes": session.metrics.session_duration_minutes,
                "bars_count": session.metrics.bars_in_session,
                "volume_traded": session.metrics.volume_traded,
                "trend_direction": session.metrics.trend_direction,
                "is_trend_day": session.metrics.is_trend_day
            }
            for session in self.session_history[-limit:]
        ]