"""
Rolling Window Manager for Zone Fade Strategy.

This module provides centralized management of rolling time windows used in the
Zone Fade strategy, including HTF zones, session context, VWAP computation,
swing detection, and intermarket analysis.
"""

import logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque
import threading
import time as time_module

from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType, MarketContext


class WindowType(Enum):
    """Types of rolling windows used in Zone Fade strategy."""
    HTF_ZONES = "htf_zones"                    # Daily and Weekly zones
    SESSION_CONTEXT = "session_context"         # Rolling RTH Session (≈ 6.5h)
    OPENING_RANGE = "opening_range"             # Fixed window at session start
    VWAP_COMPUTATION = "vwap_computation"       # Rolling from RTH open
    SWING_CHOCH = "swing_choch"                 # Short-term rolling window (5-20 bars)
    INITIATIVE_ANALYSIS = "initiative_analysis" # Micro window around zone touch
    INTERMARKET = "intermarket"                 # Parallel cross-symbol window
    QRS_ACCUMULATOR = "qrs_accumulator"         # Session-rolling accumulator


@dataclass
class WindowConfig:
    """Configuration for a rolling window."""
    window_type: WindowType
    duration_minutes: int
    max_bars: int
    reset_on_session: bool = False
    reset_on_day: bool = False
    reset_on_week: bool = False
    overlap_tolerance: float = 0.1  # 10% overlap tolerance
    memory_limit_mb: int = 50  # Memory limit per window
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class WindowState:
    """State of a rolling window."""
    window_type: WindowType
    bars: deque = field(default_factory=deque)
    last_update: Optional[datetime] = None
    session_start: Optional[datetime] = None
    is_active: bool = True
    memory_usage_mb: float = 0.0
    hit_count: int = 0
    miss_count: int = 0


class RollingWindowManager:
    """
    Centralized manager for all rolling windows in Zone Fade strategy.
    
    Manages multiple rolling time windows with different durations and purposes,
    ensuring efficient memory usage and proper synchronization.
    """
    
    def __init__(
        self,
        configs: Optional[Dict[WindowType, WindowConfig]] = None,
        evaluation_cadence_seconds: int = 30,
        memory_limit_mb: int = 500
    ):
        """
        Initialize rolling window manager.
        
        Args:
            configs: Window configurations (uses defaults if None)
            evaluation_cadence_seconds: How often to evaluate windows
            memory_limit_mb: Total memory limit for all windows
        """
        self.logger = logging.getLogger(__name__)
        self.evaluation_cadence = evaluation_cadence_seconds
        self.memory_limit_mb = memory_limit_mb
        
        # Window configurations
        self.configs = configs or self._get_default_configs()
        
        # Window states
        self.windows: Dict[WindowType, WindowState] = {}
        self._initialize_windows()
        
        # Synchronization
        self._lock = threading.RLock()
        self._last_evaluation = None
        self._is_running = False
        
        # Performance tracking
        self._total_hits = 0
        self._total_misses = 0
        self._last_cleanup = datetime.now()
        
        self.logger.info(f"RollingWindowManager initialized with {len(self.configs)} window types")
    
    def _get_default_configs(self) -> Dict[WindowType, WindowConfig]:
        """Get default window configurations."""
        return {
            WindowType.HTF_ZONES: WindowConfig(
                window_type=WindowType.HTF_ZONES,
                duration_minutes=1440,  # 24 hours
                max_bars=1000,
                reset_on_day=True,
                priority=1
            ),
            WindowType.SESSION_CONTEXT: WindowConfig(
                window_type=WindowType.SESSION_CONTEXT,
                duration_minutes=390,  # 6.5 hours (RTH)
                max_bars=500,
                reset_on_session=True,
                priority=1
            ),
            WindowType.OPENING_RANGE: WindowConfig(
                window_type=WindowType.OPENING_RANGE,
                duration_minutes=30,  # First 30 minutes
                max_bars=50,
                reset_on_session=True,
                priority=2
            ),
            WindowType.VWAP_COMPUTATION: WindowConfig(
                window_type=WindowType.VWAP_COMPUTATION,
                duration_minutes=390,  # Full RTH session
                max_bars=1000,
                reset_on_session=True,
                priority=1
            ),
            WindowType.SWING_CHOCH: WindowConfig(
                window_type=WindowType.SWING_CHOCH,
                duration_minutes=20,  # 20 minutes max
                max_bars=20,
                reset_on_session=False,
                priority=2
            ),
            WindowType.INITIATIVE_ANALYSIS: WindowConfig(
                window_type=WindowType.INITIATIVE_ANALYSIS,
                duration_minutes=15,  # 15 minutes around zone touch
                max_bars=30,
                reset_on_session=False,
                priority=3
            ),
            WindowType.INTERMARKET: WindowConfig(
                window_type=WindowType.INTERMARKET,
                duration_minutes=60,  # 1 hour for cross-symbol analysis
                max_bars=200,
                reset_on_session=False,
                priority=2
            ),
            WindowType.QRS_ACCUMULATOR: WindowConfig(
                window_type=WindowType.QRS_ACCUMULATOR,
                duration_minutes=390,  # Full RTH session
                max_bars=100,
                reset_on_session=True,
                priority=1
            )
        }
    
    def _initialize_windows(self):
        """Initialize all window states."""
        for window_type, config in self.configs.items():
            self.windows[window_type] = WindowState(
                window_type=window_type,
                bars=deque(maxlen=config.max_bars),
                is_active=True
            )
    
    def add_bar(
        self, 
        window_type: WindowType, 
        bar: OHLCVBar, 
        symbol: str = "DEFAULT"
    ) -> bool:
        """
        Add a bar to a specific window.
        
        Args:
            window_type: Type of window to add bar to
            bar: OHLCV bar to add
            symbol: Symbol for the bar (for intermarket windows)
            
        Returns:
            True if bar was added successfully
        """
        with self._lock:
            if window_type not in self.windows:
                self.logger.warning(f"Unknown window type: {window_type}")
                return False
            
            window_state = self.windows[window_type]
            config = self.configs[window_type]
            
            # Check if window should be reset
            if self._should_reset_window(window_type, bar.timestamp):
                self._reset_window(window_type)
            
            # Add bar to window
            window_state.bars.append(bar)
            window_state.last_update = bar.timestamp
            window_state.hit_count += 1
            self._total_hits += 1
            
            # Update memory usage
            self._update_memory_usage(window_type)
            
            # Check memory limits
            if self._is_memory_limit_exceeded():
                self._cleanup_old_data()
            
            return True
    
    def get_window_bars(
        self, 
        window_type: WindowType, 
        symbol: Optional[str] = None
    ) -> List[OHLCVBar]:
        """
        Get bars from a specific window.
        
        Args:
            window_type: Type of window to get bars from
            symbol: Symbol filter (for intermarket windows)
            
        Returns:
            List of OHLCV bars in the window
        """
        with self._lock:
            if window_type not in self.windows:
                self.logger.warning(f"Unknown window type: {window_type}")
                return []
            
            window_state = self.windows[window_type]
            
            if not window_state.is_active:
                return []
            
            # Return all bars (symbol filtering would be implemented here)
            return list(window_state.bars)
    
    def get_window_for_timestamp(
        self, 
        window_type: WindowType, 
        timestamp: datetime
    ) -> List[OHLCVBar]:
        """
        Get bars from a window for a specific timestamp.
        
        Args:
            window_type: Type of window
            timestamp: Timestamp to get window for
            
        Returns:
            List of bars within the window for the timestamp
        """
        with self._lock:
            if window_type not in self.windows:
                return []
            
            window_state = self.windows[window_type]
            config = self.configs[window_type]
            
            if not window_state.is_active:
                return []
            
            # Calculate window boundaries
            window_start = timestamp - timedelta(minutes=config.duration_minutes)
            
            # Filter bars within window
            window_bars = []
            for bar in window_state.bars:
                if window_start <= bar.timestamp <= timestamp:
                    window_bars.append(bar)
            
            return window_bars
    
    def is_window_ready(self, window_type: WindowType) -> bool:
        """
        Check if a window has enough data to be useful.
        
        Args:
            window_type: Type of window to check
            
        Returns:
            True if window is ready for analysis
        """
        with self._lock:
            if window_type not in self.windows:
                return False
            
            window_state = self.windows[window_type]
            config = self.configs[window_type]
            
            # Window is ready if it has minimum required bars
            min_bars = max(5, config.max_bars // 10)  # At least 10% of max bars
            return len(window_state.bars) >= min_bars
    
    def get_window_info(self, window_type: WindowType) -> Dict[str, Any]:
        """
        Get information about a window.
        
        Args:
            window_type: Type of window
            
        Returns:
            Dictionary with window information
        """
        with self._lock:
            if window_type not in self.windows:
                return {}
            
            window_state = self.windows[window_type]
            config = self.configs[window_type]
            
            return {
                'window_type': window_type.value,
                'bar_count': len(window_state.bars),
                'max_bars': config.max_bars,
                'duration_minutes': config.duration_minutes,
                'is_active': window_state.is_active,
                'last_update': window_state.last_update,
                'memory_usage_mb': window_state.memory_usage_mb,
                'hit_count': window_state.hit_count,
                'miss_count': window_state.miss_count,
                'hit_rate': window_state.hit_count / max(1, window_state.hit_count + window_state.miss_count)
            }
    
    def get_all_windows_info(self) -> Dict[str, Any]:
        """Get information about all windows."""
        with self._lock:
            return {
                window_type.value: self.get_window_info(window_type)
                for window_type in self.windows.keys()
            }
    
    def reset_window(self, window_type: WindowType):
        """Reset a specific window."""
        with self._lock:
            self._reset_window(window_type)
    
    def reset_all_windows(self):
        """Reset all windows."""
        with self._lock:
            for window_type in self.windows.keys():
                self._reset_window(window_type)
    
    def _should_reset_window(self, window_type: WindowType, timestamp: datetime) -> bool:
        """Check if a window should be reset based on timestamp."""
        config = self.configs[window_type]
        window_state = self.windows[window_type]
        
        if not window_state.last_update:
            return False
        
        # Check session reset
        if config.reset_on_session:
            if self._is_new_session(timestamp, window_state.last_update):
                return True
        
        # Check day reset
        if config.reset_on_day:
            if self._is_new_day(timestamp, window_state.last_update):
                return True
        
        # Check week reset
        if config.reset_on_week:
            if self._is_new_week(timestamp, window_state.last_update):
                return True
        
        return False
    
    def _reset_window(self, window_type: WindowType):
        """Reset a window to empty state."""
        window_state = self.windows[window_type]
        window_state.bars.clear()
        window_state.last_update = None
        window_state.memory_usage_mb = 0.0
        window_state.hit_count = 0
        window_state.miss_count = 0
        
        self.logger.debug(f"Reset window: {window_type.value}")
    
    def _is_new_session(self, current: datetime, last: datetime) -> bool:
        """Check if current timestamp is in a new RTH session."""
        # RTH sessions: 9:30 AM - 4:00 PM ET
        rth_start = time(9, 30)
        rth_end = time(16, 0)
        
        current_time = current.time()
        last_time = last.time()
        
        # Check if we've crossed session boundary
        if last_time < rth_start and current_time >= rth_start:
            return True
        
        # Check if we've crossed day boundary within session
        if current.date() != last.date():
            return True
        
        return False
    
    def _is_new_day(self, current: datetime, last: datetime) -> bool:
        """Check if current timestamp is on a new day."""
        return current.date() != last.date()
    
    def _is_new_week(self, current: datetime, last: datetime) -> bool:
        """Check if current timestamp is in a new week."""
        return current.isocalendar()[1] != last.isocalendar()[1]
    
    def _update_memory_usage(self, window_type: WindowType):
        """Update memory usage for a window."""
        window_state = self.windows[window_type]
        config = self.configs[window_type]
        
        # Estimate memory usage (rough calculation)
        bar_size = 200  # bytes per bar (estimated)
        window_state.memory_usage_mb = (len(window_state.bars) * bar_size) / (1024 * 1024)
    
    def _is_memory_limit_exceeded(self) -> bool:
        """Check if total memory usage exceeds limit."""
        total_memory = sum(
            window_state.memory_usage_mb 
            for window_state in self.windows.values()
        )
        return total_memory > self.memory_limit_mb
    
    def _cleanup_old_data(self):
        """Clean up old data to free memory."""
        self.logger.info("Cleaning up old data to free memory")
        
        # Sort windows by priority (lower number = higher priority)
        sorted_windows = sorted(
            self.windows.items(),
            key=lambda x: self.configs[x[0]].priority
        )
        
        # Remove oldest data from lowest priority windows first
        for window_type, window_state in sorted_windows:
            if len(window_state.bars) > 10:  # Keep at least 10 bars
                # Remove oldest 25% of bars
                remove_count = len(window_state.bars) // 4
                for _ in range(remove_count):
                    window_state.bars.popleft()
                
                self._update_memory_usage(window_type)
                
                if not self._is_memory_limit_exceeded():
                    break
        
        self._last_cleanup = datetime.now()
    
    async def start_evaluation_loop(self):
        """Start the evaluation loop for window management."""
        self._is_running = True
        self.logger.info("Starting rolling window evaluation loop")
        
        while self._is_running:
            try:
                await self._evaluate_windows()
                await asyncio.sleep(self.evaluation_cadence)
            except Exception as e:
                self.logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def stop_evaluation_loop(self):
        """Stop the evaluation loop."""
        self._is_running = False
        self.logger.info("Stopped rolling window evaluation loop")
    
    async def _evaluate_windows(self):
        """Evaluate all windows for maintenance and optimization."""
        current_time = datetime.now()
        
        with self._lock:
            # Check for windows that need reset
            for window_type, window_state in self.windows.items():
                if window_state.last_update and self._should_reset_window(window_type, current_time):
                    self._reset_window(window_type)
            
            # Periodic cleanup
            if (current_time - self._last_cleanup).total_seconds() > 300:  # 5 minutes
                if self._is_memory_limit_exceeded():
                    self._cleanup_old_data()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all windows."""
        with self._lock:
            total_memory = sum(
                window_state.memory_usage_mb 
                for window_state in self.windows.values()
            )
            
            return {
                'total_windows': len(self.windows),
                'active_windows': sum(1 for w in self.windows.values() if w.is_active),
                'total_memory_mb': total_memory,
                'memory_limit_mb': self.memory_limit_mb,
                'memory_usage_percent': (total_memory / self.memory_limit_mb) * 100,
                'total_hits': self._total_hits,
                'total_misses': self._total_misses,
                'hit_rate': self._total_hits / max(1, self._total_hits + self._total_misses),
                'last_cleanup': self._last_cleanup,
                'windows': self.get_all_windows_info()
            }