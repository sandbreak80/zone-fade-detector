"""
Unit tests for Rolling Window Manager.
"""

import pytest
from datetime import datetime, timedelta, time
from unittest.mock import Mock, patch

from zone_fade_detector.core.rolling_window_manager import (
    RollingWindowManager, WindowType, WindowConfig, WindowState
)
from zone_fade_detector.core.models import OHLCVBar


class TestRollingWindowManager:
    """Test cases for RollingWindowManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a RollingWindowManager instance for testing."""
        return RollingWindowManager(evaluation_cadence_seconds=1, memory_limit_mb=100)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 2, 9, 30)  # RTH start
        bars = []
        
        for i in range(100):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.2 + i * 0.1,
                volume=1000 + i * 10
            )
            bars.append(bar)
        
        return bars
    
    def test_initialization(self, manager):
        """Test RollingWindowManager initialization."""
        assert len(manager.windows) == 8  # All window types
        assert manager.evaluation_cadence == 1
        assert manager.memory_limit_mb == 100
        
        # Check all window types are initialized
        for window_type in WindowType:
            assert window_type in manager.windows
            assert manager.windows[window_type].is_active
    
    def test_add_bar(self, manager, sample_bars):
        """Test adding bars to windows."""
        bar = sample_bars[0]
        
        # Add bar to HTF_ZONES window
        result = manager.add_bar(WindowType.HTF_ZONES, bar)
        assert result is True
        
        # Check bar was added
        window_bars = manager.get_window_bars(WindowType.HTF_ZONES)
        assert len(window_bars) == 1
        assert window_bars[0] == bar
    
    def test_get_window_bars(self, manager, sample_bars):
        """Test getting bars from windows."""
        # Add multiple bars
        for i in range(5):
            manager.add_bar(WindowType.VWAP_COMPUTATION, sample_bars[i])
        
        # Get bars
        window_bars = manager.get_window_bars(WindowType.VWAP_COMPUTATION)
        assert len(window_bars) == 5
        
        # Check bars are in correct order
        for i, bar in enumerate(window_bars):
            assert bar == sample_bars[i]
    
    def test_get_window_for_timestamp(self, manager, sample_bars):
        """Test getting window bars for specific timestamp."""
        # Add bars spanning 2 hours
        for i in range(120):  # 120 minutes
            manager.add_bar(WindowType.SWING_CHOCH, sample_bars[i])
        
        # Get window for timestamp 1 hour in
        target_time = sample_bars[0].timestamp + timedelta(minutes=60)
        window_bars = manager.get_window_for_timestamp(WindowType.SWING_CHOCH, target_time)
        
        # Should get bars within the 20-minute window
        assert len(window_bars) <= 20  # Max window size
        assert all(bar.timestamp <= target_time for bar in window_bars)
    
    def test_is_window_ready(self, manager, sample_bars):
        """Test window readiness check."""
        # Empty window should not be ready
        assert not manager.is_window_ready(WindowType.HTF_ZONES)
        
        # Add minimum bars
        for i in range(5):
            manager.add_bar(WindowType.HTF_ZONES, sample_bars[i])
        
        # Should be ready now
        assert manager.is_window_ready(WindowType.HTF_ZONES)
    
    def test_window_reset_on_session(self, manager, sample_bars):
        """Test window reset on new session."""
        # Add bar from previous session
        prev_session_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 1, 15, 30),  # Previous day
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        manager.add_bar(WindowType.SESSION_CONTEXT, prev_session_bar)
        
        # Add bar from new session
        new_session_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 9, 30),  # New day, RTH start
            open=101.0, high=101.5, low=100.5, close=101.2, volume=1000
        )
        manager.add_bar(WindowType.SESSION_CONTEXT, new_session_bar)
        
        # Window should be reset (only new session bar)
        window_bars = manager.get_window_bars(WindowType.SESSION_CONTEXT)
        assert len(window_bars) == 1
        assert window_bars[0] == new_session_bar
    
    def test_window_reset_on_day(self, manager, sample_bars):
        """Test window reset on new day."""
        # Add bar from previous day
        prev_day_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 1, 15, 30),
            open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
        )
        manager.add_bar(WindowType.HTF_ZONES, prev_day_bar)
        
        # Add bar from new day
        new_day_bar = OHLCVBar(
            timestamp=datetime(2024, 1, 2, 9, 30),
            open=101.0, high=101.5, low=100.5, close=101.2, volume=1000
        )
        manager.add_bar(WindowType.HTF_ZONES, new_day_bar)
        
        # Window should be reset (only new day bar)
        window_bars = manager.get_window_bars(WindowType.HTF_ZONES)
        assert len(window_bars) == 1
        assert window_bars[0] == new_day_bar
    
    def test_memory_management(self, manager, sample_bars):
        """Test memory management and cleanup."""
        # Add many bars to exceed memory limit
        for i in range(1000):
            bar = OHLCVBar(
                timestamp=datetime(2024, 1, 2, 9, 30) + timedelta(minutes=i),
                open=100.0, high=100.5, low=99.5, close=100.2, volume=1000
            )
            manager.add_bar(WindowType.VWAP_COMPUTATION, bar)
        
        # Check memory usage
        stats = manager.get_performance_stats()
        assert stats['memory_usage_percent'] > 0
        
        # Trigger cleanup
        manager._cleanup_old_data()
        
        # Memory usage should be reduced
        new_stats = manager.get_performance_stats()
        assert new_stats['memory_usage_percent'] < stats['memory_usage_percent']
    
    def test_window_info(self, manager, sample_bars):
        """Test getting window information."""
        # Add some bars
        for i in range(10):
            manager.add_bar(WindowType.SWING_CHOCH, sample_bars[i])
        
        # Get window info
        info = manager.get_window_info(WindowType.SWING_CHOCH)
        
        assert info['window_type'] == 'swing_choch'
        assert info['bar_count'] == 10
        assert info['is_active'] is True
        assert info['hit_count'] == 10
        assert info['hit_rate'] == 1.0
    
    def test_reset_window(self, manager, sample_bars):
        """Test resetting a specific window."""
        # Add bars
        for i in range(5):
            manager.add_bar(WindowType.OPENING_RANGE, sample_bars[i])
        
        # Reset window
        manager.reset_window(WindowType.OPENING_RANGE)
        
        # Window should be empty
        window_bars = manager.get_window_bars(WindowType.OPENING_RANGE)
        assert len(window_bars) == 0
    
    def test_reset_all_windows(self, manager, sample_bars):
        """Test resetting all windows."""
        # Add bars to multiple windows
        for window_type in [WindowType.HTF_ZONES, WindowType.VWAP_COMPUTATION]:
            for i in range(5):
                manager.add_bar(window_type, sample_bars[i])
        
        # Reset all windows
        manager.reset_all_windows()
        
        # All windows should be empty
        for window_type in [WindowType.HTF_ZONES, WindowType.VWAP_COMPUTATION]:
            window_bars = manager.get_window_bars(window_type)
            assert len(window_bars) == 0
    
    def test_performance_stats(self, manager, sample_bars):
        """Test performance statistics."""
        # Add some bars
        for i in range(20):
            manager.add_bar(WindowType.HTF_ZONES, sample_bars[i])
        
        # Get performance stats
        stats = manager.get_performance_stats()
        
        assert 'total_windows' in stats
        assert 'active_windows' in stats
        assert 'total_memory_mb' in stats
        assert 'hit_rate' in stats
        assert 'windows' in stats
        
        assert stats['total_windows'] == 8
        assert stats['active_windows'] == 8
        assert stats['hit_rate'] == 1.0
    
    def test_unknown_window_type(self, manager, sample_bars):
        """Test handling of unknown window types."""
        bar = sample_bars[0]
        
        # This should not raise an exception
        result = manager.add_bar(WindowType.HTF_ZONES, bar)
        assert result is True
        
        # Unknown window type should return empty list
        bars = manager.get_window_bars(WindowType.HTF_ZONES)
        assert isinstance(bars, list)
    
    def test_concurrent_access(self, manager, sample_bars):
        """Test thread safety of the manager."""
        import threading
        import time
        
        results = []
        errors = []
        
        def add_bars():
            try:
                for i in range(10):
                    bar = sample_bars[i]
                    result = manager.add_bar(WindowType.VWAP_COMPUTATION, bar)
                    results.append(result)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=add_bars)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 bars each
        assert all(results)  # All should be True
    
    @pytest.mark.asyncio
    async def test_evaluation_loop(self, manager, sample_bars):
        """Test the evaluation loop."""
        # Add some bars
        for i in range(10):
            manager.add_bar(WindowType.HTF_ZONES, sample_bars[i])
        
        # Start evaluation loop
        task = asyncio.create_task(manager.start_evaluation_loop())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop the loop
        manager.stop_evaluation_loop()
        task.cancel()
        
        # Should not raise any exceptions
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected when cancelling task


class TestWindowConfig:
    """Test cases for WindowConfig."""
    
    def test_window_config_creation(self):
        """Test WindowConfig creation."""
        config = WindowConfig(
            window_type=WindowType.HTF_ZONES,
            duration_minutes=1440,
            max_bars=1000,
            reset_on_day=True,
            priority=1
        )
        
        assert config.window_type == WindowType.HTF_ZONES
        assert config.duration_minutes == 1440
        assert config.max_bars == 1000
        assert config.reset_on_day is True
        assert config.priority == 1


class TestWindowState:
    """Test cases for WindowState."""
    
    def test_window_state_creation(self):
        """Test WindowState creation."""
        state = WindowState(window_type=WindowType.HTF_ZONES)
        
        assert state.window_type == WindowType.HTF_ZONES
        assert len(state.bars) == 0
        assert state.last_update is None
        assert state.is_active is True
        assert state.memory_usage_mb == 0.0
        assert state.hit_count == 0
        assert state.miss_count == 0