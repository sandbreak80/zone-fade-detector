"""
Unit tests for Micro Window Analyzer.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from zone_fade_detector.core.micro_window_analyzer import (
    MicroWindowAnalyzer, InitiativeType, MicroWindowType, InitiativeMetrics,
    ZoneTouchAnalysis, MicroWindowState, MicroWindowConfig
)
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.models import OHLCVBar, Zone


class TestMicroWindowAnalyzer:
    """Test cases for MicroWindowAnalyzer."""
    
    @pytest.fixture
    def window_manager(self):
        """Create a mock window manager."""
        manager = Mock(spec=RollingWindowManager)
        manager.get_window_bars.return_value = []
        return manager
    
    @pytest.fixture
    def micro_analyzer(self, window_manager):
        """Create a MicroWindowAnalyzer instance for testing."""
        return MicroWindowAnalyzer(
            window_manager,
            pre_touch_minutes=15,
            post_touch_minutes=10,
            min_bars_for_analysis=5
        )
    
    @pytest.fixture
    def sample_zone(self):
        """Create a sample zone for testing."""
        return Zone(
            zone_id="test_zone",
            symbol="SPY",
            zone_type="supply",
            high=100.0,
            low=99.0,
            strength=0.8,
            touches=2,
            created_at=datetime.now(),
            last_touch=datetime.now()
        )
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 2, 10, 0)
        bars = []
        
        # Create bars with different patterns
        for i in range(30):
            # Simulate different patterns
            if i < 10:  # Pre-touch: normal movement
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=99.0 + i * 0.01,
                    high=99.2 + i * 0.01,
                    low=98.8 + i * 0.01,
                    close=99.1 + i * 0.01,
                    volume=1000 + i * 10
                )
            elif i == 10:  # Touch bar: high volume
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=99.1,
                    high=100.0,  # Touch the zone
                    low=99.0,
                    close=99.5,
                    volume=5000  # High volume
                )
            else:  # Post-touch: rejection pattern
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=99.5 - (i - 10) * 0.02,
                    high=99.7 - (i - 10) * 0.02,
                    low=99.3 - (i - 10) * 0.02,
                    close=99.4 - (i - 10) * 0.02,
                    volume=2000 - (i - 10) * 50
                )
            bars.append(bar)
        
        return bars
    
    def test_initialization(self, micro_analyzer):
        """Test MicroWindowAnalyzer initialization."""
        assert micro_analyzer.pre_touch_minutes == 15
        assert micro_analyzer.post_touch_minutes == 10
        assert micro_analyzer.min_bars_for_analysis == 5
        assert micro_analyzer.volume_spike_threshold == 1.5
        assert micro_analyzer.volatility_spike_threshold == 1.3
        assert len(micro_analyzer.active_windows) == 0
        assert len(micro_analyzer.completed_analyses) == 0
    
    def test_analyze_zone_touch_success(self, micro_analyzer, sample_zone, sample_bars, window_manager):
        """Test successful zone touch analysis."""
        # Mock window manager to return sample bars
        window_manager.get_window_bars.return_value = sample_bars
        
        touch_bar = sample_bars[10]  # The touch bar
        analysis = micro_analyzer.analyze_zone_touch(sample_zone, touch_bar, "SPY")
        
        assert analysis is not None
        assert analysis.zone == sample_zone
        assert analysis.touch_timestamp == touch_bar.timestamp
        assert analysis.touch_type == "supply"
        assert analysis.touch_price == 100.0  # Zone high
        assert analysis.is_significant is not None
        assert analysis.confidence_score >= 0.0
        assert analysis.confidence_score <= 1.0
        assert analysis.absorption_detected is not None
        assert analysis.exhaustion_detected is not None
        assert analysis.rejection_confirmed is not None
        assert len(analysis.micro_window_bars) > 0
        
        # Check that analysis was stored
        assert len(micro_analyzer.completed_analyses) == 1
        assert micro_analyzer.completed_analyses[0] == analysis
    
    def test_analyze_zone_touch_insufficient_bars(self, micro_analyzer, sample_zone, window_manager):
        """Test zone touch analysis with insufficient bars."""
        # Mock window manager to return few bars
        window_manager.get_window_bars.return_value = [sample_bars[0], sample_bars[1]]
        
        touch_bar = sample_bars[1]
        analysis = micro_analyzer.analyze_zone_touch(sample_zone, touch_bar, "SPY")
        
        assert analysis is None
        assert len(micro_analyzer.completed_analyses) == 0
    
    def test_determine_touch_type(self, micro_analyzer, sample_zone):
        """Test touch type determination."""
        # Supply zone
        supply_zone = Zone(
            zone_id="supply_zone",
            symbol="SPY",
            zone_type="supply",
            high=100.0,
            low=99.0,
            strength=0.8,
            touches=1,
            created_at=datetime.now(),
            last_touch=datetime.now()
        )
        
        touch_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=99.5,
            high=100.0,
            low=99.0,
            close=99.8,
            volume=1000
        )
        
        touch_type = micro_analyzer._determine_touch_type(supply_zone, touch_bar)
        assert touch_type == "supply"
        
        # Demand zone
        demand_zone = Zone(
            zone_id="demand_zone",
            symbol="SPY",
            zone_type="demand",
            high=100.0,
            low=99.0,
            strength=0.8,
            touches=1,
            created_at=datetime.now(),
            last_touch=datetime.now()
        )
        
        touch_type = micro_analyzer._determine_touch_type(demand_zone, touch_bar)
        assert touch_type == "demand"
    
    def test_get_touch_price(self, micro_analyzer, sample_zone):
        """Test touch price calculation."""
        touch_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=99.5,
            high=100.0,
            low=99.0,
            close=99.8,
            volume=1000
        )
        
        touch_price = micro_analyzer._get_touch_price(sample_zone, touch_bar)
        assert touch_price == 100.0  # Zone high for supply zone
    
    def test_calculate_volume_ratio(self, micro_analyzer, sample_bars):
        """Test volume ratio calculation."""
        volume_ratio = micro_analyzer._calculate_volume_ratio(sample_bars)
        assert volume_ratio > 0
        assert isinstance(volume_ratio, float)
    
    def test_calculate_price_momentum(self, micro_analyzer, sample_bars):
        """Test price momentum calculation."""
        momentum = micro_analyzer._calculate_price_momentum(sample_bars)
        assert isinstance(momentum, float)
    
    def test_calculate_wick_ratio(self, micro_analyzer, sample_bars):
        """Test wick ratio calculation."""
        wick_ratio = micro_analyzer._calculate_wick_ratio(sample_bars)
        assert 0.0 <= wick_ratio <= 1.0
        assert isinstance(wick_ratio, float)
    
    def test_calculate_rejection_clarity(self, micro_analyzer, sample_bars):
        """Test rejection clarity calculation."""
        clarity = micro_analyzer._calculate_rejection_clarity(sample_bars)
        assert 0.0 <= clarity <= 1.0
        assert isinstance(clarity, float)
    
    def test_detect_volatility_spike(self, micro_analyzer):
        """Test volatility spike detection."""
        # Create bars with volatility spike
        base_time = datetime.now()
        bars = []
        
        for i in range(5):
            if i == 3:  # Spike bar
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=105.0,  # High volatility
                    low=95.0,
                    close=102.0,
                    volume=1000
                )
            else:
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.5,
                    low=99.5,
                    close=100.2,
                    volume=1000
                )
            bars.append(bar)
        
        has_spike = micro_analyzer._detect_volatility_spike(bars)
        assert isinstance(has_spike, bool)
    
    def test_count_absorption_signals(self, micro_analyzer):
        """Test absorption signal counting."""
        # Create bars with absorption patterns
        base_time = datetime.now()
        bars = []
        
        for i in range(5):
            if i == 2:  # Absorption bar
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.1,  # Small range
                    low=99.9,
                    close=100.05,
                    volume=5000  # High volume
                )
            else:
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.5,
                    low=99.5,
                    close=100.2,
                    volume=1000
                )
            bars.append(bar)
        
        absorption_count = micro_analyzer._count_absorption_signals(bars)
        assert absorption_count >= 0
        assert isinstance(absorption_count, int)
    
    def test_count_exhaustion_signals(self, micro_analyzer):
        """Test exhaustion signal counting."""
        # Create bars with exhaustion patterns
        base_time = datetime.now()
        bars = []
        
        for i in range(5):
            if i == 2:  # Exhaustion bar
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=101.0,  # Large range
                    low=99.0,
                    close=100.5,
                    volume=500  # Low volume
                )
            else:
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0,
                    high=100.5,
                    low=99.5,
                    close=100.2,
                    volume=1000
                )
            bars.append(bar)
        
        exhaustion_count = micro_analyzer._count_exhaustion_signals(bars)
        assert exhaustion_count >= 0
        assert isinstance(exhaustion_count, int)
    
    def test_count_consecutive_direction_bars(self, micro_analyzer):
        """Test consecutive direction bar counting."""
        # Create bars with consecutive direction
        base_time = datetime.now()
        bars = []
        
        for i in range(5):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.2,  # All bullish
                volume=1000
            )
            bars.append(bar)
        
        consecutive_count = micro_analyzer._count_consecutive_direction_bars(bars)
        assert consecutive_count == 5
        assert isinstance(consecutive_count, int)
    
    def test_determine_initiative_type(self, micro_analyzer):
        """Test initiative type determination."""
        # Test absorption
        initiative_type = micro_analyzer._determine_initiative_type(
            price_momentum=0.0,
            volume_ratio=1.0,
            wick_ratio=0.0,
            rejection_clarity=0.0,
            absorption_signals=3,
            exhaustion_signals=0
        )
        assert initiative_type == InitiativeType.ABSORPTION
        
        # Test exhaustion
        initiative_type = micro_analyzer._determine_initiative_type(
            price_momentum=0.0,
            volume_ratio=1.0,
            wick_ratio=0.0,
            rejection_clarity=0.0,
            absorption_signals=0,
            exhaustion_signals=3
        )
        assert initiative_type == InitiativeType.EXHAUSTION
        
        # Test bullish
        initiative_type = micro_analyzer._determine_initiative_type(
            price_momentum=0.02,
            volume_ratio=1.5,
            wick_ratio=0.0,
            rejection_clarity=0.0,
            absorption_signals=0,
            exhaustion_signals=0
        )
        assert initiative_type == InitiativeType.BULLISH
        
        # Test bearish
        initiative_type = micro_analyzer._determine_initiative_type(
            price_momentum=-0.02,
            volume_ratio=1.5,
            wick_ratio=0.0,
            rejection_clarity=0.0,
            absorption_signals=0,
            exhaustion_signals=0
        )
        assert initiative_type == InitiativeType.BEARISH
        
        # Test neutral
        initiative_type = micro_analyzer._determine_initiative_type(
            price_momentum=0.0,
            volume_ratio=1.0,
            wick_ratio=0.0,
            rejection_clarity=0.8,
            absorption_signals=0,
            exhaustion_signals=0
        )
        assert initiative_type == InitiativeType.NEUTRAL
    
    def test_calculate_strength_score(self, micro_analyzer):
        """Test strength score calculation."""
        score = micro_analyzer._calculate_strength_score(
            volume_ratio=2.0,
            price_momentum=0.02,
            wick_ratio=0.5,
            rejection_clarity=0.8,
            absorption_signals=2,
            exhaustion_signals=1,
            consecutive_bars=3
        )
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_is_significant_touch(self, micro_analyzer, sample_zone):
        """Test significant touch detection."""
        # Create high-strength metrics
        high_strength = InitiativeMetrics(
            initiative_type=InitiativeType.BULLISH,
            strength_score=0.8,
            volume_ratio=2.0,
            price_momentum=0.02,
            wick_ratio=0.5,
            rejection_clarity=0.8,
            absorption_signals=2,
            exhaustion_signals=1,
            consecutive_bars=3,
            volatility_spike=True,
            volume_spike=True
        )
        
        # Create low-strength metrics
        low_strength = InitiativeMetrics(
            initiative_type=InitiativeType.NEUTRAL,
            strength_score=0.2,
            volume_ratio=1.0,
            price_momentum=0.0,
            wick_ratio=0.2,
            rejection_clarity=0.3,
            absorption_signals=0,
            exhaustion_signals=0,
            consecutive_bars=1,
            volatility_spike=False,
            volume_spike=False
        )
        
        touch_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=99.5,
            high=100.0,
            low=99.0,
            close=99.8,
            volume=1000
        )
        
        # Test with high strength
        is_significant = micro_analyzer._is_significant_touch(
            high_strength, low_strength, sample_zone, touch_bar
        )
        assert is_significant is True
        
        # Test with low strength
        is_significant = micro_analyzer._is_significant_touch(
            low_strength, low_strength, sample_zone, touch_bar
        )
        assert is_significant is False
    
    def test_calculate_confidence_score(self, micro_analyzer, sample_zone):
        """Test confidence score calculation."""
        high_strength = InitiativeMetrics(
            initiative_type=InitiativeType.ABSORPTION,
            strength_score=0.8,
            volume_ratio=2.0,
            price_momentum=0.02,
            wick_ratio=0.5,
            rejection_clarity=0.8,
            absorption_signals=2,
            exhaustion_signals=1,
            consecutive_bars=3,
            volatility_spike=True,
            volume_spike=True
        )
        
        touch_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=99.5,
            high=100.0,
            low=99.0,
            close=99.8,
            volume=1000
        )
        
        confidence = micro_analyzer._calculate_confidence_score(
            high_strength, high_strength, sample_zone, touch_bar
        )
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_detect_absorption(self, micro_analyzer):
        """Test absorption detection."""
        # Create metrics with absorption
        absorption_metrics = InitiativeMetrics(
            initiative_type=InitiativeType.ABSORPTION,
            strength_score=0.8,
            volume_ratio=2.0,
            price_momentum=0.02,
            wick_ratio=0.5,
            rejection_clarity=0.8,
            absorption_signals=3,
            exhaustion_signals=0,
            consecutive_bars=2,
            volatility_spike=True,
            volume_spike=True
        )
        
        neutral_metrics = InitiativeMetrics(
            initiative_type=InitiativeType.NEUTRAL,
            strength_score=0.3,
            volume_ratio=1.0,
            price_momentum=0.0,
            wick_ratio=0.2,
            rejection_clarity=0.3,
            absorption_signals=0,
            exhaustion_signals=0,
            consecutive_bars=1,
            volatility_spike=False,
            volume_spike=False
        )
        
        # Test with absorption
        absorption_detected = micro_analyzer._detect_absorption(
            absorption_metrics, neutral_metrics
        )
        assert absorption_detected is True
        
        # Test without absorption
        absorption_detected = micro_analyzer._detect_absorption(
            neutral_metrics, neutral_metrics
        )
        assert absorption_detected is False
    
    def test_detect_exhaustion(self, micro_analyzer):
        """Test exhaustion detection."""
        # Create metrics with exhaustion
        exhaustion_metrics = InitiativeMetrics(
            initiative_type=InitiativeType.EXHAUSTION,
            strength_score=0.8,
            volume_ratio=2.0,
            price_momentum=0.02,
            wick_ratio=0.5,
            rejection_clarity=0.8,
            absorption_signals=0,
            exhaustion_signals=3,
            consecutive_bars=2,
            volatility_spike=True,
            volume_spike=True
        )
        
        neutral_metrics = InitiativeMetrics(
            initiative_type=InitiativeType.NEUTRAL,
            strength_score=0.3,
            volume_ratio=1.0,
            price_momentum=0.0,
            wick_ratio=0.2,
            rejection_clarity=0.3,
            absorption_signals=0,
            exhaustion_signals=0,
            consecutive_bars=1,
            volatility_spike=False,
            volume_spike=False
        )
        
        # Test with exhaustion
        exhaustion_detected = micro_analyzer._detect_exhaustion(
            exhaustion_metrics, neutral_metrics
        )
        assert exhaustion_detected is True
        
        # Test without exhaustion
        exhaustion_detected = micro_analyzer._detect_exhaustion(
            neutral_metrics, neutral_metrics
        )
        assert exhaustion_detected is False
    
    def test_confirm_rejection(self, micro_analyzer, sample_zone):
        """Test rejection confirmation."""
        # Create metrics with high rejection clarity
        high_rejection = InitiativeMetrics(
            initiative_type=InitiativeType.BEARISH,
            strength_score=0.8,
            volume_ratio=2.0,
            price_momentum=-0.02,
            wick_ratio=0.8,
            rejection_clarity=0.8,
            absorption_signals=0,
            exhaustion_signals=0,
            consecutive_bars=2,
            volatility_spike=True,
            volume_spike=True
        )
        
        touch_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=99.5,
            high=100.0,
            low=99.0,
            close=99.8,
            volume=1000
        )
        
        # Test with high rejection clarity
        rejection_confirmed = micro_analyzer._confirm_rejection(
            high_rejection, sample_zone, touch_bar
        )
        assert rejection_confirmed is True
    
    def test_get_recent_analyses(self, micro_analyzer, sample_zone, sample_bars, window_manager):
        """Test getting recent analyses."""
        # Mock window manager
        window_manager.get_window_bars.return_value = sample_bars
        
        # Create multiple analyses
        for i in range(5):
            touch_bar = sample_bars[i * 5]
            analysis = micro_analyzer.analyze_zone_touch(sample_zone, touch_bar, "SPY")
            assert analysis is not None
        
        # Test getting recent analyses
        recent = micro_analyzer.get_recent_analyses(limit=3)
        assert len(recent) == 3
        assert recent[0] == micro_analyzer.completed_analyses[-1]
    
    def test_get_significant_touches(self, micro_analyzer, sample_zone, sample_bars, window_manager):
        """Test getting significant touches."""
        # Mock window manager
        window_manager.get_window_bars.return_value = sample_bars
        
        # Create analyses
        for i in range(3):
            touch_bar = sample_bars[i * 10]
            analysis = micro_analyzer.analyze_zone_touch(sample_zone, touch_bar, "SPY")
            assert analysis is not None
        
        # Test getting significant touches
        significant = micro_analyzer.get_significant_touches()
        assert isinstance(significant, list)
        assert all(isinstance(a, ZoneTouchAnalysis) for a in significant)
    
    def test_get_analysis_summary(self, micro_analyzer, sample_zone, sample_bars, window_manager):
        """Test getting analysis summary."""
        # Mock window manager
        window_manager.get_window_bars.return_value = sample_bars
        
        # Create some analyses
        for i in range(3):
            touch_bar = sample_bars[i * 10]
            analysis = micro_analyzer.analyze_zone_touch(sample_zone, touch_bar, "SPY")
            assert analysis is not None
        
        # Test getting summary
        summary = micro_analyzer.get_analysis_summary()
        assert isinstance(summary, dict)
        assert "total_analyses" in summary
        assert "significant_touches" in summary
        assert "absorption_touches" in summary
        assert "exhaustion_touches" in summary
        assert "rejection_touches" in summary
        assert "average_confidence" in summary
        assert summary["total_analyses"] == 3


class TestInitiativeType:
    """Test cases for InitiativeType enum."""
    
    def test_initiative_type_values(self):
        """Test InitiativeType enum values."""
        assert InitiativeType.BULLISH.value == "bullish"
        assert InitiativeType.BEARISH.value == "bearish"
        assert InitiativeType.NEUTRAL.value == "neutral"
        assert InitiativeType.EXHAUSTION.value == "exhaustion"
        assert InitiativeType.ABSORPTION.value == "absorption"


class TestMicroWindowType:
    """Test cases for MicroWindowType enum."""
    
    def test_micro_window_type_values(self):
        """Test MicroWindowType enum values."""
        assert MicroWindowType.PRE_TOUCH.value == "pre_touch"
        assert MicroWindowType.POST_TOUCH.value == "post_touch"
        assert MicroWindowType.ZONE_APPROACH.value == "zone_approach"
        assert MicroWindowType.ZONE_REJECTION.value == "zone_rejection"


class TestInitiativeMetrics:
    """Test cases for InitiativeMetrics dataclass."""
    
    def test_initiative_metrics_creation(self):
        """Test InitiativeMetrics creation."""
        metrics = InitiativeMetrics(
            initiative_type=InitiativeType.BULLISH,
            strength_score=0.8,
            volume_ratio=2.0,
            price_momentum=0.02,
            wick_ratio=0.5,
            rejection_clarity=0.8,
            absorption_signals=2,
            exhaustion_signals=1,
            consecutive_bars=3,
            volatility_spike=True,
            volume_spike=True
        )
        
        assert metrics.initiative_type == InitiativeType.BULLISH
        assert metrics.strength_score == 0.8
        assert metrics.volume_ratio == 2.0
        assert metrics.price_momentum == 0.02
        assert metrics.wick_ratio == 0.5
        assert metrics.rejection_clarity == 0.8
        assert metrics.absorption_signals == 2
        assert metrics.exhaustion_signals == 1
        assert metrics.consecutive_bars == 3
        assert metrics.volatility_spike is True
        assert metrics.volume_spike is True


class TestZoneTouchAnalysis:
    """Test cases for ZoneTouchAnalysis dataclass."""
    
    def test_zone_touch_analysis_creation(self):
        """Test ZoneTouchAnalysis creation."""
        zone = Zone(
            zone_id="test_zone",
            symbol="SPY",
            zone_type="supply",
            high=100.0,
            low=99.0,
            strength=0.8,
            touches=1,
            created_at=datetime.now(),
            last_touch=datetime.now()
        )
        
        pre_metrics = InitiativeMetrics(
            initiative_type=InitiativeType.BULLISH,
            strength_score=0.6,
            volume_ratio=1.5,
            price_momentum=0.01,
            wick_ratio=0.3,
            rejection_clarity=0.5,
            absorption_signals=1,
            exhaustion_signals=0,
            consecutive_bars=2,
            volatility_spike=False,
            volume_spike=True
        )
        
        post_metrics = InitiativeMetrics(
            initiative_type=InitiativeType.BEARISH,
            strength_score=0.7,
            volume_ratio=1.8,
            price_momentum=-0.01,
            wick_ratio=0.6,
            rejection_clarity=0.8,
            absorption_signals=0,
            exhaustion_signals=1,
            consecutive_bars=3,
            volatility_spike=True,
            volume_spike=True
        )
        
        analysis = ZoneTouchAnalysis(
            zone=zone,
            touch_timestamp=datetime.now(),
            touch_price=100.0,
            touch_type="supply",
            pre_touch_analysis=pre_metrics,
            post_touch_analysis=post_metrics,
            micro_window_bars=[],
            is_significant=True,
            confidence_score=0.8,
            absorption_detected=False,
            exhaustion_detected=True,
            rejection_confirmed=True
        )
        
        assert analysis.zone == zone
        assert analysis.touch_type == "supply"
        assert analysis.touch_price == 100.0
        assert analysis.pre_touch_analysis == pre_metrics
        assert analysis.post_touch_analysis == post_metrics
        assert analysis.is_significant is True
        assert analysis.confidence_score == 0.8
        assert analysis.absorption_detected is False
        assert analysis.exhaustion_detected is True
        assert analysis.rejection_confirmed is True


class TestMicroWindowConfig:
    """Test cases for MicroWindowConfig dataclass."""
    
    def test_micro_window_config_creation(self):
        """Test MicroWindowConfig creation."""
        config = MicroWindowConfig(
            window_type=MicroWindowType.PRE_TOUCH,
            duration_minutes=15,
            max_bars=30,
            priority=3
        )
        
        assert config.window_type == MicroWindowType.PRE_TOUCH
        assert config.duration_minutes == 15
        assert config.max_bars == 30
        assert config.priority == 3


class TestMicroWindowState:
    """Test cases for MicroWindowState dataclass."""
    
    def test_micro_window_state_creation(self):
        """Test MicroWindowState creation."""
        zone = Zone(
            zone_id="test_zone",
            symbol="SPY",
            zone_type="supply",
            high=100.0,
            low=99.0,
            strength=0.8,
            touches=1,
            created_at=datetime.now(),
            last_touch=datetime.now()
        )
        
        state = MicroWindowState(
            window_type=MicroWindowType.PRE_TOUCH,
            zone=zone,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=15),
            bars=[],
            is_active=True,
            analysis_complete=False,
            initiative_metrics=None
        )
        
        assert state.window_type == MicroWindowType.PRE_TOUCH
        assert state.zone == zone
        assert state.is_active is True
        assert state.analysis_complete is False
        assert state.initiative_metrics is None