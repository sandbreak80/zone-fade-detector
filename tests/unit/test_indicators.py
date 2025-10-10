"""
Unit tests for technical indicators.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from zone_fade_detector.core.models import OHLCVBar, VWAPData, OpeningRange, SwingPoint, SetupDirection
from zone_fade_detector.indicators.vwap import VWAPCalculator
from zone_fade_detector.indicators.opening_range import OpeningRangeCalculator
from zone_fade_detector.indicators.swing_structure import SwingStructureDetector
from zone_fade_detector.indicators.volume_analysis import VolumeAnalyzer


class TestVWAPCalculator:
    """Test VWAP calculator functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create VWAP calculator instance."""
        return VWAPCalculator(lookback_hours=6.5)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        for i in range(10):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.2 + i * 0.1,
                volume=1000000 + i * 10000
            )
            bars.append(bar)
        
        return bars
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.lookback_hours == 6.5
        assert calculator.logger is not None
    
    def test_calculate_vwap_with_sufficient_data(self, calculator, sample_bars):
        """Test VWAP calculation with sufficient data."""
        vwap_data = calculator.calculate_vwap(sample_bars)
        
        assert vwap_data is not None
        assert isinstance(vwap_data, VWAPData)
        assert vwap_data.vwap > 0
        assert vwap_data.upper_1sigma > vwap_data.vwap
        assert vwap_data.lower_1sigma < vwap_data.vwap
        assert vwap_data.upper_2sigma > vwap_data.upper_1sigma
        assert vwap_data.lower_2sigma < vwap_data.lower_1sigma
    
    def test_calculate_vwap_insufficient_data(self, calculator):
        """Test VWAP calculation with insufficient data."""
        # Test with empty bars
        vwap_data = calculator.calculate_vwap([])
        assert vwap_data is None
        
        # Test with single bar
        single_bar = [OHLCVBar(
            timestamp=datetime.now(),
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000000
        )]
        vwap_data = calculator.calculate_vwap(single_bar)
        assert vwap_data is None
    
    def test_vwap_properties(self, calculator, sample_bars):
        """Test VWAP data properties."""
        vwap_data = calculator.calculate_vwap(sample_bars)
        
        assert vwap_data.is_flat is False  # Should have some slope
        assert isinstance(vwap_data.is_bullish, bool)
        assert isinstance(vwap_data.is_bearish, bool)
    
    def test_is_trend_day(self, calculator):
        """Test trend day detection."""
        # Create VWAP data with high slope
        vwap_data = VWAPData(
            vwap=100.0,
            upper_1sigma=102.0,
            lower_1sigma=98.0,
            upper_2sigma=104.0,
            lower_2sigma=96.0,
            slope=0.005,  # High slope
            timestamp=datetime.now()
        )
        
        assert calculator.is_trend_day(vwap_data) is True
        
        # Test with low slope
        vwap_data.slope = 0.001
        assert calculator.is_trend_day(vwap_data) is False


class TestOpeningRangeCalculator:
    """Test Opening Range calculator functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create Opening Range calculator instance."""
        return OpeningRangeCalculator(duration_minutes=30)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        for i in range(35):  # 35 minutes of data
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.2 + i * 0.1,
                volume=1000000 + i * 10000
            )
            bars.append(bar)
        
        return bars
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.duration_minutes == 30
        assert calculator.logger is not None
    
    def test_calculate_opening_range(self, calculator, sample_bars):
        """Test opening range calculation."""
        or_data = calculator.calculate_opening_range(sample_bars)
        
        assert or_data is not None
        assert isinstance(or_data, OpeningRange)
        assert or_data.high > or_data.low
        assert or_data.range_size > 0
        assert or_data.mid_point > 0
        assert or_data.is_valid is True
    
    def test_calculate_opening_range_insufficient_data(self, calculator):
        """Test opening range calculation with insufficient data."""
        # Test with empty bars
        or_data = calculator.calculate_opening_range([])
        assert or_data is None
        
        # Test with bars outside OR period
        bars = [OHLCVBar(
            timestamp=datetime(2024, 1, 15, 10, 30),  # After OR period
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000000
        )]
        or_data = calculator.calculate_opening_range(bars)
        assert or_data is None
    
    def test_is_price_in_opening_range(self, calculator):
        """Test price in opening range detection."""
        or_data = OpeningRange(
            high=105.0,
            low=95.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            volume=1000000
        )
        
        assert calculator.is_price_in_opening_range(100.0, or_data) is True
        assert calculator.is_price_in_opening_range(110.0, or_data) is False
        assert calculator.is_price_in_opening_range(90.0, or_data) is False
    
    def test_calculate_or_breakout_levels(self, calculator):
        """Test OR breakout levels calculation."""
        or_data = OpeningRange(
            high=105.0,
            low=95.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            volume=1000000
        )
        
        upside, downside = calculator.calculate_or_breakout_levels(or_data)
        
        assert upside > or_data.high
        assert downside < or_data.low
    
    def test_calculate_or_quality_score(self, calculator):
        """Test OR quality score calculation."""
        or_data = OpeningRange(
            high=105.0,
            low=95.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            volume=1000000
        )
        
        recent_ranges = [
            OpeningRange(high=104.0, low=96.0, start_time=datetime.now(), end_time=datetime.now(), volume=900000),
            OpeningRange(high=103.0, low=97.0, start_time=datetime.now(), end_time=datetime.now(), volume=950000)
        ]
        
        score = calculator.calculate_or_quality_score(or_data, recent_ranges)
        
        assert 0.0 <= score <= 1.0


class TestSwingStructureDetector:
    """Test Swing Structure detector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create swing structure detector instance."""
        return SwingStructureDetector(lookback_bars=20, min_swing_size=0.1)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        # Create bars with swing pattern
        prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100, 101, 102]
        
        for i, price in enumerate(prices):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price + 0.1,
                volume=1000000
            )
            bars.append(bar)
        
        return bars
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.lookback_bars == 20
        assert detector.min_swing_size == 0.1
        assert detector.swing_confirmation_bars == 2
        assert detector.logger is not None
    
    def test_detect_swing_structure(self, detector, sample_bars):
        """Test swing structure detection."""
        structure = detector.detect_swing_structure(sample_bars)
        
        assert structure is not None
        assert isinstance(structure.swing_highs, list)
        assert isinstance(structure.swing_lows, list)
        assert isinstance(structure.choch_detected, bool)
    
    def test_detect_swing_structure_insufficient_data(self, detector):
        """Test swing structure detection with insufficient data."""
        # Test with empty bars
        structure = detector.detect_swing_structure([])
        assert len(structure.swing_highs) == 0
        assert len(structure.swing_lows) == 0
        
        # Test with insufficient bars
        single_bar = [OHLCVBar(
            timestamp=datetime.now(),
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000000
        )]
        structure = detector.detect_swing_structure(single_bar)
        assert len(structure.swing_highs) == 0
        assert len(structure.swing_lows) == 0
    
    def test_detect_choch(self, detector):
        """Test CHoCH detection."""
        # Create mock swing structure
        structure = Mock()
        structure.swing_highs = [
            SwingPoint(price=102.0, timestamp=datetime.now(), is_high=True),
            SwingPoint(price=104.0, timestamp=datetime.now(), is_high=True)
        ]
        structure.swing_lows = [
            SwingPoint(price=98.0, timestamp=datetime.now(), is_high=False),
            SwingPoint(price=96.0, timestamp=datetime.now(), is_high=False)
        ]
        
        choch_detected, direction, timestamp = detector.detect_choch(structure)
        
        assert isinstance(choch_detected, bool)
        if choch_detected:
            assert direction in [SetupDirection.LONG, SetupDirection.SHORT]
            assert timestamp is not None
    
    def test_calculate_swing_strength_score(self, detector):
        """Test swing strength score calculation."""
        structure = Mock()
        structure.swing_highs = [
            SwingPoint(price=102.0, timestamp=datetime.now(), is_high=True, strength=1.5)
        ]
        structure.swing_lows = [
            SwingPoint(price=98.0, timestamp=datetime.now(), is_high=False, strength=1.2)
        ]
        
        score = detector.calculate_swing_strength_score(structure)
        
        assert 0.0 <= score <= 1.0


class TestVolumeAnalyzer:
    """Test Volume Analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create volume analyzer instance."""
        return VolumeAnalyzer(lookback_bars=20, expansion_threshold=1.5, contraction_threshold=0.7)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        for i in range(25):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.2 + i * 0.1,
                volume=1000000 + i * 50000  # Increasing volume
            )
            bars.append(bar)
        
        return bars
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.lookback_bars == 20
        assert analyzer.expansion_threshold == 1.5
        assert analyzer.contraction_threshold == 0.7
        assert analyzer.logger is not None
    
    def test_analyze_volume(self, analyzer, sample_bars):
        """Test volume analysis."""
        volume_analysis = analyzer.analyze_volume(sample_bars)
        
        assert volume_analysis is not None
        assert volume_analysis.current_volume > 0
        assert volume_analysis.average_volume > 0
        assert volume_analysis.volume_ratio > 0
        assert isinstance(volume_analysis.is_expansion, bool)
        assert isinstance(volume_analysis.is_contraction, bool)
    
    def test_analyze_volume_insufficient_data(self, analyzer):
        """Test volume analysis with insufficient data."""
        # Test with empty bars
        volume_analysis = analyzer.analyze_volume([])
        assert volume_analysis is None
        
        # Test with insufficient bars
        single_bar = [OHLCVBar(
            timestamp=datetime.now(),
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000000
        )]
        volume_analysis = analyzer.analyze_volume(single_bar)
        assert volume_analysis is None
    
    def test_detect_volume_expansion(self, analyzer, sample_bars):
        """Test volume expansion detection."""
        # Test with high volume bar
        high_volume_bars = sample_bars.copy()
        high_volume_bars[-1].volume = 5000000  # High volume
        
        expansion = analyzer.detect_volume_expansion(high_volume_bars)
        assert isinstance(expansion, bool)
    
    def test_detect_volume_contraction(self, analyzer, sample_bars):
        """Test volume contraction detection."""
        # Test with low volume bar
        low_volume_bars = sample_bars.copy()
        low_volume_bars[-1].volume = 100000  # Low volume
        
        contraction = analyzer.detect_volume_contraction(low_volume_bars)
        assert isinstance(contraction, bool)
    
    def test_calculate_volume_profile(self, analyzer, sample_bars):
        """Test volume profile calculation."""
        profile = analyzer.calculate_volume_profile(sample_bars, price_levels=10)
        
        assert isinstance(profile, dict)
        assert len(profile) > 0
        assert all(isinstance(price, float) for price in profile.keys())
        assert all(isinstance(volume, int) for volume in profile.values())
    
    def test_detect_initiative_volume(self, analyzer, sample_bars):
        """Test initiative volume detection."""
        has_initiative, direction = analyzer.detect_initiative_volume(sample_bars)
        
        assert isinstance(has_initiative, bool)
        if has_initiative:
            assert direction in ["bullish", "bearish"]
    
    def test_get_volume_signals(self, analyzer, sample_bars):
        """Test comprehensive volume signals."""
        signals = analyzer.get_volume_signals(sample_bars)
        
        assert isinstance(signals, dict)
        assert 'current_volume' in signals
        assert 'average_volume' in signals
        assert 'volume_ratio' in signals
        assert 'is_expansion' in signals
        assert 'is_contraction' in signals
        assert 'has_initiative' in signals
        assert 'volume_exhaustion' in signals