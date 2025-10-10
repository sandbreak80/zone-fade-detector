"""
Unit tests for trading strategies.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from zone_fade_detector.core.models import (
    OHLCVBar, Zone, ZoneType, ZoneFadeSetup, SetupDirection, 
    QRSFactors, MarketContext, VWAPData
)
from zone_fade_detector.strategies.zone_detector import ZoneDetector
from zone_fade_detector.strategies.qrs_scorer import QRSScorer
from zone_fade_detector.strategies.market_context import MarketContextAnalyzer


class TestZoneDetector:
    """Test Zone Detector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create zone detector instance."""
        return ZoneDetector(zone_tolerance=0.002, min_zone_strength=1.0)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        # Create bars with clear high and low
        for i in range(20):
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
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.zone_tolerance == 0.002
        assert detector.min_zone_strength == 1.0
        assert detector.value_area_percentile == 70.0
        assert detector.logger is not None
    
    def test_detect_prior_day_zones(self, detector, sample_bars):
        """Test prior day zone detection."""
        zones = detector.detect_prior_day_zones(sample_bars)
        
        # Should return empty list since no prior day data
        assert isinstance(zones, list)
        assert len(zones) == 0  # No prior day data in sample
    
    def test_detect_weekly_zones(self, detector, sample_bars):
        """Test weekly zone detection."""
        zones = detector.detect_weekly_zones(sample_bars)
        
        # Should return empty list since no weekly data
        assert isinstance(zones, list)
        assert len(zones) == 0  # No weekly data in sample
    
    def test_detect_opening_range_zones(self, detector, sample_bars):
        """Test opening range zone detection."""
        zones = detector.detect_opening_range_zones(sample_bars)
        
        assert isinstance(zones, list)
        # Should have OR high and low zones
        assert len(zones) == 2
        assert all(isinstance(zone, Zone) for zone in zones)
        assert any(zone.zone_type == ZoneType.OPENING_RANGE_HIGH for zone in zones)
        assert any(zone.zone_type == ZoneType.OPENING_RANGE_LOW for zone in zones)
    
    def test_detect_value_area_zones(self, detector, sample_bars):
        """Test value area zone detection."""
        zones = detector.detect_value_area_zones(sample_bars)
        
        assert isinstance(zones, list)
        # Should have VAH and VAL zones
        assert len(zones) == 2
        assert all(isinstance(zone, Zone) for zone in zones)
        assert any(zone.zone_type == ZoneType.VALUE_AREA_HIGH for zone in zones)
        assert any(zone.zone_type == ZoneType.VALUE_AREA_LOW for zone in zones)
    
    def test_detect_all_zones(self, detector, sample_bars):
        """Test detection of all zones."""
        zones = detector.detect_all_zones(sample_bars)
        
        assert isinstance(zones, list)
        assert all(isinstance(zone, Zone) for zone in zones)
        # Should have at least OR and VA zones
        assert len(zones) >= 4
    
    def test_find_nearest_zone(self, detector):
        """Test finding nearest zone."""
        zones = [
            Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH),
            Zone(level=105.0, zone_type=ZoneType.WEEKLY_HIGH),
            Zone(level=95.0, zone_type=ZoneType.PRIOR_DAY_LOW)
        ]
        
        nearest = detector.find_nearest_zone(102.0, zones)
        assert nearest is not None
        assert nearest.level == 100.0  # Closest to 102.0
    
    def test_is_price_near_zone(self, detector):
        """Test price near zone detection."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH)
        
        assert detector.is_price_near_zone(100.1, zone) is True  # Within tolerance
        assert detector.is_price_near_zone(102.0, zone) is False  # Outside tolerance
    
    def test_zone_quality_calculation(self, detector, sample_bars):
        """Test zone quality calculation."""
        level = 100.0
        quality = detector._calculate_zone_quality(sample_bars, level, True)
        
        assert 0 <= quality <= 2
        assert isinstance(quality, int)
    
    def test_zone_strength_calculation(self, detector, sample_bars):
        """Test zone strength calculation."""
        level = 100.0
        strength = detector._calculate_zone_strength(sample_bars, level, True)
        
        assert strength >= 0
        assert isinstance(strength, float)


class TestQRSScorer:
    """Test QRS Scorer functionality."""
    
    @pytest.fixture
    def scorer(self):
        """Create QRS scorer instance."""
        return QRSScorer(a_setup_threshold=7)
    
    @pytest.fixture
    def sample_setup(self):
        """Create sample Zone Fade setup."""
        zone = Zone(
            level=500.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=2,
            strength=2.0
        )
        
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=499.0,
            high=501.0,
            low=498.0,
            close=499.5,
            volume=1000000
        )
        
        qrs_factors = QRSFactors(
            zone_quality=2,
            rejection_clarity=2,
            structure_flip=2,
            context=1,
            intermarket_divergence=1
        )
        
        return ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
    
    def test_scorer_initialization(self, scorer):
        """Test scorer initialization."""
        assert scorer.a_setup_threshold == 7
        assert scorer.logger is not None
        assert 'htf_relevance' in scorer.zone_quality_weights
        assert 'pin_bar' in scorer.rejection_weights
    
    def test_score_zone_quality(self, scorer):
        """Test zone quality scoring."""
        # High quality zone
        high_zone = Zone(
            level=500.0,
            zone_type=ZoneType.WEEKLY_HIGH,
            quality=2,
            strength=2.0
        )
        score = scorer._score_zone_quality(high_zone)
        assert 0 <= score <= 2
        
        # Low quality zone
        low_zone = Zone(
            level=500.0,
            zone_type=ZoneType.OPENING_RANGE_HIGH,
            quality=0,
            strength=1.0
        )
        score = scorer._score_zone_quality(low_zone)
        assert 0 <= score <= 2
    
    def test_score_rejection_clarity(self, scorer):
        """Test rejection clarity scoring."""
        # Pin bar
        pin_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=100.0,
            high=100.5,
            low=99.0,
            close=99.1,  # Long lower wick
            volume=1000000
        )
        score = scorer._score_rejection_clarity(pin_bar)
        assert 0 <= score <= 2
        
        # Regular bar
        regular_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.1,
            volume=1000000
        )
        score = scorer._score_rejection_clarity(regular_bar)
        assert 0 <= score <= 2
    
    def test_score_structure_flip(self, scorer):
        """Test structure flip scoring."""
        # With CHoCH
        score = scorer._score_structure_flip(True, SetupDirection.SHORT)
        assert score == 2
        
        # Without CHoCH
        score = scorer._score_structure_flip(False, SetupDirection.SHORT)
        assert score == 0
    
    def test_score_context(self, scorer, sample_setup):
        """Test context scoring."""
        # With market context
        market_context = MarketContext(
            is_trend_day=False,
            vwap_slope=0.0005,
            value_area_overlap=True,
            market_balance=0.5
        )
        score = scorer._score_context(sample_setup, market_context)
        assert 0 <= score <= 2
        
        # Without market context
        score = scorer._score_context(sample_setup, None)
        assert 0 <= score <= 2
    
    def test_score_intermarket_divergence(self, scorer):
        """Test intermarket divergence scoring."""
        # With divergence data
        intermarket_data = {
            'price_changes': {
                'SPY': 0.5,
                'QQQ': -0.3,
                'IWM': 0.2
            }
        }
        score = scorer._score_intermarket_divergence('SPY', intermarket_data)
        assert 0 <= score <= 2
        
        # Without data
        score = scorer._score_intermarket_divergence('SPY', None)
        assert score == 0
    
    def test_score_setup(self, scorer, sample_setup):
        """Test complete setup scoring."""
        qrs_factors = scorer.score_setup(sample_setup)
        
        assert isinstance(qrs_factors, QRSFactors)
        assert 0 <= qrs_factors.zone_quality <= 2
        assert 0 <= qrs_factors.rejection_clarity <= 2
        assert 0 <= qrs_factors.structure_flip <= 2
        assert 0 <= qrs_factors.context <= 2
        assert 0 <= qrs_factors.intermarket_divergence <= 2
        assert 0 <= qrs_factors.total_score <= 10
        assert isinstance(qrs_factors.is_a_setup, bool)
    
    def test_analyze_setup_quality(self, scorer, sample_setup):
        """Test detailed setup quality analysis."""
        analysis = scorer.analyze_setup_quality(sample_setup)
        
        assert isinstance(analysis, dict)
        assert 'qrs_factors' in analysis
        assert 'total_score' in analysis
        assert 'is_a_setup' in analysis
        assert 'zone_quality_breakdown' in analysis
        assert 'rejection_breakdown' in analysis
        assert 'context_breakdown' in analysis
        assert 'recommendations' in analysis


class TestMarketContextAnalyzer:
    """Test Market Context Analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create market context analyzer instance."""
        return MarketContextAnalyzer()
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        # Create bars with trending pattern
        for i in range(25):
            bar = OHLCVBar(
                timestamp=base_time + timedelta(minutes=i),
                open=100.0 + i * 0.2,  # Trending up
                high=100.5 + i * 0.2,
                low=99.5 + i * 0.2,
                close=100.2 + i * 0.2,
                volume=1000000 + i * 10000
            )
            bars.append(bar)
        
        return bars
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.trend_day_threshold == 0.002
        assert analyzer.balance_threshold == 0.001
        assert analyzer.value_area_tolerance == 0.01
        assert analyzer.logger is not None
    
    def test_analyze_market_context(self, analyzer, sample_bars):
        """Test market context analysis."""
        context = analyzer.analyze_market_context(sample_bars)
        
        assert isinstance(context, MarketContext)
        assert isinstance(context.is_trend_day, bool)
        assert isinstance(context.vwap_slope, float)
        assert isinstance(context.value_area_overlap, bool)
        assert isinstance(context.market_balance, float)
        assert context.market_balance >= 0.0 and context.market_balance <= 1.0
        assert context.volatility_regime in ['low', 'normal', 'high']
        assert context.session_type in ['premarket', 'regular', 'afterhours']
    
    def test_detect_trend_day(self, analyzer, sample_bars):
        """Test trend day detection."""
        is_trend_day = analyzer._detect_trend_day(sample_bars, len(sample_bars) - 1)
        assert isinstance(is_trend_day, bool)
    
    def test_calculate_vwap_slope(self, analyzer, sample_bars):
        """Test VWAP slope calculation."""
        slope = analyzer._calculate_vwap_slope(sample_bars, len(sample_bars) - 1)
        assert isinstance(slope, float)
    
    def test_detect_value_area_overlap(self, analyzer, sample_bars):
        """Test value area overlap detection."""
        overlap = analyzer._detect_value_area_overlap(sample_bars, len(sample_bars) - 1)
        assert isinstance(overlap, bool)
    
    def test_calculate_market_balance(self, analyzer, sample_bars):
        """Test market balance calculation."""
        balance = analyzer._calculate_market_balance(sample_bars, len(sample_bars) - 1)
        assert 0.0 <= balance <= 1.0
        assert isinstance(balance, float)
    
    def test_determine_volatility_regime(self, analyzer, sample_bars):
        """Test volatility regime determination."""
        regime = analyzer._determine_volatility_regime(sample_bars, len(sample_bars) - 1)
        assert regime in ['low', 'normal', 'high']
    
    def test_determine_session_type(self, analyzer):
        """Test session type determination."""
        # Regular hours
        regular_time = datetime(2024, 1, 15, 10, 30)
        session_type = analyzer._determine_session_type(regular_time)
        assert session_type == 'regular'
        
        # Premarket
        premarket_time = datetime(2024, 1, 15, 8, 30)
        session_type = analyzer._determine_session_type(premarket_time)
        assert session_type == 'premarket'
        
        # After hours
        afterhours_time = datetime(2024, 1, 15, 18, 30)
        session_type = analyzer._determine_session_type(afterhours_time)
        assert session_type == 'afterhours'
    
    def test_analyze_intermarket_divergence(self, analyzer):
        """Test intermarket divergence analysis."""
        symbol_data = {
            'SPY': [OHLCVBar(
                timestamp=datetime.now(),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=1000000
            )],
            'QQQ': [OHLCVBar(
                timestamp=datetime.now(),
                open=100.0, high=100.5, low=99.5, close=99.5, volume=1000000
            )]
        }
        
        analysis = analyzer.analyze_intermarket_divergence(symbol_data)
        
        assert isinstance(analysis, dict)
        assert 'has_divergence' in analysis
        assert 'price_changes' in analysis
        assert isinstance(analysis['has_divergence'], bool)
    
    def test_get_market_summary(self, analyzer, sample_bars):
        """Test market summary generation."""
        summary = analyzer.get_market_summary(sample_bars)
        
        assert isinstance(summary, dict)
        assert 'is_trend_day' in summary
        assert 'vwap_slope' in summary
        assert 'is_balanced' in summary
        assert 'value_area_overlap' in summary
        assert 'market_balance' in summary
        assert 'volatility_regime' in summary
        assert 'session_type' in summary
        assert 'recommendation' in summary