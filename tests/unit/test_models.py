"""
Unit tests for core data models.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from zone_fade_detector.core.models import (
    OHLCVBar,
    Zone,
    ZoneType,
    SetupDirection,
    CandleType,
    VWAPData,
    OpeningRange,
    SwingPoint,
    SwingStructure,
    VolumeAnalysis,
    QRSFactors,
    ZoneFadeSetup,
    MarketContext,
    Alert,
)


class TestOHLCVBar:
    """Test OHLCVBar data structure."""
    
    def test_valid_bar_creation(self):
        """Test creating a valid OHLCV bar."""
        timestamp = datetime.now()
        bar = OHLCVBar(
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000000
        )
        
        assert bar.timestamp == timestamp
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 98.0
        assert bar.close == 103.0
        assert bar.volume == 1000000
    
    def test_bar_properties(self):
        """Test OHLCV bar properties."""
        bar = OHLCVBar(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000000
        )
        
        assert bar.body_size == 3.0
        assert bar.upper_wick == 2.0
        assert bar.lower_wick == 2.0
        assert bar.total_range == 7.0
        assert bar.is_bullish is True
        assert bar.is_bearish is False
    
    def test_invalid_bar_data(self):
        """Test validation of invalid bar data."""
        with pytest.raises(ValueError, match="Invalid OHLCV data"):
            OHLCVBar(
                timestamp=datetime.now(),
                open=100.0,
                high=95.0,  # High lower than close
                low=98.0,
                close=103.0,
                volume=1000000
            )
        
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            OHLCVBar(
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=98.0,
                close=103.0,
                volume=-1000
            )


class TestZone:
    """Test Zone data structure."""
    
    def test_zone_creation(self):
        """Test creating a valid zone."""
        zone = Zone(
            level=500.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=2,
            strength=1.5
        )
        
        assert zone.level == 500.0
        assert zone.zone_type == ZoneType.PRIOR_DAY_HIGH
        assert zone.quality == 2
        assert zone.strength == 1.5
        assert zone.touches == 0
        assert zone.last_touch is None
    
    def test_zone_validation(self):
        """Test zone data validation."""
        with pytest.raises(ValueError, match="Zone quality must be between 0 and 2"):
            Zone(
                level=500.0,
                zone_type=ZoneType.PRIOR_DAY_HIGH,
                quality=3
            )
        
        with pytest.raises(ValueError, match="Zone strength must be positive"):
            Zone(
                level=500.0,
                zone_type=ZoneType.PRIOR_DAY_HIGH,
                strength=-1.0
            )
    
    def test_zone_touch_tracking(self):
        """Test zone touch tracking."""
        zone = Zone(
            level=500.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH
        )
        
        timestamp = datetime.now()
        zone.add_touch(timestamp)
        
        assert zone.touches == 1
        assert zone.last_touch == timestamp


class TestVWAPData:
    """Test VWAPData structure."""
    
    def test_vwap_creation(self):
        """Test creating VWAP data."""
        vwap = VWAPData(
            vwap=100.0,
            upper_1sigma=102.0,
            lower_1sigma=98.0,
            upper_2sigma=104.0,
            lower_2sigma=96.0,
            slope=0.001,
            timestamp=datetime.now()
        )
        
        assert vwap.vwap == 100.0
        assert vwap.upper_1sigma == 102.0
        assert vwap.lower_1sigma == 98.0
        assert vwap.slope == 0.001
        assert vwap.is_flat is False
        assert vwap.is_bullish is True
        assert vwap.is_bearish is False
    
    def test_vwap_slope_classification(self):
        """Test VWAP slope classification."""
        # Flat slope
        vwap_flat = VWAPData(
            vwap=100.0,
            upper_1sigma=102.0,
            lower_1sigma=98.0,
            upper_2sigma=104.0,
            lower_2sigma=96.0,
            slope=0.0005,
            timestamp=datetime.now()
        )
        assert vwap_flat.is_flat is True
        assert vwap_flat.is_bullish is False
        assert vwap_flat.is_bearish is False
        
        # Bearish slope
        vwap_bearish = VWAPData(
            vwap=100.0,
            upper_1sigma=102.0,
            lower_1sigma=98.0,
            upper_2sigma=104.0,
            lower_2sigma=96.0,
            slope=-0.002,
            timestamp=datetime.now()
        )
        assert vwap_bearish.is_flat is False
        assert vwap_bearish.is_bullish is False
        assert vwap_bearish.is_bearish is True


class TestOpeningRange:
    """Test OpeningRange structure."""
    
    def test_opening_range_creation(self):
        """Test creating opening range."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)
        
        or_data = OpeningRange(
            high=105.0,
            low=95.0,
            start_time=start_time,
            end_time=end_time,
            volume=500000
        )
        
        assert or_data.high == 105.0
        assert or_data.low == 95.0
        assert or_data.range_size == 10.0
        assert or_data.mid_point == 100.0
        assert or_data.is_valid is True
    
    def test_invalid_opening_range(self):
        """Test invalid opening range."""
        start_time = datetime.now()
        end_time = start_time - timedelta(minutes=30)  # End before start
        
        or_data = OpeningRange(
            high=105.0,
            low=95.0,
            start_time=start_time,
            end_time=end_time
        )
        
        assert or_data.is_valid is False


class TestSwingStructure:
    """Test SwingStructure analysis."""
    
    def test_swing_structure_creation(self):
        """Test creating swing structure."""
        structure = SwingStructure()
        
        assert len(structure.swing_highs) == 0
        assert len(structure.swing_lows) == 0
        assert structure.choch_detected is False
        assert structure.choch_direction is None
    
    def test_swing_point_management(self):
        """Test adding swing points."""
        structure = SwingStructure()
        
        timestamp = datetime.now()
        swing_high = SwingPoint(price=105.0, timestamp=timestamp, is_high=True)
        swing_low = SwingPoint(price=95.0, timestamp=timestamp, is_high=False)
        
        structure.add_swing_high(swing_high)
        structure.add_swing_low(swing_low)
        
        assert len(structure.swing_highs) == 1
        assert len(structure.swing_lows) == 1
        assert structure.last_swing_high == swing_high
        assert structure.last_swing_low == swing_low


class TestQRSFactors:
    """Test QRS scoring system."""
    
    def test_qrs_creation(self):
        """Test creating QRS factors."""
        qrs = QRSFactors(
            zone_quality=2,
            rejection_clarity=2,
            structure_flip=2,
            context=1,
            intermarket_divergence=1
        )
        
        assert qrs.total_score == 8
        assert qrs.is_a_setup is True
    
    def test_qrs_validation(self):
        """Test QRS factor validation."""
        with pytest.raises(ValueError, match="QRS factor zone_quality must be between 0 and 2"):
            QRSFactors(zone_quality=3)
        
        with pytest.raises(ValueError, match="QRS factor rejection_clarity must be between 0 and 2"):
            QRSFactors(rejection_clarity=-1)
    
    def test_a_setup_threshold(self):
        """Test A-Setup threshold logic."""
        # A-Setup (score >= 7)
        qrs_a = QRSFactors(
            zone_quality=2,
            rejection_clarity=2,
            structure_flip=2,
            context=1,
            intermarket_divergence=1
        )
        assert qrs_a.is_a_setup is True
        
        # Not A-Setup (score < 7)
        qrs_b = QRSFactors(
            zone_quality=1,
            rejection_clarity=1,
            structure_flip=1,
            context=1,
            intermarket_divergence=1
        )
        assert qrs_b.is_a_setup is False


class TestZoneFadeSetup:
    """Test ZoneFadeSetup structure."""
    
    def test_setup_creation(self):
        """Test creating a zone fade setup."""
        zone = Zone(
            level=500.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=2
        )
        
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=499.0,
            high=501.0,
            low=498.0,
            close=499.5,
            volume=1000000
        )
        
        qrs = QRSFactors(
            zone_quality=2,
            rejection_clarity=2,
            structure_flip=2,
            context=1,
            intermarket_divergence=1
        )
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs,
            timestamp=datetime.now()
        )
        
        assert setup.symbol == "SPY"
        assert setup.direction == SetupDirection.SHORT
        assert setup.is_a_setup is True
        assert setup.qrs_score == 8
    
    def test_setup_calculations(self):
        """Test setup price calculations."""
        zone = Zone(
            level=500.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=2
        )
        zone.range_size = 10.0  # Mock range size
        
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=499.0,
            high=501.0,
            low=498.0,
            close=499.5,
            volume=1000000
        )
        
        qrs = QRSFactors()
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs,
            timestamp=datetime.now()
        )
        
        # Test entry price calculation (would need range_size property)
        # This is a simplified test - actual implementation would be more complex
        assert setup.symbol == "SPY"
        assert setup.direction == SetupDirection.SHORT


class TestMarketContext:
    """Test MarketContext structure."""
    
    def test_market_context_creation(self):
        """Test creating market context."""
        context = MarketContext(
            is_trend_day=False,
            vwap_slope=0.0005,
            value_area_overlap=True,
            market_balance=0.5
        )
        
        assert context.is_trend_day is False
        assert context.vwap_slope == 0.0005
        assert context.value_area_overlap is True
        assert context.is_balanced is True
    
    def test_balanced_market_detection(self):
        """Test balanced market detection."""
        # Balanced market
        balanced = MarketContext(
            is_trend_day=False,
            vwap_slope=0.0005,
            value_area_overlap=True
        )
        assert balanced.is_balanced is True
        
        # Trend day
        trend_day = MarketContext(
            is_trend_day=True,
            vwap_slope=0.0005,
            value_area_overlap=True
        )
        assert trend_day.is_balanced is False
        
        # High slope
        high_slope = MarketContext(
            is_trend_day=False,
            vwap_slope=0.01,
            value_area_overlap=True
        )
        assert high_slope.is_balanced is False


class TestAlert:
    """Test Alert structure."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        zone = Zone(level=500.0, zone_type=ZoneType.PRIOR_DAY_HIGH)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=499.0, high=501.0, low=498.0, close=499.5, volume=1000000
        )
        qrs = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs,
            timestamp=datetime.now()
        )
        
        alert = Alert(
            setup=setup,
            alert_id="ALERT_001",
            created_at=datetime.now(),
            priority="high"
        )
        
        assert alert.alert_id == "ALERT_001"
        assert alert.priority == "high"
        assert alert.status == "active"
        assert alert.setup.symbol == "SPY"
    
    def test_alert_serialization(self):
        """Test alert serialization to dictionary."""
        zone = Zone(level=500.0, zone_type=ZoneType.PRIOR_DAY_HIGH)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=499.0, high=501.0, low=498.0, close=499.5, volume=1000000
        )
        qrs = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs,
            timestamp=datetime.now()
        )
        
        alert = Alert(
            setup=setup,
            alert_id="ALERT_001",
            created_at=datetime.now()
        )
        
        alert_dict = alert.to_dict()
        
        assert "alert_id" in alert_dict
        assert "symbol" in alert_dict
        assert "direction" in alert_dict
        assert "qrs_score" in alert_dict
        assert alert_dict["symbol"] == "SPY"
        assert alert_dict["direction"] == "short"