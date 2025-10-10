"""
Unit tests for Zone Fade strategy.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from zone_fade_detector.core.models import (
    OHLCVBar, Zone, ZoneType, ZoneFadeSetup, SetupDirection,
    QRSFactors, MarketContext, VWAPData, Alert
)
from zone_fade_detector.strategies.zone_fade_strategy import ZoneFadeStrategy
from zone_fade_detector.strategies.signal_processor import SignalProcessor, SignalProcessorConfig


class TestZoneFadeStrategy:
    """Test Zone Fade Strategy functionality."""
    
    @pytest.fixture
    def strategy(self):
        """Create Zone Fade strategy instance."""
        return ZoneFadeStrategy(min_qrs_score=7)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 15, 9, 30)
        bars = []
        
        # Create bars with a clear high and rejection pattern
        for i in range(25):
            if i < 20:
                # Normal bars
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=100.0 + i * 0.1,
                    high=100.5 + i * 0.1,
                    low=99.5 + i * 0.1,
                    close=100.2 + i * 0.1,
                    volume=1000000 + i * 10000
                )
            else:
                # Rejection bar at the end
                bar = OHLCVBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=102.0,
                    high=102.5,  # High wick
                    low=101.0,
                    close=101.2,  # Close near low
                    volume=2000000  # High volume
                )
            bars.append(bar)
        
        return bars
    
    @pytest.fixture
    def sample_zone(self):
        """Create sample zone for testing."""
        return Zone(
            level=102.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=2,
            strength=2.0
        )
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.min_qrs_score == 7
        assert strategy.zone_tolerance == 0.002
        assert strategy.rejection_candle_min_wick_ratio == 0.3
        assert strategy.choch_confirmation_bars == 2
        assert strategy.logger is not None
    
    def test_detect_setups(self, strategy, sample_bars):
        """Test setup detection."""
        setups = strategy.detect_setups("SPY", sample_bars)
        
        assert isinstance(setups, list)
        # Should detect setups if zones are found
        # Note: This depends on zone detection working properly
    
    def test_is_price_approaching_zone(self, strategy, sample_zone):
        """Test price approaching zone detection."""
        # Test high zone approach
        high_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=101.0,
            high=102.1,  # Touches zone
            low=100.5,
            close=101.5,
            volume=1000000
        )
        
        assert strategy._is_price_approaching_zone(high_bar, sample_zone) is True
        
        # Test low zone approach
        low_zone = Zone(
            level=98.0,
            zone_type=ZoneType.PRIOR_DAY_LOW,
            quality=2,
            strength=2.0
        )
        
        low_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0,
            high=99.5,
            low=97.9,  # Touches zone
            close=98.5,
            volume=1000000
        )
        
        assert strategy._is_price_approaching_zone(low_bar, low_zone) is True
    
    def test_is_rejection_candle(self, strategy):
        """Test rejection candle detection."""
        # Pin bar with long upper wick
        pin_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=100.0,
            high=102.0,  # Long upper wick
            low=99.5,
            close=100.1,
            volume=1000000
        )
        
        assert strategy._is_rejection_candle(pin_bar, SetupDirection.SHORT) is True
        assert strategy._is_rejection_candle(pin_bar, SetupDirection.LONG) is False
        
        # Regular bar
        regular_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.2,
            volume=1000000
        )
        
        assert strategy._is_rejection_candle(regular_bar, SetupDirection.SHORT) is False
        assert strategy._is_rejection_candle(regular_bar, SetupDirection.LONG) is False
    
    def test_check_choch_confirmation(self, strategy, sample_bars):
        """Test CHoCH confirmation."""
        # Test with sufficient bars
        choch_confirmed = strategy._check_choch_confirmation(
            sample_bars, len(sample_bars) - 1, SetupDirection.SHORT
        )
        
        assert isinstance(choch_confirmed, bool)
        
        # Test with insufficient bars
        choch_confirmed = strategy._check_choch_confirmation(
            sample_bars[:5], 4, SetupDirection.SHORT
        )
        
        assert choch_confirmed is False
    
    def test_validate_setup(self, strategy, sample_zone):
        """Test setup validation."""
        # Create a sample setup
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=101.0,
            high=102.0,
            low=100.5,
            close=101.2,
            volume=1000000
        )
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=sample_zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1),
            timestamp=datetime.now()
        )
        
        # Test with no additional bars
        assert strategy.validate_setup(setup, []) is True
        
        # Test with additional bars that don't invalidate
        additional_bars = [
            OHLCVBar(
                timestamp=datetime.now() + timedelta(minutes=5),
                open=101.0,
                high=101.5,
                low=100.5,
                close=101.0,
                volume=1000000
            )
        ]
        
        assert strategy.validate_setup(setup, additional_bars) is True
    
    def test_generate_alert(self, strategy, sample_zone):
        """Test alert generation."""
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=101.0,
            high=102.0,
            low=100.5,
            close=101.2,
            volume=1000000
        )
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=sample_zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1),
            timestamp=datetime.now()
        )
        
        alert = strategy.generate_alert(setup)
        
        assert isinstance(alert, Alert)
        assert alert.setup == setup
        assert alert.alert_id is not None
        assert alert.priority in ["low", "normal", "high", "critical"]
    
    def test_get_setup_summary(self, strategy, sample_zone):
        """Test setup summary generation."""
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=101.0,
            high=102.0,
            low=100.5,
            close=101.2,
            volume=1000000
        )
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=sample_zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1),
            timestamp=datetime.now()
        )
        
        summary = strategy.get_setup_summary(setup)
        
        assert isinstance(summary, dict)
        assert 'symbol' in summary
        assert 'direction' in summary
        assert 'zone_level' in summary
        assert 'qrs_score' in summary
        assert 'is_a_setup' in summary
        assert 'entry_price' in summary
        assert 'stop_loss' in summary
        assert 'target_1' in summary
        assert 'target_2' in summary
    
    def test_analyze_multiple_symbols(self, strategy, sample_bars):
        """Test multiple symbol analysis."""
        symbol_data = {
            'SPY': sample_bars,
            'QQQ': sample_bars  # Same data for simplicity
        }
        
        all_setups = strategy.analyze_multiple_symbols(symbol_data)
        
        assert isinstance(all_setups, dict)
        assert 'SPY' in all_setups
        assert 'QQQ' in all_setups
        assert isinstance(all_setups['SPY'], list)
        assert isinstance(all_setups['QQQ'], list)
    
    def test_get_strategy_stats(self, strategy):
        """Test strategy statistics."""
        # Create sample setups
        setups = [
            ZoneFadeSetup(
                symbol="SPY",
                direction=SetupDirection.SHORT,
                zone=Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH),
                rejection_candle=OHLCVBar(
                    timestamp=datetime.now(),
                    open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
                ),
                choch_confirmed=True,
                qrs_factors=QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1),
                timestamp=datetime.now()
            )
        ]
        
        stats = strategy.get_strategy_stats(setups)
        
        assert isinstance(stats, dict)
        assert 'total_setups' in stats
        assert 'a_setups' in stats
        assert 'a_setup_rate' in stats
        assert 'avg_qrs_score' in stats
        assert 'long_setups' in stats
        assert 'short_setups' in stats


class TestSignalProcessor:
    """Test Signal Processor functionality."""
    
    @pytest.fixture
    def config(self):
        """Create signal processor configuration."""
        return SignalProcessorConfig(
            min_qrs_score=7,
            max_setups_per_symbol=3,
            setup_cooldown_minutes=15
        )
    
    @pytest.fixture
    def processor(self, config):
        """Create signal processor instance."""
        return SignalProcessor(config)
    
    @pytest.fixture
    def sample_setups(self):
        """Create sample setups for testing."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        
        return [
            ZoneFadeSetup(
                symbol="SPY",
                direction=SetupDirection.SHORT,
                zone=zone,
                rejection_candle=rejection_candle,
                choch_confirmed=True,
                qrs_factors=QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1),
                timestamp=datetime.now()
            )
        ]
    
    def test_processor_initialization(self, processor, config):
        """Test processor initialization."""
        assert processor.config == config
        assert processor.logger is not None
        assert processor.strategy is not None
        assert isinstance(processor.recent_setups, dict)
        assert isinstance(processor.recent_alerts, dict)
        assert isinstance(processor.setup_history, list)
    
    def test_process_signals(self, processor, sample_setups):
        """Test signal processing."""
        symbol_data = {
            'SPY': [OHLCVBar(
                timestamp=datetime.now(),
                open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
            )]
        }
        
        alerts = processor.process_signals(symbol_data)
        
        assert isinstance(alerts, list)
        # Alerts may be empty if no valid setups are detected
    
    def test_filter_setups(self, processor, sample_setups):
        """Test setup filtering."""
        filtered = processor._filter_setups("SPY", sample_setups)
        
        assert isinstance(filtered, list)
        assert len(filtered) <= len(sample_setups)
    
    def test_passes_volume_filter(self, processor, sample_setups):
        """Test volume filtering."""
        setup = sample_setups[0]
        
        # Test with volume analysis
        from zone_fade_detector.core.models import VolumeAnalysis
        setup.volume_analysis = VolumeAnalysis(
            current_volume=1500000,
            average_volume=1000000,
            volume_ratio=1.5
        )
        
        assert processor._passes_volume_filter(setup) is True
        
        # Test with low volume
        setup.volume_analysis.volume_ratio = 0.5
        assert processor._passes_volume_filter(setup) is False
    
    def test_passes_cooldown_filter(self, processor, sample_setups):
        """Test cooldown filtering."""
        setup = sample_setups[0]
        
        # First setup should pass
        assert processor._passes_cooldown_filter("SPY", setup) is True
        
        # Track the setup
        processor._track_alert("SPY", processor.strategy.generate_alert(setup))
        
        # Second setup within cooldown should fail
        assert processor._passes_cooldown_filter("SPY", setup) is False
    
    def test_should_generate_alert(self, processor, sample_setups):
        """Test alert generation decision."""
        setup = sample_setups[0]
        
        # First alert should be generated
        assert processor._should_generate_alert("SPY", setup) is True
        
        # Track the alert
        processor._track_alert("SPY", processor.strategy.generate_alert(setup))
        
        # Duplicate alert should not be generated
        assert processor._should_generate_alert("SPY", setup) is False
    
    def test_track_alert(self, processor, sample_setups):
        """Test alert tracking."""
        setup = sample_setups[0]
        alert = processor.strategy.generate_alert(setup)
        
        processor._track_alert("SPY", alert)
        
        assert "SPY" in processor.recent_setups
        assert len(processor.recent_setups["SPY"]) == 1
        assert len(processor.setup_history) == 1
    
    def test_get_processor_stats(self, processor, sample_setups):
        """Test processor statistics."""
        # Add some setups to history
        for setup in sample_setups:
            processor.setup_history.append(setup)
        
        stats = processor.get_processor_stats()
        
        assert isinstance(stats, dict)
        assert 'total_setups_processed' in stats
        assert 'recent_setups_active' in stats
        assert 'symbols_tracked' in stats
        assert 'alerts_generated' in stats
        assert 'symbol_distribution' in stats
        assert 'direction_distribution' in stats
        assert 'zone_type_distribution' in stats
    
    def test_clear_history(self, processor, sample_setups):
        """Test history clearing."""
        # Add some data
        for setup in sample_setups:
            processor.setup_history.append(setup)
        
        processor.clear_history()
        
        assert len(processor.recent_setups) == 0
        assert len(processor.recent_alerts) == 0
        assert len(processor.setup_history) == 0
    
    def test_get_recent_setups(self, processor, sample_setups):
        """Test recent setups retrieval."""
        # Add some setups
        for setup in sample_setups:
            processor.setup_history.append(setup)
            processor.recent_setups["SPY"] = [setup]
        
        recent = processor.get_recent_setups()
        assert isinstance(recent, list)
        
        recent_spy = processor.get_recent_setups("SPY")
        assert isinstance(recent_spy, list)
        assert len(recent_spy) <= 10  # Default limit