"""
Unit tests for Parallel Cross-Symbol Processor.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock

from zone_fade_detector.core.parallel_cross_symbol_processor import (
    ParallelCrossSymbolProcessor, IntermarketSignal, SymbolType, SymbolConfig,
    IntermarketMetrics, IntermarketAnalysis, CrossSymbolWindow
)
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager
from zone_fade_detector.core.session_state_manager import SessionStateManager
from zone_fade_detector.core.micro_window_analyzer import MicroWindowAnalyzer
from zone_fade_detector.core.models import OHLCVBar


class TestParallelCrossSymbolProcessor:
    """Test cases for ParallelCrossSymbolProcessor."""
    
    @pytest.fixture
    def window_manager(self):
        """Create a mock window manager."""
        return Mock(spec=RollingWindowManager)
    
    @pytest.fixture
    def session_manager(self):
        """Create a mock session manager."""
        return Mock(spec=SessionStateManager)
    
    @pytest.fixture
    def micro_analyzer(self):
        """Create a mock micro analyzer."""
        return Mock(spec=MicroWindowAnalyzer)
    
    @pytest.fixture
    def processor(self, window_manager, session_manager, micro_analyzer):
        """Create a ParallelCrossSymbolProcessor instance for testing."""
        return ParallelCrossSymbolProcessor(
            window_manager=window_manager,
            session_manager=session_manager,
            micro_analyzer=micro_analyzer,
            max_workers=2,
            analysis_interval_seconds=30
        )
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing."""
        base_time = datetime(2024, 1, 2, 10, 0)
        bars = []
        
        for i in range(20):
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
    
    def test_initialization(self, processor):
        """Test ParallelCrossSymbolProcessor initialization."""
        assert processor.max_workers == 2
        assert processor.analysis_interval == 30
        assert processor.correlation_threshold == 0.7
        assert processor.divergence_threshold == 0.3
        assert len(processor.symbol_configs) == 0
        assert len(processor.cross_symbol_windows) == 0
        assert processor.is_running is False
        assert processor.last_analysis_time is None
    
    def test_add_symbol(self, processor):
        """Test adding symbols for analysis."""
        processor.add_symbol("SPY", SymbolType.BROAD_MARKET, weight=1.0)
        
        assert "SPY" in processor.symbol_configs
        assert "SPY" in processor.cross_symbol_windows
        
        config = processor.symbol_configs["SPY"]
        assert config.symbol == "SPY"
        assert config.symbol_type == SymbolType.BROAD_MARKET
        assert config.weight == 1.0
        assert config.enabled is True
        
        window = processor.cross_symbol_windows["SPY"]
        assert window.symbol == "SPY"
        assert window.symbol_type == SymbolType.BROAD_MARKET
        assert window.is_ready is False
    
    def test_remove_symbol(self, processor):
        """Test removing symbols from analysis."""
        processor.add_symbol("SPY", SymbolType.BROAD_MARKET)
        processor.add_symbol("QQQ", SymbolType.BROAD_MARKET)
        
        assert len(processor.symbol_configs) == 2
        assert len(processor.cross_symbol_windows) == 2
        
        processor.remove_symbol("SPY")
        
        assert "SPY" not in processor.symbol_configs
        assert "SPY" not in processor.cross_symbol_windows
        assert "QQQ" in processor.symbol_configs
        assert "QQQ" in processor.cross_symbol_windows
    
    def test_update_symbol_data(self, processor, sample_bars):
        """Test updating symbol data."""
        processor.add_symbol("SPY", SymbolType.BROAD_MARKET, min_bars=5)
        
        # Add bars one by one
        for i, bar in enumerate(sample_bars[:10]):
            should_analyze = processor.update_symbol_data("SPY", bar)
            
            if i < 4:  # Less than min_bars
                assert should_analyze is False
                assert processor.cross_symbol_windows["SPY"].is_ready is False
            else:  # At least min_bars
                assert should_analyze is True
                assert processor.cross_symbol_windows["SPY"].is_ready is True
        
        # Check that bars are stored
        window = processor.cross_symbol_windows["SPY"]
        assert len(window.bars) == 10
        assert window.last_update == sample_bars[9].timestamp
    
    def test_calculate_price_change(self, processor, sample_bars):
        """Test price change calculation."""
        # Test with multiple bars
        price_change = processor._calculate_price_change(sample_bars)
        expected = (sample_bars[-1].close - sample_bars[0].close) / sample_bars[0].close
        assert abs(price_change - expected) < 0.0001
        
        # Test with single bar
        single_bar = [sample_bars[0]]
        price_change = processor._calculate_price_change(single_bar)
        assert price_change == 0.0
        
        # Test with empty list
        price_change = processor._calculate_price_change([])
        assert price_change == 0.0
    
    def test_calculate_volume_ratio(self, processor, sample_bars):
        """Test volume ratio calculation."""
        volume_ratio = processor._calculate_volume_ratio(sample_bars)
        assert volume_ratio > 0
        assert isinstance(volume_ratio, float)
        
        # Test with empty list
        volume_ratio = processor._calculate_volume_ratio([])
        assert volume_ratio == 1.0
    
    def test_calculate_momentum(self, processor, sample_bars):
        """Test momentum calculation."""
        momentum = processor._calculate_momentum(sample_bars)
        assert isinstance(momentum, float)
        
        # Test with insufficient bars
        short_bars = sample_bars[:3]
        momentum = processor._calculate_momentum(short_bars)
        assert momentum == 0.0
    
    def test_calculate_volatility(self, processor, sample_bars):
        """Test volatility calculation."""
        volatility = processor._calculate_volatility(sample_bars)
        assert volatility >= 0
        assert isinstance(volatility, float)
        
        # Test with insufficient bars
        short_bars = sample_bars[:2]
        volatility = processor._calculate_volatility(short_bars)
        assert volatility == 0.0
    
    def test_determine_trend_direction(self, processor):
        """Test trend direction determination."""
        # Bullish trend
        trend = processor._determine_trend_direction(0.02, 0.01)
        assert trend == "bullish"
        
        # Bearish trend
        trend = processor._determine_trend_direction(-0.02, -0.01)
        assert trend == "bearish"
        
        # Neutral trend
        trend = processor._determine_trend_direction(0.005, 0.002)
        assert trend == "neutral"
    
    def test_is_outlier_move(self, processor):
        """Test outlier move detection."""
        # Outlier move
        is_outlier = processor._is_outlier_move(0.1, 0.02)
        assert is_outlier is True
        
        # Normal move
        is_outlier = processor._is_outlier_move(0.01, 0.02)
        assert is_outlier is False
    
    def test_detect_broad_market_divergences(self, processor):
        """Test broad market divergence detection."""
        # Create mock metrics
        spy_metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="SPY",
            price_change=0.02,
            volume_ratio=1.2,
            momentum=0.01,
            volatility=0.02,
            relative_strength=0.6,
            trend_direction="bullish",
            is_outlier=False,
            correlation_score=0.8
        )
        
        qqq_metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="QQQ",
            price_change=-0.01,
            volume_ratio=0.9,
            momentum=-0.005,
            volatility=0.03,
            relative_strength=0.4,
            trend_direction="bearish",
            is_outlier=False,
            correlation_score=0.7
        )
        
        broad_market = {"SPY": spy_metrics, "QQQ": qqq_metrics}
        
        signals = processor._detect_broad_market_divergences(broad_market)
        assert len(signals) > 0
        assert any(signal in [IntermarketSignal.BULLISH_DIVERGENCE, IntermarketSignal.BEARISH_DIVERGENCE] 
                  for signal in signals)
    
    def test_detect_volatility_signals(self, processor):
        """Test volatility signal detection."""
        # High volatility
        high_vol_metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="VIX",
            price_change=0.1,
            volume_ratio=2.0,
            momentum=0.05,
            volatility=0.08,
            relative_strength=0.8,
            trend_direction="bullish",
            is_outlier=True,
            correlation_score=0.3
        )
        
        volatility = {"VIX": high_vol_metrics}
        signals = processor._detect_volatility_signals(volatility)
        assert IntermarketSignal.VOLATILITY_SPIKE in signals
        
        # Low volatility
        low_vol_metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="VIX",
            price_change=0.01,
            volume_ratio=0.8,
            momentum=0.001,
            volatility=0.005,
            relative_strength=0.2,
            trend_direction="neutral",
            is_outlier=False,
            correlation_score=0.9
        )
        
        volatility = {"VIX": low_vol_metrics}
        signals = processor._detect_volatility_signals(volatility)
        assert IntermarketSignal.VOLATILITY_SUPPRESSION in signals
    
    def test_detect_risk_sentiment_signals(self, processor):
        """Test risk sentiment signal detection."""
        # Risk off scenario
        tlt_metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="TLT",
            price_change=0.02,
            volume_ratio=1.5,
            momentum=0.01,
            volatility=0.02,
            relative_strength=0.8,
            trend_direction="bullish",
            is_outlier=False,
            correlation_score=0.6
        )
        
        spy_metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="SPY",
            price_change=-0.02,
            volume_ratio=1.2,
            momentum=-0.01,
            volatility=0.03,
            relative_strength=0.4,
            trend_direction="bearish",
            is_outlier=False,
            correlation_score=0.7
        )
        
        bonds = {"TLT": tlt_metrics}
        broad_market = {"SPY": spy_metrics}
        
        signals = processor._detect_risk_sentiment_signals(bonds, broad_market)
        assert IntermarketSignal.RISK_OFF in signals
    
    def test_detect_momentum_shift_signals(self, processor):
        """Test momentum shift signal detection."""
        # Mixed momentum scenario
        symbol_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.01,
                volume_ratio=1.1,
                momentum=0.005,
                volatility=0.02,
                relative_strength=0.6,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            ),
            "QQQ": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="QQQ",
                price_change=-0.01,
                volume_ratio=0.9,
                momentum=-0.005,
                volatility=0.03,
                relative_strength=0.4,
                trend_direction="bearish",
                is_outlier=False,
                correlation_score=0.7
            ),
            "IWM": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="IWM",
                price_change=0.005,
                volume_ratio=1.0,
                momentum=0.002,
                volatility=0.025,
                relative_strength=0.5,
                trend_direction="neutral",
                is_outlier=False,
                correlation_score=0.75
            )
        }
        
        signals = processor._detect_momentum_shift_signals(symbol_metrics)
        assert IntermarketSignal.MOMENTUM_SHIFT in signals
    
    def test_calculate_correlations(self, processor):
        """Test correlation calculation."""
        symbol_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.01,
                volume_ratio=1.1,
                momentum=0.005,
                volatility=0.02,
                relative_strength=0.6,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            ),
            "QQQ": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="QQQ",
                price_change=0.008,
                volume_ratio=1.05,
                momentum=0.004,
                volatility=0.025,
                relative_strength=0.65,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.75
            )
        }
        
        correlations = processor._calculate_correlations(symbol_metrics)
        assert "SPY-QQQ" in correlations
        assert isinstance(correlations["SPY-QQQ"], float)
    
    def test_detect_divergences(self, processor):
        """Test divergence detection."""
        symbol_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.05,
                volume_ratio=1.2,
                momentum=0.02,
                volatility=0.03,
                relative_strength=0.8,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            ),
            "QQQ": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="QQQ",
                price_change=-0.02,
                volume_ratio=0.9,
                momentum=-0.01,
                volatility=0.04,
                relative_strength=0.3,
                trend_direction="bearish",
                is_outlier=False,
                correlation_score=0.6
            )
        }
        
        divergences = processor._detect_divergences(symbol_metrics)
        assert len(divergences) > 0
        assert any("SPY" in div and "QQQ" in div for div in divergences)
    
    def test_analyze_sector_rotation(self, processor):
        """Test sector rotation analysis."""
        symbol_metrics = {
            "XLK": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="XLK",
                price_change=0.03,
                volume_ratio=1.3,
                momentum=0.015,
                volatility=0.025,
                relative_strength=0.8,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.7
            ),
            "XLF": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="XLF",
                price_change=0.01,
                volume_ratio=1.1,
                momentum=0.005,
                volatility=0.02,
                relative_strength=0.6,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            )
        }
        
        # Add sector symbols to processor
        processor.add_symbol("XLK", SymbolType.SECTOR)
        processor.add_symbol("XLF", SymbolType.SECTOR)
        
        sector_rotation = processor._analyze_sector_rotation(symbol_metrics)
        assert "XLK" in sector_rotation
        assert "XLF" in sector_rotation
        assert sector_rotation["XLK"] > sector_rotation["XLF"]  # XLK stronger
    
    def test_determine_risk_sentiment(self, processor):
        """Test risk sentiment determination."""
        symbol_metrics = {
            "TLT": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="TLT",
                price_change=0.02,
                volume_ratio=1.5,
                momentum=0.01,
                volatility=0.02,
                relative_strength=0.8,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.6
            ),
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=-0.02,
                volume_ratio=1.2,
                momentum=-0.01,
                volatility=0.03,
                relative_strength=0.4,
                trend_direction="bearish",
                is_outlier=False,
                correlation_score=0.7
            )
        }
        
        # Add symbols to processor
        processor.add_symbol("TLT", SymbolType.BOND)
        processor.add_symbol("SPY", SymbolType.BROAD_MARKET)
        
        risk_sentiment = processor._determine_risk_sentiment(symbol_metrics)
        assert risk_sentiment in ["risk_on", "risk_off", "neutral"]
    
    def test_determine_volatility_regime(self, processor):
        """Test volatility regime determination."""
        # High volatility
        high_vol_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.05,
                volume_ratio=2.0,
                momentum=0.02,
                volatility=0.05,
                relative_strength=0.8,
                trend_direction="bullish",
                is_outlier=True,
                correlation_score=0.7
            )
        }
        
        regime = processor._determine_volatility_regime(high_vol_metrics)
        assert regime == "high"
        
        # Low volatility
        low_vol_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.005,
                volume_ratio=0.8,
                momentum=0.002,
                volatility=0.005,
                relative_strength=0.5,
                trend_direction="neutral",
                is_outlier=False,
                correlation_score=0.9
            )
        }
        
        regime = processor._determine_volatility_regime(low_vol_metrics)
        assert regime == "low"
    
    def test_calculate_momentum_alignment(self, processor):
        """Test momentum alignment calculation."""
        # Aligned momentum
        aligned_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.02,
                volume_ratio=1.2,
                momentum=0.01,
                volatility=0.02,
                relative_strength=0.6,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            ),
            "QQQ": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="QQQ",
                price_change=0.018,
                volume_ratio=1.15,
                momentum=0.009,
                volatility=0.025,
                relative_strength=0.65,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.75
            )
        }
        
        alignment = processor._calculate_momentum_alignment(aligned_metrics)
        assert alignment > 0.5  # Should be well aligned
        
        # Mixed momentum
        mixed_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.02,
                volume_ratio=1.2,
                momentum=0.01,
                volatility=0.02,
                relative_strength=0.6,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            ),
            "QQQ": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="QQQ",
                price_change=-0.01,
                volume_ratio=0.9,
                momentum=-0.005,
                volatility=0.03,
                relative_strength=0.4,
                trend_direction="bearish",
                is_outlier=False,
                correlation_score=0.6
            )
        }
        
        alignment = processor._calculate_momentum_alignment(mixed_metrics)
        assert alignment < 0.6  # Should be less aligned
    
    def test_calculate_analysis_confidence(self, processor):
        """Test analysis confidence calculation."""
        symbol_metrics = {
            "SPY": IntermarketMetrics(
                timestamp=datetime.now(),
                symbol="SPY",
                price_change=0.01,
                volume_ratio=1.1,
                momentum=0.005,
                volatility=0.02,
                relative_strength=0.6,
                trend_direction="bullish",
                is_outlier=False,
                correlation_score=0.8
            )
        }
        
        signals = [IntermarketSignal.BULLISH_DIVERGENCE]
        
        confidence = processor._calculate_analysis_confidence(symbol_metrics, signals)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_get_recent_analyses(self, processor):
        """Test getting recent analyses."""
        # Initially empty
        recent = processor.get_recent_analyses()
        assert len(recent) == 0
        
        # Add some mock analyses
        analysis1 = IntermarketAnalysis(
            timestamp=datetime.now(),
            primary_symbol="SPY",
            signals=[IntermarketSignal.BULLISH_DIVERGENCE],
            correlations={},
            divergences=[],
            sector_rotation={},
            risk_sentiment="neutral",
            volatility_regime="normal",
            momentum_alignment=0.8,
            confidence_score=0.9,
            market_context=Mock()
        )
        
        processor.recent_analyses.append(analysis1)
        
        recent = processor.get_recent_analyses()
        assert len(recent) == 1
        assert recent[0] == analysis1
    
    def test_get_signal_frequency(self, processor):
        """Test getting signal frequency."""
        # Initially zero
        frequency = processor.get_signal_frequency(IntermarketSignal.BULLISH_DIVERGENCE)
        assert frequency == 0
        
        # Add some signal history
        now = datetime.now()
        processor.signal_history[IntermarketSignal.BULLISH_DIVERGENCE].extend([
            now - timedelta(hours=1),
            now - timedelta(hours=2),
            now - timedelta(hours=25)  # Outside 24-hour window
        ])
        
        frequency = processor.get_signal_frequency(IntermarketSignal.BULLISH_DIVERGENCE)
        assert frequency == 2  # Only 2 within 24 hours
    
    def test_get_analysis_summary(self, processor):
        """Test getting analysis summary."""
        # No analyses
        summary = processor.get_analysis_summary()
        assert summary["status"] == "no_analyses"
        
        # Add some mock analyses
        analysis = IntermarketAnalysis(
            timestamp=datetime.now(),
            primary_symbol="SPY",
            signals=[IntermarketSignal.BULLISH_DIVERGENCE, IntermarketSignal.RISK_ON],
            correlations={"SPY-QQQ": 0.8},
            divergences=[("SPY", "QQQ", "bullish")],
            sector_rotation={"XLK": 0.8},
            risk_sentiment="risk_on",
            volatility_regime="normal",
            momentum_alignment=0.7,
            confidence_score=0.85,
            market_context=Mock()
        )
        
        processor.recent_analyses.append(analysis)
        
        summary = processor.get_analysis_summary()
        assert summary["total_analyses"] == 1
        assert summary["avg_confidence"] == 0.85
        assert summary["avg_momentum_alignment"] == 0.7
        assert "BULLISH_DIVERGENCE" in summary["signal_counts"]
        assert summary["signal_counts"]["BULLISH_DIVERGENCE"] == 1
        assert summary["risk_sentiment_distribution"]["risk_on"] == 1


class TestIntermarketSignal:
    """Test cases for IntermarketSignal enum."""
    
    def test_intermarket_signal_values(self):
        """Test IntermarketSignal enum values."""
        assert IntermarketSignal.BULLISH_DIVERGENCE.value == "bullish_divergence"
        assert IntermarketSignal.BEARISH_DIVERGENCE.value == "bearish_divergence"
        assert IntermarketSignal.CORRELATION_BREAK.value == "correlation_break"
        assert IntermarketSignal.SECTOR_ROTATION.value == "sector_rotation"
        assert IntermarketSignal.RISK_OFF.value == "risk_off"
        assert IntermarketSignal.RISK_ON.value == "risk_on"
        assert IntermarketSignal.VOLATILITY_SPIKE.value == "volatility_spike"
        assert IntermarketSignal.VOLATILITY_SUPPRESSION.value == "volatility_suppression"
        assert IntermarketSignal.MOMENTUM_SHIFT.value == "momentum_shift"
        assert IntermarketSignal.CONSOLIDATION.value == "consolidation"


class TestSymbolType:
    """Test cases for SymbolType enum."""
    
    def test_symbol_type_values(self):
        """Test SymbolType enum values."""
        assert SymbolType.BROAD_MARKET.value == "broad_market"
        assert SymbolType.SECTOR.value == "sector"
        assert SymbolType.VOLATILITY.value == "volatility"
        assert SymbolType.BOND.value == "bond"
        assert SymbolType.COMMODITY.value == "commodity"
        assert SymbolType.CURRENCY.value == "currency"
        assert SymbolType.CRYPTO.value == "crypto"


class TestSymbolConfig:
    """Test cases for SymbolConfig dataclass."""
    
    def test_symbol_config_creation(self):
        """Test SymbolConfig creation."""
        config = SymbolConfig(
            symbol="SPY",
            symbol_type=SymbolType.BROAD_MARKET,
            weight=1.0,
            enabled=True,
            min_bars=10,
            lookback_minutes=60
        )
        
        assert config.symbol == "SPY"
        assert config.symbol_type == SymbolType.BROAD_MARKET
        assert config.weight == 1.0
        assert config.enabled is True
        assert config.min_bars == 10
        assert config.lookback_minutes == 60


class TestIntermarketMetrics:
    """Test cases for IntermarketMetrics dataclass."""
    
    def test_intermarket_metrics_creation(self):
        """Test IntermarketMetrics creation."""
        metrics = IntermarketMetrics(
            timestamp=datetime.now(),
            symbol="SPY",
            price_change=0.02,
            volume_ratio=1.2,
            momentum=0.01,
            volatility=0.03,
            relative_strength=0.7,
            trend_direction="bullish",
            is_outlier=False,
            correlation_score=0.8
        )
        
        assert metrics.symbol == "SPY"
        assert metrics.price_change == 0.02
        assert metrics.volume_ratio == 1.2
        assert metrics.momentum == 0.01
        assert metrics.volatility == 0.03
        assert metrics.relative_strength == 0.7
        assert metrics.trend_direction == "bullish"
        assert metrics.is_outlier is False
        assert metrics.correlation_score == 0.8


class TestCrossSymbolWindow:
    """Test cases for CrossSymbolWindow dataclass."""
    
    def test_cross_symbol_window_creation(self):
        """Test CrossSymbolWindow creation."""
        window = CrossSymbolWindow(
            symbol="SPY",
            symbol_type=SymbolType.BROAD_MARKET,
            bars=[],
            last_update=None,
            is_ready=False,
            metrics=None
        )
        
        assert window.symbol == "SPY"
        assert window.symbol_type == SymbolType.BROAD_MARKET
        assert window.bars == []
        assert window.last_update is None
        assert window.is_ready is False
        assert window.metrics is None