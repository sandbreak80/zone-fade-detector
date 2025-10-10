"""
Zone Fade strategy implementation.

This module provides the core Zone Fade strategy logic including
setup detection, signal generation, and validation.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from zone_fade_detector.core.models import (
    OHLCVBar, Zone, ZoneType, ZoneFadeSetup, SetupDirection,
    QRSFactors, MarketContext, VWAPData, Alert
)
from zone_fade_detector.indicators.vwap import VWAPCalculator
from zone_fade_detector.indicators.opening_range import OpeningRangeCalculator
from zone_fade_detector.indicators.swing_structure import SwingStructureDetector
from zone_fade_detector.indicators.volume_analysis import VolumeAnalyzer
from zone_fade_detector.strategies.zone_detector import ZoneDetector
from zone_fade_detector.strategies.qrs_scorer import QRSScorer
from zone_fade_detector.strategies.market_context import MarketContextAnalyzer


class ZoneFadeStrategy:
    """
    Zone Fade trading strategy implementation.
    
    Implements the complete Zone Fade strategy including:
    - Zone approach detection
    - Rejection candle identification
    - CHoCH confirmation
    - QRS scoring and validation
    - Setup generation and alerts
    """
    
    def __init__(
        self,
        min_qrs_score: int = 7,
        zone_tolerance: float = 0.002,
        rejection_candle_min_wick_ratio: float = 0.1,  # Lowered from 0.3 to 0.1 (10%)
        choch_confirmation_bars: int = 2
    ):
        """
        Initialize Zone Fade strategy.
        
        Args:
            min_qrs_score: Minimum QRS score for A-Setup (default: 7)
            zone_tolerance: Price tolerance for zone approach (default: 0.2%)
            rejection_candle_min_wick_ratio: Minimum wick ratio for rejection (default: 0.3)
            choch_confirmation_bars: Bars needed for CHoCH confirmation (default: 2)
        """
        self.min_qrs_score = min_qrs_score
        self.zone_tolerance = zone_tolerance
        self.rejection_candle_min_wick_ratio = rejection_candle_min_wick_ratio
        self.choch_confirmation_bars = choch_confirmation_bars
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.zone_detector = ZoneDetector(zone_tolerance=zone_tolerance)
        self.qrs_scorer = QRSScorer(a_setup_threshold=min_qrs_score)
        self.market_context_analyzer = MarketContextAnalyzer()
        self.vwap_calculator = VWAPCalculator()
        self.or_calculator = OpeningRangeCalculator()
        self.swing_detector = SwingStructureDetector()
        self.volume_analyzer = VolumeAnalyzer()
    
    def detect_setups(
        self,
        symbol: str,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> List[ZoneFadeSetup]:
        """
        Detect Zone Fade setups from OHLCV bars.
        
        Args:
            symbol: Stock symbol
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            List of detected ZoneFadeSetup objects
        """
        if not bars:
            return []
        
        if current_index is None:
            current_index = len(bars) - 1
        
        if current_index < 20:  # Need minimum bars for analysis
            return []
        
        setups = []
        
        # Get current bar
        current_bar = bars[current_index]
        
        # Detect zones
        zones = self.zone_detector.detect_all_zones(bars, current_bar.timestamp)
        
        # Analyze market context
        market_context = self.market_context_analyzer.analyze_market_context(
            bars, current_index
        )
        
        # Check each zone for potential setups
        for zone in zones:
            if self._is_price_approaching_zone(current_bar, zone):
                setup = self._create_setup_from_zone(
                    symbol, current_bar, zone, bars, current_index, market_context
                )
                if setup:
                    setups.append(setup)
        
        return setups
    
    def _is_price_approaching_zone(
        self,
        current_bar: OHLCVBar,
        zone: Zone
    ) -> bool:
        """
        Check if price is approaching a zone.
        
        Args:
            current_bar: Current OHLCV bar
            zone: Zone to check
            
        Returns:
            True if price is approaching the zone
        """
        # Check if current bar touches or is near the zone
        if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, 
                             ZoneType.VALUE_AREA_HIGH, ZoneType.OPENING_RANGE_HIGH]:
            # For high zones, check if price reached the zone
            return current_bar.high >= zone.level * (1 - self.zone_tolerance)
        
        elif zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW,
                               ZoneType.VALUE_AREA_LOW, ZoneType.OPENING_RANGE_LOW]:
            # For low zones, check if price reached the zone
            return current_bar.low <= zone.level * (1 + self.zone_tolerance)
        
        return False
    
    def _create_setup_from_zone(
        self,
        symbol: str,
        current_bar: OHLCVBar,
        zone: Zone,
        bars: List[OHLCVBar],
        current_index: int,
        market_context: MarketContext
    ) -> Optional[ZoneFadeSetup]:
        """
        Create a Zone Fade setup from a zone approach.
        
        Args:
            symbol: Stock symbol
            current_bar: Current OHLCV bar
            zone: Zone being approached
            bars: List of OHLCV bars
            current_index: Current bar index
            market_context: Market context
            
        Returns:
            ZoneFadeSetup object or None if not valid
        """
        # Determine setup direction based on zone type
        if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH,
                             ZoneType.VALUE_AREA_HIGH, ZoneType.OPENING_RANGE_HIGH]:
            direction = SetupDirection.SHORT
        else:
            direction = SetupDirection.LONG
        
        # Check if current bar is a rejection candle with volume spike
        is_rejection, volume_metrics = self._is_rejection_candle_with_volume(
            current_bar, direction, bars, current_index
        )
        if not is_rejection:
            return None
        
        # Check for CHoCH confirmation
        choch_confirmed = self._check_choch_confirmation(bars, current_index, direction)
        
        # Get additional data
        vwap_data = self.vwap_calculator.calculate_vwap(bars[:current_index + 1])
        or_data = self.or_calculator.calculate_opening_range(bars[:current_index + 1])
        swing_structure = self.swing_detector.detect_swing_structure(
            bars, current_index
        )
        volume_analysis = self.volume_analyzer.analyze_volume(bars, current_index)
        
        # Create setup
        setup = ZoneFadeSetup(
            symbol=symbol,
            direction=direction,
            zone=zone,
            rejection_candle=current_bar,
            choch_confirmed=choch_confirmed,
            qrs_factors=QRSFactors(),  # Will be scored later
            timestamp=current_bar.timestamp,
            vwap_data=vwap_data,
            opening_range=or_data,
            swing_structure=swing_structure,
            volume_analysis=volume_analysis
        )
        
        # Score the setup
        intermarket_data = self._get_intermarket_data(symbol, bars, current_index)
        setup.qrs_factors = self.qrs_scorer.score_setup(
            setup, market_context, intermarket_data
        )
        
        # Return all setups (let the signal processor filter by QRS threshold)
        return setup
    
    def _is_rejection_candle(
        self,
        bar: OHLCVBar,
        direction: SetupDirection
    ) -> bool:
        """
        Check if a bar is a rejection candle.
        
        Args:
            bar: OHLCV bar to check
            direction: Expected setup direction
            
        Returns:
            True if bar is a rejection candle
        """
        total_range = bar.total_range
        if total_range == 0:
            return False
        
        # Check for long wick in the direction of rejection
        if direction == SetupDirection.SHORT:
            # For short setups, look for long upper wick
            upper_wick_ratio = bar.upper_wick / total_range
            return upper_wick_ratio >= self.rejection_candle_min_wick_ratio
        else:
            # For long setups, look for long lower wick
            lower_wick_ratio = bar.lower_wick / total_range
            return lower_wick_ratio >= self.rejection_candle_min_wick_ratio
    
    def _is_rejection_candle_with_volume(
        self,
        bar: OHLCVBar,
        direction: SetupDirection,
        bars: List[OHLCVBar],
        current_index: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a bar is a rejection candle with volume spike confirmation.
        
        Args:
            bar: OHLCV bar to check
            direction: Expected setup direction
            bars: List of OHLCV bars for volume analysis
            current_index: Current bar index
            
        Returns:
            Tuple of (is_rejection, volume_metrics)
        """
        # First check basic rejection candle criteria
        is_basic_rejection = self._is_rejection_candle(bar, direction)
        
        # Initialize volume metrics
        volume_metrics = {
            'is_volume_spike': False,
            'spike_ratio': 0.0,
            'volume_analysis': {}
        }
        
        if not is_basic_rejection:
            return False, volume_metrics
        
        # Check for volume spike confirmation
        try:
            is_volume_spike, spike_ratio, volume_analysis = self.volume_analyzer.detect_rejection_volume_spike(
                bars, current_index, spike_threshold=1.5, lookback_bars=15  # Lowered threshold
            )
            
            volume_metrics.update({
                'is_volume_spike': is_volume_spike,
                'spike_ratio': spike_ratio,
                'volume_analysis': volume_analysis
            })
            
            # Enhanced rejection prefers both wick rejection AND volume spike, but allows basic rejection
            is_enhanced_rejection = is_basic_rejection  # Volume spike is bonus, not required
            
            return is_enhanced_rejection, volume_metrics
            
        except Exception as e:
            self.logger.warning(f"Volume spike analysis failed: {e}")
            # Fall back to basic rejection if volume analysis fails
            return is_basic_rejection, volume_metrics
    
    def _check_choch_confirmation(
        self,
        bars: List[OHLCVBar],
        current_index: int,
        direction: SetupDirection
    ) -> bool:
        """
        Check for CHoCH confirmation.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            direction: Setup direction
            
        Returns:
            True if CHoCH is confirmed
        """
        if current_index < self.choch_confirmation_bars:
            return False
        
        # Get recent bars for CHoCH analysis
        recent_bars = bars[max(0, current_index - 10):current_index + 1]
        
        # Detect swing structure
        swing_structure = self.swing_detector.detect_swing_structure(recent_bars)
        
        # Check for CHoCH
        choch_detected, choch_direction, _ = self.swing_detector.detect_choch(
            swing_structure
        )
        
        if choch_detected and choch_direction == direction:
            return True
        
        return False
    
    def _get_intermarket_data(
        self,
        symbol: str,
        bars: List[OHLCVBar],
        current_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get intermarket analysis data.
        
        Args:
            symbol: Current symbol
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            Dictionary with intermarket data or None
        """
        # This is a simplified implementation
        # In a real system, this would fetch data from other symbols
        
        if current_index < 5:
            return None
        
        # Calculate price change for current symbol
        recent_bars = bars[max(0, current_index - 5):current_index + 1]
        if len(recent_bars) < 2:
            return None
        
        price_change = (recent_bars[-1].close - recent_bars[0].close) / recent_bars[0].close * 100
        
        return {
            'price_changes': {symbol: price_change},
            'sector_rotation': {'is_rotating': False}  # Simplified
        }
    
    def validate_setup(
        self,
        setup: ZoneFadeSetup,
        additional_bars: List[OHLCVBar]
    ) -> bool:
        """
        Validate a Zone Fade setup with additional data.
        
        Args:
            setup: ZoneFadeSetup to validate
            additional_bars: Additional bars for validation
            
        Returns:
            True if setup is still valid
        """
        if not additional_bars:
            return True
        
        # Check if price has moved away from the zone
        latest_bar = additional_bars[-1]
        
        if setup.direction == SetupDirection.SHORT:
            # For short setups, check if price is still near or below the zone
            if latest_bar.close > setup.zone.level * (1 + self.zone_tolerance):
                return False
        else:
            # For long setups, check if price is still near or above the zone
            if latest_bar.close < setup.zone.level * (1 - self.zone_tolerance):
                return False
        
        # Check if setup is still within time limits (e.g., 30 minutes)
        time_diff = latest_bar.timestamp - setup.timestamp
        if time_diff > timedelta(minutes=30):
            return False
        
        return True
    
    def generate_alert(
        self,
        setup: ZoneFadeSetup,
        alert_id: Optional[str] = None
    ) -> Alert:
        """
        Generate an alert for a Zone Fade setup.
        
        Args:
            setup: ZoneFadeSetup object
            alert_id: Optional alert ID
            
        Returns:
            Alert object
        """
        if alert_id is None:
            alert_id = f"ZF_{setup.symbol}_{setup.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Determine priority based on QRS score
        if setup.qrs_score >= 9:
            priority = "critical"
        elif setup.qrs_score >= 8:
            priority = "high"
        elif setup.qrs_score >= 7:
            priority = "normal"
        else:
            priority = "low"
        
        return Alert(
            setup=setup,
            alert_id=alert_id,
            created_at=datetime.now(),
            priority=priority
        )
    
    def get_setup_summary(
        self,
        setup: ZoneFadeSetup
    ) -> Dict[str, Any]:
        """
        Get a summary of a Zone Fade setup.
        
        Args:
            setup: ZoneFadeSetup object
            
        Returns:
            Dictionary with setup summary
        """
        return {
            'symbol': setup.symbol,
            'direction': setup.direction.value,
            'zone_level': setup.zone.level,
            'zone_type': setup.zone.zone_type.value,
            'qrs_score': setup.qrs_score,
            'is_a_setup': setup.is_a_setup,
            'choch_confirmed': setup.choch_confirmed,
            'entry_price': setup.entry_price,
            'stop_loss': setup.stop_loss,
            'target_1': setup.target_1,
            'target_2': setup.target_2,
            'timestamp': setup.timestamp.isoformat(),
            'zone_quality': setup.zone.quality,
            'zone_strength': setup.zone.strength,
            'rejection_candle': {
                'open': setup.rejection_candle.open,
                'high': setup.rejection_candle.high,
                'low': setup.rejection_candle.low,
                'close': setup.rejection_candle.close,
                'volume': setup.rejection_candle.volume
            }
        }
    
    def analyze_multiple_symbols(
        self,
        symbol_data: Dict[str, List[OHLCVBar]]
    ) -> Dict[str, List[ZoneFadeSetup]]:
        """
        Analyze multiple symbols for Zone Fade setups.
        
        Args:
            symbol_data: Dictionary mapping symbols to OHLCV bars
            
        Returns:
            Dictionary mapping symbols to their detected setups
        """
        all_setups = {}
        
        for symbol, bars in symbol_data.items():
            try:
                setups = self.detect_setups(symbol, bars)
                all_setups[symbol] = setups
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                all_setups[symbol] = []
        
        return all_setups
    
    def get_strategy_stats(
        self,
        setups: List[ZoneFadeSetup]
    ) -> Dict[str, Any]:
        """
        Get statistics for detected setups.
        
        Args:
            setups: List of ZoneFadeSetup objects
            
        Returns:
            Dictionary with strategy statistics
        """
        if not setups:
            return {}
        
        # Basic statistics
        total_setups = len(setups)
        a_setups = sum(1 for setup in setups if setup.is_a_setup)
        
        # QRS score distribution
        qrs_scores = [setup.qrs_score for setup in setups]
        avg_qrs_score = np.mean(qrs_scores) if qrs_scores else 0
        
        # Direction distribution
        long_setups = sum(1 for setup in setups if setup.direction == SetupDirection.LONG)
        short_setups = sum(1 for setup in setups if setup.direction == SetupDirection.SHORT)
        
        # Zone type distribution
        zone_types = {}
        for setup in setups:
            zone_type = setup.zone.zone_type.value
            zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
        
        # CHoCH confirmation rate
        choch_confirmed = sum(1 for setup in setups if setup.choch_confirmed)
        choch_rate = choch_confirmed / total_setups if total_setups > 0 else 0
        
        return {
            'total_setups': total_setups,
            'a_setups': a_setups,
            'a_setup_rate': a_setups / total_setups if total_setups > 0 else 0,
            'avg_qrs_score': avg_qrs_score,
            'long_setups': long_setups,
            'short_setups': short_setups,
            'zone_type_distribution': zone_types,
            'choch_confirmation_rate': choch_rate,
            'qrs_score_distribution': {
                'min': min(qrs_scores) if qrs_scores else 0,
                'max': max(qrs_scores) if qrs_scores else 0,
                'median': np.median(qrs_scores) if qrs_scores else 0
            }
        }