"""
Volume analysis for initiative and exhaustion detection.

This module provides volume analysis tools for detecting market initiative,
exhaustion, and volume-based signals for Zone Fade setups.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import numpy as np

from zone_fade_detector.core.models import OHLCVBar, VolumeAnalysis


class VolumeAnalyzer:
    """
    Analyzer for volume-based market signals.
    
    Provides methods to analyze volume patterns, detect initiative,
    and identify exhaustion signals for Zone Fade setups.
    """
    
    def __init__(
        self,
        lookback_bars: int = 20,
        expansion_threshold: float = 1.5,
        contraction_threshold: float = 0.7
    ):
        """
        Initialize volume analyzer.
        
        Args:
            lookback_bars: Number of bars to look back for volume analysis
            expansion_threshold: Volume expansion multiplier threshold
            contraction_threshold: Volume contraction multiplier threshold
        """
        self.lookback_bars = lookback_bars
        self.expansion_threshold = expansion_threshold
        self.contraction_threshold = contraction_threshold
        self.logger = logging.getLogger(__name__)
    
    def analyze_volume(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> Optional[VolumeAnalysis]:
        """
        Analyze volume for the current bar.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            VolumeAnalysis object or None if insufficient data
        """
        if not bars:
            return None
        
        if current_index is None:
            current_index = len(bars) - 1
        
        # Ensure we have enough bars
        if current_index < self.lookback_bars:
            return None
        
        current_bar = bars[current_index]
        
        # Get historical bars for comparison
        historical_bars = bars[max(0, current_index - self.lookback_bars):current_index]
        
        if not historical_bars:
            return None
        
        # Calculate average volume
        avg_volume = sum(bar.volume for bar in historical_bars) / len(historical_bars)
        
        # Create volume analysis
        return VolumeAnalysis(
            current_volume=current_bar.volume,
            average_volume=avg_volume,
            volume_ratio=current_bar.volume / avg_volume if avg_volume > 0 else 1.0,
            expansion_threshold=self.expansion_threshold,
            contraction_threshold=self.contraction_threshold
        )
    
    def detect_volume_expansion(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> bool:
        """
        Detect if current bar shows volume expansion.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            True if volume expansion detected
        """
        volume_analysis = self.analyze_volume(bars, current_index)
        return volume_analysis.is_expansion if volume_analysis else False
    
    def detect_volume_contraction(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> bool:
        """
        Detect if current bar shows volume contraction.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            True if volume contraction detected
        """
        volume_analysis = self.analyze_volume(bars, current_index)
        return volume_analysis.is_contraction if volume_analysis else False
    
    def detect_volume_spike(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None,
        spike_threshold: float = 2.0,
        lookback_bars: int = 10
    ) -> Tuple[bool, float]:
        """
        Detect volume spike on rejection candle.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            spike_threshold: Volume spike multiplier threshold (default: 2.0x)
            lookback_bars: Number of bars to look back for average volume
            
        Returns:
            Tuple of (is_spike, spike_ratio)
        """
        if not bars:
            return False, 0.0
        
        if current_index is None:
            current_index = len(bars) - 1
        
        if current_index < lookback_bars:
            return False, 0.0
        
        current_bar = bars[current_index]
        current_volume = current_bar.volume
        
        if current_volume == 0:
            return False, 0.0
        
        # Calculate average volume over lookback period
        lookback_start = max(0, current_index - lookback_bars)
        lookback_bars_data = bars[lookback_start:current_index]
        
        if not lookback_bars_data:
            return False, 0.0
        
        # Calculate average volume (excluding zero volumes)
        valid_volumes = [bar.volume for bar in lookback_bars_data if bar.volume > 0]
        if not valid_volumes:
            return False, 0.0
        
        avg_volume = sum(valid_volumes) / len(valid_volumes)
        
        if avg_volume == 0:
            return False, 0.0
        
        # Calculate spike ratio
        spike_ratio = current_volume / avg_volume
        
        # Check if it's a spike
        is_spike = spike_ratio >= spike_threshold
        
        return is_spike, spike_ratio
    
    def detect_rejection_volume_spike(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None,
        spike_threshold: float = 1.8,
        lookback_bars: int = 15
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect volume spike specifically for rejection candles.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            spike_threshold: Volume spike multiplier threshold (default: 1.8x)
            lookback_bars: Number of bars to look back for average volume
            
        Returns:
            Tuple of (is_spike, spike_ratio, volume_metrics)
        """
        if not bars:
            return False, 0.0, {}
        
        if current_index is None:
            current_index = len(bars) - 1
        
        if current_index < lookback_bars:
            return False, 0.0, {}
        
        current_bar = bars[current_index]
        current_volume = current_bar.volume
        
        if current_volume == 0:
            return False, 0.0, {}
        
        # Calculate volume metrics
        lookback_start = max(0, current_index - lookback_bars)
        lookback_bars_data = bars[lookback_start:current_index]
        
        if not lookback_bars_data:
            return False, 0.0, {}
        
        # Calculate various volume metrics
        valid_volumes = [bar.volume for bar in lookback_bars_data if bar.volume > 0]
        if not valid_volumes:
            return False, 0.0, {}
        
        avg_volume = sum(valid_volumes) / len(valid_volumes)
        max_volume = max(valid_volumes)
        median_volume = sorted(valid_volumes)[len(valid_volumes) // 2]
        
        # Calculate spike ratios
        avg_spike_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        max_spike_ratio = current_volume / max_volume if max_volume > 0 else 0
        median_spike_ratio = current_volume / median_volume if median_volume > 0 else 0
        
        # Check if it's a rejection volume spike
        # Use a combination of metrics for more accurate detection
        is_spike = (
            avg_spike_ratio >= spike_threshold and
            max_spike_ratio >= 1.2 and  # At least 20% above recent max
            median_spike_ratio >= 1.5   # At least 50% above median
        )
        
        volume_metrics = {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'max_volume': max_volume,
            'median_volume': median_volume,
            'avg_spike_ratio': avg_spike_ratio,
            'max_spike_ratio': max_spike_ratio,
            'median_spike_ratio': median_spike_ratio
        }
        
        return is_spike, avg_spike_ratio, volume_metrics
    
    def calculate_volume_profile(
        self,
        bars: List[OHLCVBar],
        price_levels: int = 20
    ) -> Dict[float, int]:
        """
        Calculate volume profile for price levels.
        
        Args:
            bars: List of OHLCV bars
            price_levels: Number of price levels to analyze
            
        Returns:
            Dictionary mapping price levels to volume
        """
        if not bars:
            return {}
        
        # Get price range
        min_price = min(bar.low for bar in bars)
        max_price = max(bar.high for bar in bars)
        
        if min_price == max_price:
            return {}
        
        # Create price levels
        price_step = (max_price - min_price) / price_levels
        volume_profile = {}
        
        for i in range(price_levels + 1):
            price_level = min_price + (i * price_step)
            volume_profile[price_level] = 0
        
        # Distribute volume across price levels
        for bar in bars:
            # Find the price level for this bar's typical price
            typical_price = (bar.high + bar.low + bar.close) / 3.0
            level_index = int((typical_price - min_price) / price_step)
            level_index = max(0, min(level_index, price_levels))
            
            price_level = min_price + (level_index * price_step)
            volume_profile[price_level] += bar.volume
        
        return volume_profile
    
    def detect_volume_exhaustion(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None,
        lookback_periods: int = 5
    ) -> bool:
        """
        Detect volume exhaustion at current level.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            lookback_periods: Number of periods to look back
            
        Returns:
            True if volume exhaustion detected
        """
        if not bars or current_index is None:
            current_index = len(bars) - 1
        
        if current_index < lookback_periods:
            return False
        
        # Get recent bars
        recent_bars = bars[current_index - lookback_periods:current_index + 1]
        
        # Check for decreasing volume trend
        volumes = [bar.volume for bar in recent_bars]
        
        # Calculate volume trend (simple linear regression slope)
        if len(volumes) < 3:
            return False
        
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        
        # Check if volume is decreasing and current volume is low
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])  # Exclude current bar
        
        volume_exhaustion = (
            slope < 0 and  # Decreasing trend
            current_volume < avg_volume * self.contraction_threshold
        )
        
        return volume_exhaustion
    
    def calculate_volume_weighted_price(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> Optional[float]:
        """
        Calculate volume weighted price for recent bars.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            Volume weighted price or None if insufficient data
        """
        if not bars:
            return None
        
        if current_index is None:
            current_index = len(bars) - 1
        
        # Get recent bars
        recent_bars = bars[max(0, current_index - self.lookback_bars):current_index + 1]
        
        if not recent_bars:
            return None
        
        # Calculate volume weighted price
        total_volume = sum(bar.volume for bar in recent_bars)
        if total_volume == 0:
            return None
        
        weighted_price = sum(
            bar.volume * (bar.high + bar.low + bar.close) / 3.0
            for bar in recent_bars
        ) / total_volume
        
        return weighted_price
    
    def detect_initiative_volume(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if current bar shows initiative volume.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            Tuple of (has_initiative, direction)
        """
        if not bars or current_index is None:
            current_index = len(bars) - 1
        
        if current_index < 1:
            return False, None
        
        current_bar = bars[current_index]
        prev_bar = bars[current_index - 1]
        
        # Check for volume expansion
        volume_analysis = self.analyze_volume(bars, current_index)
        if not volume_analysis or not volume_analysis.is_expansion:
            return False, None
        
        # Check price direction
        if current_bar.close > prev_bar.close:
            return True, "bullish"
        elif current_bar.close < prev_bar.close:
            return True, "bearish"
        else:
            return False, None
    
    def calculate_volume_oscillator(
        self,
        bars: List[OHLCVBar],
        short_period: int = 5,
        long_period: int = 20
    ) -> Optional[float]:
        """
        Calculate volume oscillator.
        
        Args:
            bars: List of OHLCV bars
            short_period: Short period for moving average
            long_period: Long period for moving average
            
        Returns:
            Volume oscillator value or None if insufficient data
        """
        if len(bars) < long_period:
            return None
        
        # Get recent bars
        recent_bars = bars[-long_period:]
        
        # Calculate moving averages
        short_ma = np.mean([bar.volume for bar in recent_bars[-short_period:]])
        long_ma = np.mean([bar.volume for bar in recent_bars])
        
        # Calculate oscillator
        oscillator = ((short_ma - long_ma) / long_ma) * 100 if long_ma > 0 else 0
        
        return oscillator
    
    def get_volume_signals(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> Dict:
        """
        Get comprehensive volume signals.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            Dictionary with volume signals
        """
        if not bars:
            return {}
        
        if current_index is None:
            current_index = len(bars) - 1
        
        volume_analysis = self.analyze_volume(bars, current_index)
        if not volume_analysis:
            return {}
        
        has_initiative, direction = self.detect_initiative_volume(bars, current_index)
        volume_exhaustion = self.detect_volume_exhaustion(bars, current_index)
        volume_oscillator = self.calculate_volume_oscillator(bars)
        
        return {
            'current_volume': volume_analysis.current_volume,
            'average_volume': volume_analysis.average_volume,
            'volume_ratio': volume_analysis.volume_ratio,
            'is_expansion': volume_analysis.is_expansion,
            'is_contraction': volume_analysis.is_contraction,
            'has_initiative': has_initiative,
            'initiative_direction': direction,
            'volume_exhaustion': volume_exhaustion,
            'volume_oscillator': volume_oscillator
        }