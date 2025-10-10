"""
VWAP (Volume Weighted Average Price) calculator.

This module provides VWAP calculation with standard deviation bands
and slope analysis for trend identification.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from zone_fade_detector.core.models import OHLCVBar, VWAPData


class VWAPCalculator:
    """
    Calculator for VWAP and related indicators.
    
    Provides methods to calculate VWAP, standard deviation bands,
    and slope analysis for trend identification.
    """
    
    def __init__(self, lookback_hours: float = 6.5):
        """
        Initialize VWAP calculator.
        
        Args:
            lookback_hours: Hours of data to use for VWAP calculation
        """
        self.lookback_hours = lookback_hours
        self.logger = logging.getLogger(__name__)
    
    def calculate_vwap(
        self,
        bars: List[OHLCVBar],
        start_time: Optional[datetime] = None
    ) -> Optional[VWAPData]:
        """
        Calculate VWAP from OHLCV bars.
        
        Args:
            bars: List of OHLCV bars
            start_time: Start time for VWAP calculation (defaults to session open)
            
        Returns:
            VWAPData object or None if insufficient data
        """
        if not bars:
            self.logger.warning("No bars provided for VWAP calculation")
            return None
        
        if len(bars) < 2:
            self.logger.warning("Insufficient bars for VWAP calculation")
            return None
        
        # Determine start time
        if start_time is None:
            # Use the first bar's timestamp as session start
            start_time = bars[0].timestamp
        
        # Filter bars for the session
        session_bars = [
            bar for bar in bars
            if bar.timestamp >= start_time
        ]
        
        if len(session_bars) < 2:
            self.logger.warning("Insufficient session bars for VWAP calculation")
            return None
        
        # Calculate VWAP
        vwap, volume_sum, price_volume_sum = self._calculate_vwap_values(session_bars)
        
        # Calculate standard deviation bands
        upper_1sigma, lower_1sigma, upper_2sigma, lower_2sigma = self._calculate_std_bands(
            session_bars, vwap
        )
        
        # Calculate slope
        slope = self._calculate_slope(session_bars, vwap)
        
        # Get the latest timestamp
        latest_timestamp = max(bar.timestamp for bar in session_bars)
        
        return VWAPData(
            vwap=vwap,
            upper_1sigma=upper_1sigma,
            lower_1sigma=lower_1sigma,
            upper_2sigma=upper_2sigma,
            lower_2sigma=lower_2sigma,
            slope=slope,
            timestamp=latest_timestamp,
            volume_sum=volume_sum,
            price_volume_sum=price_volume_sum
        )
    
    def _calculate_vwap_values(
        self,
        bars: List[OHLCVBar]
    ) -> Tuple[float, float, float]:
        """
        Calculate VWAP values.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            Tuple of (vwap, volume_sum, price_volume_sum)
        """
        volume_sum = 0.0
        price_volume_sum = 0.0
        
        for bar in bars:
            # Use typical price (HLC/3) for VWAP calculation
            typical_price = (bar.high + bar.low + bar.close) / 3.0
            volume = float(bar.volume)
            
            volume_sum += volume
            price_volume_sum += typical_price * volume
        
        vwap = price_volume_sum / volume_sum if volume_sum > 0 else 0.0
        
        return vwap, volume_sum, price_volume_sum
    
    def _calculate_std_bands(
        self,
        bars: List[OHLCVBar],
        vwap: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate standard deviation bands.
        
        Args:
            bars: List of OHLCV bars
            vwap: VWAP value
            
        Returns:
            Tuple of (upper_1sigma, lower_1sigma, upper_2sigma, lower_2sigma)
        """
        if len(bars) < 2:
            return vwap, vwap, vwap, vwap
        
        # Calculate price deviations from VWAP
        deviations = []
        for bar in bars:
            typical_price = (bar.high + bar.low + bar.close) / 3.0
            deviation = (typical_price - vwap) ** 2
            deviations.append(deviation)
        
        # Calculate standard deviation
        variance = np.mean(deviations)
        std_dev = np.sqrt(variance)
        
        # Calculate bands
        upper_1sigma = vwap + std_dev
        lower_1sigma = vwap - std_dev
        upper_2sigma = vwap + (2 * std_dev)
        lower_2sigma = vwap - (2 * std_dev)
        
        return upper_1sigma, lower_1sigma, upper_2sigma, lower_2sigma
    
    def _calculate_slope(
        self,
        bars: List[OHLCVBar],
        current_vwap: float
    ) -> float:
        """
        Calculate VWAP slope.
        
        Args:
            bars: List of OHLCV bars
            current_vwap: Current VWAP value
            
        Returns:
            VWAP slope (price change per minute)
        """
        if len(bars) < 2:
            return 0.0
        
        # Calculate VWAP for the first half of the session
        mid_point = len(bars) // 2
        first_half = bars[:mid_point]
        second_half = bars[mid_point:]
        
        if not first_half or not second_half:
            return 0.0
        
        # Calculate VWAP for first half
        first_vwap, _, _ = self._calculate_vwap_values(first_half)
        
        # Calculate time difference in minutes
        time_diff = (bars[-1].timestamp - bars[0].timestamp).total_seconds() / 60.0
        
        if time_diff <= 0:
            return 0.0
        
        # Calculate slope (price change per minute)
        slope = (current_vwap - first_vwap) / time_diff
        
        return slope
    
    def calculate_session_vwap(
        self,
        bars: List[OHLCVBar],
        session_start_hour: int = 9,
        session_start_minute: int = 30
    ) -> Optional[VWAPData]:
        """
        Calculate VWAP for a trading session.
        
        Args:
            bars: List of OHLCV bars
            session_start_hour: Session start hour (default: 9)
            session_start_minute: Session start minute (default: 30)
            
        Returns:
            VWAPData object or None if insufficient data
        """
        if not bars:
            return None
        
        # Find session start time
        first_bar = bars[0]
        session_start = first_bar.timestamp.replace(
            hour=session_start_hour,
            minute=session_start_minute,
            second=0,
            microsecond=0
        )
        
        return self.calculate_vwap(bars, session_start)
    
    def calculate_rolling_vwap(
        self,
        bars: List[OHLCVBar],
        window_minutes: int = 30
    ) -> List[VWAPData]:
        """
        Calculate rolling VWAP for each bar.
        
        Args:
            bars: List of OHLCV bars
            window_minutes: Rolling window in minutes
            
        Returns:
            List of VWAPData objects
        """
        if not bars:
            return []
        
        vwap_data = []
        
        for i, current_bar in enumerate(bars):
            # Get bars within the rolling window
            window_start = current_bar.timestamp - timedelta(minutes=window_minutes)
            window_bars = [
                bar for bar in bars[:i+1]
                if bar.timestamp >= window_start
            ]
            
            if len(window_bars) < 2:
                continue
            
            # Calculate VWAP for this window
            vwap = self.calculate_vwap(window_bars)
            if vwap:
                vwap_data.append(vwap)
        
        return vwap_data
    
    def is_trend_day(self, vwap_data: VWAPData, threshold: float = 0.002) -> bool:
        """
        Determine if it's a trend day based on VWAP slope.
        
        Args:
            vwap_data: VWAPData object
            threshold: Slope threshold for trend day (default: 0.002)
            
        Returns:
            True if it's a trend day, False otherwise
        """
        return abs(vwap_data.slope) > threshold
    
    def get_vwap_levels(self, vwap_data: VWAPData) -> dict:
        """
        Get all VWAP levels for analysis.
        
        Args:
            vwap_data: VWAPData object
            
        Returns:
            Dictionary with all VWAP levels
        """
        return {
            'vwap': vwap_data.vwap,
            'upper_1sigma': vwap_data.upper_1sigma,
            'lower_1sigma': vwap_data.lower_1sigma,
            'upper_2sigma': vwap_data.upper_2sigma,
            'lower_2sigma': vwap_data.lower_2sigma,
            'slope': vwap_data.slope,
            'is_flat': vwap_data.is_flat,
            'is_bullish': vwap_data.is_bullish,
            'is_bearish': vwap_data.is_bearish
        }