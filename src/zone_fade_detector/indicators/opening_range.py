"""
Opening Range (OR) calculator.

This module provides calculation of opening range high and low
from the first 30 minutes of regular trading hours.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from zone_fade_detector.core.models import OHLCVBar, OpeningRange


class OpeningRangeCalculator:
    """
    Calculator for Opening Range indicators.
    
    Provides methods to calculate opening range high and low
    from the first 30 minutes of trading.
    """
    
    def __init__(self, duration_minutes: int = 30):
        """
        Initialize Opening Range calculator.
        
        Args:
            duration_minutes: Duration of opening range in minutes (default: 30)
        """
        self.duration_minutes = duration_minutes
        self.logger = logging.getLogger(__name__)
    
    def calculate_opening_range(
        self,
        bars: List[OHLCVBar],
        session_start_hour: int = 9,
        session_start_minute: int = 30
    ) -> Optional[OpeningRange]:
        """
        Calculate opening range from OHLCV bars.
        
        Args:
            bars: List of OHLCV bars
            session_start_hour: Session start hour (default: 9)
            session_start_minute: Session start minute (default: 30)
            
        Returns:
            OpeningRange object or None if insufficient data
        """
        if not bars:
            self.logger.warning("No bars provided for opening range calculation")
            return None
        
        # Find session start time
        first_bar = bars[0]
        session_start = first_bar.timestamp.replace(
            hour=session_start_hour,
            minute=session_start_minute,
            second=0,
            microsecond=0
        )
        
        # Calculate opening range end time
        session_end = session_start + timedelta(minutes=self.duration_minutes)
        
        # Filter bars for opening range period
        or_bars = [
            bar for bar in bars
            if session_start <= bar.timestamp <= session_end
        ]
        
        if not or_bars:
            self.logger.warning("No bars found in opening range period")
            return None
        
        # Calculate opening range high and low
        or_high = max(bar.high for bar in or_bars)
        or_low = min(bar.low for bar in or_bars)
        
        # Calculate total volume
        total_volume = sum(bar.volume for bar in or_bars)
        
        return OpeningRange(
            high=or_high,
            low=or_low,
            start_time=session_start,
            end_time=session_end,
            volume=total_volume
        )
    
    def calculate_daily_opening_range(
        self,
        bars: List[OHLCVBar],
        date: Optional[datetime] = None
    ) -> Optional[OpeningRange]:
        """
        Calculate opening range for a specific date.
        
        Args:
            bars: List of OHLCV bars
            date: Date to calculate OR for (defaults to first bar's date)
            
        Returns:
            OpeningRange object or None if insufficient data
        """
        if not bars:
            return None
        
        if date is None:
            date = bars[0].timestamp.date()
        
        # Filter bars for the specific date
        date_bars = [
            bar for bar in bars
            if bar.timestamp.date() == date
        ]
        
        if not date_bars:
            self.logger.warning(f"No bars found for date {date}")
            return None
        
        return self.calculate_opening_range(date_bars)
    
    def calculate_multiple_days_or(
        self,
        bars: List[OHLCVBar],
        days: int = 5
    ) -> List[OpeningRange]:
        """
        Calculate opening range for multiple days.
        
        Args:
            bars: List of OHLCV bars
            days: Number of days to calculate OR for
            
        Returns:
            List of OpeningRange objects
        """
        if not bars:
            return []
        
        # Get unique dates from bars
        dates = sorted(list(set(bar.timestamp.date() for bar in bars)))
        
        # Take the last N days
        recent_dates = dates[-days:] if len(dates) >= days else dates
        
        or_data = []
        for date in recent_dates:
            or_range = self.calculate_daily_opening_range(bars, date)
            if or_range:
                or_data.append(or_range)
        
        return or_data
    
    def is_price_in_opening_range(
        self,
        current_price: float,
        opening_range: OpeningRange
    ) -> bool:
        """
        Check if current price is within opening range.
        
        Args:
            current_price: Current price to check
            opening_range: OpeningRange object
            
        Returns:
            True if price is within OR, False otherwise
        """
        return opening_range.low <= current_price <= opening_range.high
    
    def calculate_or_breakout_levels(
        self,
        opening_range: OpeningRange,
        breakout_threshold: float = 0.001
    ) -> Tuple[float, float]:
        """
        Calculate breakout levels for opening range.
        
        Args:
            opening_range: OpeningRange object
            breakout_threshold: Threshold for breakout (default: 0.1%)
            
        Returns:
            Tuple of (upside_breakout_level, downside_breakout_level)
        """
        or_range = opening_range.range_size
        threshold_amount = or_range * breakout_threshold
        
        upside_breakout = opening_range.high + threshold_amount
        downside_breakout = opening_range.low - threshold_amount
        
        return upside_breakout, downside_breakout
    
    def calculate_or_retest_levels(
        self,
        opening_range: OpeningRange
    ) -> Tuple[float, float]:
        """
        Calculate retest levels for opening range.
        
        Args:
            opening_range: OpeningRange object
            
        Returns:
            Tuple of (high_retest_level, low_retest_level)
        """
        # Retest levels are typically at the OR high and low
        return opening_range.high, opening_range.low
    
    def get_or_statistics(
        self,
        opening_ranges: List[OpeningRange]
    ) -> dict:
        """
        Calculate statistics for multiple opening ranges.
        
        Args:
            opening_ranges: List of OpeningRange objects
            
        Returns:
            Dictionary with OR statistics
        """
        if not opening_ranges:
            return {}
        
        ranges = [or_data.range_size for or_data in opening_ranges]
        volumes = [or_data.volume for or_data in opening_ranges]
        
        return {
            'count': len(opening_ranges),
            'avg_range_size': sum(ranges) / len(ranges),
            'min_range_size': min(ranges),
            'max_range_size': max(ranges),
            'avg_volume': sum(volumes) / len(volumes),
            'min_volume': min(volumes),
            'max_volume': max(volumes)
        }
    
    def calculate_or_quality_score(
        self,
        opening_range: OpeningRange,
        recent_ranges: List[OpeningRange]
    ) -> float:
        """
        Calculate quality score for opening range.
        
        Args:
            opening_range: Current OpeningRange object
            recent_ranges: Recent OpeningRange objects for comparison
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not recent_ranges:
            return 0.5  # Default score if no comparison data
        
        # Calculate relative range size
        avg_range = sum(or_data.range_size for or_data in recent_ranges) / len(recent_ranges)
        range_ratio = opening_range.range_size / avg_range if avg_range > 0 else 1.0
        
        # Calculate relative volume
        avg_volume = sum(or_data.volume for or_data in recent_ranges) / len(recent_ranges)
        volume_ratio = opening_range.volume / avg_volume if avg_volume > 0 else 1.0
        
        # Quality score based on range and volume
        # Higher range and volume = higher quality
        range_score = min(range_ratio, 2.0) / 2.0  # Cap at 2x average
        volume_score = min(volume_ratio, 2.0) / 2.0  # Cap at 2x average
        
        quality_score = (range_score + volume_score) / 2.0
        
        return min(quality_score, 1.0)  # Cap at 1.0