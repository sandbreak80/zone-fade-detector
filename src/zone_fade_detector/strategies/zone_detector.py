"""
Zone detection for higher-timeframe support and resistance levels.

This module provides detection of various zone types including prior day
highs/lows, weekly levels, and value area calculations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import numpy as np

from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType
from zone_fade_detector.indicators.opening_range import OpeningRangeCalculator


class ZoneDetector:
    """
    Detector for higher-timeframe zones.
    
    Provides methods to identify and validate various types of zones
    including prior day levels, weekly levels, and value areas.
    """
    
    def __init__(
        self,
        zone_tolerance: float = 0.002,  # 0.2% tolerance
        min_zone_strength: float = 1.0,
        value_area_percentile: float = 70.0
    ):
        """
        Initialize zone detector.
        
        Args:
            zone_tolerance: Price tolerance for zone validation (percentage)
            min_zone_strength: Minimum strength for zone validation
            value_area_percentile: Percentile for value area calculation
        """
        self.zone_tolerance = zone_tolerance
        self.min_zone_strength = min_zone_strength
        self.value_area_percentile = value_area_percentile
        self.logger = logging.getLogger(__name__)
        self.or_calculator = OpeningRangeCalculator()
    
    def detect_prior_day_zones(
        self,
        bars: List[OHLCVBar],
        date: Optional[datetime] = None
    ) -> List[Zone]:
        """
        Detect prior day high and low zones.
        
        Args:
            bars: List of OHLCV bars
            date: Reference date (defaults to today)
            
        Returns:
            List of Zone objects for prior day levels
        """
        if not bars:
            return []
        
        if date is None:
            date = datetime.now()
        
        # Get prior day bars
        prior_day = date - timedelta(days=1)
        prior_day_bars = [
            bar for bar in bars
            if bar.timestamp.date() == prior_day.date()
        ]
        
        if not prior_day_bars:
            self.logger.warning(f"No bars found for prior day {prior_day.date()}")
            return []
        
        zones = []
        
        # Calculate prior day high and low
        prior_day_high = max(bar.high for bar in prior_day_bars)
        prior_day_low = min(bar.low for bar in prior_day_bars)
        
        # Create zones
        high_zone = Zone(
            level=prior_day_high,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=self._calculate_zone_quality(prior_day_bars, prior_day_high, True),
            strength=self._calculate_zone_strength(prior_day_bars, prior_day_high, True)
        )
        zones.append(high_zone)
        
        low_zone = Zone(
            level=prior_day_low,
            zone_type=ZoneType.PRIOR_DAY_LOW,
            quality=self._calculate_zone_quality(prior_day_bars, prior_day_low, False),
            strength=self._calculate_zone_strength(prior_day_bars, prior_day_low, False)
        )
        zones.append(low_zone)
        
        return zones
    
    def detect_weekly_zones(
        self,
        bars: List[OHLCVBar],
        week_start: Optional[datetime] = None
    ) -> List[Zone]:
        """
        Detect weekly high and low zones.
        
        Args:
            bars: List of OHLCV bars
            week_start: Start of the week (defaults to current week)
            
        Returns:
            List of Zone objects for weekly levels
        """
        if not bars:
            return []
        
        if week_start is None:
            # Get start of current week (Monday)
            today = datetime.now()
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)
        
        week_end = week_start + timedelta(days=5)  # Friday
        
        # Get weekly bars
        weekly_bars = [
            bar for bar in bars
            if week_start.date() <= bar.timestamp.date() <= week_end.date()
        ]
        
        if not weekly_bars:
            self.logger.warning(f"No bars found for week starting {week_start.date()}")
            return []
        
        zones = []
        
        # Calculate weekly high and low
        weekly_high = max(bar.high for bar in weekly_bars)
        weekly_low = min(bar.low for bar in weekly_bars)
        
        # Create zones
        high_zone = Zone(
            level=weekly_high,
            zone_type=ZoneType.WEEKLY_HIGH,
            quality=self._calculate_zone_quality(weekly_bars, weekly_high, True),
            strength=self._calculate_zone_strength(weekly_bars, weekly_high, True)
        )
        zones.append(high_zone)
        
        low_zone = Zone(
            level=weekly_low,
            zone_type=ZoneType.WEEKLY_LOW,
            quality=self._calculate_zone_quality(weekly_bars, weekly_low, False),
            strength=self._calculate_zone_strength(weekly_bars, weekly_low, False)
        )
        zones.append(low_zone)
        
        return zones
    
    def detect_opening_range_zones(
        self,
        bars: List[OHLCVBar],
        date: Optional[datetime] = None
    ) -> List[Zone]:
        """
        Detect opening range high and low zones.
        
        Args:
            bars: List of OHLCV bars
            date: Reference date (defaults to today)
            
        Returns:
            List of Zone objects for opening range levels
        """
        if not bars:
            return []
        
        if date is None:
            date = datetime.now()
        
        # Get opening range data
        or_data = self.or_calculator.calculate_daily_opening_range(bars, date)
        
        if not or_data:
            return []
        
        zones = []
        
        # Create zones
        high_zone = Zone(
            level=or_data.high,
            zone_type=ZoneType.OPENING_RANGE_HIGH,
            quality=self._calculate_or_zone_quality(or_data, bars),
            strength=1.0  # OR zones have standard strength
        )
        zones.append(high_zone)
        
        low_zone = Zone(
            level=or_data.low,
            zone_type=ZoneType.OPENING_RANGE_LOW,
            quality=self._calculate_or_zone_quality(or_data, bars),
            strength=1.0
        )
        zones.append(low_zone)
        
        return zones
    
    def detect_value_area_zones(
        self,
        bars: List[OHLCVBar],
        date: Optional[datetime] = None
    ) -> List[Zone]:
        """
        Detect value area high and low zones.
        
        Args:
            bars: List of OHLCV bars
            date: Reference date (defaults to today)
            
        Returns:
            List of Zone objects for value area levels
        """
        if not bars:
            return []
        
        if date is None:
            date = datetime.now()
        
        # Get daily bars
        daily_bars = [
            bar for bar in bars
            if bar.timestamp.date() == date.date()
        ]
        
        if not daily_bars:
            self.logger.warning(f"No bars found for date {date.date()}")
            return []
        
        # Calculate value area
        vah, val = self._calculate_value_area(daily_bars)
        
        if vah is None or val is None:
            return []
        
        zones = []
        
        # Create zones
        high_zone = Zone(
            level=vah,
            zone_type=ZoneType.VALUE_AREA_HIGH,
            quality=self._calculate_zone_quality(daily_bars, vah, True),
            strength=self._calculate_zone_strength(daily_bars, vah, True)
        )
        zones.append(high_zone)
        
        low_zone = Zone(
            level=val,
            zone_type=ZoneType.VALUE_AREA_LOW,
            quality=self._calculate_zone_quality(daily_bars, val, False),
            strength=self._calculate_zone_strength(daily_bars, val, False)
        )
        zones.append(low_zone)
        
        return zones
    
    def detect_all_zones(
        self,
        bars: List[OHLCVBar],
        date: Optional[datetime] = None
    ) -> List[Zone]:
        """
        Detect all available zones for a given date.
        
        Args:
            bars: List of OHLCV bars
            date: Reference date (defaults to today)
            
        Returns:
            List of all detected Zone objects
        """
        all_zones = []
        
        # Detect different zone types
        all_zones.extend(self.detect_prior_day_zones(bars, date))
        all_zones.extend(self.detect_weekly_zones(bars, date))
        all_zones.extend(self.detect_opening_range_zones(bars, date))
        all_zones.extend(self.detect_value_area_zones(bars, date))
        
        # Filter out low-quality zones
        quality_zones = [
            zone for zone in all_zones
            if zone.quality >= 1 and zone.strength >= self.min_zone_strength
        ]
        
        return quality_zones
    
    def find_nearest_zone(
        self,
        price: float,
        zones: List[Zone],
        max_distance: float = 0.01  # 1% max distance
    ) -> Optional[Zone]:
        """
        Find the nearest zone to a given price.
        
        Args:
            price: Current price
            zones: List of zones to search
            max_distance: Maximum distance as percentage
            
        Returns:
            Nearest Zone object or None if no zone within distance
        """
        if not zones:
            return None
        
        nearest_zone = None
        min_distance = float('inf')
        
        for zone in zones:
            distance = abs(zone.level - price) / price
            
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                nearest_zone = zone
        
        return nearest_zone
    
    def is_price_near_zone(
        self,
        price: float,
        zone: Zone,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Check if price is near a zone.
        
        Args:
            price: Current price
            zone: Zone to check
            tolerance: Price tolerance (defaults to zone_tolerance)
            
        Returns:
            True if price is near the zone
        """
        if tolerance is None:
            tolerance = self.zone_tolerance
        
        distance = abs(zone.level - price) / price
        return distance <= tolerance
    
    def _calculate_zone_quality(
        self,
        bars: List[OHLCVBar],
        level: float,
        is_high: bool
    ) -> int:
        """
        Calculate zone quality score (0-2).
        
        Args:
            bars: List of OHLCV bars
            level: Zone level
            is_high: True for high zone, False for low zone
            
        Returns:
            Quality score between 0 and 2
        """
        if not bars:
            return 0
        
        # Count touches of the zone
        touches = 0
        for bar in bars:
            if is_high:
                if abs(bar.high - level) / level <= self.zone_tolerance:
                    touches += 1
            else:
                if abs(bar.low - level) / level <= self.zone_tolerance:
                    touches += 1
        
        # Quality based on touches and volume
        if touches >= 3:
            return 2
        elif touches >= 2:
            return 1
        else:
            return 0
    
    def _calculate_zone_strength(
        self,
        bars: List[OHLCVBar],
        level: float,
        is_high: bool
    ) -> float:
        """
        Calculate zone strength.
        
        Args:
            bars: List of OHLCV bars
            level: Zone level
            is_high: True for high zone, False for low zone
            
        Returns:
            Zone strength multiplier
        """
        if not bars:
            return 1.0
        
        # Calculate average volume at the zone
        zone_volume = 0
        zone_touches = 0
        
        for bar in bars:
            if is_high:
                if abs(bar.high - level) / level <= self.zone_tolerance:
                    zone_volume += bar.volume
                    zone_touches += 1
            else:
                if abs(bar.low - level) / level <= self.zone_tolerance:
                    zone_volume += bar.volume
                    zone_touches += 1
        
        if zone_touches == 0:
            return 1.0
        
        avg_zone_volume = zone_volume / zone_touches
        avg_total_volume = sum(bar.volume for bar in bars) / len(bars)
        
        # Strength based on relative volume
        strength = avg_zone_volume / avg_total_volume if avg_total_volume > 0 else 1.0
        
        return min(strength, 3.0)  # Cap at 3x average volume
    
    def _calculate_or_zone_quality(
        self,
        or_data,
        bars: List[OHLCVBar]
    ) -> int:
        """
        Calculate opening range zone quality.
        
        Args:
            or_data: OpeningRange object
            bars: List of OHLCV bars
            
        Returns:
            Quality score between 0 and 2
        """
        if not or_data or not bars:
            return 0
        
        # Quality based on OR range size and volume
        range_size = or_data.range_size
        avg_price = sum((bar.high + bar.low) / 2 for bar in bars) / len(bars)
        range_percentage = range_size / avg_price * 100 if avg_price > 0 else 0
        
        if range_percentage >= 2.0:  # 2% or more range
            return 2
        elif range_percentage >= 1.0:  # 1% or more range
            return 1
        else:
            return 0
    
    def _calculate_value_area(
        self,
        bars: List[OHLCVBar]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate value area high and low.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            Tuple of (value_area_high, value_area_low)
        """
        if not bars:
            return None, None
        
        # Create price levels
        min_price = min(bar.low for bar in bars)
        max_price = max(bar.high for bar in bars)
        
        if min_price == max_price:
            return None, None
        
        # Create price bins
        num_bins = 20
        bin_size = (max_price - min_price) / num_bins
        
        # Calculate volume at each price level
        price_volume = {}
        for bar in bars:
            typical_price = (bar.high + bar.low + bar.close) / 3.0
            bin_index = int((typical_price - min_price) / bin_size)
            bin_index = max(0, min(bin_index, num_bins - 1))
            
            price_level = min_price + (bin_index * bin_size)
            if price_level not in price_volume:
                price_volume[price_level] = 0
            price_volume[price_level] += bar.volume
        
        # Sort by volume and find value area
        sorted_prices = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
        
        total_volume = sum(volume for _, volume in price_volume.items())
        target_volume = total_volume * (self.value_area_percentile / 100.0)
        
        accumulated_volume = 0
        value_area_prices = []
        
        for price, volume in sorted_prices:
            accumulated_volume += volume
            value_area_prices.append(price)
            
            if accumulated_volume >= target_volume:
                break
        
        if not value_area_prices:
            return None, None
        
        vah = max(value_area_prices)
        val = min(value_area_prices)
        
        return vah, val