"""
Swing structure detector for identifying swing highs, lows, and CHoCH.

This module provides detection of swing structure patterns including
swing highs, swing lows, and Change of Character (CHoCH) patterns.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import numpy as np

from zone_fade_detector.core.models import (
    OHLCVBar, 
    SwingPoint, 
    SwingStructure, 
    SetupDirection
)


class SwingStructureDetector:
    """
    Detector for swing structure patterns.
    
    Provides methods to identify swing highs, swing lows, and
    Change of Character (CHoCH) patterns in price action.
    """
    
    def __init__(
        self,
        lookback_bars: int = 20,  # Restored to original 20
        min_swing_size: float = 0.1,  # Restored to original 10%
        swing_confirmation_bars: int = 2  # Restored to original 2
    ):
        """
        Initialize swing structure detector.
        
        Args:
            lookback_bars: Number of bars to look back for swing detection
            min_swing_size: Minimum price move for swing (percentage)
            swing_confirmation_bars: Bars needed to confirm a swing
        """
        self.lookback_bars = lookback_bars
        self.min_swing_size = min_swing_size
        self.swing_confirmation_bars = swing_confirmation_bars
        self.logger = logging.getLogger(__name__)
    
    def detect_swing_structure(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> SwingStructure:
        """
        Detect swing structure from OHLCV bars.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            SwingStructure object with detected swings and CHoCH
        """
        if not bars:
            return SwingStructure()
        
        if current_index is None:
            current_index = len(bars) - 1
        
        # Ensure we have enough bars
        if current_index < self.lookback_bars:
            return SwingStructure()
        
        # Get the relevant bars for analysis
        analysis_bars = bars[max(0, current_index - self.lookback_bars):current_index + 1]
        
        # Detect swing highs and lows
        swing_highs = self._detect_swing_highs(analysis_bars)
        swing_lows = self._detect_swing_lows(analysis_bars)
        
        # Create swing structure
        structure = SwingStructure()
        
        # Add swing points
        for swing in swing_highs:
            structure.add_swing_high(swing)
        
        for swing in swing_lows:
            structure.add_swing_low(swing)
        
        return structure
    
    def _detect_swing_highs(self, bars: List[OHLCVBar]) -> List[SwingPoint]:
        """
        Detect swing highs in the given bars.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            List of SwingPoint objects representing swing highs
        """
        swing_highs = []
        
        if len(bars) < 3:
            return swing_highs
        
        for i in range(1, len(bars) - 1):
            current_bar = bars[i]
            
            # Check if this is a potential swing high
            if self._is_potential_swing_high(bars, i):
                # Calculate swing strength
                strength = self._calculate_swing_strength(bars, i, is_high=True)
                
                # Create swing point
                swing_point = SwingPoint(
                    price=current_bar.high,
                    timestamp=current_bar.timestamp,
                    is_high=True,
                    strength=strength
                )
                
                swing_highs.append(swing_point)
        
        return swing_highs
    
    def _detect_swing_lows(self, bars: List[OHLCVBar]) -> List[SwingPoint]:
        """
        Detect swing lows in the given bars.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            List of SwingPoint objects representing swing lows
        """
        swing_lows = []
        
        if len(bars) < 3:
            return swing_lows
        
        for i in range(1, len(bars) - 1):
            current_bar = bars[i]
            
            # Check if this is a potential swing low
            if self._is_potential_swing_low(bars, i):
                # Calculate swing strength
                strength = self._calculate_swing_strength(bars, i, is_high=False)
                
                # Create swing point
                swing_point = SwingPoint(
                    price=current_bar.low,
                    timestamp=current_bar.timestamp,
                    is_high=False,
                    strength=strength
                )
                
                swing_lows.append(swing_point)
        
        return swing_lows
    
    def _is_potential_swing_high(
        self,
        bars: List[OHLCVBar],
        index: int
    ) -> bool:
        """
        Check if a bar is a potential swing high.
        
        Args:
            bars: List of OHLCV bars
            index: Index of the bar to check
            
        Returns:
            True if the bar is a potential swing high
        """
        if index < self.swing_confirmation_bars or index >= len(bars) - self.swing_confirmation_bars:
            return False
        
        current_bar = bars[index]
        
        # Check if current high is higher than surrounding bars
        for i in range(max(0, index - self.swing_confirmation_bars), 
                      min(len(bars), index + self.swing_confirmation_bars + 1)):
            if i != index and bars[i].high >= current_bar.high:
                return False
        
        # Check minimum swing size (more permissive)
        swing_size = self._calculate_swing_size(bars, index, is_high=True)
        if swing_size < self.min_swing_size * 0.5:  # More permissive threshold
            return False
        
        return True
    
    def _is_potential_swing_low(
        self,
        bars: List[OHLCVBar],
        index: int
    ) -> bool:
        """
        Check if a bar is a potential swing low.
        
        Args:
            bars: List of OHLCV bars
            index: Index of the bar to check
            
        Returns:
            True if the bar is a potential swing low
        """
        if index < self.swing_confirmation_bars or index >= len(bars) - self.swing_confirmation_bars:
            return False
        
        current_bar = bars[index]
        
        # Check if current low is lower than surrounding bars
        for i in range(max(0, index - self.swing_confirmation_bars), 
                      min(len(bars), index + self.swing_confirmation_bars + 1)):
            if i != index and bars[i].low <= current_bar.low:
                return False
        
        # Check minimum swing size (more permissive)
        swing_size = self._calculate_swing_size(bars, index, is_high=False)
        if swing_size < self.min_swing_size * 0.5:  # More permissive threshold
            return False
        
        return True
    
    def _calculate_swing_size(
        self,
        bars: List[OHLCVBar],
        index: int,
        is_high: bool
    ) -> float:
        """
        Calculate the size of a swing as a percentage.
        
        Args:
            bars: List of OHLCV bars
            index: Index of the swing bar
            is_high: True for swing high, False for swing low
            
        Returns:
            Swing size as a percentage
        """
        if index < 1 or index >= len(bars) - 1:
            return 0.0
        
        current_bar = bars[index]
        
        if is_high:
            # For swing high, measure from previous low
            prev_low = min(bars[i].low for i in range(max(0, index - 5), index))
            swing_size = (current_bar.high - prev_low) / prev_low * 100
        else:
            # For swing low, measure from previous high
            prev_high = max(bars[i].high for i in range(max(0, index - 5), index))
            swing_size = (prev_high - current_bar.low) / prev_high * 100
        
        return swing_size
    
    def _calculate_swing_strength(
        self,
        bars: List[OHLCVBar],
        index: int,
        is_high: bool
    ) -> float:
        """
        Calculate the strength of a swing point.
        
        Args:
            bars: List of OHLCV bars
            index: Index of the swing bar
            is_high: True for swing high, False for swing low
            
        Returns:
            Swing strength (1.0 = normal, >1.0 = stronger)
        """
        swing_size = self._calculate_swing_size(bars, index, is_high)
        
        # Base strength on swing size
        if swing_size < 0.5:
            return 0.5
        elif swing_size < 1.0:
            return 1.0
        elif swing_size < 2.0:
            return 1.5
        else:
            return 2.0
    
    def detect_choch(
        self,
        swing_structure: SwingStructure
    ) -> Tuple[bool, Optional[SetupDirection], Optional[datetime]]:
        """
        Detect Change of Character (CHoCH) in swing structure.
        
        Args:
            swing_structure: SwingStructure object to analyze
            
        Returns:
            Tuple of (choch_detected, direction, timestamp)
        """
        if not swing_structure.swing_highs or not swing_structure.swing_lows:
            return False, None, None
        
        # Check for CHoCH patterns
        choch_detected, direction, timestamp = self._check_choch_pattern(
            swing_structure.swing_highs,
            swing_structure.swing_lows
        )
        
        return choch_detected, direction, timestamp
    
    def _check_choch_pattern(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Tuple[bool, Optional[SetupDirection], Optional[datetime]]:
        """
        Check for CHoCH pattern between swing highs and lows.
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            
        Returns:
            Tuple of (choch_detected, direction, timestamp)
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return False, None, None
        
        # Get the last two swing highs and lows
        last_swing_high = swing_highs[-1]
        prev_swing_high = swing_highs[-2]
        last_swing_low = swing_lows[-1]
        prev_swing_low = swing_lows[-2]
        
        # Check for bullish CHoCH (break of swing high structure)
        if last_swing_high.timestamp > last_swing_low.timestamp:
            # Last swing was a high, check if it broke above previous high
            if last_swing_high.price > prev_swing_high.price:
                return True, SetupDirection.SHORT, last_swing_high.timestamp
        
        # Check for bearish CHoCH (break of swing low structure)
        if last_swing_low.timestamp > last_swing_high.timestamp:
            # Last swing was a low, check if it broke below previous low
            if last_swing_low.price < prev_swing_low.price:
                return True, SetupDirection.LONG, last_swing_low.timestamp
        
        return False, None, None
    
    def calculate_swing_strength_score(
        self,
        swing_structure: SwingStructure
    ) -> float:
        """
        Calculate overall swing structure strength score.
        
        Args:
            swing_structure: SwingStructure object to analyze
            
        Returns:
            Strength score between 0.0 and 1.0
        """
        if not swing_structure.swing_highs and not swing_structure.swing_lows:
            return 0.0
        
        # Calculate average swing strength
        all_swings = swing_structure.swing_highs + swing_structure.swing_lows
        avg_strength = sum(swing.strength for swing in all_swings) / len(all_swings)
        
        # Normalize to 0-1 range
        strength_score = min(avg_strength / 2.0, 1.0)
        
        return strength_score
    
    def get_swing_analysis(
        self,
        swing_structure: SwingStructure
    ) -> Dict:
        """
        Get comprehensive swing structure analysis.
        
        Args:
            swing_structure: SwingStructure object to analyze
            
        Returns:
            Dictionary with swing analysis data
        """
        return {
            'swing_highs_count': len(swing_structure.swing_highs),
            'swing_lows_count': len(swing_structure.swing_lows),
            'choch_detected': swing_structure.choch_detected,
            'choch_direction': swing_structure.choch_direction.value if swing_structure.choch_direction else None,
            'choch_timestamp': swing_structure.choch_timestamp.isoformat() if swing_structure.choch_timestamp else None,
            'strength_score': self.calculate_swing_strength_score(swing_structure),
            'last_swing_high': swing_structure.last_swing_high.price if swing_structure.last_swing_high else None,
            'last_swing_low': swing_structure.last_swing_low.price if swing_structure.last_swing_low else None
        }