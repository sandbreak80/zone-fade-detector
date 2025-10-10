"""
Market context analyzer for Zone Fade setups.

This module provides analysis of market context including trend day detection,
VWAP analysis, and value area overlap detection.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import numpy as np

from zone_fade_detector.core.models import OHLCVBar, MarketContext, VWAPData
from zone_fade_detector.indicators.vwap import VWAPCalculator
from zone_fade_detector.indicators.opening_range import OpeningRangeCalculator


class MarketContextAnalyzer:
    """
    Analyzer for market context and environment.
    
    Provides methods to analyze market conditions including trend day detection,
    VWAP analysis, and value area overlap for Zone Fade setups.
    """
    
    def __init__(
        self,
        trend_day_threshold: float = 0.002,  # 0.2% VWAP slope threshold
        balance_threshold: float = 0.001,    # 0.1% VWAP slope for balance
        value_area_tolerance: float = 0.01   # 1% tolerance for value area overlap
    ):
        """
        Initialize market context analyzer.
        
        Args:
            trend_day_threshold: VWAP slope threshold for trend day detection
            balance_threshold: VWAP slope threshold for balanced market
            value_area_tolerance: Tolerance for value area overlap detection
        """
        self.trend_day_threshold = trend_day_threshold
        self.balance_threshold = balance_threshold
        self.value_area_tolerance = value_area_tolerance
        self.logger = logging.getLogger(__name__)
        
        # Initialize calculators
        self.vwap_calculator = VWAPCalculator()
        self.or_calculator = OpeningRangeCalculator()
    
    def analyze_market_context(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> MarketContext:
        """
        Analyze market context from OHLCV bars.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            MarketContext object with analysis results
        """
        if not bars:
            return MarketContext()
        
        if current_index is None:
            current_index = len(bars) - 1
        
        # Analyze different aspects
        is_trend_day = self._detect_trend_day(bars, current_index)
        vwap_slope = self._calculate_vwap_slope(bars, current_index)
        value_area_overlap = self._detect_value_area_overlap(bars, current_index)
        market_balance = self._calculate_market_balance(bars, current_index)
        volatility_regime = self._determine_volatility_regime(bars, current_index)
        session_type = self._determine_session_type(bars[current_index].timestamp)
        
        return MarketContext(
            is_trend_day=is_trend_day,
            vwap_slope=vwap_slope,
            value_area_overlap=value_area_overlap,
            market_balance=market_balance,
            volatility_regime=volatility_regime,
            session_type=session_type
        )
    
    def _detect_trend_day(
        self,
        bars: List[OHLCVBar],
        current_index: int
    ) -> bool:
        """
        Detect if current session is a trend day.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            True if it's a trend day
        """
        # Get VWAP data
        vwap_data = self.vwap_calculator.calculate_vwap(bars[:current_index + 1])
        
        if not vwap_data:
            return False
        
        # Check VWAP slope
        if abs(vwap_data.slope) > self.trend_day_threshold:
            return True
        
        # Additional trend day criteria
        # Check for sustained directional movement
        recent_bars = bars[max(0, current_index - 10):current_index + 1]
        if len(recent_bars) < 5:
            return False
        
        # Check for consistent directional movement
        price_changes = []
        for i in range(1, len(recent_bars)):
            change = (recent_bars[i].close - recent_bars[i-1].close) / recent_bars[i-1].close
            price_changes.append(change)
        
        if not price_changes:
            return False
        
        # Check if most changes are in the same direction
        positive_changes = sum(1 for change in price_changes if change > 0)
        negative_changes = sum(1 for change in price_changes if change < 0)
        
        # If 80% or more changes are in same direction, it's a trend day
        total_changes = len(price_changes)
        if total_changes > 0:
            if positive_changes / total_changes >= 0.8 or negative_changes / total_changes >= 0.8:
                return True
        
        return False
    
    def _calculate_vwap_slope(
        self,
        bars: List[OHLCVBar],
        current_index: int
    ) -> float:
        """
        Calculate VWAP slope.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            VWAP slope value
        """
        vwap_data = self.vwap_calculator.calculate_vwap(bars[:current_index + 1])
        
        if not vwap_data:
            return 0.0
        
        return vwap_data.slope
    
    def _detect_value_area_overlap(
        self,
        bars: List[OHLCVBar],
        current_index: int
    ) -> bool:
        """
        Detect value area overlap.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            True if value areas overlap
        """
        # Get opening range
        or_data = self.or_calculator.calculate_opening_range(bars[:current_index + 1])
        
        if not or_data:
            return False
        
        # Get VWAP data
        vwap_data = self.vwap_calculator.calculate_vwap(bars[:current_index + 1])
        
        if not vwap_data:
            return False
        
        # Check if VWAP is within opening range
        vwap_in_or = or_data.low <= vwap_data.vwap <= or_data.high
        
        # Check if opening range overlaps with VWAP bands
        or_overlaps_vwap = (
            or_data.low <= vwap_data.upper_1sigma and
            or_data.high >= vwap_data.lower_1sigma
        )
        
        return vwap_in_or or or_overlaps_vwap
    
    def _calculate_market_balance(
        self,
        bars: List[OHLCVBar],
        current_index: int
    ) -> float:
        """
        Calculate market balance score (0.0 to 1.0).
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            Market balance score (0.0 = bearish, 0.5 = balanced, 1.0 = bullish)
        """
        if current_index < 10:
            return 0.5  # Default balanced
        
        recent_bars = bars[max(0, current_index - 10):current_index + 1]
        
        # Calculate price momentum
        price_changes = []
        for i in range(1, len(recent_bars)):
            change = (recent_bars[i].close - recent_bars[i-1].close) / recent_bars[i-1].close
            price_changes.append(change)
        
        if not price_changes:
            return 0.5
        
        # Calculate average change
        avg_change = np.mean(price_changes)
        
        # Normalize to 0-1 range
        # -0.01 to +0.01 maps to 0.0 to 1.0
        normalized_balance = (avg_change + 0.01) / 0.02
        normalized_balance = max(0.0, min(1.0, normalized_balance))
        
        return normalized_balance
    
    def _determine_volatility_regime(
        self,
        bars: List[OHLCVBar],
        current_index: int
    ) -> str:
        """
        Determine current volatility regime.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            Volatility regime: 'low', 'normal', or 'high'
        """
        if current_index < 20:
            return 'normal'  # Default
        
        recent_bars = bars[max(0, current_index - 20):current_index + 1]
        
        # Calculate price ranges
        ranges = []
        for bar in recent_bars:
            range_pct = (bar.high - bar.low) / bar.close * 100
            ranges.append(range_pct)
        
        if not ranges:
            return 'normal'
        
        avg_range = np.mean(ranges)
        std_range = np.std(ranges)
        
        # Determine regime based on average range
        if avg_range < 0.5:  # Less than 0.5% average range
            return 'low'
        elif avg_range > 2.0:  # More than 2% average range
            return 'high'
        else:
            return 'normal'
    
    def _determine_session_type(self, timestamp: datetime) -> str:
        """
        Determine session type based on timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Session type: 'premarket', 'regular', or 'afterhours'
        """
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Regular trading hours: 9:30 AM - 4:00 PM ET
        if 9 <= hour < 16 or (hour == 9 and minute >= 30):
            return 'regular'
        elif hour < 9 or (hour == 9 and minute < 30):
            return 'premarket'
        else:
            return 'afterhours'
    
    def analyze_intermarket_divergence(
        self,
        symbol_data: Dict[str, List[OHLCVBar]]
    ) -> Dict[str, Any]:
        """
        Analyze intermarket divergence between symbols.
        
        Args:
            symbol_data: Dictionary mapping symbols to their OHLCV bars
            
        Returns:
            Dictionary with divergence analysis
        """
        if len(symbol_data) < 2:
            return {'has_divergence': False}
        
        # Calculate price changes for each symbol
        price_changes = {}
        for symbol, bars in symbol_data.items():
            if not bars:
                continue
            
            # Calculate percentage change from first to last bar
            first_price = bars[0].close
            last_price = bars[-1].close
            change_pct = (last_price - first_price) / first_price * 100
            
            price_changes[symbol] = change_pct
        
        if len(price_changes) < 2:
            return {'has_divergence': False}
        
        # Check for divergence
        symbols = list(price_changes.keys())
        diverging_pairs = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                change1, change2 = price_changes[symbol1], price_changes[symbol2]
                
                # Check if changes are in opposite directions
                if (change1 > 0 and change2 < 0) or (change1 < 0 and change2 > 0):
                    diverging_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'change1': change1,
                        'change2': change2
                    })
        
        return {
            'has_divergence': len(diverging_pairs) > 0,
            'diverging_pairs': diverging_pairs,
            'price_changes': price_changes,
            'divergence_strength': len(diverging_pairs) / (len(symbols) * (len(symbols) - 1) / 2)
        }
    
    def get_market_summary(
        self,
        bars: List[OHLCVBar],
        current_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive market summary.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index (defaults to last bar)
            
        Returns:
            Dictionary with market summary
        """
        if not bars:
            return {}
        
        if current_index is None:
            current_index = len(bars) - 1
        
        context = self.analyze_market_context(bars, current_index)
        
        return {
            'is_trend_day': context.is_trend_day,
            'vwap_slope': context.vwap_slope,
            'is_balanced': context.is_balanced,
            'value_area_overlap': context.value_area_overlap,
            'market_balance': context.market_balance,
            'volatility_regime': context.volatility_regime,
            'session_type': context.session_type,
            'recommendation': self._get_market_recommendation(context)
        }
    
    def _get_market_recommendation(self, context: MarketContext) -> str:
        """
        Get market recommendation based on context.
        
        Args:
            context: MarketContext object
            
        Returns:
            Recommendation string
        """
        if context.is_trend_day:
            return "Avoid Zone Fade setups - trend day detected"
        elif context.is_balanced:
            return "Favorable for Zone Fade setups - balanced market"
        elif context.value_area_overlap:
            return "Good for Zone Fade setups - value area overlap"
        else:
            return "Neutral - monitor for better context"