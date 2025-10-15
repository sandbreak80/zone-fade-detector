"""
Market Type Detection Filter

This module implements market type detection to classify trading days as either
TREND_DAY or RANGE_BOUND. This is a critical filter that prevents zone fade
signals on trend days, as per trading methodology Rule #6.

Features:
- NYSE TICK analysis (30-bar rolling mean)
- A/D Line slope analysis (60-bar window)
- Related markets alignment check
- ATR expansion detection
- Directional bars analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class MarketType(Enum):
    """Market type classifications."""
    TREND_DAY = "TREND_DAY"
    RANGE_BOUND = "RANGE_BOUND"
    UNKNOWN = "UNKNOWN"

@dataclass
class MarketTypeResult:
    """Result of market type detection."""
    market_type: MarketType
    confidence: float
    contributing_factors: Dict[str, bool]
    tick_mean: float
    ad_slope: float
    atr_expansion: float
    directional_bars_pct: float
    related_markets_aligned: bool
    timestamp: datetime
    reasoning: str

class MarketTypeDetector:
    """
    Detects market type (trend vs range-bound) using multiple indicators.
    
    This is a critical filter that prevents zone fade signals on trend days.
    Zone fade strategies work best in range-bound markets where price
    reverts to mean (VWAP) rather than trending in one direction.
    """
    
    def __init__(self, 
                 tick_threshold: float = 800.0,
                 ad_slope_threshold: float = 1000.0,
                 atr_expansion_threshold: float = 1.3,
                 directional_bars_threshold: float = 0.7,
                 tick_window: int = 30,
                 ad_window: int = 60,
                 atr_window: int = 20):
        """
        Initialize market type detector.
        
        Args:
            tick_threshold: TICK mean threshold for trend detection
            ad_slope_threshold: A/D Line slope threshold for trend detection
            atr_expansion_threshold: ATR expansion threshold for trend detection
            directional_bars_threshold: Directional bars percentage threshold
            tick_window: Window size for TICK analysis
            ad_window: Window size for A/D Line analysis
            atr_window: Window size for ATR calculation
        """
        self.tick_threshold = tick_threshold
        self.ad_slope_threshold = ad_slope_threshold
        self.atr_expansion_threshold = atr_expansion_threshold
        self.directional_bars_threshold = directional_bars_threshold
        self.tick_window = tick_window
        self.ad_window = ad_window
        self.atr_window = atr_window
        
        # Statistics tracking
        self.trend_days_detected = 0
        self.range_bound_days_detected = 0
        self.total_classifications = 0
    
    def detect_market_type(self, 
                          price_bars: List,
                          tick_data: List[float],
                          ad_line_data: List[float],
                          related_markets: Optional[Dict[str, List]] = None) -> MarketTypeResult:
        """
        Detect market type using multiple indicators.
        
        Args:
            price_bars: List of OHLCV bars
            tick_data: List of NYSE TICK values
            ad_line_data: List of Advance/Decline Line values
            related_markets: Dict of related market data (ES, NQ, RTY)
            
        Returns:
            MarketTypeResult with classification and details
        """
        if len(price_bars) < max(self.tick_window, self.ad_window, self.atr_window):
            return MarketTypeResult(
                market_type=MarketType.UNKNOWN,
                confidence=0.0,
                contributing_factors={},
                tick_mean=0.0,
                ad_slope=0.0,
                atr_expansion=0.0,
                directional_bars_pct=0.0,
                related_markets_aligned=False,
                timestamp=datetime.now(),
                reasoning="Insufficient data for analysis"
            )
        
        # Analyze each indicator
        tick_analysis = self._analyze_tick(tick_data)
        ad_analysis = self._analyze_ad_line(ad_line_data)
        atr_analysis = self._analyze_atr_expansion(price_bars)
        directional_analysis = self._analyze_directional_bars(price_bars)
        related_markets_analysis = self._analyze_related_markets(related_markets)
        
        # Determine market type based on indicators
        contributing_factors = {
            'tick_skewed': tick_analysis['is_skewed'],
            'ad_trending': ad_analysis['is_trending'],
            'atr_expanded': atr_analysis['is_expanded'],
            'directional_bars': directional_analysis['is_directional'],
            'related_markets_aligned': related_markets_analysis['is_aligned']
        }
        
        # Count positive indicators
        positive_indicators = sum(contributing_factors.values())
        
        # Classification logic: 2 or more indicators = TREND_DAY
        if positive_indicators >= 2:
            market_type = MarketType.TREND_DAY
            confidence = min(0.9, 0.5 + (positive_indicators * 0.1))
            reasoning = f"Trend day detected: {positive_indicators}/5 indicators positive"
        else:
            market_type = MarketType.RANGE_BOUND
            confidence = min(0.9, 0.5 + ((5 - positive_indicators) * 0.1))
            reasoning = f"Range-bound day: {positive_indicators}/5 indicators positive"
        
        # Update statistics
        self.total_classifications += 1
        if market_type == MarketType.TREND_DAY:
            self.trend_days_detected += 1
        else:
            self.range_bound_days_detected += 1
        
        return MarketTypeResult(
            market_type=market_type,
            confidence=confidence,
            contributing_factors=contributing_factors,
            tick_mean=tick_analysis['mean'],
            ad_slope=ad_analysis['slope'],
            atr_expansion=atr_analysis['expansion_ratio'],
            directional_bars_pct=directional_analysis['percentage'],
            related_markets_aligned=related_markets_analysis['is_aligned'],
            timestamp=datetime.now(),
            reasoning=reasoning
        )
    
    def _analyze_tick(self, tick_data: List[float]) -> Dict:
        """Analyze NYSE TICK for sustained skew."""
        if len(tick_data) < self.tick_window:
            return {'is_skewed': False, 'mean': 0.0, 'std': 0.0}
        
        recent_ticks = tick_data[-self.tick_window:]
        mean_tick = np.mean(recent_ticks)
        std_tick = np.std(recent_ticks)
        
        is_skewed = abs(mean_tick) > self.tick_threshold
        
        return {
            'is_skewed': is_skewed,
            'mean': mean_tick,
            'std': std_tick
        }
    
    def _analyze_ad_line(self, ad_line_data: List[float]) -> Dict:
        """Analyze Advance/Decline Line for trending behavior."""
        if len(ad_line_data) < self.ad_window:
            return {'is_trending': False, 'slope': 0.0, 'r_squared': 0.0}
        
        recent_ad = ad_line_data[-self.ad_window:]
        x = np.arange(len(recent_ad))
        
        # Linear regression
        try:
            coeffs = np.polyfit(x, recent_ad, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate R-squared manually
            y_pred = slope * x + intercept
            ss_res = np.sum((recent_ad - y_pred) ** 2)
            ss_tot = np.sum((recent_ad - np.mean(recent_ad)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        except:
            slope = 0
            r_squared = 0
        
        is_trending = abs(slope) > self.ad_slope_threshold
        
        return {
            'is_trending': is_trending,
            'slope': slope,
            'r_squared': r_squared
        }
    
    def _analyze_atr_expansion(self, price_bars: List) -> Dict:
        """Analyze ATR expansion for volatility increase."""
        if len(price_bars) < self.atr_window * 2:
            return {'is_expanded': False, 'expansion_ratio': 1.0, 'current_atr': 0.0, 'avg_atr': 0.0}
        
        # Calculate ATR for recent period
        recent_bars = price_bars[-self.atr_window:]
        current_atr = self._calculate_atr(recent_bars)
        
        # Calculate ATR for baseline period
        baseline_bars = price_bars[-(self.atr_window * 2):-self.atr_window]
        avg_atr = self._calculate_atr(baseline_bars)
        
        expansion_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        is_expanded = expansion_ratio > self.atr_expansion_threshold
        
        return {
            'is_expanded': is_expanded,
            'expansion_ratio': expansion_ratio,
            'current_atr': current_atr,
            'avg_atr': avg_atr
        }
    
    def _analyze_directional_bars(self, price_bars: List) -> Dict:
        """Analyze percentage of directional bars."""
        if len(price_bars) < self.atr_window:
            return {'is_directional': False, 'percentage': 0.0, 'bullish_bars': 0, 'bearish_bars': 0}
        
        recent_bars = price_bars[-self.atr_window:]
        bullish_bars = 0
        bearish_bars = 0
        
        for bar in recent_bars:
            if hasattr(bar, 'close') and hasattr(bar, 'open'):
                if bar.close > bar.open:
                    bullish_bars += 1
                elif bar.close < bar.open:
                    bearish_bars += 1
        
        total_directional = bullish_bars + bearish_bars
        percentage = total_directional / len(recent_bars) if len(recent_bars) > 0 else 0.0
        
        # Check if one direction dominates
        max_directional = max(bullish_bars, bearish_bars)
        is_directional = (max_directional / len(recent_bars)) > self.directional_bars_threshold
        
        return {
            'is_directional': is_directional,
            'percentage': percentage,
            'bullish_bars': bullish_bars,
            'bearish_bars': bearish_bars
        }
    
    def _analyze_related_markets(self, related_markets: Optional[Dict[str, List]]) -> Dict:
        """Analyze related markets for directional alignment."""
        if not related_markets or len(related_markets) == 0:
            return {'is_aligned': False, 'aligned_count': 0, 'total_count': 0}
        
        aligned_count = 0
        total_count = 0
        
        for symbol, bars in related_markets.items():
            if len(bars) < 2:
                continue
            
            total_count += 1
            
            # Simple trend detection: compare first and last close
            first_close = bars[0].close if hasattr(bars[0], 'close') else bars[0]['close']
            last_close = bars[-1].close if hasattr(bars[-1], 'close') else bars[-1]['close']
            
            if last_close > first_close:
                aligned_count += 1
        
        is_aligned = aligned_count >= (total_count * 0.6) if total_count > 0 else False
        
        return {
            'is_aligned': is_aligned,
            'aligned_count': aligned_count,
            'total_count': total_count
        }
    
    def _calculate_atr(self, bars: List, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.0
        
        true_ranges = []
        
        for i in range(1, len(bars)):
            current = bars[i]
            previous = bars[i-1]
            
            # Get OHLC values
            if hasattr(current, 'high'):
                high = current.high
                low = current.low
                prev_close = previous.close
            else:
                high = current['high']
                low = current['low']
                prev_close = previous['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        return np.mean(true_ranges) if true_ranges else 0.0
    
    def is_range_bound(self, result: MarketTypeResult) -> bool:
        """Check if market is range-bound (suitable for zone fades)."""
        return result.market_type == MarketType.RANGE_BOUND
    
    def is_trend_day(self, result: MarketTypeResult) -> bool:
        """Check if market is trending (not suitable for zone fades)."""
        return result.market_type == MarketType.TREND_DAY
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        if self.total_classifications == 0:
            return {
                'total_classifications': 0,
                'trend_days_detected': 0,
                'range_bound_days_detected': 0,
                'trend_day_percentage': 0.0,
                'range_bound_percentage': 0.0
            }
        
        return {
            'total_classifications': self.total_classifications,
            'trend_days_detected': self.trend_days_detected,
            'range_bound_days_detected': self.range_bound_days_detected,
            'trend_day_percentage': (self.trend_days_detected / self.total_classifications) * 100,
            'range_bound_percentage': (self.range_bound_days_detected / self.total_classifications) * 100
        }
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.trend_days_detected = 0
        self.range_bound_days_detected = 0
        self.total_classifications = 0


class MarketTypeFilter:
    """
    Filter that applies market type detection to zone fade signals.
    
    This filter implements the critical requirement that zone fade signals
    should only be generated on range-bound days, not on trend days.
    """
    
    def __init__(self, detector: MarketTypeDetector):
        """
        Initialize market type filter.
        
        Args:
            detector: MarketTypeDetector instance
        """
        self.detector = detector
        self.signals_vetoed = 0
        self.signals_passed = 0
    
    def filter_signal(self, signal, market_data: Dict) -> Optional[Dict]:
        """
        Filter signal based on market type.
        
        Args:
            signal: Zone fade signal to filter
            market_data: Market data including price bars, TICK, A/D Line
            
        Returns:
            Filtered signal if range-bound, None if trend day
        """
        # Detect market type
        result = self.detector.detect_market_type(
            price_bars=market_data.get('price_bars', []),
            tick_data=market_data.get('tick_data', []),
            ad_line_data=market_data.get('ad_line_data', []),
            related_markets=market_data.get('related_markets', {})
        )
        
        # Apply filter
        if self.detector.is_trend_day(result):
            self.signals_vetoed += 1
            return None  # VETO: Trend day detected
        
        # Add market type info to signal
        if isinstance(signal, dict):
            signal['market_type'] = result.market_type.value
            signal['market_type_confidence'] = result.confidence
            signal['market_type_reasoning'] = result.reasoning
        else:
            # If signal is an object, add attributes
            signal.market_type = result.market_type.value
            signal.market_type_confidence = result.confidence
            signal.market_type_reasoning = result.reasoning
        
        self.signals_passed += 1
        return signal
    
    def get_filter_statistics(self) -> Dict:
        """Get filter statistics."""
        total_signals = self.signals_vetoed + self.signals_passed
        
        return {
            'total_signals_processed': total_signals,
            'signals_vetoed': self.signals_vetoed,
            'signals_passed': self.signals_passed,
            'veto_percentage': (self.signals_vetoed / total_signals * 100) if total_signals > 0 else 0.0,
            'pass_percentage': (self.signals_passed / total_signals * 100) if total_signals > 0 else 0.0,
            'detector_statistics': self.detector.get_statistics()
        }