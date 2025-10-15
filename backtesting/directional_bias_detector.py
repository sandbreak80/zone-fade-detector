#!/usr/bin/env python3
"""
Directional Bias Detection System

This module implements the core directional bias detection logic
as outlined in the original Zone-Based Intraday Trading Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class DirectionalBias(Enum):
    """Directional bias states."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SessionType(Enum):
    """Session type classifications."""
    TREND_DAY = "trend_day"
    BALANCED_DAY = "balanced_day"
    CHOPPY_DAY = "choppy_day"


@dataclass
class BiasAnalysis:
    """Results of directional bias analysis."""
    bias: DirectionalBias
    confidence: float  # 0.0 to 1.0
    session_type: SessionType
    session_confidence: float
    choch_confirmed: bool
    structure_broken: bool
    momentum_score: float  # -1.0 to 1.0
    volume_profile: str  # "increasing", "decreasing", "stable"
    key_levels: Dict[str, float]  # Support/resistance levels
    analysis_timestamp: datetime


class DirectionalBiasDetector:
    """Detects directional bias using multiple timeframes and market structure."""
    
    def __init__(self, 
                 lookback_bars: int = 50,
                 choch_lookback: int = 20,
                 volume_threshold: float = 1.5):
        """
        Initialize directional bias detector.
        
        Args:
            lookback_bars: Number of bars to look back for structure analysis
            choch_lookback: Number of bars to look back for CHoCH detection
            volume_threshold: Volume spike threshold (1.5 = 50% above average)
        """
        self.lookback_bars = lookback_bars
        self.choch_lookback = choch_lookback
        self.volume_threshold = volume_threshold
        
        # Track zone touch history for first touch rule
        self.zone_touch_history = {}
    
    def detect_directional_bias(self, bars: List, current_index: int) -> BiasAnalysis:
        """
        Detect current directional bias using multiple analysis methods.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            BiasAnalysis with bias, confidence, and supporting data
        """
        if current_index < self.lookback_bars:
            return self._create_neutral_bias()
        
        # Get analysis window
        start_index = max(0, current_index - self.lookback_bars)
        analysis_bars = bars[start_index:current_index + 1]
        
        # Multiple analysis methods
        structure_analysis = self._analyze_market_structure(analysis_bars)
        choch_analysis = self._analyze_choch_confirmation(analysis_bars)
        momentum_analysis = self._analyze_momentum(analysis_bars)
        volume_analysis = self._analyze_volume_profile(analysis_bars)
        session_analysis = self._analyze_session_type(analysis_bars)
        
        # Combine analyses to determine bias
        bias, confidence = self._combine_bias_analyses(
            structure_analysis, choch_analysis, momentum_analysis, volume_analysis
        )
        
        # Determine session type
        session_type, session_confidence = self._determine_session_type(
            session_analysis, structure_analysis, volume_analysis
        )
        
        # Extract key levels
        key_levels = self._extract_key_levels(analysis_bars)
        
        return BiasAnalysis(
            bias=bias,
            confidence=confidence,
            session_type=session_type,
            session_confidence=session_confidence,
            choch_confirmed=choch_analysis['confirmed'],
            structure_broken=structure_analysis['structure_broken'],
            momentum_score=momentum_analysis['score'],
            volume_profile=volume_analysis['profile'],
            key_levels=key_levels,
            analysis_timestamp=analysis_bars[-1].timestamp
        )
    
    def _analyze_market_structure(self, bars: List) -> Dict:
        """Analyze market structure for trend direction."""
        if len(bars) < 10:
            return {'direction': 'neutral', 'strength': 0.0, 'structure_broken': False}
        
        # Find swing highs and lows
        swing_highs = self._find_swing_highs(bars)
        swing_lows = self._find_swing_lows(bars)
        
        # Analyze structure progression
        structure_score = 0.0
        structure_broken = False
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for higher highs and higher lows (bullish)
            recent_highs = swing_highs[-2:]
            recent_lows = swing_lows[-2:]
            
            if (recent_highs[1] > recent_highs[0] and 
                recent_lows[1] > recent_lows[0]):
                structure_score = 0.8  # Strong bullish structure
            elif (recent_highs[1] > recent_highs[0] or 
                  recent_lows[1] > recent_lows[0]):
                structure_score = 0.4  # Weak bullish structure
            elif (recent_highs[1] < recent_highs[0] and 
                  recent_lows[1] < recent_lows[0]):
                structure_score = -0.8  # Strong bearish structure
            elif (recent_highs[1] < recent_highs[0] or 
                  recent_lows[1] < recent_lows[0]):
                structure_score = -0.4  # Weak bearish structure
            else:
                structure_score = 0.0  # Neutral structure
        
        # Determine direction
        if structure_score > 0.3:
            direction = 'bullish'
        elif structure_score < -0.3:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'direction': direction,
            'strength': abs(structure_score),
            'structure_broken': structure_broken,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }
    
    def _analyze_choch_confirmation(self, bars: List) -> Dict:
        """Analyze Change of Character (CHoCH) confirmation."""
        if len(bars) < self.choch_lookback:
            return {'confirmed': False, 'direction': 'neutral', 'strength': 0.0}
        
        # Look for CHoCH patterns in recent bars
        recent_bars = bars[-self.choch_lookback:]
        
        # Simple CHoCH detection: look for break of previous swing high/low
        choch_confirmed = False
        choch_direction = 'neutral'
        choch_strength = 0.0
        
        if len(recent_bars) >= 10:
            # Find recent swing high and low
            recent_high = max(bar.high for bar in recent_bars[:-5])  # Exclude last 5 bars
            recent_low = min(bar.low for bar in recent_bars[:-5])
            
            # Check if current bars break these levels
            current_bars = recent_bars[-5:]
            
            # Check for bullish CHoCH (break above recent high)
            if any(bar.close > recent_high for bar in current_bars):
                choch_confirmed = True
                choch_direction = 'bullish'
                choch_strength = 0.8
            
            # Check for bearish CHoCH (break below recent low)
            elif any(bar.close < recent_low for bar in current_bars):
                choch_confirmed = True
                choch_direction = 'bearish'
                choch_strength = 0.8
        
        return {
            'confirmed': choch_confirmed,
            'direction': choch_direction,
            'strength': choch_strength
        }
    
    def _analyze_momentum(self, bars: List) -> Dict:
        """Analyze price momentum using multiple indicators."""
        if len(bars) < 20:
            return {'score': 0.0, 'direction': 'neutral'}
        
        # Calculate momentum indicators
        closes = [bar.close for bar in bars]
        
        # Rate of Change (ROC)
        roc_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
        roc_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0
        
        # Simple Moving Average slope
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else sma_10
        sma_slope = (sma_10 - sma_20) / sma_20 if sma_20 != 0 else 0
        
        # Price position relative to recent range
        recent_high = max(bar.high for bar in bars[-10:])
        recent_low = min(bar.low for bar in bars[-10:])
        price_position = (closes[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        
        # Combine momentum indicators
        momentum_score = (roc_5 * 0.4 + roc_10 * 0.3 + sma_slope * 0.2 + (price_position - 0.5) * 0.1)
        
        # Normalize to -1 to 1 range
        momentum_score = max(-1.0, min(1.0, momentum_score * 10))
        
        if momentum_score > 0.2:
            direction = 'bullish'
        elif momentum_score < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'score': momentum_score,
            'direction': direction,
            'roc_5': roc_5,
            'roc_10': roc_10,
            'sma_slope': sma_slope,
            'price_position': price_position
        }
    
    def _analyze_volume_profile(self, bars: List) -> Dict:
        """Analyze volume profile for confirmation."""
        if len(bars) < 10:
            return {'profile': 'stable', 'trend': 'neutral'}
        
        volumes = [bar.volume for bar in bars]
        avg_volume = np.mean(volumes)
        
        # Recent volume trend
        recent_volumes = volumes[-5:]
        recent_avg = np.mean(recent_volumes)
        
        # Volume trend analysis
        if recent_avg > avg_volume * 1.2:
            profile = 'increasing'
        elif recent_avg < avg_volume * 0.8:
            profile = 'decreasing'
        else:
            profile = 'stable'
        
        # Volume-price relationship
        recent_closes = [bar.close for bar in bars[-5:]]
        price_trend = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        volume_trend = (recent_avg - avg_volume) / avg_volume
        
        if price_trend > 0 and volume_trend > 0:
            trend = 'bullish'
        elif price_trend < 0 and volume_trend > 0:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'profile': profile,
            'trend': trend,
            'avg_volume': avg_volume,
            'recent_avg': recent_avg,
            'volume_ratio': recent_avg / avg_volume if avg_volume > 0 else 1.0
        }
    
    def _analyze_session_type(self, bars: List) -> Dict:
        """Analyze session type (Trend Day vs Balanced Day)."""
        if len(bars) < 20:
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Opening range analysis (first 30 minutes)
        opening_bars = bars[:min(30, len(bars))]
        if len(opening_bars) < 5:
            return {'type': 'unknown', 'confidence': 0.0}
        
        opening_high = max(bar.high for bar in opening_bars)
        opening_low = min(bar.low for bar in opening_bars)
        opening_range = opening_high - opening_low
        
        # Current price relative to opening range
        current_price = bars[-1].close
        opening_mid = (opening_high + opening_low) / 2
        
        # Price action analysis
        recent_bars = bars[-10:]
        recent_high = max(bar.high for bar in recent_bars)
        recent_low = min(bar.low for bar in recent_bars)
        recent_range = recent_high - recent_low
        
        # Trend day characteristics
        # 1. Price breaks out of opening range
        # 2. Sustained directional movement
        # 3. Higher volume on breakouts
        
        breakout_above = current_price > opening_high
        breakout_below = current_price < opening_low
        sustained_move = abs(current_price - opening_mid) > opening_range * 0.5
        
        # Volume analysis for trend confirmation
        opening_volume = np.mean([bar.volume for bar in opening_bars])
        recent_volume = np.mean([bar.volume for bar in recent_bars])
        volume_increase = recent_volume > opening_volume * 1.2
        
        # Determine session type
        if (breakout_above or breakout_below) and sustained_move and volume_increase:
            session_type = 'trend_day'
            confidence = 0.8
        elif abs(current_price - opening_mid) < opening_range * 0.3:
            session_type = 'balanced_day'
            confidence = 0.7
        else:
            session_type = 'choppy_day'
            confidence = 0.6
        
        return {
            'type': session_type,
            'confidence': confidence,
            'opening_range': opening_range,
            'current_vs_opening': (current_price - opening_mid) / opening_mid if opening_mid > 0 else 0,
            'breakout_above': breakout_above,
            'breakout_below': breakout_below,
            'sustained_move': sustained_move,
            'volume_increase': volume_increase
        }
    
    def _combine_bias_analyses(self, structure: Dict, choch: Dict, 
                              momentum: Dict, volume: Dict) -> Tuple[DirectionalBias, float]:
        """Combine all analyses to determine final bias."""
        # Weighted scoring system
        structure_weight = 0.3
        choch_weight = 0.25
        momentum_weight = 0.25
        volume_weight = 0.2
        
        # Convert to scores
        structure_score = 0.0
        if structure['direction'] == 'bullish':
            structure_score = structure['strength']
        elif structure['direction'] == 'bearish':
            structure_score = -structure['strength']
        
        choch_score = 0.0
        if choch['confirmed']:
            if choch['direction'] == 'bullish':
                choch_score = choch['strength']
            elif choch['direction'] == 'bearish':
                choch_score = -choch['strength']
        
        momentum_score = momentum['score']
        
        volume_score = 0.0
        if volume['trend'] == 'bullish':
            volume_score = 0.5
        elif volume['trend'] == 'bearish':
            volume_score = -0.5
        
        # Calculate weighted score
        total_score = (
            structure_score * structure_weight +
            choch_score * choch_weight +
            momentum_score * momentum_weight +
            volume_score * volume_weight
        )
        
        # Determine bias and confidence
        if total_score > 0.3:
            bias = DirectionalBias.BULLISH
            confidence = min(0.95, total_score)
        elif total_score < -0.3:
            bias = DirectionalBias.BEARISH
            confidence = min(0.95, abs(total_score))
        else:
            bias = DirectionalBias.NEUTRAL
            confidence = 0.5
        
        return bias, confidence
    
    def _determine_session_type(self, session: Dict, structure: Dict, 
                               volume: Dict) -> Tuple[SessionType, float]:
        """Determine session type based on analysis."""
        session_type_str = session['type']
        confidence = session['confidence']
        
        # Adjust confidence based on structure and volume
        if session_type_str == 'trend_day':
            if structure['strength'] > 0.6 and volume['profile'] == 'increasing':
                confidence = min(0.95, confidence + 0.1)
            elif structure['strength'] < 0.3:
                confidence = max(0.3, confidence - 0.2)
        
        # Convert to enum
        if session_type_str == 'trend_day':
            return SessionType.TREND_DAY, confidence
        elif session_type_str == 'balanced_day':
            return SessionType.BALANCED_DAY, confidence
        else:
            return SessionType.CHOPPY_DAY, confidence
    
    def _extract_key_levels(self, bars: List) -> Dict[str, float]:
        """Extract key support and resistance levels."""
        if len(bars) < 10:
            return {}
        
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        
        # Find recent swing points
        swing_highs = self._find_swing_highs(bars)
        swing_lows = self._find_swing_lows(bars)
        
        # Current levels
        current_high = max(highs[-5:])
        current_low = min(lows[-5:])
        
        return {
            'recent_high': current_high,
            'recent_low': current_low,
            'swing_high': swing_highs[-1] if swing_highs else current_high,
            'swing_low': swing_lows[-1] if swing_lows else current_low,
            'session_high': max(highs),
            'session_low': min(lows)
        }
    
    def _find_swing_highs(self, bars: List, lookback: int = 5) -> List[float]:
        """Find swing highs in the bars."""
        if len(bars) < lookback * 2 + 1:
            return []
        
        swing_highs = []
        for i in range(lookback, len(bars) - lookback):
            is_swing_high = True
            current_high = bars[i].high
            
            # Check if current bar is higher than surrounding bars
            for j in range(i - lookback, i + lookback + 1):
                if j != i and bars[j].high >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(current_high)
        
        return swing_highs
    
    def _find_swing_lows(self, bars: List, lookback: int = 5) -> List[float]:
        """Find swing lows in the bars."""
        if len(bars) < lookback * 2 + 1:
            return []
        
        swing_lows = []
        for i in range(lookback, len(bars) - lookback):
            is_swing_low = True
            current_low = bars[i].low
            
            # Check if current bar is lower than surrounding bars
            for j in range(i - lookback, i + lookback + 1):
                if j != i and bars[j].low <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(current_low)
        
        return swing_lows
    
    def _create_neutral_bias(self) -> BiasAnalysis:
        """Create neutral bias when insufficient data."""
        return BiasAnalysis(
            bias=DirectionalBias.NEUTRAL,
            confidence=0.5,
            session_type=SessionType.BALANCED_DAY,
            session_confidence=0.5,
            choch_confirmed=False,
            structure_broken=False,
            momentum_score=0.0,
            volume_profile='stable',
            key_levels={},
            analysis_timestamp=datetime.now()
        )
    
    def is_first_touch(self, zone_id: str, current_time: datetime) -> bool:
        """Check if this is the first touch of a zone."""
        if zone_id not in self.zone_touch_history:
            self.zone_touch_history[zone_id] = current_time
            return True
        
        # Check if enough time has passed since last touch (e.g., 1 hour)
        time_since_last = current_time - self.zone_touch_history[zone_id]
        if time_since_last.total_seconds() > 3600:  # 1 hour
            self.zone_touch_history[zone_id] = current_time
            return True
        
        return False


def test_directional_bias_detector():
    """Test the directional bias detector with sample data."""
    detector = DirectionalBiasDetector()
    
    # Create sample bars
    bars = []
    base_price = 100.0
    for i in range(100):
        # Create trending data
        price = base_price + i * 0.1 + np.random.normal(0, 0.5)
        bar = type('Bar', (), {
            'open': price - 0.2,
            'high': price + 0.3,
            'low': price - 0.3,
            'close': price,
            'volume': 1000 + np.random.randint(-200, 200),
            'timestamp': datetime.now() + timedelta(minutes=i)
        })()
        bars.append(bar)
    
    # Test bias detection
    bias_analysis = detector.detect_directional_bias(bars, len(bars) - 1)
    
    print(f"Bias: {bias_analysis.bias.value}")
    print(f"Confidence: {bias_analysis.confidence:.2f}")
    print(f"Session Type: {bias_analysis.session_type.value}")
    print(f"CHoCH Confirmed: {bias_analysis.choch_confirmed}")
    print(f"Momentum Score: {bias_analysis.momentum_score:.2f}")
    print(f"Volume Profile: {bias_analysis.volume_profile}")


if __name__ == "__main__":
    test_directional_bias_detector()