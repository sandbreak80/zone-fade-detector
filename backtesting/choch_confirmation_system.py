#!/usr/bin/env python3
"""
CHoCH Confirmation System

This module implements enhanced Change of Character (CHoCH) detection
and integration with trading decisions as outlined in the framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from directional_bias_detector import DirectionalBias, SessionType, BiasAnalysis
from session_aware_trading import TradeType, TradeDirection, TradeDecision


class CHoCHType(Enum):
    """Types of Change of Character."""
    BULLISH = "bullish"  # Structure shift from bearish to bullish
    BEARISH = "bearish"  # Structure shift from bullish to bearish
    NONE = "none"        # No CHoCH detected


@dataclass
class CHoCHSignal:
    """CHoCH signal data."""
    choch_type: CHoCHType
    confirmation_time: datetime
    confirmation_price: float
    previous_structure: str  # "bullish", "bearish", "neutral"
    new_structure: str
    confidence: float
    volume_confirmation: bool
    momentum_confirmation: bool
    key_level_break: bool
    invalidation_level: float


class CHoCHConfirmationSystem:
    """Enhanced CHoCH detection and confirmation system."""
    
    def __init__(self, 
                 lookback_bars: int = 20,
                 min_structure_bars: int = 5,
                 volume_threshold: float = 1.3):
        """
        Initialize CHoCH confirmation system.
        
        Args:
            lookback_bars: Number of bars to look back for structure analysis
            min_structure_bars: Minimum bars required for structure confirmation
            volume_threshold: Volume threshold for CHoCH confirmation
        """
        self.lookback_bars = lookback_bars
        self.min_structure_bars = min_structure_bars
        self.volume_threshold = volume_threshold
        
        # Track recent CHoCH signals
        self.recent_choch_signals = []
        self.max_recent_signals = 10
    
    def detect_choch(self, bars: List, current_index: int) -> Optional[CHoCHSignal]:
        """
        Detect Change of Character in the market structure.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            CHoCHSignal if CHoCH detected, None otherwise
        """
        if current_index < self.lookback_bars:
            return None
        
        # Get analysis window
        start_index = max(0, current_index - self.lookback_bars)
        analysis_bars = bars[start_index:current_index + 1]
        
        if len(analysis_bars) < self.min_structure_bars * 2:
            return None
        
        # Analyze structure shift
        structure_shift = self._analyze_structure_shift(analysis_bars)
        if not structure_shift['shift_detected']:
            return None
        
        # Confirm CHoCH with additional criteria
        confirmation = self._confirm_choch(analysis_bars, structure_shift)
        if not confirmation['confirmed']:
            return None
        
        # Create CHoCH signal
        choch_signal = CHoCHSignal(
            choch_type=structure_shift['choch_type'],
            confirmation_time=analysis_bars[-1].timestamp,
            confirmation_price=analysis_bars[-1].close,
            previous_structure=structure_shift['previous_structure'],
            new_structure=structure_shift['new_structure'],
            confidence=confirmation['confidence'],
            volume_confirmation=confirmation['volume_confirmation'],
            momentum_confirmation=confirmation['momentum_confirmation'],
            key_level_break=confirmation['key_level_break'],
            invalidation_level=confirmation['invalidation_level']
        )
        
        # Store recent signal
        self._store_choch_signal(choch_signal)
        
        return choch_signal
    
    def _analyze_structure_shift(self, bars: List) -> Dict:
        """Analyze for structure shift in the bars."""
        
        # Split bars into two periods for comparison
        mid_point = len(bars) // 2
        early_bars = bars[:mid_point]
        recent_bars = bars[mid_point:]
        
        if len(early_bars) < self.min_structure_bars or len(recent_bars) < self.min_structure_bars:
            return {'shift_detected': False}
        
        # Analyze early period structure
        early_structure = self._analyze_period_structure(early_bars)
        
        # Analyze recent period structure
        recent_structure = self._analyze_period_structure(recent_bars)
        
        # Check for structure shift
        shift_detected = False
        choch_type = CHoCHType.NONE
        
        if (early_structure['trend'] == 'bearish' and 
            recent_structure['trend'] == 'bullish' and
            recent_structure['strength'] > 0.6):
            shift_detected = True
            choch_type = CHoCHType.BULLISH
        
        elif (early_structure['trend'] == 'bullish' and 
              recent_structure['trend'] == 'bearish' and
              recent_structure['strength'] > 0.6):
            shift_detected = True
            choch_type = CHoCHType.BEARISH
        
        return {
            'shift_detected': shift_detected,
            'choch_type': choch_type,
            'previous_structure': early_structure['trend'],
            'new_structure': recent_structure['trend'],
            'early_structure': early_structure,
            'recent_structure': recent_structure
        }
    
    def _analyze_period_structure(self, bars: List) -> Dict:
        """Analyze structure for a specific period."""
        
        if len(bars) < 3:
            return {'trend': 'neutral', 'strength': 0.0}
        
        # Find swing points
        swing_highs = self._find_swing_highs(bars)
        swing_lows = self._find_swing_lows(bars)
        
        # Analyze trend direction
        trend = 'neutral'
        strength = 0.0
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for higher highs and higher lows (bullish)
            if (swing_highs[-1] > swing_highs[-2] and 
                swing_lows[-1] > swing_lows[-2]):
                trend = 'bullish'
                strength = 0.8
            # Check for lower highs and lower lows (bearish)
            elif (swing_highs[-1] < swing_highs[-2] and 
                  swing_lows[-1] < swing_lows[-2]):
                trend = 'bearish'
                strength = 0.8
            # Check for partial structure
            elif swing_highs[-1] > swing_highs[-2] or swing_lows[-1] > swing_lows[-2]:
                trend = 'bullish'
                strength = 0.4
            elif swing_highs[-1] < swing_highs[-2] or swing_lows[-1] < swing_lows[-2]:
                trend = 'bearish'
                strength = 0.4
        
        # Additional momentum confirmation
        closes = [bar.close for bar in bars]
        if len(closes) >= 3:
            price_change = (closes[-1] - closes[0]) / closes[0]
            if abs(price_change) > 0.02:  # 2% move
                if price_change > 0 and trend == 'bullish':
                    strength = min(1.0, strength + 0.2)
                elif price_change < 0 and trend == 'bearish':
                    strength = min(1.0, strength + 0.2)
        
        return {
            'trend': trend,
            'strength': strength,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'price_change': (closes[-1] - closes[0]) / closes[0] if len(closes) > 0 else 0
        }
    
    def _confirm_choch(self, bars: List, structure_shift: Dict) -> Dict:
        """Confirm CHoCH with additional criteria."""
        
        recent_bars = bars[-self.min_structure_bars:]
        
        # Volume confirmation
        volume_confirmation = self._check_volume_confirmation(recent_bars)
        
        # Momentum confirmation
        momentum_confirmation = self._check_momentum_confirmation(recent_bars, structure_shift)
        
        # Key level break confirmation
        key_level_break = self._check_key_level_break(bars, structure_shift)
        
        # Calculate overall confidence
        confidence = 0.0
        if volume_confirmation:
            confidence += 0.3
        if momentum_confirmation:
            confidence += 0.3
        if key_level_break:
            confidence += 0.4
        
        # Require at least 60% confidence
        confirmed = confidence >= 0.6
        
        # Calculate invalidation level
        invalidation_level = self._calculate_invalidation_level(bars, structure_shift)
        
        return {
            'confirmed': confirmed,
            'confidence': confidence,
            'volume_confirmation': volume_confirmation,
            'momentum_confirmation': momentum_confirmation,
            'key_level_break': key_level_break,
            'invalidation_level': invalidation_level
        }
    
    def _check_volume_confirmation(self, bars: List) -> bool:
        """Check if volume confirms the CHoCH."""
        
        if len(bars) < 3:
            return False
        
        # Calculate average volume
        volumes = [bar.volume for bar in bars]
        avg_volume = np.mean(volumes)
        
        # Check if recent volume is above threshold
        recent_volume = np.mean(volumes[-2:])  # Last 2 bars
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        return volume_ratio >= self.volume_threshold
    
    def _check_momentum_confirmation(self, bars: List, structure_shift: Dict) -> bool:
        """Check if momentum confirms the CHoCH."""
        
        if len(bars) < 3:
            return False
        
        closes = [bar.close for bar in bars]
        
        # Calculate momentum indicators
        roc = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
        
        # Check if momentum aligns with CHoCH type
        if structure_shift['choch_type'] == CHoCHType.BULLISH:
            return roc > 0.01  # At least 1% positive momentum
        elif structure_shift['choch_type'] == CHoCHType.BEARISH:
            return roc < -0.01  # At least 1% negative momentum
        
        return False
    
    def _check_key_level_break(self, bars: List, structure_shift: Dict) -> bool:
        """Check if CHoCH involves breaking a key level."""
        
        if len(bars) < 10:
            return False
        
        # Find recent swing points
        recent_high = max(bar.high for bar in bars[-10:])
        recent_low = min(bar.low for bar in bars[-10:])
        
        current_price = bars[-1].close
        
        # Check if current price breaks recent swing levels
        if structure_shift['choch_type'] == CHoCHType.BULLISH:
            return current_price > recent_high
        elif structure_shift['choch_type'] == CHoCHType.BEARISH:
            return current_price < recent_low
        
        return False
    
    def _calculate_invalidation_level(self, bars: List, structure_shift: Dict) -> float:
        """Calculate invalidation level for the CHoCH."""
        
        if len(bars) < 5:
            return bars[-1].close
        
        # Use recent swing points as invalidation levels
        recent_bars = bars[-5:]
        
        if structure_shift['choch_type'] == CHoCHType.BULLISH:
            # For bullish CHoCH, invalidation is below recent low
            return min(bar.low for bar in recent_bars)
        elif structure_shift['choch_type'] == CHoCHType.BEARISH:
            # For bearish CHoCH, invalidation is above recent high
            return max(bar.high for bar in recent_bars)
        
        return bars[-1].close
    
    def _find_swing_highs(self, bars: List, lookback: int = 3) -> List[float]:
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
    
    def _find_swing_lows(self, bars: List, lookback: int = 3) -> List[float]:
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
    
    def _store_choch_signal(self, choch_signal: CHoCHSignal):
        """Store recent CHoCH signal."""
        self.recent_choch_signals.append(choch_signal)
        
        # Keep only recent signals
        if len(self.recent_choch_signals) > self.max_recent_signals:
            self.recent_choch_signals = self.recent_choch_signals[-self.max_recent_signals:]
    
    def is_choch_valid(self, choch_signal: CHoCHSignal, current_price: float) -> bool:
        """Check if CHoCH signal is still valid (not invalidated)."""
        
        if choch_signal.choch_type == CHoCHType.BULLISH:
            return current_price > choch_signal.invalidation_level
        elif choch_signal.choch_type == CHoCHType.BEARISH:
            return current_price < choch_signal.invalidation_level
        
        return True
    
    def get_recent_choch(self, time_window_minutes: int = 60) -> Optional[CHoCHSignal]:
        """Get most recent CHoCH signal within time window."""
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=time_window_minutes)
        
        # Find most recent CHoCH within time window
        recent_signals = [
            signal for signal in self.recent_choch_signals
            if signal.confirmation_time >= cutoff_time
        ]
        
        if not recent_signals:
            return None
        
        # Return most recent
        return max(recent_signals, key=lambda s: s.confirmation_time)
    
    def enhance_trade_decision(self, trade_decision: TradeDecision, 
                             choch_signal: Optional[CHoCHSignal]) -> TradeDecision:
        """Enhance trade decision with CHoCH confirmation."""
        
        if not choch_signal:
            # No CHoCH signal - only allow fades on balanced days
            if (trade_decision.trade_type == TradeType.CONTINUATION and 
                trade_decision.bias_analysis.session_type != SessionType.TREND_DAY):
                # Downgrade continuation trades without CHoCH
                return TradeDecision(
                    trade_type=TradeType.NO_TRADE,
                    direction=None,
                    confidence=0.0,
                    reason="Continuation trade requires CHoCH confirmation",
                    bias_analysis=trade_decision.bias_analysis,
                    zone_type=trade_decision.zone_type,
                    is_first_touch=trade_decision.is_first_touch,
                    choch_aligned=False,
                    session_appropriate=trade_decision.session_appropriate
                )
            return trade_decision
        
        # Check if CHoCH is still valid
        if not self.is_choch_valid(choch_signal, trade_decision.bias_analysis.key_levels.get('recent_high', 0)):
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason="CHoCH signal invalidated",
                bias_analysis=trade_decision.bias_analysis,
                zone_type=trade_decision.zone_type,
                is_first_touch=trade_decision.is_first_touch,
                choch_aligned=False,
                session_appropriate=trade_decision.session_appropriate
            )
        
        # Enhance trade decision based on CHoCH alignment
        if trade_decision.trade_type == TradeType.CONTINUATION:
            # Continuation trades should align with CHoCH
            choch_aligned = self._check_choch_trade_alignment(choch_signal, trade_decision)
            
            if choch_aligned:
                # Boost confidence for aligned trades
                enhanced_confidence = min(0.95, trade_decision.confidence * 1.2)
                return TradeDecision(
                    trade_type=trade_decision.trade_type,
                    direction=trade_decision.direction,
                    confidence=enhanced_confidence,
                    reason=f"{trade_decision.reason} + CHoCH confirmed",
                    bias_analysis=trade_decision.bias_analysis,
                    zone_type=trade_decision.zone_type,
                    is_first_touch=trade_decision.is_first_touch,
                    choch_aligned=True,
                    session_appropriate=trade_decision.session_appropriate
                )
            else:
                # Reject misaligned continuation trades
                return TradeDecision(
                    trade_type=TradeType.NO_TRADE,
                    direction=None,
                    confidence=0.0,
                    reason="Continuation trade misaligned with CHoCH",
                    bias_analysis=trade_decision.bias_analysis,
                    zone_type=trade_decision.zone_type,
                    is_first_touch=trade_decision.is_first_touch,
                    choch_aligned=False,
                    session_appropriate=trade_decision.session_appropriate
                )
        
        # For fade trades, CHoCH can provide additional context but not required
        return trade_decision
    
    def _check_choch_trade_alignment(self, choch_signal: CHoCHSignal, 
                                   trade_decision: TradeDecision) -> bool:
        """Check if trade aligns with CHoCH signal."""
        
        if not trade_decision.direction:
            return False
        
        # Check alignment based on CHoCH type and trade direction
        if (choch_signal.choch_type == CHoCHType.BULLISH and 
            trade_decision.direction == TradeDirection.LONG):
            return True
        elif (choch_signal.choch_type == CHoCHType.BEARISH and 
              trade_decision.direction == TradeDirection.SHORT):
            return True
        
        return False


def test_choch_confirmation_system():
    """Test the CHoCH confirmation system."""
    
    choch_system = CHoCHConfirmationSystem()
    
    # Create sample bars with structure shift
    bars = []
    base_price = 100.0
    
    # Create bearish structure first
    for i in range(20):
        price = base_price - i * 0.2 + np.random.normal(0, 0.3)
        bar = type('Bar', (), {
            'open': price - 0.1,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1000 + np.random.randint(-100, 100),
            'timestamp': datetime.now() + timedelta(minutes=i)
        })()
        bars.append(bar)
    
    # Create bullish structure shift
    for i in range(20, 40):
        price = base_price - 20 * 0.2 + (i - 20) * 0.3 + np.random.normal(0, 0.3)
        bar = type('Bar', (), {
            'open': price - 0.1,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1500 + np.random.randint(-100, 200),  # Higher volume
            'timestamp': datetime.now() + timedelta(minutes=i)
        })()
        bars.append(bar)
    
    # Test CHoCH detection
    choch_signal = choch_system.detect_choch(bars, len(bars) - 1)
    
    print("CHoCH Confirmation System Test:")
    print("=" * 40)
    
    if choch_signal:
        print(f"CHoCH Detected: {choch_signal.choch_type.value}")
        print(f"Confidence: {choch_signal.confidence:.2f}")
        print(f"Previous Structure: {choch_signal.previous_structure}")
        print(f"New Structure: {choch_signal.new_structure}")
        print(f"Volume Confirmation: {choch_signal.volume_confirmation}")
        print(f"Momentum Confirmation: {choch_signal.momentum_confirmation}")
        print(f"Key Level Break: {choch_signal.key_level_break}")
        print(f"Invalidation Level: {choch_signal.invalidation_level:.2f}")
    else:
        print("No CHoCH detected")


if __name__ == "__main__":
    test_choch_confirmation_system()