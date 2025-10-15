#!/usr/bin/env python3
"""
Session-Aware Trading Rules

This module implements the session-aware trading rules from the original
Zone-Based Intraday Trading Framework, integrating with directional bias detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from directional_bias_detector import DirectionalBias, SessionType, BiasAnalysis, DirectionalBiasDetector


class TradeType(Enum):
    """Types of trades based on framework rules."""
    FADE = "fade"  # Fade the zone (mean reversion)
    CONTINUATION = "continuation"  # Continue in direction of bias
    NO_TRADE = "no_trade"  # No trade allowed


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradeDecision:
    """Result of trade decision analysis."""
    trade_type: TradeType
    direction: Optional[TradeDirection]
    confidence: float
    reason: str
    bias_analysis: BiasAnalysis
    zone_type: str
    is_first_touch: bool
    choch_aligned: bool
    session_appropriate: bool


class SessionAwareTradingRules:
    """Implements session-aware trading rules from the framework."""
    
    def __init__(self, bias_detector: DirectionalBiasDetector):
        """
        Initialize session-aware trading rules.
        
        Args:
            bias_detector: Directional bias detector instance
        """
        self.bias_detector = bias_detector
        
        # Zone type mappings
        self.resistance_zones = [
            'prior_day_high', 'weekly_high', 'value_area_high',
            'monthly_high', 'quarterly_high'
        ]
        self.support_zones = [
            'prior_day_low', 'weekly_low', 'value_area_low',
            'monthly_low', 'quarterly_low'
        ]
    
    def analyze_trade_opportunity(self, 
                                bars: List, 
                                current_index: int,
                                zone_type: str,
                                zone_level: float,
                                current_price: float) -> TradeDecision:
        """
        Analyze a trade opportunity using session-aware rules.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            zone_type: Type of zone being touched
            zone_level: Price level of the zone
            current_price: Current price touching the zone
            
        Returns:
            TradeDecision with trade type, direction, and reasoning
        """
        # Get current bias analysis
        bias_analysis = self.bias_detector.detect_directional_bias(bars, current_index)
        
        # Check if this is first touch of zone
        zone_id = f"{zone_type}_{zone_level:.2f}"
        is_first_touch = self.bias_detector.is_first_touch(
            zone_id, bars[current_index].timestamp
        )
        
        # Determine trade decision based on framework rules
        trade_decision = self._determine_trade_type(
            bias_analysis, zone_type, current_price, zone_level, is_first_touch
        )
        
        return trade_decision
    
    def _determine_trade_type(self, 
                            bias_analysis: BiasAnalysis,
                            zone_type: str,
                            current_price: float,
                            zone_level: float,
                            is_first_touch: bool) -> TradeDecision:
        """Determine trade type based on framework rules."""
        
        # Rule 1: First Touch Rule
        if not is_first_touch:
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason="Not first touch of zone - framework requires first touch only",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=is_first_touch,
                choch_aligned=False,
                session_appropriate=False
            )
        
        # Rule 2: Session Type Awareness
        session_appropriate, session_reason = self._check_session_appropriateness(
            bias_analysis, zone_type
        )
        
        if not session_appropriate:
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason=f"Session inappropriate: {session_reason}",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=is_first_touch,
                choch_aligned=False,
                session_appropriate=False
            )
        
        # Rule 3: Directional Bias Alignment
        if bias_analysis.bias == DirectionalBias.NEUTRAL:
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason="Neutral bias - framework requires clear directional bias",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=is_first_touch,
                choch_aligned=False,
                session_appropriate=True
            )
        
        # Rule 4: CHoCH Confirmation for Continuation Trades
        choch_aligned = self._check_choch_alignment(bias_analysis, zone_type)
        
        # Rule 5: Determine Trade Type and Direction
        if bias_analysis.session_type == SessionType.TREND_DAY:
            # Trend Day: Only continuation trades in direction of bias
            return self._handle_trend_day_trade(
                bias_analysis, zone_type, current_price, zone_level, choch_aligned
            )
        else:
            # Balanced/Choppy Day: Fading is valid
            return self._handle_balanced_day_trade(
                bias_analysis, zone_type, current_price, zone_level, choch_aligned
            )
    
    def _check_session_appropriateness(self, 
                                     bias_analysis: BiasAnalysis,
                                     zone_type: str) -> Tuple[bool, str]:
        """Check if trade is appropriate for current session type."""
        
        if bias_analysis.session_type == SessionType.TREND_DAY:
            # On trend days, only take continuation trades
            if bias_analysis.bias == DirectionalBias.BULLISH and zone_type in self.support_zones:
                return True, "Trend day - continuation trade at support"
            elif bias_analysis.bias == DirectionalBias.BEARISH and zone_type in self.resistance_zones:
                return True, "Trend day - continuation trade at resistance"
            else:
                return False, "Trend day - no fading allowed"
        
        elif bias_analysis.session_type == SessionType.BALANCED_DAY:
            # On balanced days, both fades and continuations are valid
            return True, "Balanced day - both fades and continuations valid"
        
        else:  # CHOPPY_DAY
            # On choppy days, be more selective
            if bias_analysis.confidence > 0.7:
                return True, "Choppy day - high confidence trade"
            else:
                return False, "Choppy day - low confidence, avoid trade"
    
    def _check_choch_alignment(self, 
                             bias_analysis: BiasAnalysis,
                             zone_type: str) -> bool:
        """Check if CHoCH aligns with trade direction."""
        
        if not bias_analysis.choch_confirmed:
            return False
        
        # CHoCH should align with bias direction
        if (bias_analysis.bias == DirectionalBias.BULLISH and 
            bias_analysis.choch_confirmed):
            return True
        elif (bias_analysis.bias == DirectionalBias.BEARISH and 
              bias_analysis.choch_confirmed):
            return True
        
        return False
    
    def _handle_trend_day_trade(self, 
                              bias_analysis: BiasAnalysis,
                              zone_type: str,
                              current_price: float,
                              zone_level: float,
                              choch_aligned: bool) -> TradeDecision:
        """Handle trade decision on trend days."""
        
        # Trend days: Only continuation trades
        if bias_analysis.bias == DirectionalBias.BULLISH and zone_type in self.support_zones:
            # Bullish bias + support zone = continuation long
            return TradeDecision(
                trade_type=TradeType.CONTINUATION,
                direction=TradeDirection.LONG,
                confidence=bias_analysis.confidence * 0.9,  # Slightly lower for trend days
                reason="Trend day - continuation long at support",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
        
        elif bias_analysis.bias == DirectionalBias.BEARISH and zone_type in self.resistance_zones:
            # Bearish bias + resistance zone = continuation short
            return TradeDecision(
                trade_type=TradeType.CONTINUATION,
                direction=TradeDirection.SHORT,
                confidence=bias_analysis.confidence * 0.9,
                reason="Trend day - continuation short at resistance",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
        
        else:
            # Wrong zone type for bias
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason="Trend day - wrong zone type for bias direction",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
    
    def _handle_balanced_day_trade(self, 
                                 bias_analysis: BiasAnalysis,
                                 zone_type: str,
                                 current_price: float,
                                 zone_level: float,
                                 choch_aligned: bool) -> TradeDecision:
        """Handle trade decision on balanced days."""
        
        # Balanced days: Both fades and continuations are valid
        
        # Check if we should fade (opposite to bias)
        if bias_analysis.bias == DirectionalBias.BULLISH and zone_type in self.resistance_zones:
            # Bullish bias + resistance zone = fade short
            return TradeDecision(
                trade_type=TradeType.FADE,
                direction=TradeDirection.SHORT,
                confidence=bias_analysis.confidence * 0.7,  # Lower confidence for fades
                reason="Balanced day - fade short at resistance (bias bullish)",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
        
        elif bias_analysis.bias == DirectionalBias.BEARISH and zone_type in self.support_zones:
            # Bearish bias + support zone = fade long
            return TradeDecision(
                trade_type=TradeType.FADE,
                direction=TradeDirection.LONG,
                confidence=bias_analysis.confidence * 0.7,
                reason="Balanced day - fade long at support (bias bearish)",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
        
        # Check if we should continue (same as bias)
        elif bias_analysis.bias == DirectionalBias.BULLISH and zone_type in self.support_zones:
            # Bullish bias + support zone = continuation long
            return TradeDecision(
                trade_type=TradeType.CONTINUATION,
                direction=TradeDirection.LONG,
                confidence=bias_analysis.confidence * 0.8,
                reason="Balanced day - continuation long at support",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
        
        elif bias_analysis.bias == DirectionalBias.BEARISH and zone_type in self.resistance_zones:
            # Bearish bias + resistance zone = continuation short
            return TradeDecision(
                trade_type=TradeType.CONTINUATION,
                direction=TradeDirection.SHORT,
                confidence=bias_analysis.confidence * 0.8,
                reason="Balanced day - continuation short at resistance",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
        
        else:
            # No valid trade setup
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason="Balanced day - no valid trade setup",
                bias_analysis=bias_analysis,
                zone_type=zone_type,
                is_first_touch=True,
                choch_aligned=choch_aligned,
                session_appropriate=True
            )
    
    def should_take_trade(self, trade_decision: TradeDecision, 
                        min_confidence: float = 0.6) -> bool:
        """Determine if trade should be taken based on decision and confidence."""
        
        if trade_decision.trade_type == TradeType.NO_TRADE:
            return False
        
        if trade_decision.confidence < min_confidence:
            return False
        
        # Additional filters
        if not trade_decision.session_appropriate:
            return False
        
        if trade_decision.trade_type == TradeType.CONTINUATION and not trade_decision.choch_aligned:
            return False  # Continuation trades require CHoCH confirmation
        
        return True
    
    def get_trade_summary(self, trade_decision: TradeDecision) -> str:
        """Get a summary of the trade decision."""
        
        if trade_decision.trade_type == TradeType.NO_TRADE:
            return f"NO TRADE: {trade_decision.reason}"
        
        direction_str = trade_decision.direction.value if trade_decision.direction else "N/A"
        trade_type_str = trade_decision.trade_type.value.upper()
        
        return (f"{trade_type_str} {direction_str} | "
                f"Confidence: {trade_decision.confidence:.2f} | "
                f"Bias: {trade_decision.bias_analysis.bias.value} | "
                f"Session: {trade_decision.bias_analysis.session_type.value} | "
                f"CHoCH: {trade_decision.choch_aligned} | "
                f"Reason: {trade_decision.reason}")


def test_session_aware_trading():
    """Test the session-aware trading rules."""
    
    # Create bias detector
    bias_detector = DirectionalBiasDetector()
    
    # Create trading rules
    trading_rules = SessionAwareTradingRules(bias_detector)
    
    # Create sample bars for testing
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
    
    # Test different scenarios
    test_cases = [
        ("prior_day_high", 105.0, 104.8),  # Resistance zone
        ("prior_day_low", 95.0, 95.2),     # Support zone
        ("weekly_high", 110.0, 109.8),     # Resistance zone
        ("weekly_low", 90.0, 90.2),        # Support zone
    ]
    
    print("Testing Session-Aware Trading Rules:")
    print("=" * 60)
    
    for zone_type, zone_level, current_price in test_cases:
        decision = trading_rules.analyze_trade_opportunity(
            bars, len(bars) - 1, zone_type, zone_level, current_price
        )
        
        print(f"\nZone: {zone_type} at {zone_level}")
        print(f"Current Price: {current_price}")
        print(f"Decision: {trading_rules.get_trade_summary(decision)}")
        print(f"Should Take Trade: {trading_rules.should_take_trade(decision)}")


if __name__ == "__main__":
    test_session_aware_trading()