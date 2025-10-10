"""
Quality Rating System (QRS) scorer for Zone Fade setups.

This module provides the QRS scoring system that evaluates Zone Fade setups
based on 5 key factors: zone quality, rejection clarity, structure flip,
context, and intermarket divergence.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np

from zone_fade_detector.core.models import (
    ZoneFadeSetup, 
    QRSFactors, 
    OHLCVBar, 
    Zone, 
    ZoneType,
    SetupDirection,
    VWAPData,
    MarketContext
)
from zone_fade_detector.indicators.vwap import VWAPCalculator
from zone_fade_detector.indicators.swing_structure import SwingStructureDetector
from zone_fade_detector.indicators.volume_analysis import VolumeAnalyzer


class QRSScorer:
    """
    Quality Rating System scorer for Zone Fade setups.
    
    Evaluates setups based on 5 factors:
    1. Zone Quality (0-2 points)
    2. Rejection Clarity (0-2 points)
    3. Structure Flip (0-2 points)
    4. Context (0-2 points)
    5. Intermarket Divergence (0-2 points)
    """
    
    def __init__(
        self,
        a_setup_threshold: int = 7,
        zone_quality_weights: Dict[str, float] = None,
        rejection_weights: Dict[str, float] = None,
        context_weights: Dict[str, float] = None
    ):
        """
        Initialize QRS scorer.
        
        Args:
            a_setup_threshold: Minimum score for A-Setup (default: 7)
            zone_quality_weights: Weights for zone quality factors
            rejection_weights: Weights for rejection clarity factors
            context_weights: Weights for context factors
        """
        self.a_setup_threshold = a_setup_threshold
        self.logger = logging.getLogger(__name__)
        
        # Default weights
        self.zone_quality_weights = zone_quality_weights or {
            'htf_relevance': 2.0,
            'zone_strength': 1.0
        }
        
        self.rejection_weights = rejection_weights or {
            'pin_bar': 2.0,
            'engulfing': 1.5,
            'long_wick': 1.0
        }
        
        self.context_weights = context_weights or {
            'vwap_slope': 1.0,
            'value_area_overlap': 1.0,
            'trend_day_penalty': -1.0
        }
        
        # Initialize indicator calculators
        self.vwap_calculator = VWAPCalculator()
        self.swing_detector = SwingStructureDetector()
        self.volume_analyzer = VolumeAnalyzer()
    
    def score_setup(
        self,
        setup: ZoneFadeSetup,
        market_context: Optional[MarketContext] = None,
        intermarket_data: Optional[Dict[str, Any]] = None
    ) -> QRSFactors:
        """
        Score a Zone Fade setup using QRS.
        
        Args:
            setup: ZoneFadeSetup object to score
            market_context: Market context data
            intermarket_data: Intermarket analysis data
            
        Returns:
            QRSFactors object with detailed scoring
        """
        # Score each factor
        zone_quality = self._score_zone_quality(setup.zone)
        rejection_clarity = self._score_rejection_clarity(setup.rejection_candle)
        structure_flip = self._score_structure_flip(setup.choch_confirmed, setup.direction)
        context = self._score_context(setup, market_context)
        intermarket_divergence = self._score_intermarket_divergence(
            setup.symbol, 
            intermarket_data
        )
        
        return QRSFactors(
            zone_quality=zone_quality,
            rejection_clarity=rejection_clarity,
            structure_flip=structure_flip,
            context=context,
            intermarket_divergence=intermarket_divergence
        )
    
    def _score_zone_quality(self, zone: Zone) -> int:
        """
        Score zone quality (0-2 points).
        
        Args:
            zone: Zone object to score
            
        Returns:
            Zone quality score (0-2)
        """
        score = 0
        
        # HTF relevance (higher timeframe zones get more points)
        if zone.zone_type in [ZoneType.WEEKLY_HIGH, ZoneType.WEEKLY_LOW]:
            score += 2
        elif zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.PRIOR_DAY_LOW]:
            score += 1
        elif zone.zone_type in [ZoneType.VALUE_AREA_HIGH, ZoneType.VALUE_AREA_LOW]:
            score += 1
        else:
            score += 0
        
        # Zone strength and quality
        if zone.quality >= 2 and zone.strength >= 2.0:
            score += 1
        elif zone.quality >= 1 and zone.strength >= 1.5:
            score += 0.5
        
        # Cap at 2 points
        return min(int(score), 2)
    
    def _score_rejection_clarity(self, rejection_candle: OHLCVBar) -> int:
        """
        Score rejection candle clarity (0-2 points).
        
        Args:
            rejection_candle: OHLCVBar object to analyze
            
        Returns:
            Rejection clarity score (0-2)
        """
        score = 0
        
        # Analyze candle characteristics
        body_size = rejection_candle.body_size
        upper_wick = rejection_candle.upper_wick
        lower_wick = rejection_candle.lower_wick
        total_range = rejection_candle.total_range
        
        if total_range == 0:
            return 0
        
        # Pin bar detection
        if upper_wick / total_range >= 0.6 or lower_wick / total_range >= 0.6:
            score += 2
        elif upper_wick / total_range >= 0.4 or lower_wick / total_range >= 0.4:
            score += 1
        
        # Engulfing pattern detection
        # This would need previous candle for full analysis
        # For now, score based on body size relative to range
        if body_size / total_range >= 0.8:
            score += 1
        
        # Long wick detection
        if upper_wick / total_range >= 0.3 or lower_wick / total_range >= 0.3:
            score += 0.5
        
        # Cap at 2 points
        return min(int(score), 2)
    
    def _score_structure_flip(
        self, 
        choch_confirmed: bool, 
        direction: SetupDirection
    ) -> int:
        """
        Score structure flip (CHoCH) (0-2 points).
        
        Args:
            choch_confirmed: Whether CHoCH is confirmed
            direction: Setup direction
            
        Returns:
            Structure flip score (0-2)
        """
        if choch_confirmed:
            return 2
        else:
            return 0
    
    def _score_context(
        self, 
        setup: ZoneFadeSetup, 
        market_context: Optional[MarketContext]
    ) -> int:
        """
        Score market context (0-2 points).
        
        Args:
            setup: ZoneFadeSetup object
            market_context: Market context data
            
        Returns:
            Context score (0-2)
        """
        score = 0
        
        if market_context is None:
            # Use setup data to infer context
            if setup.vwap_data:
                if setup.vwap_data.is_flat:
                    score += 1
                elif abs(setup.vwap_data.slope) < 0.001:
                    score += 1
        else:
            # Use provided market context
            if market_context.is_balanced:
                score += 2
            elif not market_context.is_trend_day and market_context.value_area_overlap:
                score += 1
        
        # VWAP slope analysis
        if setup.vwap_data:
            if setup.vwap_data.is_flat:
                score += 1
            elif abs(setup.vwap_data.slope) < 0.001:
                score += 0.5
        
        # Value area overlap (simplified check)
        if setup.opening_range and setup.vwap_data:
            or_mid = setup.opening_range.mid_point
            vwap = setup.vwap_data.vwap
            if abs(or_mid - vwap) / vwap < 0.01:  # Within 1%
                score += 0.5
        
        # Cap at 2 points
        return min(int(score), 2)
    
    def _score_intermarket_divergence(
        self, 
        symbol: str, 
        intermarket_data: Optional[Dict[str, Any]]
    ) -> int:
        """
        Score intermarket divergence (0-2 points).
        
        Args:
            symbol: Current symbol
            intermarket_data: Intermarket analysis data
            
        Returns:
            Intermarket divergence score (0-2)
        """
        if not intermarket_data:
            return 0
        
        score = 0
        
        # Check for divergence between ETFs
        etf_symbols = ['SPY', 'QQQ', 'IWM']
        if symbol in etf_symbols:
            other_symbols = [s for s in etf_symbols if s != symbol]
            
            # Check if current symbol is diverging from others
            if 'price_changes' in intermarket_data:
                price_changes = intermarket_data['price_changes']
                current_change = price_changes.get(symbol, 0)
                
                # Count diverging symbols
                diverging_count = 0
                for other_symbol in other_symbols:
                    other_change = price_changes.get(other_symbol, 0)
                    if (current_change > 0 and other_change < 0) or \
                       (current_change < 0 and other_change > 0):
                        diverging_count += 1
                
                if diverging_count >= 1:
                    score += 2
                elif diverging_count >= 0.5:  # Partial divergence
                    score += 1
        
        # Check sector rotation
        if 'sector_rotation' in intermarket_data:
            sector_rotation = intermarket_data['sector_rotation']
            if sector_rotation.get('is_rotating', False):
                score += 1
        
        # Cap at 2 points
        return min(int(score), 2)
    
    def analyze_setup_quality(
        self,
        setup: ZoneFadeSetup,
        market_context: Optional[MarketContext] = None,
        intermarket_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Provide detailed analysis of setup quality.
        
        Args:
            setup: ZoneFadeSetup object to analyze
            market_context: Market context data
            intermarket_data: Intermarket analysis data
            
        Returns:
            Dictionary with detailed quality analysis
        """
        qrs_factors = self.score_setup(setup, market_context, intermarket_data)
        
        return {
            'qrs_factors': qrs_factors,
            'total_score': qrs_factors.total_score,
            'is_a_setup': qrs_factors.is_a_setup,
            'zone_quality_breakdown': self._get_zone_quality_breakdown(setup.zone),
            'rejection_breakdown': self._get_rejection_breakdown(setup.rejection_candle),
            'context_breakdown': self._get_context_breakdown(setup, market_context),
            'recommendations': self._get_recommendations(qrs_factors)
        }
    
    def _get_zone_quality_breakdown(self, zone: Zone) -> Dict[str, Any]:
        """Get detailed zone quality breakdown."""
        return {
            'zone_type': zone.zone_type.value,
            'quality': zone.quality,
            'strength': zone.strength,
            'touches': zone.touches,
            'htf_relevance': zone.zone_type in [ZoneType.WEEKLY_HIGH, ZoneType.WEEKLY_LOW]
        }
    
    def _get_rejection_breakdown(self, rejection_candle: OHLCVBar) -> Dict[str, Any]:
        """Get detailed rejection candle breakdown."""
        total_range = rejection_candle.total_range
        
        return {
            'body_size': rejection_candle.body_size,
            'upper_wick': rejection_candle.upper_wick,
            'lower_wick': rejection_candle.lower_wick,
            'total_range': total_range,
            'body_ratio': rejection_candle.body_size / total_range if total_range > 0 else 0,
            'upper_wick_ratio': rejection_candle.upper_wick / total_range if total_range > 0 else 0,
            'lower_wick_ratio': rejection_candle.lower_wick / total_range if total_range > 0 else 0,
            'is_pin_bar': max(rejection_candle.upper_wick, rejection_candle.lower_wick) / total_range >= 0.6 if total_range > 0 else False
        }
    
    def _get_context_breakdown(
        self, 
        setup: ZoneFadeSetup, 
        market_context: Optional[MarketContext]
    ) -> Dict[str, Any]:
        """Get detailed context breakdown."""
        breakdown = {
            'is_trend_day': False,
            'vwap_slope': 0.0,
            'is_balanced': False,
            'value_area_overlap': False
        }
        
        if market_context:
            breakdown.update({
                'is_trend_day': market_context.is_trend_day,
                'vwap_slope': market_context.vwap_slope,
                'is_balanced': market_context.is_balanced,
                'value_area_overlap': market_context.value_area_overlap
            })
        
        if setup.vwap_data:
            breakdown['vwap_slope'] = setup.vwap_data.slope
            breakdown['is_flat'] = setup.vwap_data.is_flat
        
        return breakdown
    
    def _get_recommendations(self, qrs_factors: QRSFactors) -> List[str]:
        """Get recommendations based on QRS factors."""
        recommendations = []
        
        if qrs_factors.zone_quality < 2:
            recommendations.append("Consider waiting for higher quality zones")
        
        if qrs_factors.rejection_clarity < 2:
            recommendations.append("Look for clearer rejection patterns")
        
        if qrs_factors.structure_flip < 2:
            recommendations.append("Wait for CHoCH confirmation")
        
        if qrs_factors.context < 2:
            recommendations.append("Ensure market context is favorable")
        
        if qrs_factors.intermarket_divergence < 1:
            recommendations.append("Check for intermarket divergence signals")
        
        if qrs_factors.total_score >= self.a_setup_threshold:
            recommendations.append("A-Setup confirmed - consider entry")
        else:
            recommendations.append(f"Score {qrs_factors.total_score}/10 - wait for better setup")
        
        return recommendations