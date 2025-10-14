"""
Enhanced Quality Rating System (QRS)

This module implements the enhanced 5-factor QRS system with veto power
for Factor 3 (Market Type & Internals). The system provides detailed
scoring and filtering for zone fade setups.

Features:
- 5-factor QRS system (0-10 point scale)
- Factor 3 veto power (Market Type & Internals)
- Enhanced Factor 4 (Structure & Touch)
- Detailed QRS breakdown in alerts
- Configurable threshold (default: 7.0)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class QRSGrade(Enum):
    """QRS grade classifications."""
    A_PLUS = "A+"  # 9.0-10.0
    A = "A"        # 8.0-8.9
    B_PLUS = "B+"  # 7.0-7.9
    B = "B"        # 6.0-6.9
    C = "C"        # 5.0-5.9
    D = "D"        # 4.0-4.9
    F = "F"        # 0.0-3.9

@dataclass
class QRSFactor:
    """Individual QRS factor result."""
    name: str
    score: float
    max_score: float
    weight: float
    details: Dict[str, Any]
    reasoning: str

@dataclass
class QRSResult:
    """Complete QRS scoring result."""
    total_score: float
    grade: QRSGrade
    factors: Dict[str, QRSFactor]
    veto: bool
    veto_reason: Optional[str]
    timestamp: datetime
    setup_id: str

class EnhancedQRSScorer:
    """
    Enhanced Quality Rating System with 5 factors and veto power.
    
    This system provides comprehensive scoring for zone fade setups
    with critical veto functionality for unfavorable market conditions.
    """
    
    def __init__(self, 
                 threshold: float = 7.0,
                 factor_weights: Optional[Dict[str, float]] = None):
        """
        Initialize enhanced QRS scorer.
        
        Args:
            threshold: Minimum QRS score for signal generation
            factor_weights: Custom factor weights (default: equal weights)
        """
        self.threshold = threshold
        
        # Default factor weights (equal weight)
        self.factor_weights = factor_weights or {
            'zone_quality': 0.2,      # Factor 1
            'rejection_volume': 0.2,   # Factor 2
            'market_internals': 0.2,   # Factor 3 (VETO)
            'structure_touch': 0.2,    # Factor 4
            'context_intermarket': 0.2 # Factor 5
        }
        
        # Statistics tracking
        self.total_scored = 0
        self.signals_generated = 0
        self.signals_vetoed = 0
        self.avg_score = 0.0
    
    def score_setup(self, 
                   setup: Dict[str, Any],
                   market_data: Dict[str, Any]) -> Optional[QRSResult]:
        """
        Score a zone fade setup using enhanced QRS.
        
        Args:
            setup: Zone fade setup data
            market_data: Market data including internals, zones, etc.
            
        Returns:
            QRSResult if setup passes, None if vetoed
        """
        setup_id = setup.get('setup_id', f'setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Score each factor
        factors = {}
        
        # Factor 1: Zone Quality (0-2 points)
        factors['zone_quality'] = self._score_zone_quality(setup, market_data)
        
        # Factor 2: Rejection + Volume (0-2 points)
        factors['rejection_volume'] = self._score_rejection_volume(setup, market_data)
        
        # Factor 3: Market Type & Internals (0-2 points) - VETO FACTOR
        factors['market_internals'] = self._score_market_internals(setup, market_data)
        
        # Check for veto condition
        if factors['market_internals'].score == 0.0:
            self.signals_vetoed += 1
            return QRSResult(
                total_score=0.0,
                grade=QRSGrade.F,
                factors=factors,
                veto=True,
                veto_reason=factors['market_internals'].reasoning,
                timestamp=datetime.now(),
                setup_id=setup_id
            )
        
        # Factor 4: Structure & Touch (0-2 points)
        factors['structure_touch'] = self._score_structure_touch(setup, market_data)
        
        # Factor 5: Context & Intermarket (0-2 points)
        factors['context_intermarket'] = self._score_context_intermarket(setup, market_data)
        
        # Calculate total score
        total_score = sum(factor.score * self.factor_weights[factor.name] 
                         for factor in factors.values())
        
        # Determine grade
        grade = self._determine_grade(total_score)
        
        # Check if meets threshold
        meets_threshold = total_score >= self.threshold
        
        # Update statistics
        self.total_scored += 1
        if meets_threshold:
            self.signals_generated += 1
        else:
            self.signals_vetoed += 1
        
        # Update average score
        self.avg_score = ((self.avg_score * (self.total_scored - 1)) + total_score) / self.total_scored
        
        return QRSResult(
            total_score=total_score,
            grade=grade,
            factors=factors,
            veto=not meets_threshold,
            veto_reason=f"Score {total_score:.1f} below threshold {self.threshold}" if not meets_threshold else None,
            timestamp=datetime.now(),
            setup_id=setup_id
        )
    
    def _score_zone_quality(self, setup: Dict[str, Any], market_data: Dict[str, Any]) -> QRSFactor:
        """Score Factor 1: Zone Quality (0-2 points)."""
        zone_type = setup.get('zone_type', 'unknown')
        zone_timeframe = setup.get('zone_timeframe', 'M5')
        zone_strength = setup.get('zone_strength', 1.0)
        prior_touches = setup.get('prior_touches', 0)
        confluence_factors = setup.get('confluence_factors', [])
        
        score = 0.0
        details = {}
        reasoning_parts = []
        
        # Timeframe scoring
        if zone_timeframe in ['Daily', 'Weekly']:
            score += 1.0
            details['timeframe'] = 'HTF'
            reasoning_parts.append('HTF zone')
        elif zone_timeframe in ['H4', 'H1']:
            score += 0.5
            details['timeframe'] = 'MTF'
            reasoning_parts.append('MTF zone')
        else:
            details['timeframe'] = 'LTF'
            reasoning_parts.append('LTF zone')
        
        # Zone type scoring
        if zone_type in ['BCZ', 'Balance_Core_Zone']:
            score += 1.0
            details['zone_type'] = 'BCZ'
            reasoning_parts.append('BCZ zone')
        elif zone_type in ['BchZ', 'Balance_Challenge_Zone']:
            score += 0.5
            details['zone_type'] = 'BchZ'
            reasoning_parts.append('BchZ zone')
        else:
            details['zone_type'] = 'Standard'
            reasoning_parts.append('Standard zone')
        
        # Confluence factors
        confluence_bonus = min(0.5, len(confluence_factors) * 0.1)
        score += confluence_bonus
        details['confluence_factors'] = len(confluence_factors)
        if confluence_factors:
            reasoning_parts.append(f'{len(confluence_factors)} confluence factors')
        
        # Cap at 2.0
        score = min(score, 2.0)
        
        return QRSFactor(
            name='zone_quality',
            score=score,
            max_score=2.0,
            weight=self.factor_weights['zone_quality'],
            details=details,
            reasoning=' + '.join(reasoning_parts) if reasoning_parts else 'No zone quality factors'
        )
    
    def _score_rejection_volume(self, setup: Dict[str, Any], market_data: Dict[str, Any]) -> QRSFactor:
        """Score Factor 2: Rejection + Volume (0-2 points)."""
        rejection_bar = setup.get('rejection_bar', {})
        volume_data = market_data.get('volume_data', {})
        
        # Calculate wick ratio
        if all(key in rejection_bar for key in ['open', 'high', 'low', 'close']):
            total_range = rejection_bar['high'] - rejection_bar['low']
            wick_length = max(
                rejection_bar['high'] - rejection_bar['close'],
                rejection_bar['close'] - rejection_bar['low']
            )
            wick_ratio = wick_length / total_range if total_range > 0 else 0
        else:
            wick_ratio = 0
        
        # Calculate volume spike
        current_volume = rejection_bar.get('volume', 0)
        avg_volume = volume_data.get('avg_volume', current_volume)
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Score based on both factors
        score = 0.0
        details = {
            'wick_ratio': wick_ratio,
            'volume_spike': volume_spike
        }
        reasoning_parts = []
        
        # Wick quality
        if wick_ratio >= 0.3:
            score += 1.0
            reasoning_parts.append(f'Clean rejection ({wick_ratio:.1%} wick)')
        elif wick_ratio >= 0.2:
            score += 0.5
            reasoning_parts.append(f'Good rejection ({wick_ratio:.1%} wick)')
        else:
            reasoning_parts.append(f'Weak rejection ({wick_ratio:.1%} wick)')
        
        # Volume quality
        if volume_spike >= 1.8:
            score += 1.0
            reasoning_parts.append(f'Strong volume ({volume_spike:.1f}x)')
        elif volume_spike >= 1.5:
            score += 0.5
            reasoning_parts.append(f'Good volume ({volume_spike:.1f}x)')
        else:
            reasoning_parts.append(f'Weak volume ({volume_spike:.1f}x)')
        
        return QRSFactor(
            name='rejection_volume',
            score=score,
            max_score=2.0,
            weight=self.factor_weights['rejection_volume'],
            details=details,
            reasoning=' + '.join(reasoning_parts)
        )
    
    def _score_market_internals(self, setup: Dict[str, Any], market_data: Dict[str, Any]) -> QRSFactor:
        """Score Factor 3: Market Type & Internals (0-2 points) - VETO FACTOR."""
        market_type = market_data.get('market_type', 'UNKNOWN')
        internals_favorable = market_data.get('internals_favorable', False)
        internals_quality_score = market_data.get('internals_quality_score', 0.0)
        
        # This factor has veto power
        if market_type == 'TREND_DAY' or not internals_favorable:
            return QRSFactor(
                name='market_internals',
                score=0.0,
                max_score=2.0,
                weight=self.factor_weights['market_internals'],
                details={
                    'market_type': market_type,
                    'internals_favorable': internals_favorable,
                    'quality_score': internals_quality_score
                },
                reasoning='Trend day or initiative activity - VETO'
            )
        
        # Score based on internals quality
        score = internals_quality_score  # Should be 1.0 or 2.0
        
        details = {
            'market_type': market_type,
            'internals_favorable': internals_favorable,
            'quality_score': internals_quality_score
        }
        
        if score == 2.0:
            reasoning = 'Perfect internals - balanced TICK + flat A/D'
        elif score == 1.0:
            reasoning = 'Mixed internals - one factor favorable'
        else:
            reasoning = 'Poor internals'
        
        return QRSFactor(
            name='market_internals',
            score=score,
            max_score=2.0,
            weight=self.factor_weights['market_internals'],
            details=details,
            reasoning=reasoning
        )
    
    def _score_structure_touch(self, setup: Dict[str, Any], market_data: Dict[str, Any]) -> QRSFactor:
        """Score Factor 4: Structure & Touch (0-2 points) - ENHANCED."""
        choch_confirmed = setup.get('choch_confirmed', False)
        touch_number = setup.get('touch_number', 1)
        has_balance_before = setup.get('has_balance_before', False)
        zone_position = setup.get('zone_position', 'middle')
        setup_type = setup.get('setup_type', 'ZFR')
        
        score = 0.0
        details = {
            'choch_confirmed': choch_confirmed,
            'touch_number': touch_number,
            'has_balance_before': has_balance_before,
            'zone_position': zone_position,
            'setup_type': setup_type
        }
        reasoning_parts = []
        
        # CHoCH confirmed
        if choch_confirmed:
            score += 0.5
            reasoning_parts.append('CHoCH confirmed')
        
        # Touch number
        if touch_number == 1:
            score += 0.5
            reasoning_parts.append('1st touch')
        elif touch_number == 2:
            score += 0.25
            reasoning_parts.append('2nd touch')
        
        # No balance before
        if not has_balance_before:
            score += 0.5
            reasoning_parts.append('No balance before')
        
        # Zone position vs setup type
        if setup_type == 'ZFR':  # Aggressive 3+R setup
            if zone_position in ['front', 'middle']:
                score += 0.5
                reasoning_parts.append('ZFR at front/middle')
            elif zone_position == 'back':
                score += 0.25
                reasoning_parts.append('ZFR at back')
        elif setup_type == 'ZF-TR':  # Defensive 2-3R setup
            if zone_position in ['middle', 'back']:
                score += 0.5
                reasoning_parts.append('ZF-TR at middle/back')
            elif zone_position == 'front':
                score += 0.25
                reasoning_parts.append('ZF-TR at front')
        
        # Cap at 2.0
        score = min(score, 2.0)
        
        return QRSFactor(
            name='structure_touch',
            score=score,
            max_score=2.0,
            weight=self.factor_weights['structure_touch'],
            details=details,
            reasoning=' + '.join(reasoning_parts) if reasoning_parts else 'No structure factors'
        )
    
    def _score_context_intermarket(self, setup: Dict[str, Any], market_data: Dict[str, Any]) -> QRSFactor:
        """Score Factor 5: Context & Intermarket (0-2 points)."""
        htf_alignment = setup.get('htf_alignment', 'neutral')
        related_markets_aligned = setup.get('related_markets_aligned', False)
        divergences_confirm = setup.get('divergences_confirm', False)
        
        score = 0.0
        details = {
            'htf_alignment': htf_alignment,
            'related_markets_aligned': related_markets_aligned,
            'divergences_confirm': divergences_confirm
        }
        reasoning_parts = []
        
        # HTF alignment
        if htf_alignment == 'aligned':
            score += 1.0
            reasoning_parts.append('HTF aligned')
        elif htf_alignment == 'neutral':
            score += 0.5
            reasoning_parts.append('HTF neutral')
        
        # Related markets
        if related_markets_aligned:
            score += 0.5
            reasoning_parts.append('Related markets aligned')
        
        # Divergences
        if divergences_confirm:
            score += 0.5
            reasoning_parts.append('Divergences confirm')
        
        # Cap at 2.0
        score = min(score, 2.0)
        
        return QRSFactor(
            name='context_intermarket',
            score=score,
            max_score=2.0,
            weight=self.factor_weights['context_intermarket'],
            details=details,
            reasoning=' + '.join(reasoning_parts) if reasoning_parts else 'No context factors'
        )
    
    def _determine_grade(self, score: float) -> QRSGrade:
        """Determine QRS grade based on score."""
        if score >= 9.0:
            return QRSGrade.A_PLUS
        elif score >= 8.0:
            return QRSGrade.A
        elif score >= 7.0:
            return QRSGrade.B_PLUS
        elif score >= 6.0:
            return QRSGrade.B
        elif score >= 5.0:
            return QRSGrade.C
        elif score >= 4.0:
            return QRSGrade.D
        else:
            return QRSGrade.F
    
    def should_generate_signal(self, result: QRSResult) -> bool:
        """Check if signal should be generated based on QRS result."""
        return not result.veto and result.total_score >= self.threshold
    
    def get_statistics(self) -> Dict:
        """Get QRS scoring statistics."""
        if self.total_scored == 0:
            return {
                'total_scored': 0,
                'signals_generated': 0,
                'signals_vetoed': 0,
                'generation_rate': 0.0,
                'veto_rate': 0.0,
                'avg_score': 0.0
            }
        
        return {
            'total_scored': self.total_scored,
            'signals_generated': self.signals_generated,
            'signals_vetoed': self.signals_vetoed,
            'generation_rate': (self.signals_generated / self.total_scored) * 100,
            'veto_rate': (self.signals_vetoed / self.total_scored) * 100,
            'avg_score': self.avg_score
        }
    
    def reset_statistics(self):
        """Reset QRS scoring statistics."""
        self.total_scored = 0
        self.signals_generated = 0
        self.signals_vetoed = 0
        self.avg_score = 0.0