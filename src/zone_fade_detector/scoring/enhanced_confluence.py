"""
Enhanced Zone Confluence Scoring

This module implements enhanced confluence scoring for zone quality assessment:
- Multi-factor weighted algorithm
- Volume-based confirmation
- Time-based prioritization  
- Dynamic zone quality scoring
- Confidence levels
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import statistics


class ConfluenceFactor(Enum):
    """Confluence factors for zone quality."""
    HTF_ZONE = "HTF_ZONE"                    # Higher timeframe zone
    VOLUME_NODE = "VOLUME_NODE"              # High volume area
    MULTIPLE_TOUCHES = "MULTIPLE_TOUCHES"    # Zone tested multiple times
    RECENT_FORMATION = "RECENT_FORMATION"    # Recently formed zone
    PRICE_ACTION = "PRICE_ACTION"            # Clean price action around zone
    STRUCTURE_LEVEL = "STRUCTURE_LEVEL"      # Important structure level
    ROUND_NUMBER = "ROUND_NUMBER"            # Round psychological level
    VWAP_ALIGNMENT = "VWAP_ALIGNMENT"        # Aligns with VWAP
    SESSION_EXTREME = "SESSION_EXTREME"      # Session high/low


class ZoneQuality(Enum):
    """Zone quality classifications."""
    ELITE = "ELITE"              # 90-100% score
    EXCELLENT = "EXCELLENT"      # 80-90% score
    GOOD = "GOOD"                # 70-80% score
    ACCEPTABLE = "ACCEPTABLE"    # 60-70% score
    POOR = "POOR"                # <60% score


@dataclass
class ConfluenceScore:
    """Individual confluence factor score."""
    factor: ConfluenceFactor
    score: float  # 0.0-1.0
    weight: float  # 0.0-1.0
    weighted_score: float
    evidence: str


@dataclass
class ZoneConfluenceResult:
    """Complete zone confluence scoring result."""
    total_score: float  # 0.0-100.0
    quality: ZoneQuality
    confluence_factors: List[ConfluenceScore]
    factor_count: int
    weighted_average: float
    confidence: float
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str


class EnhancedConfluenceScorer:
    """
    Enhanced confluence scorer for zone quality assessment.
    
    Features:
    - Multi-factor weighted scoring
    - Volume-based confirmation
    - Time-based prioritization
    - Dynamic quality assessment
    - Confidence levels
    """
    
    def __init__(self,
                 htf_weight: float = 0.20,
                 volume_weight: float = 0.20,
                 time_weight: float = 0.15,
                 structure_weight: float = 0.15,
                 price_action_weight: float = 0.15,
                 psychological_weight: float = 0.10,
                 vwap_weight: float = 0.05):
        """
        Initialize enhanced confluence scorer.
        
        Args:
            htf_weight: Weight for higher timeframe zones
            volume_weight: Weight for volume confirmation
            time_weight: Weight for time-based factors
            structure_weight: Weight for structure levels
            price_action_weight: Weight for price action quality
            psychological_weight: Weight for round numbers
            vwap_weight: Weight for VWAP alignment
        """
        self.weights = {
            'htf': htf_weight,
            'volume': volume_weight,
            'time': time_weight,
            'structure': structure_weight,
            'price_action': price_action_weight,
            'psychological': psychological_weight,
            'vwap': vwap_weight
        }
        
        # Statistics
        self.total_scored = 0
        self.elite_zones = 0
        self.excellent_zones = 0
        self.good_zones = 0
    
    def score_zone_confluence(self,
                             zone_level: float,
                             zone_type: str,
                             zone_age_hours: float,
                             touch_count: int,
                             volume_profile: Optional[Dict[float, float]] = None,
                             recent_bars: Optional[List] = None,
                             vwap: Optional[float] = None) -> ZoneConfluenceResult:
        """
        Score zone confluence using multiple factors.
        
        Args:
            zone_level: Price level of the zone
            zone_type: Type of zone (e.g., 'PRIOR_DAY_HIGH')
            zone_age_hours: Age of zone in hours
            touch_count: Number of times zone was touched
            volume_profile: Volume profile data
            recent_bars: Recent price bars for analysis
            vwap: Current VWAP value
            
        Returns:
            ZoneConfluenceResult with complete scoring
        """
        self.total_scored += 1
        
        confluence_factors = []
        
        # 1. HTF Zone Factor
        htf_score = self._score_htf_zone(zone_type)
        confluence_factors.append(htf_score)
        
        # 2. Volume Factor
        if volume_profile:
            volume_score = self._score_volume_confluence(zone_level, volume_profile)
            confluence_factors.append(volume_score)
        
        # 3. Time Factor
        time_score = self._score_time_factor(zone_age_hours, touch_count)
        confluence_factors.append(time_score)
        
        # 4. Structure Factor
        structure_score = self._score_structure_level(zone_type, zone_level)
        confluence_factors.append(structure_score)
        
        # 5. Price Action Factor
        if recent_bars:
            price_action_score = self._score_price_action(zone_level, recent_bars)
            confluence_factors.append(price_action_score)
        
        # 6. Psychological Level Factor
        psychological_score = self._score_psychological_level(zone_level)
        confluence_factors.append(psychological_score)
        
        # 7. VWAP Alignment Factor
        if vwap:
            vwap_score = self._score_vwap_alignment(zone_level, vwap)
            confluence_factors.append(vwap_score)
        
        # Calculate total score
        total_weighted_score = sum(cf.weighted_score for cf in confluence_factors)
        total_score = total_weighted_score * 100  # Convert to 0-100 scale
        
        # Determine quality
        quality = self._determine_quality(total_score)
        
        # Track statistics
        if quality == ZoneQuality.ELITE:
            self.elite_zones += 1
        elif quality == ZoneQuality.EXCELLENT:
            self.excellent_zones += 1
        elif quality == ZoneQuality.GOOD:
            self.good_zones += 1
        
        # Calculate confidence
        confidence = len(confluence_factors) / 7.0  # Max 7 factors
        
        # Generate insights
        strengths, weaknesses = self._generate_insights(confluence_factors, quality)
        recommendation = self._generate_recommendation(total_score, quality, confidence)
        
        return ZoneConfluenceResult(
            total_score=total_score,
            quality=quality,
            confluence_factors=confluence_factors,
            factor_count=len(confluence_factors),
            weighted_average=total_weighted_score,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation
        )
    
    def _score_htf_zone(self, zone_type: str) -> ConfluenceScore:
        """Score higher timeframe zone quality."""
        # HTF zones are more reliable
        htf_zones = ['PRIOR_DAY_HIGH', 'PRIOR_DAY_LOW', 'WEEKLY_HIGH', 'WEEKLY_LOW', 
                     'MONTHLY_HIGH', 'MONTHLY_LOW', 'VALUE_AREA_HIGH', 'VALUE_AREA_LOW']
        
        if any(htf in zone_type.upper() for htf in htf_zones):
            score = 1.0
            evidence = f"HTF zone: {zone_type}"
        elif 'OPENING_RANGE' in zone_type.upper():
            score = 0.8
            evidence = f"Opening range zone: {zone_type}"
        else:
            score = 0.5
            evidence = f"Standard zone: {zone_type}"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.HTF_ZONE,
            score=score,
            weight=self.weights['htf'],
            weighted_score=score * self.weights['htf'],
            evidence=evidence
        )
    
    def _score_volume_confluence(self, zone_level: float, 
                                 volume_profile: Dict[float, float]) -> ConfluenceScore:
        """Score volume confluence at zone level."""
        if not volume_profile:
            return ConfluenceScore(
                factor=ConfluenceFactor.VOLUME_NODE,
                score=0.5,
                weight=self.weights['volume'],
                weighted_score=0.5 * self.weights['volume'],
                evidence="No volume data available"
            )
        
        # Find volume at zone level (within 0.1% tolerance)
        tolerance = zone_level * 0.001
        zone_volumes = [vol for price, vol in volume_profile.items() 
                       if abs(price - zone_level) <= tolerance]
        
        if not zone_volumes:
            score = 0.3
            evidence = "Low volume at zone level"
        else:
            avg_zone_volume = statistics.mean(zone_volumes)
            total_avg_volume = statistics.mean(volume_profile.values())
            
            volume_ratio = avg_zone_volume / total_avg_volume if total_avg_volume > 0 else 0
            
            if volume_ratio >= 2.0:
                score = 1.0
                evidence = f"High volume node: {volume_ratio:.1f}x average"
            elif volume_ratio >= 1.5:
                score = 0.8
                evidence = f"Good volume node: {volume_ratio:.1f}x average"
            elif volume_ratio >= 1.2:
                score = 0.6
                evidence = f"Moderate volume: {volume_ratio:.1f}x average"
            else:
                score = 0.4
                evidence = f"Low volume: {volume_ratio:.1f}x average"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.VOLUME_NODE,
            score=score,
            weight=self.weights['volume'],
            weighted_score=score * self.weights['volume'],
            evidence=evidence
        )
    
    def _score_time_factor(self, zone_age_hours: float, touch_count: int) -> ConfluenceScore:
        """Score time-based factors (freshness and touch count)."""
        # Fresh zones (< 24 hours) are generally better
        # But zones with 1-2 touches are ideal (not virgin, not overtraded)
        
        freshness_score = 0.0
        if zone_age_hours < 4:
            freshness_score = 1.0
        elif zone_age_hours < 12:
            freshness_score = 0.9
        elif zone_age_hours < 24:
            freshness_score = 0.7
        elif zone_age_hours < 48:
            freshness_score = 0.5
        else:
            freshness_score = 0.3
        
        touch_score = 0.0
        if touch_count == 0:
            touch_score = 0.9  # Virgin zone
        elif touch_count == 1:
            touch_score = 1.0  # Ideal: first retest
        elif touch_count == 2:
            touch_score = 0.8  # Still good
        else:
            touch_score = 0.4  # Overtraded
        
        # Combined score
        score = (freshness_score * 0.6 + touch_score * 0.4)
        
        evidence = f"Age: {zone_age_hours:.1f}h, Touches: {touch_count}"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.RECENT_FORMATION,
            score=score,
            weight=self.weights['time'],
            weighted_score=score * self.weights['time'],
            evidence=evidence
        )
    
    def _score_structure_level(self, zone_type: str, zone_level: float) -> ConfluenceScore:
        """Score structural importance of zone."""
        # Session extremes and value areas are important structure
        important_types = ['HIGH', 'LOW', 'VALUE_AREA', 'POC']
        
        if any(imp in zone_type.upper() for imp in important_types):
            score = 0.9
            evidence = f"Important structure level: {zone_type}"
        elif 'OPENING_RANGE' in zone_type.upper():
            score = 0.7
            evidence = "Opening range level"
        else:
            score = 0.5
            evidence = "Standard level"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.STRUCTURE_LEVEL,
            score=score,
            weight=self.weights['structure'],
            weighted_score=score * self.weights['structure'],
            evidence=evidence
        )
    
    def _score_price_action(self, zone_level: float, recent_bars: List) -> ConfluenceScore:
        """Score price action quality around zone."""
        if len(recent_bars) < 10:
            return ConfluenceScore(
                factor=ConfluenceFactor.PRICE_ACTION,
                score=0.5,
                weight=self.weights['price_action'],
                weighted_score=0.5 * self.weights['price_action'],
                evidence="Insufficient price data"
            )
        
        # Check for clean rejection patterns
        recent_prices = []
        for bar in recent_bars[-10:]:
            if hasattr(bar, 'close'):
                recent_prices.append(bar.close)
            else:
                recent_prices.append(bar['close'])
        
        # Calculate how many times price approached zone
        tolerance = zone_level * 0.002  # 0.2% tolerance
        approaches = sum(1 for price in recent_prices if abs(price - zone_level) <= tolerance)
        
        # Clean price action means few approaches but strong reactions
        if approaches == 0:
            score = 0.8  # Virgin zone
            evidence = "Clean, untested zone"
        elif approaches == 1:
            score = 1.0  # One clean rejection
            evidence = "Single clean rejection"
        elif approaches == 2:
            score = 0.7  # Two touches
            evidence = "Two touches, still valid"
        else:
            score = 0.4  # Multiple touches, weakening
            evidence = f"{approaches} touches, weakening"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.PRICE_ACTION,
            score=score,
            weight=self.weights['price_action'],
            weighted_score=score * self.weights['price_action'],
            evidence=evidence
        )
    
    def _score_psychological_level(self, zone_level: float) -> ConfluenceScore:
        """Score psychological level (round numbers)."""
        # Check if near round numbers (00, 50, etc.)
        # More significant for lower prices: e.g., 100.00, 150.00, 200.00
        
        # Check for full round numbers (100, 150, 200, etc.)
        remainder_full = zone_level % 50
        if remainder_full < 0.50 or remainder_full > 49.50:
            score = 1.0
            evidence = f"Major round number: {zone_level:.2f}"
        # Check for half round numbers (100.50, 150.50, etc.)
        elif abs(remainder_full - 25) < 0.50:
            score = 0.8
            evidence = f"Half round number: {zone_level:.2f}"
        # Check for 10s (110, 120, 130, etc.)
        else:
            remainder_ten = zone_level % 10
            if remainder_ten < 0.50 or remainder_ten > 9.50:
                score = 0.6
                evidence = f"Minor round number: {zone_level:.2f}"
            else:
                score = 0.3
                evidence = "Not a psychological level"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.ROUND_NUMBER,
            score=score,
            weight=self.weights['psychological'],
            weighted_score=score * self.weights['psychological'],
            evidence=evidence
        )
    
    def _score_vwap_alignment(self, zone_level: float, vwap: float) -> ConfluenceScore:
        """Score alignment with VWAP."""
        if vwap == 0:
            return ConfluenceScore(
                factor=ConfluenceFactor.VWAP_ALIGNMENT,
                score=0.5,
                weight=self.weights['vwap'],
                weighted_score=0.5 * self.weights['vwap'],
                evidence="No VWAP data"
            )
        
        distance_pct = abs((zone_level - vwap) / vwap * 100)
        
        if distance_pct < 0.5:
            score = 1.0
            evidence = f"Excellent VWAP alignment: {distance_pct:.2f}%"
        elif distance_pct < 1.0:
            score = 0.8
            evidence = f"Good VWAP alignment: {distance_pct:.2f}%"
        elif distance_pct < 2.0:
            score = 0.6
            evidence = f"Moderate VWAP distance: {distance_pct:.2f}%"
        else:
            score = 0.3
            evidence = f"Far from VWAP: {distance_pct:.2f}%"
        
        return ConfluenceScore(
            factor=ConfluenceFactor.VWAP_ALIGNMENT,
            score=score,
            weight=self.weights['vwap'],
            weighted_score=score * self.weights['vwap'],
            evidence=evidence
        )
    
    def _determine_quality(self, total_score: float) -> ZoneQuality:
        """Determine zone quality from total score."""
        if total_score >= 90:
            return ZoneQuality.ELITE
        elif total_score >= 80:
            return ZoneQuality.EXCELLENT
        elif total_score >= 70:
            return ZoneQuality.GOOD
        elif total_score >= 60:
            return ZoneQuality.ACCEPTABLE
        else:
            return ZoneQuality.POOR
    
    def _generate_insights(self, factors: List[ConfluenceScore], 
                          quality: ZoneQuality) -> Tuple[List[str], List[str]]:
        """Generate strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        for factor in factors:
            if factor.score >= 0.8:
                strengths.append(f"{factor.factor.value}: {factor.evidence}")
            elif factor.score < 0.5:
                weaknesses.append(f"{factor.factor.value}: {factor.evidence}")
        
        return strengths, weaknesses
    
    def _generate_recommendation(self, total_score: float, 
                                 quality: ZoneQuality, 
                                 confidence: float) -> str:
        """Generate trading recommendation."""
        if quality in [ZoneQuality.ELITE, ZoneQuality.EXCELLENT] and confidence >= 0.8:
            return f"High-quality zone ({total_score:.1f}/100) - excellent confluence"
        elif quality == ZoneQuality.GOOD and confidence >= 0.7:
            return f"Good zone ({total_score:.1f}/100) - acceptable confluence"
        elif quality == ZoneQuality.ACCEPTABLE:
            return f"Acceptable zone ({total_score:.1f}/100) - consider with other confirmation"
        else:
            return f"Poor zone ({total_score:.1f}/100) - look for better setups"
    
    def get_statistics(self) -> Dict[str, any]:
        """Get confluence scoring statistics."""
        return {
            'total_scored': self.total_scored,
            'elite_zones': self.elite_zones,
            'excellent_zones': self.excellent_zones,
            'good_zones': self.good_zones,
            'elite_rate': (self.elite_zones / self.total_scored * 100) if self.total_scored > 0 else 0,
            'quality_rate': ((self.elite_zones + self.excellent_zones + self.good_zones) / self.total_scored * 100) if self.total_scored > 0 else 0
        }
