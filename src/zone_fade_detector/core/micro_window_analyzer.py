"""
Micro Window Analyzer for Zone Fade Strategy.

This module provides pre/post zone touch analysis for initiative/lack-of-initiative
patterns, enabling precise detection of absorption and exhaustion around key levels.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque

from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType, MarketContext
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType


class InitiativeType(Enum):
    """Types of market initiative."""
    BULLISH = "bullish"           # Strong buying pressure
    BEARISH = "bearish"           # Strong selling pressure
    NEUTRAL = "neutral"           # Balanced/indecisive
    EXHAUSTION = "exhaustion"     # Initiative exhaustion
    ABSORPTION = "absorption"     # Absorption at key levels


class MicroWindowType(Enum):
    """Types of micro windows for analysis."""
    PRE_TOUCH = "pre_touch"       # Before zone touch
    POST_TOUCH = "post_touch"     # After zone touch
    ZONE_APPROACH = "zone_approach"  # Approaching zone
    ZONE_REJECTION = "zone_rejection"  # Rejecting from zone


@dataclass
class MicroWindowConfig:
    """Configuration for micro window analysis."""
    window_type: MicroWindowType
    duration_minutes: int
    max_bars: int
    priority: int = 3  # Higher priority for micro analysis


@dataclass
class InitiativeMetrics:
    """Metrics for initiative analysis."""
    initiative_type: InitiativeType
    strength_score: float  # 0.0 to 1.0
    volume_ratio: float    # Current vs average volume
    price_momentum: float  # Price change rate
    wick_ratio: float      # Wick to body ratio
    rejection_clarity: float  # Clear rejection signals
    absorption_signals: int   # Number of absorption signals
    exhaustion_signals: int   # Number of exhaustion signals
    consecutive_bars: int     # Consecutive bars in same direction
    volatility_spike: bool    # Volatility spike detected
    volume_spike: bool        # Volume spike detected


@dataclass
class ZoneTouchAnalysis:
    """Analysis of a zone touch event."""
    zone: Zone
    touch_timestamp: datetime
    touch_price: float
    touch_type: str  # "supply", "demand"
    pre_touch_analysis: InitiativeMetrics
    post_touch_analysis: InitiativeMetrics
    micro_window_bars: List[OHLCVBar]
    is_significant: bool
    confidence_score: float  # 0.0 to 1.0
    absorption_detected: bool
    exhaustion_detected: bool
    rejection_confirmed: bool


@dataclass
class MicroWindowState:
    """State of a micro window."""
    window_type: MicroWindowType
    zone: Zone
    start_time: datetime
    end_time: datetime
    bars: List[OHLCVBar] = field(default_factory=list)
    is_active: bool = False
    analysis_complete: bool = False
    initiative_metrics: Optional[InitiativeMetrics] = None


class MicroWindowAnalyzer:
    """
    Analyzes micro windows around zone touches for initiative/lack-of-initiative patterns.
    
    Provides detailed analysis of pre/post zone touch behavior to detect absorption,
    exhaustion, and initiative changes that are critical for Zone Fade setups.
    """
    
    def __init__(
        self,
        window_manager: RollingWindowManager,
        pre_touch_minutes: int = 15,  # 15 minutes before touch
        post_touch_minutes: int = 10,  # 10 minutes after touch
        min_bars_for_analysis: int = 5,
        volume_spike_threshold: float = 1.5,
        volatility_spike_threshold: float = 1.3
    ):
        """
        Initialize micro window analyzer.
        
        Args:
            window_manager: Rolling window manager instance
            pre_touch_minutes: Minutes to analyze before zone touch
            post_touch_minutes: Minutes to analyze after zone touch
            min_bars_for_analysis: Minimum bars required for analysis
            volume_spike_threshold: Threshold for volume spike detection
            volatility_spike_threshold: Threshold for volatility spike detection
        """
        self.window_manager = window_manager
        self.pre_touch_minutes = pre_touch_minutes
        self.post_touch_minutes = post_touch_minutes
        self.min_bars_for_analysis = min_bars_for_analysis
        self.volume_spike_threshold = volume_spike_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Active micro windows
        self.active_windows: Dict[str, MicroWindowState] = {}
        self.completed_analyses: List[ZoneTouchAnalysis] = []
        
        # Configuration
        self.configs = {
            MicroWindowType.PRE_TOUCH: MicroWindowConfig(
                window_type=MicroWindowType.PRE_TOUCH,
                duration_minutes=pre_touch_minutes,
                max_bars=pre_touch_minutes * 2,  # Assume 2 bars per minute
                priority=3
            ),
            MicroWindowType.POST_TOUCH: MicroWindowConfig(
                window_type=MicroWindowType.POST_TOUCH,
                duration_minutes=post_touch_minutes,
                max_bars=post_touch_minutes * 2,
                priority=3
            )
        }
        
        self.logger.info("MicroWindowAnalyzer initialized")
    
    def analyze_zone_touch(
        self, 
        zone: Zone, 
        touch_bar: OHLCVBar, 
        symbol: str = "DEFAULT"
    ) -> Optional[ZoneTouchAnalysis]:
        """
        Analyze a zone touch event with micro window analysis.
        
        Args:
            zone: The zone that was touched
            touch_bar: The bar that touched the zone
            symbol: Symbol for the analysis
            
        Returns:
            ZoneTouchAnalysis if analysis is possible, None otherwise
        """
        try:
            # Determine touch type
            touch_type = self._determine_touch_type(zone, touch_bar)
            touch_price = self._get_touch_price(zone, touch_bar)
            
            # Get micro window bars
            micro_bars = self._get_micro_window_bars(zone, touch_bar, symbol)
            
            if len(micro_bars) < self.min_bars_for_analysis:
                self.logger.warning(f"Insufficient bars for micro analysis: {len(micro_bars)}")
                return None
            
            # Analyze pre-touch initiative
            pre_touch_bars = [b for b in micro_bars if b.timestamp < touch_bar.timestamp]
            pre_touch_analysis = self._analyze_initiative(pre_touch_bars, "pre_touch")
            
            # Analyze post-touch initiative
            post_touch_bars = [b for b in micro_bars if b.timestamp > touch_bar.timestamp]
            post_touch_analysis = self._analyze_initiative(post_touch_bars, "post_touch")
            
            # Determine if touch is significant
            is_significant = self._is_significant_touch(
                pre_touch_analysis, post_touch_analysis, zone, touch_bar
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                pre_touch_analysis, post_touch_analysis, zone, touch_bar
            )
            
            # Detect absorption and exhaustion
            absorption_detected = self._detect_absorption(pre_touch_analysis, post_touch_analysis)
            exhaustion_detected = self._detect_exhaustion(pre_touch_analysis, post_touch_analysis)
            rejection_confirmed = self._confirm_rejection(post_touch_analysis, zone, touch_bar)
            
            # Create analysis result
            analysis = ZoneTouchAnalysis(
                zone=zone,
                touch_timestamp=touch_bar.timestamp,
                touch_price=touch_price,
                touch_type=touch_type,
                pre_touch_analysis=pre_touch_analysis,
                post_touch_analysis=post_touch_analysis,
                micro_window_bars=micro_bars,
                is_significant=is_significant,
                confidence_score=confidence_score,
                absorption_detected=absorption_detected,
                exhaustion_detected=exhaustion_detected,
                rejection_confirmed=rejection_confirmed
            )
            
            # Store completed analysis
            self.completed_analyses.append(analysis)
            
            self.logger.info(
                f"Zone touch analysis completed: {touch_type} at {touch_price:.2f}, "
                f"confidence: {confidence_score:.2f}, significant: {is_significant}"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing zone touch: {e}")
            return None
    
    def _determine_touch_type(self, zone: Zone, touch_bar: OHLCVBar) -> str:
        """Determine if this is a supply or demand zone touch."""
        # Determine based on zone type
        if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH, ZoneType.OPENING_RANGE_HIGH, ZoneType.OVERNIGHT_HIGH]:
            return "supply"
        elif zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW, ZoneType.VALUE_AREA_LOW, ZoneType.OPENING_RANGE_LOW, ZoneType.OVERNIGHT_LOW]:
            return "demand"
        else:
            # Default to supply for high levels, demand for low levels
            return "supply"
    
    def _get_touch_price(self, zone: Zone, touch_bar: OHLCVBar) -> float:
        """Get the price at which the zone was touched."""
        if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH, ZoneType.OPENING_RANGE_HIGH, ZoneType.OVERNIGHT_HIGH]:
            return min(touch_bar.low, zone.level)
        else:  # demand zones
            return max(touch_bar.high, zone.level)
    
    def _get_micro_window_bars(
        self, 
        zone: Zone, 
        touch_bar: OHLCVBar, 
        symbol: str
    ) -> List[OHLCVBar]:
        """Get bars for micro window analysis around zone touch."""
        # Get bars from rolling windows
        session_bars = self.window_manager.get_window_bars(WindowType.SESSION_CONTEXT, symbol)
        
        if not session_bars:
            return []
        
        # Calculate time window
        pre_touch_time = touch_bar.timestamp - timedelta(minutes=self.pre_touch_minutes)
        post_touch_time = touch_bar.timestamp + timedelta(minutes=self.post_touch_minutes)
        
        # Filter bars within micro window
        micro_bars = [
            bar for bar in session_bars
            if pre_touch_time <= bar.timestamp <= post_touch_time
        ]
        
        return sorted(micro_bars, key=lambda b: b.timestamp)
    
    def _analyze_initiative(
        self, 
        bars: List[OHLCVBar], 
        window_type: str
    ) -> InitiativeMetrics:
        """Analyze initiative patterns in a micro window."""
        if not bars:
            return InitiativeMetrics(
                initiative_type=InitiativeType.NEUTRAL,
                strength_score=0.0,
                volume_ratio=1.0,
                price_momentum=0.0,
                wick_ratio=0.0,
                rejection_clarity=0.0,
                absorption_signals=0,
                exhaustion_signals=0,
                consecutive_bars=0,
                volatility_spike=False,
                volume_spike=False
            )
        
        # Calculate basic metrics
        volume_ratio = self._calculate_volume_ratio(bars)
        price_momentum = self._calculate_price_momentum(bars)
        wick_ratio = self._calculate_wick_ratio(bars)
        rejection_clarity = self._calculate_rejection_clarity(bars)
        
        # Detect spikes
        volume_spike = volume_ratio > self.volume_spike_threshold
        volatility_spike = self._detect_volatility_spike(bars)
        
        # Count signals
        absorption_signals = self._count_absorption_signals(bars)
        exhaustion_signals = self._count_exhaustion_signals(bars)
        consecutive_bars = self._count_consecutive_direction_bars(bars)
        
        # Determine initiative type
        initiative_type = self._determine_initiative_type(
            price_momentum, volume_ratio, wick_ratio, rejection_clarity,
            absorption_signals, exhaustion_signals
        )
        
        # Calculate strength score
        strength_score = self._calculate_strength_score(
            volume_ratio, price_momentum, wick_ratio, rejection_clarity,
            absorption_signals, exhaustion_signals, consecutive_bars
        )
        
        return InitiativeMetrics(
            initiative_type=initiative_type,
            strength_score=strength_score,
            volume_ratio=volume_ratio,
            price_momentum=price_momentum,
            wick_ratio=wick_ratio,
            rejection_clarity=rejection_clarity,
            absorption_signals=absorption_signals,
            exhaustion_signals=exhaustion_signals,
            consecutive_bars=consecutive_bars,
            volatility_spike=volatility_spike,
            volume_spike=volume_spike
        )
    
    def _calculate_volume_ratio(self, bars: List[OHLCVBar]) -> float:
        """Calculate current volume vs average volume ratio."""
        if not bars:
            return 1.0
        
        current_volume = bars[-1].volume
        avg_volume = sum(b.volume for b in bars) / len(bars)
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_price_momentum(self, bars: List[OHLCVBar]) -> float:
        """Calculate price momentum in the micro window."""
        if len(bars) < 2:
            return 0.0
        
        first_price = bars[0].close
        last_price = bars[-1].close
        time_diff = (bars[-1].timestamp - bars[0].timestamp).total_seconds() / 3600  # hours
        
        if time_diff == 0:
            return 0.0
        
        return (last_price - first_price) / first_price / time_diff if first_price > 0 else 0.0
    
    def _calculate_wick_ratio(self, bars: List[OHLCVBar]) -> float:
        """Calculate average wick to body ratio."""
        if not bars:
            return 0.0
        
        wick_ratios = []
        for bar in bars:
            body_size = abs(bar.close - bar.open)
            total_range = bar.high - bar.low
            
            if total_range > 0:
                wick_ratio = (total_range - body_size) / total_range
                wick_ratios.append(wick_ratio)
        
        return sum(wick_ratios) / len(wick_ratios) if wick_ratios else 0.0
    
    def _calculate_rejection_clarity(self, bars: List[OHLCVBar]) -> float:
        """Calculate rejection clarity score."""
        if not bars:
            return 0.0
        
        rejection_scores = []
        for bar in bars:
            body_size = abs(bar.close - bar.open)
            total_range = bar.high - bar.low
            
            if total_range > 0:
                # Higher wick ratio indicates clearer rejection
                wick_ratio = (total_range - body_size) / total_range
                rejection_scores.append(wick_ratio)
        
        return sum(rejection_scores) / len(rejection_scores) if rejection_scores else 0.0
    
    def _detect_volatility_spike(self, bars: List[OHLCVBar]) -> bool:
        """Detect if there's a volatility spike in the micro window."""
        if len(bars) < 3:
            return False
        
        # Calculate rolling volatility
        volatilities = []
        for i in range(1, len(bars)):
            price_change = abs(bars[i].close - bars[i-1].close) / bars[i-1].close
            volatilities.append(price_change)
        
        if not volatilities:
            return False
        
        current_volatility = volatilities[-1]
        avg_volatility = sum(volatilities[:-1]) / len(volatilities[:-1]) if len(volatilities) > 1 else current_volatility
        
        return current_volatility > avg_volatility * self.volatility_spike_threshold
    
    def _count_absorption_signals(self, bars: List[OHLCVBar]) -> int:
        """Count absorption signals in the micro window."""
        if len(bars) < 3:
            return 0
        
        absorption_count = 0
        
        for i in range(1, len(bars) - 1):
            prev_bar = bars[i-1]
            curr_bar = bars[i]
            next_bar = bars[i+1]
            
            # Absorption pattern: high volume with small price movement
            volume_ratio = curr_bar.volume / ((prev_bar.volume + next_bar.volume) / 2)
            price_change = abs(curr_bar.close - curr_bar.open) / curr_bar.open
            
            if volume_ratio > 1.2 and price_change < 0.005:  # High volume, small move
                absorption_count += 1
        
        return absorption_count
    
    def _count_exhaustion_signals(self, bars: List[OHLCVBar]) -> int:
        """Count exhaustion signals in the micro window."""
        if len(bars) < 3:
            return 0
        
        exhaustion_count = 0
        
        for i in range(1, len(bars) - 1):
            prev_bar = bars[i-1]
            curr_bar = bars[i]
            next_bar = bars[i+1]
            
            # Exhaustion pattern: decreasing volume with increasing price movement
            volume_decrease = curr_bar.volume < prev_bar.volume * 0.8
            price_increase = abs(curr_bar.close - curr_bar.open) > abs(prev_bar.close - prev_bar.open) * 1.2
            
            if volume_decrease and price_increase:
                exhaustion_count += 1
        
        return exhaustion_count
    
    def _count_consecutive_direction_bars(self, bars: List[OHLCVBar]) -> int:
        """Count consecutive bars in the same direction."""
        if len(bars) < 2:
            return 0
        
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(bars)):
            prev_direction = 1 if bars[i-1].close > bars[i-1].open else -1
            curr_direction = 1 if bars[i].close > bars[i].open else -1
            
            if prev_direction == curr_direction:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        return max(max_consecutive, current_consecutive)
    
    def _determine_initiative_type(
        self,
        price_momentum: float,
        volume_ratio: float,
        wick_ratio: float,
        rejection_clarity: float,
        absorption_signals: int,
        exhaustion_signals: int
    ) -> InitiativeType:
        """Determine the type of initiative based on metrics."""
        # High absorption signals indicate absorption
        if absorption_signals >= 2:
            return InitiativeType.ABSORPTION
        
        # High exhaustion signals indicate exhaustion
        if exhaustion_signals >= 2:
            return InitiativeType.EXHAUSTION
        
        # Strong bullish momentum
        if price_momentum > 0.01 and volume_ratio > 1.2:
            return InitiativeType.BULLISH
        
        # Strong bearish momentum
        if price_momentum < -0.01 and volume_ratio > 1.2:
            return InitiativeType.BEARISH
        
        # High rejection clarity indicates neutral/indecisive
        if rejection_clarity > 0.6:
            return InitiativeType.NEUTRAL
        
        # Default to neutral
        return InitiativeType.NEUTRAL
    
    def _calculate_strength_score(
        self,
        volume_ratio: float,
        price_momentum: float,
        wick_ratio: float,
        rejection_clarity: float,
        absorption_signals: int,
        exhaustion_signals: int,
        consecutive_bars: int
    ) -> float:
        """Calculate overall strength score for the initiative."""
        # Volume strength (0-0.3)
        volume_score = min(volume_ratio / 2.0, 0.3)
        
        # Momentum strength (0-0.3)
        momentum_score = min(abs(price_momentum) * 10, 0.3)
        
        # Rejection clarity (0-0.2)
        rejection_score = rejection_clarity * 0.2
        
        # Signal strength (0-0.2)
        signal_score = min((absorption_signals + exhaustion_signals) * 0.1, 0.2)
        
        # Consecutive bars strength (0-0.1)
        consecutive_score = min(consecutive_bars * 0.05, 0.1)
        
        return volume_score + momentum_score + rejection_score + signal_score + consecutive_score
    
    def _is_significant_touch(
        self,
        pre_touch: InitiativeMetrics,
        post_touch: InitiativeMetrics,
        zone: Zone,
        touch_bar: OHLCVBar
    ) -> bool:
        """Determine if the zone touch is significant."""
        # High confidence score
        if pre_touch.strength_score > 0.6 or post_touch.strength_score > 0.6:
            return True
        
        # Clear absorption or exhaustion
        if (pre_touch.initiative_type in [InitiativeType.ABSORPTION, InitiativeType.EXHAUSTION] or
            post_touch.initiative_type in [InitiativeType.ABSORPTION, InitiativeType.EXHAUSTION]):
            return True
        
        # High volume spike
        if pre_touch.volume_spike or post_touch.volume_spike:
            return True
        
        # Clear rejection in post-touch
        if post_touch.rejection_clarity > 0.7:
            return True
        
        return False
    
    def _calculate_confidence_score(
        self,
        pre_touch: InitiativeMetrics,
        post_touch: InitiativeMetrics,
        zone: Zone,
        touch_bar: OHLCVBar
    ) -> float:
        """Calculate confidence score for the zone touch analysis."""
        # Base score from strength metrics
        base_score = (pre_touch.strength_score + post_touch.strength_score) / 2
        
        # Bonus for clear patterns
        pattern_bonus = 0.0
        if pre_touch.initiative_type in [InitiativeType.ABSORPTION, InitiativeType.EXHAUSTION]:
            pattern_bonus += 0.2
        if post_touch.initiative_type in [InitiativeType.ABSORPTION, InitiativeType.EXHAUSTION]:
            pattern_bonus += 0.2
        
        # Bonus for volume spikes
        volume_bonus = 0.0
        if pre_touch.volume_spike:
            volume_bonus += 0.1
        if post_touch.volume_spike:
            volume_bonus += 0.1
        
        # Bonus for rejection clarity
        rejection_bonus = post_touch.rejection_clarity * 0.2
        
        return min(base_score + pattern_bonus + volume_bonus + rejection_bonus, 1.0)
    
    def _detect_absorption(
        self,
        pre_touch: InitiativeMetrics,
        post_touch: InitiativeMetrics
    ) -> bool:
        """Detect if absorption occurred at the zone."""
        # Pre-touch absorption signals
        pre_absorption = (pre_touch.initiative_type == InitiativeType.ABSORPTION or
                         pre_touch.absorption_signals >= 2)
        
        # Post-touch absorption signals
        post_absorption = (post_touch.initiative_type == InitiativeType.ABSORPTION or
                          post_touch.absorption_signals >= 2)
        
        return pre_absorption or post_absorption
    
    def _detect_exhaustion(
        self,
        pre_touch: InitiativeMetrics,
        post_touch: InitiativeMetrics
    ) -> bool:
        """Detect if exhaustion occurred at the zone."""
        # Pre-touch exhaustion signals
        pre_exhaustion = (pre_touch.initiative_type == InitiativeType.EXHAUSTION or
                         pre_touch.exhaustion_signals >= 2)
        
        # Post-touch exhaustion signals
        post_exhaustion = (post_touch.initiative_type == InitiativeType.EXHAUSTION or
                          post_touch.exhaustion_signals >= 2)
        
        return pre_exhaustion or post_exhaustion
    
    def _confirm_rejection(
        self,
        post_touch: InitiativeMetrics,
        zone: Zone,
        touch_bar: OHLCVBar
    ) -> bool:
        """Confirm if rejection occurred after zone touch."""
        # High rejection clarity
        if post_touch.rejection_clarity > 0.6:
            return True
        
        # Clear bearish rejection from supply zone
        if (zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH, ZoneType.OPENING_RANGE_HIGH, ZoneType.OVERNIGHT_HIGH] and 
            post_touch.initiative_type == InitiativeType.BEARISH and
            post_touch.strength_score > 0.4):
            return True
        
        # Clear bullish rejection from demand zone
        if (zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW, ZoneType.VALUE_AREA_LOW, ZoneType.OPENING_RANGE_LOW, ZoneType.OVERNIGHT_LOW] and 
            post_touch.initiative_type == InitiativeType.BULLISH and
            post_touch.strength_score > 0.4):
            return True
        
        return False
    
    def get_recent_analyses(self, limit: int = 10) -> List[ZoneTouchAnalysis]:
        """Get recent zone touch analyses."""
        return self.completed_analyses[-limit:]
    
    def get_significant_touches(self) -> List[ZoneTouchAnalysis]:
        """Get all significant zone touches."""
        return [analysis for analysis in self.completed_analyses if analysis.is_significant]
    
    def get_absorption_touches(self) -> List[ZoneTouchAnalysis]:
        """Get zone touches with absorption detected."""
        return [analysis for analysis in self.completed_analyses if analysis.absorption_detected]
    
    def get_exhaustion_touches(self) -> List[ZoneTouchAnalysis]:
        """Get zone touches with exhaustion detected."""
        return [analysis for analysis in self.completed_analyses if analysis.exhaustion_detected]
    
    def get_rejection_touches(self) -> List[ZoneTouchAnalysis]:
        """Get zone touches with confirmed rejection."""
        return [analysis for analysis in self.completed_analyses if analysis.rejection_confirmed]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses."""
        total_analyses = len(self.completed_analyses)
        significant_touches = len(self.get_significant_touches())
        absorption_touches = len(self.get_absorption_touches())
        exhaustion_touches = len(self.get_exhaustion_touches())
        rejection_touches = len(self.get_rejection_touches())
        
        avg_confidence = (
            sum(a.confidence_score for a in self.completed_analyses) / total_analyses
            if total_analyses > 0 else 0.0
        )
        
        return {
            "total_analyses": total_analyses,
            "significant_touches": significant_touches,
            "absorption_touches": absorption_touches,
            "exhaustion_touches": exhaustion_touches,
            "rejection_touches": rejection_touches,
            "average_confidence": avg_confidence,
            "significance_rate": significant_touches / total_analyses if total_analyses > 0 else 0.0,
            "absorption_rate": absorption_touches / total_analyses if total_analyses > 0 else 0.0,
            "exhaustion_rate": exhaustion_touches / total_analyses if total_analyses > 0 else 0.0,
            "rejection_rate": rejection_touches / total_analyses if total_analyses > 0 else 0.0
        }