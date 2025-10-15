"""
Enhanced Market Context Analysis

This module implements enhanced market context detection including:
- Improved trend detection with multiple timeframes
- Volatility regime classification
- Market structure analysis
- Momentum analysis
- Context-based filtering
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics


class TrendStrength(Enum):
    """Trend strength classifications."""
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGE_BOUND = "RANGE_BOUND"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "LOW"              # ATR < 1%
    NORMAL = "NORMAL"        # ATR 1-2%
    HIGH = "HIGH"            # ATR 2-3%
    EXTREME = "EXTREME"      # ATR > 3%


class MarketStructure(Enum):
    """Market structure classifications."""
    BULLISH = "BULLISH"          # Higher highs, higher lows
    BEARISH = "BEARISH"          # Lower highs, lower lows
    CONSOLIDATING = "CONSOLIDATING"  # Sideways movement
    TRANSITIONING = "TRANSITIONING"  # Changing structure


@dataclass
class TrendAnalysis:
    """Trend analysis result."""
    trend_strength: TrendStrength
    trend_angle: float  # degrees
    price_vs_vwap: float  # percentage
    directional_bars: int
    total_bars: int
    directional_pct: float
    confidence: float


@dataclass
class VolatilityAnalysis:
    """Volatility analysis result."""
    regime: VolatilityRegime
    current_atr: float
    atr_percentile: float
    avg_atr: float
    atr_ratio: float  # current / average
    volatility_expanding: bool


@dataclass
class StructureAnalysis:
    """Market structure analysis result."""
    structure: MarketStructure
    swing_high: float
    swing_low: float
    recent_highs: List[float]
    recent_lows: List[float]
    structure_breaks: int
    structure_confidence: float


@dataclass
class MarketContextResult:
    """Complete market context result."""
    trend_analysis: TrendAnalysis
    volatility_analysis: VolatilityAnalysis
    structure_analysis: StructureAnalysis
    is_favorable_for_fade: bool
    context_score: float  # 0.0-1.0
    warnings: List[str]
    recommendations: List[str]


class EnhancedMarketContext:
    """
    Enhanced market context analyzer with multi-timeframe analysis.
    
    Features:
    - Improved trend detection using VWAP and price action
    - Volatility regime classification
    - Market structure analysis (HH/HL, LH/LL)
    - Momentum analysis
    - Context-based trade filtering
    """
    
    def __init__(self,
                 trend_lookback: int = 50,
                 structure_lookback: int = 100,
                 volatility_lookback: int = 20,
                 trend_threshold: float = 0.5,
                 structure_break_threshold: int = 2):
        """
        Initialize enhanced market context analyzer.
        
        Args:
            trend_lookback: Lookback period for trend analysis
            structure_lookback: Lookback period for structure analysis
            volatility_lookback: Lookback period for volatility analysis
            trend_threshold: Threshold for trend detection (% of bars)
            structure_break_threshold: Threshold for structure break count
        """
        self.trend_lookback = trend_lookback
        self.structure_lookback = structure_lookback
        self.volatility_lookback = volatility_lookback
        self.trend_threshold = trend_threshold
        self.structure_break_threshold = structure_break_threshold
        
        # Statistics
        self.total_analyzed = 0
        self.favorable_contexts = 0
        self.unfavorable_contexts = 0
    
    def analyze_context(self,
                       bars: List,
                       current_index: int,
                       direction: str) -> MarketContextResult:
        """
        Analyze complete market context.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            direction: Trade direction ('LONG' or 'SHORT')
            
        Returns:
            MarketContextResult with complete analysis
        """
        self.total_analyzed += 1
        
        # Analyze trend
        trend_analysis = self._analyze_trend(bars, current_index)
        
        # Analyze volatility
        volatility_analysis = self._analyze_volatility(bars, current_index)
        
        # Analyze structure
        structure_analysis = self._analyze_structure(bars, current_index)
        
        # Determine if context is favorable for fade
        is_favorable = self._is_favorable_for_fade(
            trend_analysis, volatility_analysis, structure_analysis, direction
        )
        
        # Calculate context score
        context_score = self._calculate_context_score(
            trend_analysis, volatility_analysis, structure_analysis, direction
        )
        
        # Generate warnings and recommendations
        warnings, recommendations = self._generate_insights(
            trend_analysis, volatility_analysis, structure_analysis, direction, is_favorable
        )
        
        if is_favorable:
            self.favorable_contexts += 1
        else:
            self.unfavorable_contexts += 1
        
        return MarketContextResult(
            trend_analysis=trend_analysis,
            volatility_analysis=volatility_analysis,
            structure_analysis=structure_analysis,
            is_favorable_for_fade=is_favorable,
            context_score=context_score,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _analyze_trend(self, bars: List, current_index: int) -> TrendAnalysis:
        """Analyze trend using multiple indicators."""
        if current_index < self.trend_lookback:
            return self._create_neutral_trend()
        
        # Get recent bars
        lookback_start = max(0, current_index - self.trend_lookback)
        recent_bars = bars[lookback_start:current_index + 1]
        
        current_bar = bars[current_index]
        current_close = current_bar.close if hasattr(current_bar, 'close') else current_bar['close']
        
        # Calculate VWAP
        vwap = self._calculate_vwap(recent_bars)
        price_vs_vwap = ((current_close - vwap) / vwap * 100) if vwap > 0 else 0
        
        # Count directional bars
        directional_bars = 0
        total_bars = len(recent_bars) - 1
        
        for i in range(1, len(recent_bars)):
            prev_bar = recent_bars[i-1]
            curr_bar = recent_bars[i]
            
            prev_close = prev_bar.close if hasattr(prev_bar, 'close') else prev_bar['close']
            curr_close = curr_bar.close if hasattr(curr_bar, 'close') else curr_bar['close']
            
            if curr_close > prev_close:
                directional_bars += 1  # Bullish bar
        
        directional_pct = (directional_bars / total_bars) if total_bars > 0 else 0.5
        
        # Calculate trend angle (simplified)
        first_bar = recent_bars[0]
        last_bar = recent_bars[-1]
        first_close = first_bar.close if hasattr(first_bar, 'close') else first_bar['close']
        last_close = last_bar.close if hasattr(last_bar, 'close') else last_bar['close']
        
        price_change_pct = ((last_close - first_close) / first_close * 100) if first_close > 0 else 0
        trend_angle = price_change_pct  # Simplified angle representation
        
        # Determine trend strength
        if directional_pct >= 0.70 and price_vs_vwap > 1.0:
            trend_strength = TrendStrength.STRONG_UPTREND
            confidence = 0.9
        elif directional_pct >= 0.60 and price_vs_vwap > 0.3:
            trend_strength = TrendStrength.WEAK_UPTREND
            confidence = 0.7
        elif directional_pct <= 0.30 and price_vs_vwap < -1.0:
            trend_strength = TrendStrength.STRONG_DOWNTREND
            confidence = 0.9
        elif directional_pct <= 0.40 and price_vs_vwap < -0.3:
            trend_strength = TrendStrength.WEAK_DOWNTREND
            confidence = 0.7
        else:
            trend_strength = TrendStrength.RANGE_BOUND
            confidence = 0.8
        
        return TrendAnalysis(
            trend_strength=trend_strength,
            trend_angle=trend_angle,
            price_vs_vwap=price_vs_vwap,
            directional_bars=directional_bars,
            total_bars=total_bars,
            directional_pct=directional_pct,
            confidence=confidence
        )
    
    def _analyze_volatility(self, bars: List, current_index: int) -> VolatilityAnalysis:
        """Analyze volatility regime."""
        if current_index < self.volatility_lookback + 14:
            return self._create_normal_volatility()
        
        # Calculate current ATR
        current_atr = self._calculate_atr(bars, current_index, period=14)
        
        # Get current price for percentage calculation
        current_bar = bars[current_index]
        current_close = current_bar.close if hasattr(current_bar, 'close') else current_bar['close']
        current_atr_pct = (current_atr / current_close * 100) if current_close > 0 else 0
        
        # Calculate ATR over longer period
        lookback_start = max(0, current_index - self.volatility_lookback)
        historical_atrs = []
        
        for i in range(lookback_start, current_index):
            if i >= 14:
                atr = self._calculate_atr(bars, i, period=14)
                bar = bars[i]
                close = bar.close if hasattr(bar, 'close') else bar['close']
                atr_pct = (atr / close * 100) if close > 0 else 0
                historical_atrs.append(atr_pct)
        
        avg_atr_pct = statistics.mean(historical_atrs) if historical_atrs else current_atr_pct
        
        # Calculate ATR ratio and percentile
        atr_ratio = current_atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1.0
        atr_percentile = self._calculate_percentile(current_atr_pct, historical_atrs)
        
        # Determine volatility regime
        if current_atr_pct > 3.0:
            regime = VolatilityRegime.EXTREME
        elif current_atr_pct > 2.0:
            regime = VolatilityRegime.HIGH
        elif current_atr_pct > 1.0:
            regime = VolatilityRegime.NORMAL
        else:
            regime = VolatilityRegime.LOW
        
        # Check if volatility is expanding
        volatility_expanding = atr_ratio > 1.2
        
        return VolatilityAnalysis(
            regime=regime,
            current_atr=current_atr_pct,
            atr_percentile=atr_percentile,
            avg_atr=avg_atr_pct,
            atr_ratio=atr_ratio,
            volatility_expanding=volatility_expanding
        )
    
    def _analyze_structure(self, bars: List, current_index: int) -> StructureAnalysis:
        """Analyze market structure (HH/HL, LH/LL)."""
        if current_index < self.structure_lookback:
            return self._create_neutral_structure()
        
        # Get recent bars
        lookback_start = max(0, current_index - self.structure_lookback)
        recent_bars = bars[lookback_start:current_index + 1]
        
        # Find swing highs and lows
        swing_highs, swing_lows = self._find_swing_points(recent_bars)
        
        # Determine structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for higher highs and higher lows (bullish)
            higher_highs = all(swing_highs[i] > swing_highs[i-1] for i in range(1, len(swing_highs)))
            higher_lows = all(swing_lows[i] > swing_lows[i-1] for i in range(1, len(swing_lows)))
            
            # Check for lower highs and lower lows (bearish)
            lower_highs = all(swing_highs[i] < swing_highs[i-1] for i in range(1, len(swing_highs)))
            lower_lows = all(swing_lows[i] < swing_lows[i-1] for i in range(1, len(swing_lows)))
            
            if higher_highs and higher_lows:
                structure = MarketStructure.BULLISH
                confidence = 0.9
            elif lower_highs and lower_lows:
                structure = MarketStructure.BEARISH
                confidence = 0.9
            elif (higher_highs and not lower_lows) or (not lower_highs and higher_lows):
                structure = MarketStructure.TRANSITIONING
                confidence = 0.6
            else:
                structure = MarketStructure.CONSOLIDATING
                confidence = 0.7
        else:
            structure = MarketStructure.CONSOLIDATING
            confidence = 0.5
        
        # Count structure breaks
        structure_breaks = self._count_structure_breaks(swing_highs, swing_lows)
        
        return StructureAnalysis(
            structure=structure,
            swing_high=swing_highs[-1] if swing_highs else 0,
            swing_low=swing_lows[-1] if swing_lows else 0,
            recent_highs=swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs,
            recent_lows=swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows,
            structure_breaks=structure_breaks,
            structure_confidence=confidence
        )
    
    def _is_favorable_for_fade(self,
                               trend: TrendAnalysis,
                               volatility: VolatilityAnalysis,
                               structure: StructureAnalysis,
                               direction: str) -> bool:
        """Determine if context is favorable for fade trade."""
        # Favorable conditions for fades:
        # 1. Range-bound market (not strong trend)
        # 2. Normal/low volatility (not extreme)
        # 3. Consolidating or favorable structure
        # 4. Direction aligns with mean reversion
        
        # Check trend (fades work best in range-bound markets)
        trend_favorable = trend.trend_strength == TrendStrength.RANGE_BOUND
        
        # Check volatility (fades work best in normal/low volatility)
        volatility_favorable = volatility.regime in [VolatilityRegime.LOW, VolatilityRegime.NORMAL]
        
        # Check structure
        if direction == 'LONG':
            # LONG fades work best when fading downtrends or in consolidation
            structure_favorable = structure.structure in [
                MarketStructure.BEARISH,
                MarketStructure.CONSOLIDATING
            ]
        else:  # SHORT
            # SHORT fades work best when fading uptrends or in consolidation
            structure_favorable = structure.structure in [
                MarketStructure.BULLISH,
                MarketStructure.CONSOLIDATING
            ]
        
        # Need at least 2 out of 3 favorable conditions
        favorable_count = sum([trend_favorable, volatility_favorable, structure_favorable])
        
        return favorable_count >= 2
    
    def _calculate_context_score(self,
                                 trend: TrendAnalysis,
                                 volatility: VolatilityAnalysis,
                                 structure: StructureAnalysis,
                                 direction: str) -> float:
        """Calculate overall context score (0.0-1.0)."""
        score = 0.0
        
        # Trend score (30%)
        if trend.trend_strength == TrendStrength.RANGE_BOUND:
            score += 0.30
        elif trend.trend_strength in [TrendStrength.WEAK_UPTREND, TrendStrength.WEAK_DOWNTREND]:
            score += 0.20
        else:
            score += 0.10
        
        # Volatility score (30%)
        if volatility.regime == VolatilityRegime.NORMAL:
            score += 0.30
        elif volatility.regime == VolatilityRegime.LOW:
            score += 0.25
        elif volatility.regime == VolatilityRegime.HIGH:
            score += 0.15
        else:
            score += 0.05
        
        # Structure score (40%)
        if direction == 'LONG' and structure.structure == MarketStructure.BEARISH:
            score += 0.40  # Fading bearish structure
        elif direction == 'SHORT' and structure.structure == MarketStructure.BULLISH:
            score += 0.40  # Fading bullish structure
        elif structure.structure == MarketStructure.CONSOLIDATING:
            score += 0.30
        else:
            score += 0.15
        
        return min(1.0, score)
    
    def _generate_insights(self,
                          trend: TrendAnalysis,
                          volatility: VolatilityAnalysis,
                          structure: StructureAnalysis,
                          direction: str,
                          is_favorable: bool) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations."""
        warnings = []
        recommendations = []
        
        # Trend warnings
        if trend.trend_strength in [TrendStrength.STRONG_UPTREND, TrendStrength.STRONG_DOWNTREND]:
            warnings.append(f"Strong {trend.trend_strength.value.lower()} - fades risky")
        
        # Volatility warnings
        if volatility.regime == VolatilityRegime.EXTREME:
            warnings.append("Extreme volatility - wider stops recommended")
        elif volatility.volatility_expanding:
            warnings.append("Volatility expanding - use caution")
        
        # Structure warnings
        if structure.structure_breaks >= self.structure_break_threshold:
            warnings.append(f"{structure.structure_breaks} structure breaks detected")
        
        # Recommendations
        if is_favorable:
            recommendations.append("Context favorable for fade trades")
        else:
            recommendations.append("Context not ideal - consider passing")
        
        if volatility.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            recommendations.append("Use wider stops in high volatility")
        
        return warnings, recommendations
    
    def _calculate_vwap(self, bars: List) -> float:
        """Calculate volume-weighted average price."""
        if not bars:
            return 0.0
        
        total_volume = 0
        vwap_sum = 0
        
        for bar in bars:
            volume = bar.volume if hasattr(bar, 'volume') else bar['volume']
            close = bar.close if hasattr(bar, 'close') else bar['close']
            
            vwap_sum += close * volume
            total_volume += volume
        
        return vwap_sum / total_volume if total_volume > 0 else 0.0
    
    def _calculate_atr(self, bars: List, index: int, period: int = 14) -> float:
        """Calculate Average True Range."""
        if index < period:
            return 0.0
        
        true_ranges = []
        for i in range(index - period + 1, index + 1):
            if i < 1:
                continue
            
            curr_bar = bars[i]
            prev_bar = bars[i-1]
            
            high = curr_bar.high if hasattr(curr_bar, 'high') else curr_bar['high']
            low = curr_bar.low if hasattr(curr_bar, 'low') else curr_bar['low']
            prev_close = prev_bar.close if hasattr(prev_bar, 'close') else prev_bar['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return statistics.mean(true_ranges) if true_ranges else 0.0
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank."""
        if not values:
            return 50.0
        
        sorted_values = sorted(values)
        count_below = sum(1 for v in sorted_values if v < value)
        return (count_below / len(sorted_values)) * 100
    
    def _find_swing_points(self, bars: List, lookback: int = 5) -> Tuple[List[float], List[float]]:
        """Find swing highs and lows."""
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(bars) - lookback):
            bar = bars[i]
            high = bar.high if hasattr(bar, 'high') else bar['high']
            low = bar.low if hasattr(bar, 'low') else bar['low']
            
            # Check if swing high
            is_swing_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j == i:
                    continue
                compare_bar = bars[j]
                compare_high = compare_bar.high if hasattr(compare_bar, 'high') else compare_bar['high']
                if compare_high > high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(high)
            
            # Check if swing low
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j == i:
                    continue
                compare_bar = bars[j]
                compare_low = compare_bar.low if hasattr(compare_bar, 'low') else compare_bar['low']
                if compare_low < low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(low)
        
        return swing_highs, swing_lows
    
    def _count_structure_breaks(self, swing_highs: List[float], swing_lows: List[float]) -> int:
        """Count structure breaks in recent swings."""
        breaks = 0
        
        # Count breaks in highs
        for i in range(1, len(swing_highs)):
            if i >= 2:
                # Check if high broke previous pattern
                if (swing_highs[i] > swing_highs[i-1] > swing_highs[i-2] or
                    swing_highs[i] < swing_highs[i-1] < swing_highs[i-2]):
                    if not (swing_highs[i] > swing_highs[i-1] or swing_highs[i] < swing_highs[i-1]):
                        breaks += 1
        
        # Count breaks in lows
        for i in range(1, len(swing_lows)):
            if i >= 2:
                # Check if low broke previous pattern
                if (swing_lows[i] > swing_lows[i-1] > swing_lows[i-2] or
                    swing_lows[i] < swing_lows[i-1] < swing_lows[i-2]):
                    if not (swing_lows[i] > swing_lows[i-1] or swing_lows[i] < swing_lows[i-1]):
                        breaks += 1
        
        return breaks
    
    def _create_neutral_trend(self) -> TrendAnalysis:
        """Create neutral trend result."""
        return TrendAnalysis(
            trend_strength=TrendStrength.RANGE_BOUND,
            trend_angle=0.0,
            price_vs_vwap=0.0,
            directional_bars=0,
            total_bars=0,
            directional_pct=0.5,
            confidence=0.5
        )
    
    def _create_normal_volatility(self) -> VolatilityAnalysis:
        """Create normal volatility result."""
        return VolatilityAnalysis(
            regime=VolatilityRegime.NORMAL,
            current_atr=1.5,
            atr_percentile=50.0,
            avg_atr=1.5,
            atr_ratio=1.0,
            volatility_expanding=False
        )
    
    def _create_neutral_structure(self) -> StructureAnalysis:
        """Create neutral structure result."""
        return StructureAnalysis(
            structure=MarketStructure.CONSOLIDATING,
            swing_high=0.0,
            swing_low=0.0,
            recent_highs=[],
            recent_lows=[],
            structure_breaks=0,
            structure_confidence=0.5
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get context analysis statistics."""
        return {
            'total_analyzed': self.total_analyzed,
            'favorable_contexts': self.favorable_contexts,
            'unfavorable_contexts': self.unfavorable_contexts,
            'favorable_rate': (self.favorable_contexts / self.total_analyzed * 100) if self.total_analyzed > 0 else 0
        }
