"""
Session Analysis

This module implements session analysis to apply session-specific rules
and adjustments for zone fade setups.

Features:
- Session type detection (ON/AM/PM)
- ON range calculation and comparison
- PM-specific rules and QRS adjustments
- Short-term bias detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from enum import Enum

class SessionType(Enum):
    """Trading session type classifications."""
    ON = "ON"              # Overnight: 6:00 PM - 9:30 AM
    AM = "AM"              # Morning: 9:30 AM - 12:00 PM
    PM = "PM"              # Afternoon: 12:00 PM - 4:00 PM
    AFTER_HOURS = "AFTER_HOURS"  # After hours: 4:00 PM - 6:00 PM

class ShortTermBias(Enum):
    """Short-term bias classifications."""
    FAVOR = "FAVOR"        # Bias aligns with trade direction
    AGAINST = "AGAINST"    # Bias opposes trade direction
    NEUTRAL = "NEUTRAL"    # No clear bias

@dataclass
class ONRangeAnalysis:
    """ON range analysis result."""
    on_high: float
    on_low: float
    on_range: float
    typical_range: float
    range_multiplier: float
    is_large_range: bool
    analysis_period: int

@dataclass
class ShortTermBiasAnalysis:
    """Short-term bias analysis result."""
    bias: ShortTermBias
    bias_strength: float
    swing_structure: str
    analysis_period: int
    reasoning: str

@dataclass
class SessionAnalysisResult:
    """Complete session analysis result."""
    session_type: SessionType
    session_start: datetime
    session_end: datetime
    time_remaining: timedelta
    on_range_analysis: Optional[ONRangeAnalysis]
    short_term_bias: Optional[ShortTermBiasAnalysis]
    qrs_adjustment: float
    warnings: List[str]
    pm_rules_applied: bool
    recommendation: str

class SessionAnalyzer:
    """
    Analyzes trading sessions and applies session-specific rules.
    
    This analyzer implements the requirement to apply PM-specific rules
    and adjustments based on session type and market conditions.
    """
    
    def __init__(self,
                 on_session_start_hour: int = 18,  # 6:00 PM
                 am_session_start_hour: int = 9,
                 am_session_start_minute: int = 30,
                 am_session_end_hour: int = 12,
                 pm_session_start_hour: int = 12,
                 pm_session_end_hour: int = 16,  # 4:00 PM
                 after_hours_end_hour: int = 18,
                 on_range_lookback: int = 20):
        """
        Initialize session analyzer.
        
        Args:
            on_session_start_hour: Hour when ON session starts (ET)
            am_session_start_hour: Hour when AM session starts (ET)
            am_session_start_minute: Minute when AM session starts (ET)
            am_session_end_hour: Hour when AM session ends (ET)
            pm_session_start_hour: Hour when PM session starts (ET)
            pm_session_end_hour: Hour when PM session ends (ET)
            after_hours_end_hour: Hour when after hours end (ET)
            on_range_lookback: Days to look back for typical ON range
        """
        self.on_session_start_hour = on_session_start_hour
        self.am_session_start_hour = am_session_start_hour
        self.am_session_start_minute = am_session_start_minute
        self.am_session_end_hour = am_session_end_hour
        self.pm_session_start_hour = pm_session_start_hour
        self.pm_session_end_hour = pm_session_end_hour
        self.after_hours_end_hour = after_hours_end_hour
        self.on_range_lookback = on_range_lookback
        
        # Statistics
        self.total_analyzed = 0
        self.pm_sessions = 0
        self.pm_rules_applied = 0
        self.qrs_adjustments_made = 0
    
    def analyze_session(self, 
                       current_time: datetime,
                       price_bars: List,
                       trade_direction: str,
                       zone_quality: str = "STANDARD") -> SessionAnalysisResult:
        """
        Analyze current trading session and apply session-specific rules.
        
        Args:
            current_time: Current timestamp
            price_bars: List of OHLCV bars for analysis
            trade_direction: 'LONG' or 'SHORT'
            zone_quality: Zone quality rating ('IDEAL', 'GOOD', 'STANDARD')
            
        Returns:
            SessionAnalysisResult with analysis and recommendations
        """
        # Detect session type
        session_type, session_start, session_end = self._detect_session_type(current_time)
        
        # Calculate time remaining
        time_remaining = session_end - current_time
        
        # Initialize result
        result = SessionAnalysisResult(
            session_type=session_type,
            session_start=session_start,
            session_end=session_end,
            time_remaining=time_remaining,
            on_range_analysis=None,
            short_term_bias=None,
            qrs_adjustment=0.0,
            warnings=[],
            pm_rules_applied=False,
            recommendation="No session-specific adjustments needed"
        )
        
        # Apply session-specific analysis
        if session_type == SessionType.PM:
            result = self._apply_pm_rules(result, price_bars, trade_direction, zone_quality)
            self.pm_sessions += 1
            if result.pm_rules_applied:
                self.pm_rules_applied += 1
        
        # Update statistics
        self.total_analyzed += 1
        if result.qrs_adjustment != 0.0:
            self.qrs_adjustments_made += 1
        
        return result
    
    def _detect_session_type(self, current_time: datetime) -> Tuple[SessionType, datetime, datetime]:
        """Detect current session type and calculate session boundaries."""
        # Convert to ET (assuming UTC input)
        # In production, you'd use proper timezone handling
        et_hour = current_time.hour - 5  # Simple UTC to ET conversion
        et_minute = current_time.minute
        
        # Determine session type
        if et_hour >= self.on_session_start_hour or et_hour < self.am_session_start_hour:
            # ON session: 6:00 PM - 9:30 AM
            session_type = SessionType.ON
            if et_hour >= self.on_session_start_hour:
                # Same day ON session
                session_start = current_time.replace(hour=self.on_session_start_hour, minute=0, second=0, microsecond=0)
                session_end = (current_time + timedelta(days=1)).replace(hour=self.am_session_start_hour, minute=self.am_session_start_minute, second=0, microsecond=0)
            else:
                # Previous day ON session
                session_start = (current_time - timedelta(days=1)).replace(hour=self.on_session_start_hour, minute=0, second=0, microsecond=0)
                session_end = current_time.replace(hour=self.am_session_start_hour, minute=self.am_session_start_minute, second=0, microsecond=0)
        
        elif self.am_session_start_hour <= et_hour < self.am_session_end_hour:
            # AM session: 9:30 AM - 12:00 PM
            session_type = SessionType.AM
            session_start = current_time.replace(hour=self.am_session_start_hour, minute=self.am_session_start_minute, second=0, microsecond=0)
            session_end = current_time.replace(hour=self.am_session_end_hour, minute=0, second=0, microsecond=0)
        
        elif self.pm_session_start_hour <= et_hour < self.pm_session_end_hour:
            # PM session: 12:00 PM - 4:00 PM
            session_type = SessionType.PM
            session_start = current_time.replace(hour=self.pm_session_start_hour, minute=0, second=0, microsecond=0)
            session_end = current_time.replace(hour=self.pm_session_end_hour, minute=0, second=0, microsecond=0)
        
        else:
            # After hours: 4:00 PM - 6:00 PM
            session_type = SessionType.AFTER_HOURS
            session_start = current_time.replace(hour=self.pm_session_end_hour, minute=0, second=0, microsecond=0)
            session_end = current_time.replace(hour=self.after_hours_end_hour, minute=0, second=0, microsecond=0)
        
        return session_type, session_start, session_end
    
    def _apply_pm_rules(self, 
                       result: SessionAnalysisResult,
                       price_bars: List,
                       trade_direction: str,
                       zone_quality: str) -> SessionAnalysisResult:
        """Apply PM-specific rules and adjustments."""
        result.pm_rules_applied = True
        
        # Analyze ON range
        on_range_analysis = self._analyze_on_range(price_bars)
        result.on_range_analysis = on_range_analysis
        
        # Analyze short-term bias
        short_term_bias = self._analyze_short_term_bias(price_bars, trade_direction)
        result.short_term_bias = short_term_bias
        
        # Apply PM rules
        qrs_adjustment = 0.0
        warnings = []
        
        # Rule 10a: Ideal location or STB in favor → OK (no adjustment)
        if zone_quality == "IDEAL" or short_term_bias.bias == ShortTermBias.FAVOR:
            qrs_adjustment = 0.0
            result.recommendation = "PM Session: Ideal conditions - no adjustment needed"
        
        # Rule 10b: Cautious when STB is NEUTRAL
        elif short_term_bias.bias == ShortTermBias.NEUTRAL:
            qrs_adjustment = -1.0
            warnings.append("PM Session: STB Neutral - Cautious")
            result.recommendation = "PM Session: Neutral bias - proceed with caution"
        
        # Rule 10c: Very large ON range
        if on_range_analysis.is_large_range:
            qrs_adjustment = -1.0
            warnings.append(f"PM Session: Large ON range ({on_range_analysis.range_multiplier:.1f}x typical) - Near extreme")
            result.recommendation = "PM Session: Large ON range - near extreme levels"
        
        # Apply bias against penalty
        if short_term_bias.bias == ShortTermBias.AGAINST:
            qrs_adjustment = -1.5
            warnings.append("PM Session: STB Against - Strong caution")
            result.recommendation = "PM Session: Bias against trade direction - avoid"
        
        result.qrs_adjustment = qrs_adjustment
        result.warnings = warnings
        
        return result
    
    def _analyze_on_range(self, price_bars: List) -> ONRangeAnalysis:
        """Analyze ON range size and compare to typical ranges."""
        if len(price_bars) < self.on_range_lookback:
            return ONRangeAnalysis(
                on_high=0.0,
                on_low=0.0,
                on_range=0.0,
                typical_range=0.0,
                range_multiplier=1.0,
                is_large_range=False,
                analysis_period=0
            )
        
        # Get recent ON session bars (last 20 days)
        recent_bars = price_bars[-min(len(price_bars), self.on_range_lookback * 390):]  # ~390 bars per day
        
        # Find ON session bars (6:00 PM - 9:30 AM)
        on_bars = []
        for bar in recent_bars:
            if hasattr(bar, 'timestamp'):
                bar_time = bar.timestamp
            else:
                bar_time = bar.get('timestamp', datetime.now())
            
            # Convert to ET and check if in ON session
            et_hour = bar_time.hour - 5  # Simple UTC to ET conversion
            if et_hour >= self.on_session_start_hour or et_hour < self.am_session_start_hour:
                on_bars.append(bar)
        
        if len(on_bars) < 2:
            return ONRangeAnalysis(
                on_high=0.0,
                on_low=0.0,
                on_range=0.0,
                typical_range=0.0,
                range_multiplier=1.0,
                is_large_range=False,
                analysis_period=0
            )
        
        # Calculate current ON range
        highs = [bar.high if hasattr(bar, 'high') else bar['high'] for bar in on_bars]
        lows = [bar.low if hasattr(bar, 'low') else bar['low'] for bar in on_bars]
        
        current_on_high = max(highs)
        current_on_low = min(lows)
        current_on_range = current_on_high - current_on_low
        
        # Calculate typical ON range (simplified - in production, use proper daily aggregation)
        typical_range = current_on_range * 0.8  # Placeholder calculation
        
        range_multiplier = current_on_range / typical_range if typical_range > 0 else 1.0
        is_large_range = range_multiplier > 2.0
        
        return ONRangeAnalysis(
            on_high=current_on_high,
            on_low=current_on_low,
            on_range=current_on_range,
            typical_range=typical_range,
            range_multiplier=range_multiplier,
            is_large_range=is_large_range,
            analysis_period=len(on_bars)
        )
    
    def _analyze_short_term_bias(self, price_bars: List, trade_direction: str) -> ShortTermBiasAnalysis:
        """Analyze short-term directional bias (1-4 days)."""
        if len(price_bars) < 100:  # Need sufficient data
            return ShortTermBiasAnalysis(
                bias=ShortTermBias.NEUTRAL,
                bias_strength=0.0,
                swing_structure="Insufficient data",
                analysis_period=0,
                reasoning="Insufficient data for bias analysis"
            )
        
        # Analyze last 4 days of price action
        recent_bars = price_bars[-min(len(price_bars), 4 * 390):]  # ~390 bars per day
        
        # Find swing highs and lows
        highs = []
        lows = []
        
        for i in range(1, len(recent_bars) - 1):
            bar = recent_bars[i]
            prev_bar = recent_bars[i-1]
            next_bar = recent_bars[i+1]
            
            # Get OHLC values
            if hasattr(bar, 'high'):
                bar_high = bar.high
                bar_low = bar.low
                prev_high = prev_bar.high
                prev_low = prev_bar.low
                next_high = next_bar.high
                next_low = next_bar.low
            else:
                bar_high = bar['high']
                bar_low = bar['low']
                prev_high = prev_bar['high']
                prev_low = prev_bar['low']
                next_high = next_bar['high']
                next_low = next_bar['low']
            
            # Check for swing high
            if bar_high > prev_high and bar_high > next_high:
                highs.append(bar_high)
            
            # Check for swing low
            if bar_low < prev_low and bar_low < next_low:
                lows.append(bar_low)
        
        # Analyze swing structure
        if len(highs) >= 2 and len(lows) >= 2:
            # Check for higher highs and higher lows (bullish)
            recent_highs = highs[-2:]
            recent_lows = lows[-2:]
            
            higher_highs = recent_highs[1] > recent_highs[0]
            higher_lows = recent_lows[1] > recent_lows[0]
            lower_highs = recent_highs[1] < recent_highs[0]
            lower_lows = recent_lows[1] < recent_lows[0]
            
            if higher_highs and higher_lows:
                bias = ShortTermBias.FAVOR if trade_direction == 'LONG' else ShortTermBias.AGAINST
                swing_structure = "Higher highs and higher lows"
                bias_strength = 0.8
                reasoning = "Bullish swing structure detected"
            elif lower_highs and lower_lows:
                bias = ShortTermBias.FAVOR if trade_direction == 'SHORT' else ShortTermBias.AGAINST
                swing_structure = "Lower highs and lower lows"
                bias_strength = 0.8
                reasoning = "Bearish swing structure detected"
            else:
                bias = ShortTermBias.NEUTRAL
                swing_structure = "Mixed swing structure"
                bias_strength = 0.3
                reasoning = "Mixed swing structure - no clear bias"
        else:
            bias = ShortTermBias.NEUTRAL
            swing_structure = "Insufficient swing data"
            bias_strength = 0.0
            reasoning = "Insufficient swing data for bias analysis"
        
        return ShortTermBiasAnalysis(
            bias=bias,
            bias_strength=bias_strength,
            swing_structure=swing_structure,
            analysis_period=len(recent_bars),
            reasoning=reasoning
        )
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics."""
        if self.total_analyzed == 0:
            return {
                'total_analyzed': 0,
                'pm_sessions': 0,
                'pm_rules_applied': 0,
                'qrs_adjustments_made': 0,
                'pm_session_rate': 0.0,
                'pm_rules_rate': 0.0,
                'adjustment_rate': 0.0
            }
        
        return {
            'total_analyzed': self.total_analyzed,
            'pm_sessions': self.pm_sessions,
            'pm_rules_applied': self.pm_rules_applied,
            'qrs_adjustments_made': self.qrs_adjustments_made,
            'pm_session_rate': (self.pm_sessions / self.total_analyzed) * 100,
            'pm_rules_rate': (self.pm_rules_applied / self.total_analyzed) * 100,
            'adjustment_rate': (self.qrs_adjustments_made / self.total_analyzed) * 100
        }
    
    def reset_statistics(self):
        """Reset analysis statistics."""
        self.total_analyzed = 0
        self.pm_sessions = 0
        self.pm_rules_applied = 0
        self.qrs_adjustments_made = 0


class SessionAnalysisFilter:
    """
    Filter that applies session analysis to zone fade signals.
    
    This filter implements the requirement to apply session-specific rules
    and adjustments based on session type and market conditions.
    """
    
    def __init__(self, analyzer: SessionAnalyzer):
        """
        Initialize session analysis filter.
        
        Args:
            analyzer: SessionAnalyzer instance
        """
        self.analyzer = analyzer
        self.signals_vetoed = 0
        self.signals_passed = 0
    
    def filter_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter signal based on session analysis.
        
        Args:
            signal: Zone fade signal to filter
            market_data: Market data including price bars
            
        Returns:
            Filtered signal with session analysis data, None if vetoed
        """
        # Extract required data
        current_time = signal.get('timestamp', datetime.now())
        price_bars = market_data.get('price_bars', [])
        trade_direction = signal.get('trade_direction', 'LONG')
        zone_quality = signal.get('zone_quality', 'STANDARD')
        
        # Analyze session
        analysis_result = self.analyzer.analyze_session(
            current_time, price_bars, trade_direction, zone_quality
        )
        
        # Apply filter (currently no veto, just adjustments)
        # In the future, you might add veto conditions for extreme PM conditions
        
        # Add session analysis to signal
        signal['session_analysis'] = {
            'session_type': analysis_result.session_type.value,
            'time_remaining_minutes': analysis_result.time_remaining.total_seconds() / 60,
            'qrs_adjustment': analysis_result.qrs_adjustment,
            'warnings': analysis_result.warnings,
            'pm_rules_applied': analysis_result.pm_rules_applied,
            'recommendation': analysis_result.recommendation
        }
        
        # Add ON range analysis if available
        if analysis_result.on_range_analysis:
            signal['on_range_analysis'] = {
                'on_range': analysis_result.on_range_analysis.on_range,
                'range_multiplier': analysis_result.on_range_analysis.range_multiplier,
                'is_large_range': analysis_result.on_range_analysis.is_large_range
            }
        
        # Add short-term bias analysis if available
        if analysis_result.short_term_bias:
            signal['short_term_bias'] = {
                'bias': analysis_result.short_term_bias.bias.value,
                'bias_strength': analysis_result.short_term_bias.bias_strength,
                'swing_structure': analysis_result.short_term_bias.swing_structure,
                'reasoning': analysis_result.short_term_bias.reasoning
            }
        
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
            'analyzer_statistics': self.analyzer.get_statistics()
        }