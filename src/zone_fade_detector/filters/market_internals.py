"""
Market Internals Monitoring

This module implements market internals monitoring to ensure zone fade signals
are only generated when market conditions favor responsive (fading) behavior.

Features:
- NYSE TICK analysis for balanced vs skewed conditions
- Advance/Decline Line analysis for flat vs trending conditions
- Internals favorability check for zone fading
- QRS Factor 3 scoring with veto power
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class TICKStatus(Enum):
    """NYSE TICK status classifications."""
    BALANCED = "BALANCED"
    SKEWED_POSITIVE = "SKEWED_POSITIVE"
    SKEWED_NEGATIVE = "SKEWED_NEGATIVE"

class ADStatus(Enum):
    """Advance/Decline Line status classifications."""
    FLAT = "FLAT"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"

@dataclass
class TICKAnalysis:
    """NYSE TICK analysis result."""
    status: TICKStatus
    mean_value: float
    std_deviation: float
    window_size: int
    timestamp: datetime

@dataclass
class ADAnalysis:
    """Advance/Decline Line analysis result."""
    status: ADStatus
    slope: float
    r_squared: float
    window_size: int
    timestamp: datetime

@dataclass
class InternalsResult:
    """Market internals analysis result."""
    is_favorable: bool
    quality_score: float  # 0.0, 1.0, or 2.0 for QRS Factor 3
    tick_analysis: TICKAnalysis
    ad_analysis: ADAnalysis
    recommendation: str
    timestamp: datetime

class MarketInternalsMonitor:
    """
    Monitors market internals to determine if conditions favor zone fading.
    
    Zone fade strategies work best when market internals are balanced,
    indicating responsive (mean-reverting) behavior rather than initiative
    (trending) behavior.
    """
    
    def __init__(self,
                 tick_threshold: float = 200.0,
                 ad_slope_threshold: float = 100.0,
                 tick_window: int = 30,
                 ad_window: int = 60):
        """
        Initialize market internals monitor.
        
        Args:
            tick_threshold: TICK threshold for balanced vs skewed classification
            ad_slope_threshold: A/D Line slope threshold for flat vs trending
            tick_window: Window size for TICK analysis
            ad_window: Window size for A/D Line analysis
        """
        self.tick_threshold = tick_threshold
        self.ad_slope_threshold = ad_slope_threshold
        self.tick_window = tick_window
        self.ad_window = ad_window
        
        # Statistics tracking
        self.favorable_checks = 0
        self.unfavorable_checks = 0
        self.total_checks = 0
    
    def check_fade_conditions(self, 
                             tick_data: List[float],
                             ad_line_data: List[float]) -> InternalsResult:
        """
        Check if market internals favor zone fading.
        
        Args:
            tick_data: List of NYSE TICK values
            ad_line_data: List of Advance/Decline Line values
            
        Returns:
            InternalsResult with favorability assessment
        """
        # Analyze TICK
        tick_analysis = self._analyze_tick(tick_data)
        
        # Analyze A/D Line
        ad_analysis = self._analyze_ad_line(ad_line_data)
        
        # Determine favorability
        is_favorable, quality_score, recommendation = self._assess_favorability(
            tick_analysis, ad_analysis
        )
        
        # Update statistics
        self.total_checks += 1
        if is_favorable:
            self.favorable_checks += 1
        else:
            self.unfavorable_checks += 1
        
        return InternalsResult(
            is_favorable=is_favorable,
            quality_score=quality_score,
            tick_analysis=tick_analysis,
            ad_analysis=ad_analysis,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
    
    def _analyze_tick(self, tick_data: List[float]) -> TICKAnalysis:
        """Analyze NYSE TICK for balanced vs skewed conditions."""
        if len(tick_data) < self.tick_window:
            return TICKAnalysis(
                status=TICKStatus.BALANCED,
                mean_value=0.0,
                std_deviation=0.0,
                window_size=len(tick_data),
                timestamp=datetime.now()
            )
        
        recent_ticks = tick_data[-self.tick_window:]
        mean_value = np.mean(recent_ticks)
        std_deviation = np.std(recent_ticks)
        
        # Classify TICK status
        if abs(mean_value) <= self.tick_threshold:
            status = TICKStatus.BALANCED
        elif mean_value > self.tick_threshold:
            status = TICKStatus.SKEWED_POSITIVE
        else:
            status = TICKStatus.SKEWED_NEGATIVE
        
        return TICKAnalysis(
            status=status,
            mean_value=mean_value,
            std_deviation=std_deviation,
            window_size=self.tick_window,
            timestamp=datetime.now()
        )
    
    def _analyze_ad_line(self, ad_line_data: List[float]) -> ADAnalysis:
        """Analyze Advance/Decline Line for flat vs trending conditions."""
        if len(ad_line_data) < self.ad_window:
            return ADAnalysis(
                status=ADStatus.FLAT,
                slope=0.0,
                r_squared=0.0,
                window_size=len(ad_line_data),
                timestamp=datetime.now()
            )
        
        recent_ad = ad_line_data[-self.ad_window:]
        x = np.arange(len(recent_ad))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = np.polyfit(x, recent_ad, 1, full=False)
        r_squared = r_value ** 2
        
        # Classify A/D status
        slope_per_bar = slope  # slope is already per bar
        if abs(slope_per_bar) < self.ad_slope_threshold:
            status = ADStatus.FLAT
        elif slope_per_bar >= self.ad_slope_threshold:
            status = ADStatus.TRENDING_UP
        else:
            status = ADStatus.TRENDING_DOWN
        
        return ADAnalysis(
            status=status,
            slope=slope,
            r_squared=r_squared,
            window_size=self.ad_window,
            timestamp=datetime.now()
        )
    
    def _assess_favorability(self, 
                           tick_analysis: TICKAnalysis,
                           ad_analysis: ADAnalysis) -> Tuple[bool, float, str]:
        """
        Assess if internals favor zone fading.
        
        Returns:
            Tuple of (is_favorable, quality_score, recommendation)
        """
        # Perfect conditions: TICK balanced AND A/D flat
        if (tick_analysis.status == TICKStatus.BALANCED and 
            ad_analysis.status == ADStatus.FLAT):
            return True, 2.0, "OK to fade - balanced internals"
        
        # Mixed conditions: One favorable, one not
        elif (tick_analysis.status == TICKStatus.BALANCED or 
              ad_analysis.status == ADStatus.FLAT):
            return False, 1.0, "Mixed internals - caution advised"
        
        # Unfavorable conditions: Both showing initiative activity
        else:
            return False, 0.0, "Initiative activity - do not fade"
    
    def is_favorable(self, result: InternalsResult) -> bool:
        """Check if internals are favorable for zone fading."""
        return result.is_favorable
    
    def get_quality_score(self, result: InternalsResult) -> float:
        """Get quality score for QRS Factor 3."""
        return result.quality_score
    
    def should_veto(self, result: InternalsResult) -> bool:
        """Check if signal should be vetoed based on internals."""
        return result.quality_score == 0.0
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        if self.total_checks == 0:
            return {
                'total_checks': 0,
                'favorable_checks': 0,
                'unfavorable_checks': 0,
                'favorable_percentage': 0.0,
                'unfavorable_percentage': 0.0
            }
        
        return {
            'total_checks': self.total_checks,
            'favorable_checks': self.favorable_checks,
            'unfavorable_checks': self.unfavorable_checks,
            'favorable_percentage': (self.favorable_checks / self.total_checks) * 100,
            'unfavorable_percentage': (self.unfavorable_checks / self.total_checks) * 100
        }
    
    def reset_statistics(self):
        """Reset monitoring statistics."""
        self.favorable_checks = 0
        self.unfavorable_checks = 0
        self.total_checks = 0


class InternalsFilter:
    """
    Filter that applies market internals check to zone fade signals.
    
    This filter ensures signals are only generated when market internals
    favor responsive (fading) behavior rather than initiative (trending) behavior.
    """
    
    def __init__(self, monitor: MarketInternalsMonitor):
        """
        Initialize internals filter.
        
        Args:
            monitor: MarketInternalsMonitor instance
        """
        self.monitor = monitor
        self.signals_vetoed = 0
        self.signals_passed = 0
    
    def filter_signal(self, signal, market_data: Dict) -> Optional[Dict]:
        """
        Filter signal based on market internals.
        
        Args:
            signal: Zone fade signal to filter
            market_data: Market data including TICK and A/D Line data
            
        Returns:
            Filtered signal if internals are favorable, None if vetoed
        """
        # Check internals
        result = self.monitor.check_fade_conditions(
            tick_data=market_data.get('tick_data', []),
            ad_line_data=market_data.get('ad_line_data', [])
        )
        
        # Apply filter
        if self.monitor.should_veto(result):
            self.signals_vetoed += 1
            return None  # VETO: Initiative activity detected
        
        # Add internals info to signal
        if isinstance(signal, dict):
            signal['internals_favorable'] = result.is_favorable
            signal['internals_quality_score'] = result.quality_score
            signal['tick_status'] = result.tick_analysis.status.value
            signal['tick_mean'] = result.tick_analysis.mean_value
            signal['ad_status'] = result.ad_analysis.status.value
            signal['ad_slope'] = result.ad_analysis.slope
            signal['internals_recommendation'] = result.recommendation
        else:
            # If signal is an object, add attributes
            signal.internals_favorable = result.is_favorable
            signal.internals_quality_score = result.quality_score
            signal.tick_status = result.tick_analysis.status.value
            signal.tick_mean = result.tick_analysis.mean_value
            signal.ad_status = result.ad_analysis.status.value
            signal.ad_slope = result.ad_analysis.slope
            signal.internals_recommendation = result.recommendation
        
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
            'monitor_statistics': self.monitor.get_statistics()
        }


class DataProvider:
    """
    Mock data provider for NYSE TICK and A/D Line data.
    
    In production, this would connect to real data feeds.
    """
    
    def __init__(self):
        """Initialize data provider."""
        self.tick_data = []
        self.ad_line_data = []
    
    def add_tick_data(self, tick_value: float):
        """Add TICK data point."""
        self.tick_data.append(tick_value)
        # Keep only recent data
        if len(self.tick_data) > 1000:
            self.tick_data = self.tick_data[-500:]
    
    def add_ad_line_data(self, ad_value: float):
        """Add A/D Line data point."""
        self.ad_line_data.append(ad_value)
        # Keep only recent data
        if len(self.ad_line_data) > 1000:
            self.ad_line_data = self.ad_line_data[-500:]
    
    def get_tick_data(self, window: int = 30) -> List[float]:
        """Get recent TICK data."""
        return self.tick_data[-window:] if len(self.tick_data) >= window else self.tick_data
    
    def get_ad_line_data(self, window: int = 60) -> List[float]:
        """Get recent A/D Line data."""
        return self.ad_line_data[-window:] if len(self.ad_line_data) >= window else self.ad_line_data