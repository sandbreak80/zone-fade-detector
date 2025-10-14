"""
Zone Approach Analysis

This module implements zone approach analysis to detect balance/consolidation
before zone touches, filtering out low-probability setups.

Features:
- Balance detection before zone approaches
- ATR compression analysis (10-bar lookback)
- Approach quality scoring
- Filter low-probability setups
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class ApproachQuality(Enum):
    """Zone approach quality classifications."""
    EXCELLENT = "EXCELLENT"  # Clean, direct approach
    GOOD = "GOOD"           # Decent approach with minor issues
    POOR = "POOR"           # Choppy or balanced approach
    UNACCEPTABLE = "UNACCEPTABLE"  # Heavy balance/consolidation

@dataclass
class BalanceDetection:
    """Balance detection result."""
    has_balance: bool
    atr_ratio: float
    pre_zone_atr: float
    avg_atr: float
    compression_percentage: float
    lookback_period: int
    balance_strength: float  # 0.0-1.0

@dataclass
class ApproachAnalysis:
    """Complete approach analysis result."""
    quality: ApproachQuality
    balance_detection: BalanceDetection
    momentum_score: float  # 0.0-1.0
    cleanliness_score: float  # 0.0-1.0
    overall_score: float  # 0.0-1.0
    recommendation: str
    details: Dict[str, Any]

class ZoneApproachAnalyzer:
    """
    Analyzes zone approaches to detect balance/consolidation patterns.
    
    This analyzer implements the requirement to filter out low-probability
    setups where price has been consolidating/balancing before approaching
    the zone, as these have higher breakout probability.
    """
    
    def __init__(self,
                 lookback_period: int = 10,
                 balance_threshold: float = 0.7,
                 atr_period: int = 14,
                 baseline_period: int = 20,
                 momentum_lookback: int = 5):
        """
        Initialize zone approach analyzer.
        
        Args:
            lookback_period: Bars to analyze before zone touch
            balance_threshold: ATR ratio threshold for balance detection
            atr_period: Period for ATR calculation
            baseline_period: Period for baseline ATR calculation
            momentum_lookback: Bars to analyze for momentum
        """
        self.lookback_period = lookback_period
        self.balance_threshold = balance_threshold
        self.atr_period = atr_period
        self.baseline_period = baseline_period
        self.momentum_lookback = momentum_lookback
        
        # Statistics tracking
        self.total_analyzed = 0
        self.balance_detected = 0
        self.clean_approaches = 0
        self.excellent_approaches = 0
    
    def analyze_approach(self, 
                        price_bars: List,
                        zone_touch_index: int,
                        zone_level: float,
                        trade_direction: str) -> ApproachAnalysis:
        """
        Analyze the approach to a zone for balance/consolidation patterns.
        
        Args:
            price_bars: List of OHLCV bars
            zone_touch_index: Index of the zone touch bar
            zone_level: Price level of the zone
            trade_direction: 'LONG' or 'SHORT'
            
        Returns:
            ApproachAnalysis with quality assessment
        """
        if zone_touch_index < self.lookback_period + self.baseline_period:
            return self._create_insufficient_data_result()
        
        # Detect balance before zone approach
        balance_detection = self._detect_balance(price_bars, zone_touch_index)
        
        # Analyze momentum
        momentum_score = self._analyze_momentum(price_bars, zone_touch_index, trade_direction)
        
        # Analyze approach cleanliness
        cleanliness_score = self._analyze_cleanliness(price_bars, zone_touch_index, zone_level, trade_direction)
        
        # Determine overall quality
        quality, overall_score, recommendation = self._assess_quality(
            balance_detection, momentum_score, cleanliness_score
        )
        
        # Update statistics
        self.total_analyzed += 1
        if balance_detection.has_balance:
            self.balance_detected += 1
        if cleanliness_score >= 0.8:
            self.clean_approaches += 1
        if overall_score >= 0.9:
            self.excellent_approaches += 1
        
        return ApproachAnalysis(
            quality=quality,
            balance_detection=balance_detection,
            momentum_score=momentum_score,
            cleanliness_score=cleanliness_score,
            overall_score=overall_score,
            recommendation=recommendation,
            details={
                'lookback_period': self.lookback_period,
                'balance_threshold': self.balance_threshold,
                'zone_level': zone_level,
                'trade_direction': trade_direction
            }
        )
    
    def _detect_balance(self, price_bars: List, zone_touch_index: int) -> BalanceDetection:
        """Detect balance/consolidation before zone approach."""
        # Get pre-zone period (lookback_period bars before touch)
        pre_zone_start = zone_touch_index - self.lookback_period
        pre_zone_end = zone_touch_index
        
        if pre_zone_start < 0:
            return BalanceDetection(
                has_balance=False,
                atr_ratio=1.0,
                pre_zone_atr=0.0,
                avg_atr=0.0,
                compression_percentage=0.0,
                lookback_period=0,
                balance_strength=0.0
            )
        
        pre_zone_bars = price_bars[pre_zone_start:pre_zone_end]
        pre_zone_atr = self._calculate_atr(pre_zone_bars)
        
        # Get baseline period (baseline_period bars before pre-zone)
        baseline_start = pre_zone_start - self.baseline_period
        baseline_end = pre_zone_start
        
        if baseline_start < 0:
            baseline_start = 0
        
        baseline_bars = price_bars[baseline_start:baseline_end]
        avg_atr = self._calculate_atr(baseline_bars)
        
        # Calculate ATR ratio
        atr_ratio = pre_zone_atr / avg_atr if avg_atr > 0 else 1.0
        compression_percentage = (1.0 - atr_ratio) * 100
        
        # Determine if balance is present
        has_balance = atr_ratio < self.balance_threshold
        
        # Calculate balance strength (0.0 = no balance, 1.0 = heavy balance)
        balance_strength = max(0.0, (self.balance_threshold - atr_ratio) / self.balance_threshold)
        
        return BalanceDetection(
            has_balance=has_balance,
            atr_ratio=atr_ratio,
            pre_zone_atr=pre_zone_atr,
            avg_atr=avg_atr,
            compression_percentage=compression_percentage,
            lookback_period=self.lookback_period,
            balance_strength=balance_strength
        )
    
    def _analyze_momentum(self, price_bars: List, zone_touch_index: int, trade_direction: str) -> float:
        """Analyze momentum in the approach to the zone."""
        if zone_touch_index < self.momentum_lookback:
            return 0.5  # Neutral if insufficient data
        
        # Get momentum period
        momentum_start = zone_touch_index - self.momentum_lookback
        momentum_bars = price_bars[momentum_start:zone_touch_index]
        
        if len(momentum_bars) < 2:
            return 0.5
        
        # Calculate directional consistency
        directional_bars = 0
        total_bars = len(momentum_bars)
        
        for bar in momentum_bars:
            if hasattr(bar, 'close') and hasattr(bar, 'open'):
                if trade_direction == 'LONG':
                    # For long trades, we want upward momentum
                    if bar.close > bar.open:
                        directional_bars += 1
                else:  # SHORT
                    # For short trades, we want downward momentum
                    if bar.close < bar.open:
                        directional_bars += 1
            else:
                # Handle dictionary format
                if trade_direction == 'LONG':
                    if bar['close'] > bar['open']:
                        directional_bars += 1
                else:  # SHORT
                    if bar['close'] < bar['open']:
                        directional_bars += 1
        
        # Calculate momentum score (0.0-1.0)
        momentum_score = directional_bars / total_bars if total_bars > 0 else 0.5
        
        return momentum_score
    
    def _analyze_cleanliness(self, price_bars: List, zone_touch_index: int, 
                           zone_level: float, trade_direction: str) -> float:
        """Analyze the cleanliness of the approach to the zone."""
        if zone_touch_index < self.lookback_period:
            return 0.5  # Neutral if insufficient data
        
        # Get approach period
        approach_start = zone_touch_index - self.lookback_period
        approach_bars = price_bars[approach_start:zone_touch_index]
        
        if len(approach_bars) < 2:
            return 0.5
        
        # Calculate price range and volatility
        highs = [bar.high if hasattr(bar, 'high') else bar['high'] for bar in approach_bars]
        lows = [bar.low if hasattr(bar, 'low') else bar['low'] for bar in approach_bars]
        
        price_range = max(highs) - min(lows)
        
        # Calculate average bar range
        bar_ranges = []
        for bar in approach_bars:
            if hasattr(bar, 'high') and hasattr(bar, 'low'):
                bar_range = bar.high - bar.low
            else:
                bar_range = bar['high'] - bar['low']
            bar_ranges.append(bar_range)
        
        avg_bar_range = np.mean(bar_ranges) if bar_ranges else 0.0
        
        # Calculate cleanliness score
        # Clean approaches have consistent, moderate volatility
        if avg_bar_range == 0:
            cleanliness_score = 0.5
        else:
            # Score based on consistency of bar ranges
            range_consistency = 1.0 - (np.std(bar_ranges) / avg_bar_range) if avg_bar_range > 0 else 0.5
            cleanliness_score = min(1.0, max(0.0, range_consistency))
        
        return cleanliness_score
    
    def _assess_quality(self, balance_detection: BalanceDetection, 
                       momentum_score: float, cleanliness_score: float) -> Tuple[ApproachQuality, float, str]:
        """Assess overall approach quality."""
        # Base score from cleanliness and momentum
        base_score = (cleanliness_score + momentum_score) / 2
        
        # Apply balance penalty
        if balance_detection.has_balance:
            # Heavy penalty for balance detection
            balance_penalty = balance_detection.balance_strength * 0.5
            final_score = max(0.0, base_score - balance_penalty)
            recommendation = f"Balance detected (ATR ratio: {balance_detection.atr_ratio:.2f}) - SKIP setup"
        else:
            final_score = base_score
            recommendation = "Clean approach - proceed with setup"
        
        # Determine quality grade
        if final_score >= 0.9:
            quality = ApproachQuality.EXCELLENT
        elif final_score >= 0.7:
            quality = ApproachQuality.GOOD
        elif final_score >= 0.5:
            quality = ApproachQuality.POOR
        else:
            quality = ApproachQuality.UNACCEPTABLE
        
        return quality, final_score, recommendation
    
    def _calculate_atr(self, bars: List, period: int = None) -> float:
        """Calculate Average True Range."""
        if not bars or len(bars) < 2:
            return 0.0
        
        period = period or self.atr_period
        if len(bars) < period:
            period = len(bars)
        
        true_ranges = []
        
        for i in range(1, len(bars)):
            current = bars[i]
            previous = bars[i-1]
            
            # Get OHLC values
            if hasattr(current, 'high'):
                high = current.high
                low = current.low
                prev_close = previous.close
            else:
                high = current['high']
                low = current['low']
                prev_close = previous['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        return np.mean(true_ranges) if true_ranges else 0.0
    
    def _create_insufficient_data_result(self) -> ApproachAnalysis:
        """Create result for insufficient data."""
        return ApproachAnalysis(
            quality=ApproachQuality.POOR,
            balance_detection=BalanceDetection(
                has_balance=False,
                atr_ratio=1.0,
                pre_zone_atr=0.0,
                avg_atr=0.0,
                compression_percentage=0.0,
                lookback_period=0,
                balance_strength=0.0
            ),
            momentum_score=0.5,
            cleanliness_score=0.5,
            overall_score=0.5,
            recommendation="Insufficient data for analysis",
            details={'error': 'insufficient_data'}
        )
    
    def should_filter_setup(self, analysis: ApproachAnalysis) -> bool:
        """Check if setup should be filtered based on approach analysis."""
        return analysis.balance_detection.has_balance or analysis.quality == ApproachQuality.UNACCEPTABLE
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics."""
        if self.total_analyzed == 0:
            return {
                'total_analyzed': 0,
                'balance_detected': 0,
                'clean_approaches': 0,
                'excellent_approaches': 0,
                'balance_detection_rate': 0.0,
                'clean_approach_rate': 0.0,
                'excellent_approach_rate': 0.0
            }
        
        return {
            'total_analyzed': self.total_analyzed,
            'balance_detected': self.balance_detected,
            'clean_approaches': self.clean_approaches,
            'excellent_approaches': self.excellent_approaches,
            'balance_detection_rate': (self.balance_detected / self.total_analyzed) * 100,
            'clean_approach_rate': (self.clean_approaches / self.total_analyzed) * 100,
            'excellent_approach_rate': (self.excellent_approaches / self.total_analyzed) * 100
        }
    
    def reset_statistics(self):
        """Reset analysis statistics."""
        self.total_analyzed = 0
        self.balance_detected = 0
        self.clean_approaches = 0
        self.excellent_approaches = 0


class ZoneApproachFilter:
    """
    Filter that applies zone approach analysis to zone fade signals.
    
    This filter implements the requirement to skip setups where balance
    is detected before zone approach, as these have higher breakout probability.
    """
    
    def __init__(self, analyzer: ZoneApproachAnalyzer):
        """
        Initialize zone approach filter.
        
        Args:
            analyzer: ZoneApproachAnalyzer instance
        """
        self.analyzer = analyzer
        self.signals_vetoed = 0
        self.signals_passed = 0
    
    def filter_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter signal based on zone approach analysis.
        
        Args:
            signal: Zone fade signal to filter
            market_data: Market data including price bars
            
        Returns:
            Filtered signal if approach is acceptable, None if vetoed
        """
        # Extract required data
        price_bars = market_data.get('price_bars', [])
        zone_touch_index = signal.get('zone_touch_index', 0)
        zone_level = signal.get('zone_level', 0.0)
        trade_direction = signal.get('trade_direction', 'LONG')
        
        # Analyze approach
        analysis = self.analyzer.analyze_approach(
            price_bars, zone_touch_index, zone_level, trade_direction
        )
        
        # Apply filter
        if self.analyzer.should_filter_setup(analysis):
            self.signals_vetoed += 1
            return None  # VETO: Balance detected or poor approach
        
        # Add approach analysis to signal
        signal['approach_analysis'] = {
            'quality': analysis.quality.value,
            'has_balance': analysis.balance_detection.has_balance,
            'atr_ratio': analysis.balance_detection.atr_ratio,
            'momentum_score': analysis.momentum_score,
            'cleanliness_score': analysis.cleanliness_score,
            'overall_score': analysis.overall_score,
            'recommendation': analysis.recommendation
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