"""
Entry Optimization

This module implements entry optimization to calculate optimal entry prices
and validate risk/reward ratios for zone fade setups.

Features:
- Zone position classification (front/middle/back)
- Optimal entry price calculation
- Risk/reward ratio validation
- Setup type specific logic (ZFR vs ZF-TR)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class ZonePosition(Enum):
    """Zone position classifications."""
    FRONT = "FRONT"      # 0-33% into zone
    MIDDLE = "MIDDLE"    # 33-67% into zone
    BACK = "BACK"        # 67-100% into zone

class SetupType(Enum):
    """Setup type classifications."""
    ZFR = "ZFR"          # Aggressive 3+R setup
    ZF_TR = "ZF-TR"      # Defensive 2-3R setup

@dataclass
class ZonePositionAnalysis:
    """Zone position analysis result."""
    position: ZonePosition
    position_percentage: float
    distance_from_front: float
    distance_from_back: float
    zone_range: float

@dataclass
class EntryCalculation:
    """Entry price calculation result."""
    optimal_entry_price: float
    entry_position: ZonePosition
    setup_type: SetupType
    target_percentage: float
    distance_to_boundaries: Dict[str, float]
    rationale: str

@dataclass
class RiskRewardAnalysis:
    """Risk/reward analysis result."""
    stop_loss_price: float
    target_price: float
    risk_points: float
    reward_points: float
    risk_reward_ratio: float
    meets_minimum: bool
    minimum_required: float
    validation_result: str

@dataclass
class EntryOptimizationResult:
    """Complete entry optimization result."""
    zone_position: ZonePositionAnalysis
    entry_calculation: EntryCalculation
    risk_reward: RiskRewardAnalysis
    is_optimized: bool
    recommendation: str
    qrs_adjustment: float  # QRS score adjustment based on entry quality

class EntryOptimizer:
    """
    Optimizes entry prices and validates risk/reward ratios for zone fade setups.
    
    This optimizer implements the requirement to calculate optimal entry prices
    based on setup type and zone position, ensuring proper risk/reward ratios.
    """
    
    def __init__(self,
                 tick_size: float = 0.25,
                 zfr_entry_pct: float = 0.25,
                 zf_tr_entry_pct: float = 0.60,
                 zfr_min_rr: float = 3.0,
                 zf_tr_min_rr: float = 2.0):
        """
        Initialize entry optimizer.
        
        Args:
            tick_size: Minimum price increment (0.25 for /ES, /NQ, /RTY)
            zfr_entry_pct: Target entry percentage for ZFR setups (front entry)
            zf_tr_entry_pct: Target entry percentage for ZF-TR setups (middle-back entry)
            zfr_min_rr: Minimum R:R ratio for ZFR setups
            zf_tr_min_rr: Minimum R:R ratio for ZF-TR setups
        """
        self.tick_size = tick_size
        self.zfr_entry_pct = zfr_entry_pct
        self.zf_tr_entry_pct = zf_tr_entry_pct
        self.zfr_min_rr = zfr_min_rr
        self.zf_tr_min_rr = zf_tr_min_rr
        
        # Statistics
        self.total_optimized = 0
        self.zfr_optimizations = 0
        self.zf_tr_optimizations = 0
        self.valid_rr_ratios = 0
        self.invalid_rr_ratios = 0
    
    def optimize_entry(self, 
                      zone_high: float,
                      zone_low: float,
                      current_price: float,
                      setup_type: str,
                      trade_direction: str,
                      target_zone_high: Optional[float] = None,
                      target_zone_low: Optional[float] = None) -> EntryOptimizationResult:
        """
        Optimize entry for a zone fade setup.
        
        Args:
            zone_high: High boundary of the zone
            zone_low: Low boundary of the zone
            current_price: Current market price
            setup_type: 'ZFR' or 'ZF-TR'
            trade_direction: 'LONG' or 'SHORT'
            target_zone_high: High boundary of target zone (for R:R calculation)
            target_zone_low: Low boundary of target zone (for R:R calculation)
            
        Returns:
            EntryOptimizationResult with optimization details
        """
        # Analyze zone position
        zone_position = self._analyze_zone_position(zone_high, zone_low, current_price)
        
        # Calculate optimal entry price
        entry_calculation = self._calculate_optimal_entry(
            zone_high, zone_low, setup_type, trade_direction
        )
        
        # Calculate risk/reward ratio
        risk_reward = self._calculate_risk_reward(
            entry_calculation.optimal_entry_price,
            zone_high, zone_low,
            trade_direction,
            target_zone_high, target_zone_low,
            setup_type
        )
        
        # Determine if setup is optimized
        is_optimized = self._is_setup_optimized(zone_position, entry_calculation, risk_reward)
        
        # Calculate QRS adjustment
        qrs_adjustment = self._calculate_qrs_adjustment(zone_position, entry_calculation, risk_reward)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            zone_position, entry_calculation, risk_reward, is_optimized
        )
        
        # Update statistics
        self.total_optimized += 1
        if setup_type == 'ZFR':
            self.zfr_optimizations += 1
        else:
            self.zf_tr_optimizations += 1
        
        if risk_reward.meets_minimum:
            self.valid_rr_ratios += 1
        else:
            self.invalid_rr_ratios += 1
        
        return EntryOptimizationResult(
            zone_position=zone_position,
            entry_calculation=entry_calculation,
            risk_reward=risk_reward,
            is_optimized=is_optimized,
            recommendation=recommendation,
            qrs_adjustment=qrs_adjustment
        )
    
    def _analyze_zone_position(self, zone_high: float, zone_low: float, current_price: float) -> ZonePositionAnalysis:
        """Analyze current price position within the zone."""
        zone_range = zone_high - zone_low
        
        if zone_range <= 0:
            return ZonePositionAnalysis(
                position=ZonePosition.MIDDLE,
                position_percentage=0.5,
                distance_from_front=0.0,
                distance_from_back=0.0,
                zone_range=0.0
            )
        
        # Calculate position percentage (0.0 = at low, 1.0 = at high)
        position_percentage = (current_price - zone_low) / zone_range
        
        # Determine position classification
        if position_percentage <= 0.33:
            position = ZonePosition.FRONT
        elif position_percentage <= 0.67:
            position = ZonePosition.MIDDLE
        else:
            position = ZonePosition.BACK
        
        # Calculate distances
        distance_from_front = current_price - zone_low
        distance_from_back = zone_high - current_price
        
        return ZonePositionAnalysis(
            position=position,
            position_percentage=position_percentage,
            distance_from_front=distance_from_front,
            distance_from_back=distance_from_back,
            zone_range=zone_range
        )
    
    def _calculate_optimal_entry(self, 
                               zone_high: float, 
                               zone_low: float, 
                               setup_type: str, 
                               trade_direction: str) -> EntryCalculation:
        """Calculate optimal entry price based on setup type."""
        zone_range = zone_high - zone_low
        
        # Determine target percentage based on setup type
        if setup_type == 'ZFR':
            target_percentage = self.zfr_entry_pct
            setup_enum = SetupType.ZFR
            rationale = "ZFR: Aggressive front entry for 3+R setup"
        else:  # ZF-TR
            target_percentage = self.zf_tr_entry_pct
            setup_enum = SetupType.ZF-TR
            rationale = "ZF-TR: Defensive middle-back entry for 2-3R setup"
        
        # Calculate optimal entry price
        if trade_direction == 'LONG':
            # For long trades (fading at support), we want to enter from the low side
            optimal_entry_price = zone_low + (zone_range * target_percentage)
        else:  # SHORT
            # For short trades (fading at resistance), we want to enter from the high side
            optimal_entry_price = zone_high - (zone_range * target_percentage)
        
        # Round to nearest tick
        optimal_entry_price = round(optimal_entry_price / self.tick_size) * self.tick_size
        
        # Calculate distances to boundaries
        distance_to_boundaries = {
            'to_zone_low': abs(optimal_entry_price - zone_low),
            'to_zone_high': abs(optimal_entry_price - zone_high)
        }
        
        return EntryCalculation(
            optimal_entry_price=optimal_entry_price,
            entry_position=self._classify_entry_position(target_percentage),
            setup_type=setup_enum,
            target_percentage=target_percentage,
            distance_to_boundaries=distance_to_boundaries,
            rationale=rationale
        )
    
    def _classify_entry_position(self, target_percentage: float) -> ZonePosition:
        """Classify entry position based on target percentage."""
        if target_percentage <= 0.33:
            return ZonePosition.FRONT
        elif target_percentage <= 0.67:
            return ZonePosition.MIDDLE
        else:
            return ZonePosition.BACK
    
    def _calculate_risk_reward(self, 
                             entry_price: float,
                             zone_high: float,
                             zone_low: float,
                             trade_direction: str,
                             target_zone_high: Optional[float],
                             target_zone_low: Optional[float],
                             setup_type: str) -> RiskRewardAnalysis:
        """Calculate risk/reward ratio for the setup."""
        # Calculate stop loss (just beyond zone boundary)
        if trade_direction == 'LONG':
            stop_loss_price = zone_low - (self.tick_size * 2)  # 2 ticks below zone
        else:  # SHORT
            stop_loss_price = zone_high + (self.tick_size * 2)  # 2 ticks above zone
        
        # Calculate risk
        risk_points = abs(entry_price - stop_loss_price)
        
        # Calculate target price
        if target_zone_high is not None and target_zone_low is not None:
            # Use provided target zone
            if trade_direction == 'LONG':
                target_price = target_zone_high
            else:  # SHORT
                target_price = target_zone_low
        else:
            # Use opposite side of current zone as target
            if trade_direction == 'LONG':
                target_price = zone_high
            else:  # SHORT
                target_price = zone_low
        
        # Calculate reward
        reward_points = abs(target_price - entry_price)
        
        # Calculate risk/reward ratio
        risk_reward_ratio = reward_points / risk_points if risk_points > 0 else 0.0
        
        # Check if meets minimum requirement
        min_required = self.zfr_min_rr if setup_type == 'ZFR' else self.zf_tr_min_rr
        meets_minimum = risk_reward_ratio >= min_required
        
        # Generate validation result
        if meets_minimum:
            validation_result = f"✓ R:R {risk_reward_ratio:.1f} meets minimum {min_required:.1f}"
        else:
            validation_result = f"✗ R:R {risk_reward_ratio:.1f} below minimum {min_required:.1f}"
        
        return RiskRewardAnalysis(
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            risk_points=risk_points,
            reward_points=reward_points,
            risk_reward_ratio=risk_reward_ratio,
            meets_minimum=meets_minimum,
            minimum_required=min_required,
            validation_result=validation_result
        )
    
    def _is_setup_optimized(self, 
                          zone_position: ZonePositionAnalysis,
                          entry_calculation: EntryCalculation,
                          risk_reward: RiskRewardAnalysis) -> bool:
        """Determine if the setup is properly optimized."""
        # Check if R:R meets minimum
        if not risk_reward.meets_minimum:
            return False
        
        # Check if entry position is appropriate for setup type
        if entry_calculation.setup_type == SetupType.ZFR:
            # ZFR should be at front or middle
            if entry_calculation.entry_position == ZonePosition.BACK:
                return False
        else:  # ZF-TR
            # ZF-TR should be at middle or back
            if entry_calculation.entry_position == ZonePosition.FRONT:
                return False
        
        return True
    
    def _calculate_qrs_adjustment(self, 
                                zone_position: ZonePositionAnalysis,
                                entry_calculation: EntryCalculation,
                                risk_reward: RiskRewardAnalysis) -> float:
        """Calculate QRS score adjustment based on entry optimization quality."""
        adjustment = 0.0
        
        # R:R ratio adjustment
        if risk_reward.risk_reward_ratio >= 5.0:
            adjustment += 0.5  # Excellent R:R
        elif risk_reward.risk_reward_ratio >= 3.0:
            adjustment += 0.3  # Good R:R
        elif risk_reward.risk_reward_ratio >= 2.0:
            adjustment += 0.1  # Acceptable R:R
        else:
            adjustment -= 0.2  # Poor R:R
        
        # Entry position adjustment
        if entry_calculation.setup_type == SetupType.ZFR:
            if entry_calculation.entry_position == ZonePosition.FRONT:
                adjustment += 0.3  # Ideal for ZFR
            elif entry_calculation.entry_position == ZonePosition.MIDDLE:
                adjustment += 0.1  # Good for ZFR
            else:  # BACK
                adjustment -= 0.2  # Poor for ZFR
        else:  # ZF-TR
            if entry_calculation.entry_position == ZonePosition.BACK:
                adjustment += 0.3  # Ideal for ZF-TR
            elif entry_calculation.entry_position == ZonePosition.MIDDLE:
                adjustment += 0.1  # Good for ZF-TR
            else:  # FRONT
                adjustment -= 0.2  # Poor for ZF-TR
        
        # Zone position quality adjustment
        if zone_position.position_percentage <= 0.2 or zone_position.position_percentage >= 0.8:
            adjustment += 0.1  # Near zone boundaries (good for fading)
        elif 0.4 <= zone_position.position_percentage <= 0.6:
            adjustment -= 0.1  # Middle of zone (less ideal for fading)
        
        return max(-1.0, min(1.0, adjustment))  # Cap between -1.0 and +1.0
    
    def _generate_recommendation(self, 
                               zone_position: ZonePositionAnalysis,
                               entry_calculation: EntryCalculation,
                               risk_reward: RiskRewardAnalysis,
                               is_optimized: bool) -> str:
        """Generate recommendation based on optimization results."""
        if not is_optimized:
            if not risk_reward.meets_minimum:
                return f"SKIP: R:R {risk_reward.risk_reward_ratio:.1f} below minimum {risk_reward.minimum_required:.1f}"
            else:
                return f"SKIP: Entry position {entry_calculation.entry_position.value} not ideal for {entry_calculation.setup_type.value}"
        
        # Generate positive recommendation
        rr_quality = "Excellent" if risk_reward.risk_reward_ratio >= 5.0 else "Good" if risk_reward.risk_reward_ratio >= 3.0 else "Acceptable"
        position_quality = "Ideal" if entry_calculation.entry_position == ZonePosition.FRONT else "Good"
        
        return f"PROCEED: {rr_quality} R:R {risk_reward.risk_reward_ratio:.1f}, {position_quality} entry position"
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        if self.total_optimized == 0:
            return {
                'total_optimized': 0,
                'zfr_optimizations': 0,
                'zf_tr_optimizations': 0,
                'valid_rr_ratios': 0,
                'invalid_rr_ratios': 0,
                'valid_rr_rate': 0.0,
                'zfr_rate': 0.0,
                'zf_tr_rate': 0.0
            }
        
        return {
            'total_optimized': self.total_optimized,
            'zfr_optimizations': self.zfr_optimizations,
            'zf_tr_optimizations': self.zf_tr_optimizations,
            'valid_rr_ratios': self.valid_rr_ratios,
            'invalid_rr_ratios': self.invalid_rr_ratios,
            'valid_rr_rate': (self.valid_rr_ratios / self.total_optimized) * 100,
            'zfr_rate': (self.zfr_optimizations / self.total_optimized) * 100,
            'zf_tr_rate': (self.zf_tr_optimizations / self.total_optimized) * 100
        }
    
    def reset_statistics(self):
        """Reset optimization statistics."""
        self.total_optimized = 0
        self.zfr_optimizations = 0
        self.zf_tr_optimizations = 0
        self.valid_rr_ratios = 0
        self.invalid_rr_ratios = 0


class EntryOptimizationFilter:
    """
    Filter that applies entry optimization to zone fade signals.
    
    This filter implements the requirement to optimize entry prices and
    validate risk/reward ratios for zone fade setups.
    """
    
    def __init__(self, optimizer: EntryOptimizer):
        """
        Initialize entry optimization filter.
        
        Args:
            optimizer: EntryOptimizer instance
        """
        self.optimizer = optimizer
        self.signals_vetoed = 0
        self.signals_passed = 0
    
    def filter_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter signal based on entry optimization.
        
        Args:
            signal: Zone fade signal to filter
            market_data: Market data (not used but kept for consistency)
            
        Returns:
            Filtered signal with optimization data, None if vetoed
        """
        # Extract required data
        zone_high = signal.get('zone_high', 0.0)
        zone_low = signal.get('zone_low', 0.0)
        current_price = signal.get('current_price', 0.0)
        setup_type = signal.get('setup_type', 'ZFR')
        trade_direction = signal.get('trade_direction', 'LONG')
        target_zone_high = signal.get('target_zone_high')
        target_zone_low = signal.get('target_zone_low')
        
        # Optimize entry
        optimization_result = self.optimizer.optimize_entry(
            zone_high, zone_low, current_price, setup_type, trade_direction,
            target_zone_high, target_zone_low
        )
        
        # Apply filter
        if not optimization_result.is_optimized:
            self.signals_vetoed += 1
            return None  # VETO: Poor optimization or R:R
        
        # Add optimization data to signal
        signal['entry_optimization'] = {
            'optimal_entry_price': optimization_result.entry_calculation.optimal_entry_price,
            'entry_position': optimization_result.entry_calculation.entry_position.value,
            'setup_type': optimization_result.entry_calculation.setup_type.value,
            'risk_reward_ratio': optimization_result.risk_reward.risk_reward_ratio,
            'meets_minimum_rr': optimization_result.risk_reward.meets_minimum,
            'stop_loss_price': optimization_result.risk_reward.stop_loss_price,
            'target_price': optimization_result.risk_reward.target_price,
            'qrs_adjustment': optimization_result.qrs_adjustment,
            'recommendation': optimization_result.recommendation
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
            'optimizer_statistics': self.optimizer.get_statistics()
        }