"""
Risk Management Module

This module implements dynamic risk management including:
- ATR-based stop placement
- Volatility-based position sizing
- Risk/reward validation
- Dynamic stop adjustments
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import statistics


class StopType(Enum):
    """Stop loss types."""
    FIXED = "FIXED"              # Fixed percentage stop
    ATR_BASED = "ATR_BASED"      # ATR multiple stop
    SWING_BASED = "SWING_BASED"  # Based on swing levels
    ZONE_BASED = "ZONE_BASED"    # Based on zone invalidation


class PositionSizeMethod(Enum):
    """Position sizing methods."""
    FIXED = "FIXED"                    # Fixed % of capital
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"  # Based on ATR
    RISK_ADJUSTED = "RISK_ADJUSTED"    # Based on $ risk per trade


@dataclass
class StopLossCalculation:
    """Stop loss calculation result."""
    stop_price: float
    stop_type: StopType
    distance_points: float
    distance_percent: float
    atr_multiple: Optional[float]
    swing_level: Optional[float]
    rationale: str


@dataclass
class PositionSizeCalculation:
    """Position size calculation result."""
    shares: int
    position_value: float
    risk_amount: float
    risk_percent: float
    method: PositionSizeMethod
    volatility_factor: Optional[float]
    rationale: str


@dataclass
class RiskManagementResult:
    """Complete risk management result."""
    stop_loss: StopLossCalculation
    position_size: PositionSizeCalculation
    risk_reward_ratio: float
    meets_minimum_rr: bool
    max_loss_amount: float
    max_loss_percent: float
    recommendations: List[str]


class RiskManager:
    """
    Manages risk for zone fade trades with dynamic stops and position sizing.
    
    Features:
    - ATR-based stop placement (1.5x ATR default)
    - Volatility-based position sizing
    - Minimum stop distance enforcement (0.5% of price)
    - Risk/reward validation
    - Dynamic stop adjustments
    """
    
    def __init__(self,
                 default_stop_type: StopType = StopType.ATR_BASED,
                 atr_multiplier: float = 1.5,
                 min_stop_distance_pct: float = 0.5,
                 max_stop_distance_pct: float = 2.0,
                 default_risk_percent: float = 1.0,
                 min_risk_reward: float = 2.0,
                 max_position_percent: float = 10.0):
        """
        Initialize risk manager.
        
        Args:
            default_stop_type: Default stop loss type
            atr_multiplier: ATR multiplier for stops (1.5x default)
            min_stop_distance_pct: Minimum stop distance as % of price
            max_stop_distance_pct: Maximum stop distance as % of price
            default_risk_percent: Default risk per trade (% of capital)
            min_risk_reward: Minimum acceptable R:R ratio
            max_position_percent: Maximum position size (% of capital)
        """
        self.default_stop_type = default_stop_type
        self.atr_multiplier = atr_multiplier
        self.min_stop_distance_pct = min_stop_distance_pct
        self.max_stop_distance_pct = max_stop_distance_pct
        self.default_risk_percent = default_risk_percent
        self.min_risk_reward = min_risk_reward
        self.max_position_percent = max_position_percent
        
        # Statistics
        self.total_calculations = 0
        self.atr_stops_used = 0
        self.adjusted_stops = 0
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           direction: str,
                           atr: float,
                           zone_level: float,
                           recent_swing: Optional[float] = None,
                           stop_type: Optional[StopType] = None) -> StopLossCalculation:
        """
        Calculate optimal stop loss based on multiple factors.
        
        Args:
            entry_price: Entry price for the trade
            direction: 'LONG' or 'SHORT'
            atr: Average True Range (14-period)
            zone_level: Zone price level
            recent_swing: Recent swing high/low if available
            stop_type: Override default stop type
            
        Returns:
            StopLossCalculation with stop price and details
        """
        stop_type = stop_type or self.default_stop_type
        
        if stop_type == StopType.ATR_BASED:
            stop_calc = self._calculate_atr_stop(entry_price, direction, atr)
            self.atr_stops_used += 1
        elif stop_type == StopType.ZONE_BASED:
            stop_calc = self._calculate_zone_stop(entry_price, direction, zone_level)
        elif stop_type == StopType.SWING_BASED and recent_swing:
            stop_calc = self._calculate_swing_stop(entry_price, direction, recent_swing)
        else:
            # Default to ATR-based
            stop_calc = self._calculate_atr_stop(entry_price, direction, atr)
            self.atr_stops_used += 1
        
        # Enforce minimum stop distance
        min_distance = entry_price * (self.min_stop_distance_pct / 100)
        if stop_calc.distance_points < min_distance:
            stop_calc = self._adjust_stop_to_minimum(stop_calc, entry_price, direction, min_distance)
            self.adjusted_stops += 1
        
        # Enforce maximum stop distance
        max_distance = entry_price * (self.max_stop_distance_pct / 100)
        if stop_calc.distance_points > max_distance:
            stop_calc = self._adjust_stop_to_maximum(stop_calc, entry_price, direction, max_distance)
            self.adjusted_stops += 1
        
        self.total_calculations += 1
        return stop_calc
    
    def _calculate_atr_stop(self, entry_price: float, direction: str, atr: float) -> StopLossCalculation:
        """Calculate ATR-based stop loss."""
        atr_distance = atr * self.atr_multiplier
        
        if direction == 'LONG':
            stop_price = entry_price - atr_distance
        else:  # SHORT
            stop_price = entry_price + atr_distance
        
        distance_points = abs(entry_price - stop_price)
        distance_percent = (distance_points / entry_price) * 100
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopType.ATR_BASED,
            distance_points=distance_points,
            distance_percent=distance_percent,
            atr_multiple=self.atr_multiplier,
            swing_level=None,
            rationale=f"ATR-based stop: {self.atr_multiplier}x ATR ({atr:.2f})"
        )
    
    def _calculate_zone_stop(self, entry_price: float, direction: str, zone_level: float) -> StopLossCalculation:
        """Calculate zone invalidation stop."""
        # Place stop beyond zone with small buffer
        buffer = zone_level * 0.002  # 0.2% buffer
        
        if direction == 'LONG':
            stop_price = zone_level - buffer
        else:  # SHORT
            stop_price = zone_level + buffer
        
        distance_points = abs(entry_price - stop_price)
        distance_percent = (distance_points / entry_price) * 100
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopType.ZONE_BASED,
            distance_points=distance_points,
            distance_percent=distance_percent,
            atr_multiple=None,
            swing_level=zone_level,
            rationale=f"Zone invalidation stop at {zone_level:.2f} + buffer"
        )
    
    def _calculate_swing_stop(self, entry_price: float, direction: str, swing_level: float) -> StopLossCalculation:
        """Calculate swing-based stop."""
        # Place stop beyond swing with small buffer
        buffer = swing_level * 0.001  # 0.1% buffer
        
        if direction == 'LONG':
            stop_price = swing_level - buffer
        else:  # SHORT
            stop_price = swing_level + buffer
        
        distance_points = abs(entry_price - stop_price)
        distance_percent = (distance_points / entry_price) * 100
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopType.SWING_BASED,
            distance_points=distance_points,
            distance_percent=distance_percent,
            atr_multiple=None,
            swing_level=swing_level,
            rationale=f"Swing-based stop at {swing_level:.2f} + buffer"
        )
    
    def _adjust_stop_to_minimum(self, stop_calc: StopLossCalculation, entry_price: float,
                                direction: str, min_distance: float) -> StopLossCalculation:
        """Adjust stop to meet minimum distance requirement."""
        if direction == 'LONG':
            new_stop = entry_price - min_distance
        else:
            new_stop = entry_price + min_distance
        
        return StopLossCalculation(
            stop_price=new_stop,
            stop_type=stop_calc.stop_type,
            distance_points=min_distance,
            distance_percent=(min_distance / entry_price) * 100,
            atr_multiple=stop_calc.atr_multiple,
            swing_level=stop_calc.swing_level,
            rationale=f"{stop_calc.rationale} (adjusted to minimum {self.min_stop_distance_pct}%)"
        )
    
    def _adjust_stop_to_maximum(self, stop_calc: StopLossCalculation, entry_price: float,
                                direction: str, max_distance: float) -> StopLossCalculation:
        """Adjust stop to meet maximum distance requirement."""
        if direction == 'LONG':
            new_stop = entry_price - max_distance
        else:
            new_stop = entry_price + max_distance
        
        return StopLossCalculation(
            stop_price=new_stop,
            stop_type=stop_calc.stop_type,
            distance_points=max_distance,
            distance_percent=(max_distance / entry_price) * 100,
            atr_multiple=stop_calc.atr_multiple,
            swing_level=stop_calc.swing_level,
            rationale=f"{stop_calc.rationale} (capped at maximum {self.max_stop_distance_pct}%)"
        )
    
    def calculate_position_size(self,
                                account_balance: float,
                                entry_price: float,
                                stop_price: float,
                                atr: Optional[float] = None,
                                method: PositionSizeMethod = PositionSizeMethod.RISK_ADJUSTED) -> PositionSizeCalculation:
        """
        Calculate position size based on risk management rules.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_price: Stop loss price
            atr: Average True Range (for volatility adjustment)
            method: Position sizing method
            
        Returns:
            PositionSizeCalculation with shares and details
        """
        if method == PositionSizeMethod.RISK_ADJUSTED:
            return self._calculate_risk_adjusted_size(account_balance, entry_price, stop_price)
        elif method == PositionSizeMethod.VOLATILITY_ADJUSTED and atr:
            return self._calculate_volatility_adjusted_size(account_balance, entry_price, stop_price, atr)
        else:
            return self._calculate_fixed_size(account_balance, entry_price)
    
    def _calculate_risk_adjusted_size(self, account_balance: float, entry_price: float,
                                     stop_price: float) -> PositionSizeCalculation:
        """Calculate position size based on fixed risk per trade."""
        risk_amount = account_balance * (self.default_risk_percent / 100)
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share > 0:
            shares = int(risk_amount / risk_per_share)
        else:
            shares = 0
        
        # Enforce maximum position size
        max_shares = int((account_balance * (self.max_position_percent / 100)) / entry_price)
        shares = min(shares, max_shares)
        
        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        
        return PositionSizeCalculation(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percent=(actual_risk / account_balance) * 100,
            method=PositionSizeMethod.RISK_ADJUSTED,
            volatility_factor=None,
            rationale=f"Risk-adjusted: ${risk_amount:.2f} at risk ({self.default_risk_percent}% of capital)"
        )
    
    def _calculate_volatility_adjusted_size(self, account_balance: float, entry_price: float,
                                           stop_price: float, atr: float) -> PositionSizeCalculation:
        """Calculate position size adjusted for volatility."""
        # Base risk amount
        base_risk = account_balance * (self.default_risk_percent / 100)
        
        # Volatility factor (reduce size in high volatility)
        avg_atr_pct = (atr / entry_price) * 100
        if avg_atr_pct > 2.0:  # High volatility
            volatility_factor = 0.75
        elif avg_atr_pct > 1.0:  # Medium volatility
            volatility_factor = 0.90
        else:  # Low volatility
            volatility_factor = 1.0
        
        adjusted_risk = base_risk * volatility_factor
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share > 0:
            shares = int(adjusted_risk / risk_per_share)
        else:
            shares = 0
        
        # Enforce maximum position size
        max_shares = int((account_balance * (self.max_position_percent / 100)) / entry_price)
        shares = min(shares, max_shares)
        
        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        
        return PositionSizeCalculation(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percent=(actual_risk / account_balance) * 100,
            method=PositionSizeMethod.VOLATILITY_ADJUSTED,
            volatility_factor=volatility_factor,
            rationale=f"Volatility-adjusted: {volatility_factor:.0%} multiplier (ATR: {avg_atr_pct:.2f}%)"
        )
    
    def _calculate_fixed_size(self, account_balance: float, entry_price: float) -> PositionSizeCalculation:
        """Calculate fixed percentage position size."""
        position_value = account_balance * (self.default_risk_percent / 100)
        shares = int(position_value / entry_price)
        
        return PositionSizeCalculation(
            shares=shares,
            position_value=position_value,
            risk_amount=0,  # Not calculated for fixed method
            risk_percent=self.default_risk_percent,
            method=PositionSizeMethod.FIXED,
            volatility_factor=None,
            rationale=f"Fixed {self.default_risk_percent}% of capital"
        )
    
    def calculate_risk_management(self,
                                  account_balance: float,
                                  entry_price: float,
                                  direction: str,
                                  atr: float,
                                  zone_level: float,
                                  target_price: float,
                                  recent_swing: Optional[float] = None) -> RiskManagementResult:
        """
        Calculate complete risk management for a trade.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            direction: 'LONG' or 'SHORT'
            atr: Average True Range
            zone_level: Zone price level
            target_price: Target price for the trade
            recent_swing: Recent swing level if available
            
        Returns:
            RiskManagementResult with complete risk management details
        """
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(
            entry_price, direction, atr, zone_level, recent_swing
        )
        
        # Calculate position size
        position_size = self.calculate_position_size(
            account_balance, entry_price, stop_loss.stop_price, atr,
            PositionSizeMethod.VOLATILITY_ADJUSTED
        )
        
        # Calculate R:R ratio
        risk_points = stop_loss.distance_points
        reward_points = abs(target_price - entry_price)
        risk_reward_ratio = reward_points / risk_points if risk_points > 0 else 0
        
        # Check if meets minimum R:R
        meets_minimum_rr = risk_reward_ratio >= self.min_risk_reward
        
        # Calculate max loss
        max_loss_amount = position_size.shares * risk_points
        max_loss_percent = (max_loss_amount / account_balance) * 100
        
        # Generate recommendations
        recommendations = []
        if not meets_minimum_rr:
            recommendations.append(f"R:R ratio {risk_reward_ratio:.2f} below minimum {self.min_risk_reward}")
        if max_loss_percent > self.default_risk_percent * 1.5:
            recommendations.append(f"Max loss {max_loss_percent:.2f}% exceeds target {self.default_risk_percent}%")
        if position_size.shares == 0:
            recommendations.append("Position size calculated as 0 shares - check parameters")
        
        return RiskManagementResult(
            stop_loss=stop_loss,
            position_size=position_size,
            risk_reward_ratio=risk_reward_ratio,
            meets_minimum_rr=meets_minimum_rr,
            max_loss_amount=max_loss_amount,
            max_loss_percent=max_loss_percent,
            recommendations=recommendations
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get risk management statistics."""
        return {
            'total_calculations': self.total_calculations,
            'atr_stops_used': self.atr_stops_used,
            'adjusted_stops': self.adjusted_stops,
            'atr_stops_pct': (self.atr_stops_used / self.total_calculations * 100) if self.total_calculations > 0 else 0,
            'adjusted_stops_pct': (self.adjusted_stops / self.total_calculations * 100) if self.total_calculations > 0 else 0
        }


# Filter class for pipeline integration
class RiskManagementFilter:
    """Filter for risk management validation in the pipeline."""
    
    def __init__(self, risk_manager: RiskManager):
        """Initialize with a risk manager instance."""
        self.risk_manager = risk_manager
        self.name = "RiskManagementFilter"
    
    def filter(self, signal: Dict, account_balance: float = 100000.0) -> Tuple[bool, str]:
        """
        Filter signal based on risk management rules.
        
        Args:
            signal: Trading signal dictionary
            account_balance: Current account balance
            
        Returns:
            Tuple of (passed, reason)
        """
        try:
            # Extract required fields
            entry_price = signal.get('entry_price', 0)
            direction = signal.get('direction', 'LONG')
            atr = signal.get('atr', entry_price * 0.01)  # Default 1% if not provided
            zone_level = signal.get('zone_level', entry_price)
            target_price = signal.get('target_price', entry_price * 1.02)  # Default 2% target
            
            # Calculate risk management
            risk_mgmt = self.risk_manager.calculate_risk_management(
                account_balance, entry_price, direction, atr, zone_level, target_price
            )
            
            # Check if meets minimum requirements
            if not risk_mgmt.meets_minimum_rr:
                return False, f"R:R ratio {risk_mgmt.risk_reward_ratio:.2f} below minimum {self.risk_manager.min_risk_reward}"
            
            if risk_mgmt.position_size.shares == 0:
                return False, "Position size calculated as 0 shares"
            
            if risk_mgmt.max_loss_percent > self.risk_manager.default_risk_percent * 2:
                return False, f"Max loss {risk_mgmt.max_loss_percent:.2f}% too high"
            
            # Add risk management to signal
            signal['risk_management'] = risk_mgmt
            
            return True, "Risk management validated"
            
        except Exception as e:
            return False, f"Risk management error: {str(e)}"
