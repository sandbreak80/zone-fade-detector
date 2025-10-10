"""
Core data models for the Zone Fade Detector.

This module defines the fundamental data structures used throughout the system,
including OHLCV bars, zones, setups, and indicator data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from decimal import Decimal


class ZoneType(Enum):
    """Types of higher-timeframe zones."""
    PRIOR_DAY_HIGH = "prior_day_high"
    PRIOR_DAY_LOW = "prior_day_low"
    WEEKLY_HIGH = "weekly_high"
    WEEKLY_LOW = "weekly_low"
    VALUE_AREA_HIGH = "value_area_high"
    VALUE_AREA_LOW = "value_area_low"
    OPENING_RANGE_HIGH = "opening_range_high"
    OPENING_RANGE_LOW = "opening_range_low"
    OVERNIGHT_HIGH = "overnight_high"
    OVERNIGHT_LOW = "overnight_low"


class SetupDirection(Enum):
    """Direction of the Zone Fade setup."""
    LONG = "long"
    SHORT = "short"


class CandleType(Enum):
    """Types of rejection candles."""
    PIN_BAR = "pin_bar"
    ENGULFING = "engulfing"
    LONG_WICK = "long_wick"
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"


@dataclass
class OHLCVBar:
    """OHLCV bar data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    def __post_init__(self):
        """Validate bar data after initialization."""
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Invalid OHLCV data: high/low inconsistent with open/close")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def body_size(self) -> float:
        """Calculate the body size of the candle."""
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        """Calculate the upper wick size."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Calculate the lower wick size."""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Calculate the total range of the candle."""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if the candle is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if the candle is bearish (close < open)."""
        return self.close < self.open


@dataclass
class Zone:
    """Higher-timeframe zone definition."""
    level: float
    zone_type: ZoneType
    quality: int = 0  # 0-2 quality score
    created_at: datetime = field(default_factory=datetime.now)
    strength: float = 1.0  # Zone strength multiplier
    touches: int = 0  # Number of times price has touched this zone
    last_touch: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate zone data after initialization."""
        if not 0 <= self.quality <= 2:
            raise ValueError("Zone quality must be between 0 and 2")
        if self.strength <= 0:
            raise ValueError("Zone strength must be positive")
        if self.touches < 0:
            raise ValueError("Zone touches cannot be negative")
    
    def add_touch(self, timestamp: datetime) -> None:
        """Record a touch of this zone."""
        self.touches += 1
        self.last_touch = timestamp


@dataclass
class VWAPData:
    """VWAP calculation data."""
    vwap: float
    upper_1sigma: float
    lower_1sigma: float
    upper_2sigma: float
    lower_2sigma: float
    slope: float
    timestamp: datetime
    volume_sum: float = 0.0
    price_volume_sum: float = 0.0
    
    @property
    def is_flat(self) -> bool:
        """Check if VWAP slope is flat (within threshold)."""
        return abs(self.slope) < 0.001  # 0.1% threshold
    
    @property
    def is_bullish(self) -> bool:
        """Check if VWAP slope is bullish."""
        return self.slope > 0.001
    
    @property
    def is_bearish(self) -> bool:
        """Check if VWAP slope is bearish."""
        return self.slope < -0.001


@dataclass
class OpeningRange:
    """Opening range data."""
    high: float
    low: float
    start_time: datetime
    end_time: datetime
    volume: int = 0
    
    @property
    def range_size(self) -> float:
        """Calculate the size of the opening range."""
        return self.high - self.low
    
    @property
    def mid_point(self) -> float:
        """Calculate the midpoint of the opening range."""
        return (self.high + self.low) / 2
    
    @property
    def is_valid(self) -> bool:
        """Check if the opening range is valid."""
        return self.high > self.low and self.end_time > self.start_time


@dataclass
class SwingPoint:
    """Swing high or low point."""
    price: float
    timestamp: datetime
    is_high: bool
    strength: float = 1.0  # Swing strength (1.0 = normal, >1.0 = stronger)
    
    def __post_init__(self):
        """Validate swing point data."""
        if self.strength <= 0:
            raise ValueError("Swing strength must be positive")


@dataclass
class SwingStructure:
    """Swing structure analysis."""
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    choch_detected: bool = False
    choch_direction: Optional[SetupDirection] = None
    choch_timestamp: Optional[datetime] = None
    
    def add_swing_high(self, swing: SwingPoint) -> None:
        """Add a swing high point."""
        self.swing_highs.append(swing)
        self.last_swing_high = swing
        self._check_choch()
    
    def add_swing_low(self, swing: SwingPoint) -> None:
        """Add a swing low point."""
        self.swing_lows.append(swing)
        self.last_swing_low = swing
        self._check_choch()
    
    def _check_choch(self) -> None:
        """Check for Change of Character (CHoCH)."""
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return
        
        # Check for CHoCH logic (simplified for now)
        # This would be implemented with more sophisticated logic
        if self.last_swing_high and self.last_swing_low:
            if self.last_swing_high.timestamp > self.last_swing_low.timestamp:
                # Last swing was a high, check if we broke below last swing low
                if len(self.swing_lows) >= 2:
                    prev_swing_low = self.swing_lows[-2]
                    if self.last_swing_low.price < prev_swing_low.price:
                        self.choch_detected = True
                        self.choch_direction = SetupDirection.LONG
                        self.choch_timestamp = self.last_swing_low.timestamp
            else:
                # Last swing was a low, check if we broke above last swing high
                if len(self.swing_highs) >= 2:
                    prev_swing_high = self.swing_highs[-2]
                    if self.last_swing_high.price > prev_swing_high.price:
                        self.choch_detected = True
                        self.choch_direction = SetupDirection.SHORT
                        self.choch_timestamp = self.last_swing_high.timestamp


@dataclass
class VolumeAnalysis:
    """Volume analysis data."""
    current_volume: int
    average_volume: float
    volume_ratio: float
    is_expansion: bool = False
    is_contraction: bool = False
    expansion_threshold: float = 1.5
    contraction_threshold: float = 0.7
    
    def __post_init__(self):
        """Calculate volume analysis metrics."""
        self.volume_ratio = self.current_volume / self.average_volume if self.average_volume > 0 else 1.0
        self.is_expansion = self.volume_ratio >= self.expansion_threshold
        self.is_contraction = self.volume_ratio <= self.contraction_threshold


@dataclass
class QRSFactors:
    """Quality Rating System factors."""
    zone_quality: int = 0  # 0-2 points
    rejection_clarity: int = 0  # 0-2 points
    structure_flip: int = 0  # 0-2 points
    context: int = 0  # 0-2 points
    intermarket_divergence: int = 0  # 0-2 points
    
    @property
    def total_score(self) -> int:
        """Calculate total QRS score."""
        return (self.zone_quality + self.rejection_clarity + 
                self.structure_flip + self.context + self.intermarket_divergence)
    
    @property
    def is_a_setup(self) -> bool:
        """Check if this qualifies as an A-Setup (≥7 points)."""
        return self.total_score >= 7
    
    def __post_init__(self):
        """Validate QRS factors."""
        for field_name, value in self.__dict__.items():
            if not 0 <= value <= 2:
                raise ValueError(f"QRS factor {field_name} must be between 0 and 2")


@dataclass
class ZoneFadeSetup:
    """Complete Zone Fade setup data."""
    symbol: str
    direction: SetupDirection
    zone: Zone
    rejection_candle: OHLCVBar
    choch_confirmed: bool
    qrs_factors: QRSFactors
    timestamp: datetime
    vwap_data: Optional[VWAPData] = None
    opening_range: Optional[OpeningRange] = None
    swing_structure: Optional[SwingStructure] = None
    volume_analysis: Optional[VolumeAnalysis] = None
    intermarket_context: Optional[Dict[str, Any]] = None
    
    @property
    def is_a_setup(self) -> bool:
        """Check if this is an A-Setup."""
        return self.qrs_factors.is_a_setup
    
    @property
    def qrs_score(self) -> int:
        """Get the QRS score."""
        return self.qrs_factors.total_score
    
    @property
    def entry_price(self) -> float:
        """Calculate suggested entry price."""
        if self.direction == SetupDirection.LONG:
            return self.zone.level - (self.zone.range_size * 0.1)  # 10% into zone
        else:
            return self.zone.level + (self.zone.range_size * 0.1)  # 10% into zone
    
    @property
    def stop_loss(self) -> float:
        """Calculate suggested stop loss."""
        if self.direction == SetupDirection.LONG:
            return self.zone.level - (self.zone.range_size * 0.2)  # 20% beyond zone
        else:
            return self.zone.level + (self.zone.range_size * 0.2)  # 20% beyond zone
    
    @property
    def target_1(self) -> float:
        """Calculate first target (VWAP or range mid)."""
        if self.vwap_data:
            return self.vwap_data.vwap
        elif self.opening_range:
            return self.opening_range.mid_point
        else:
            return self.zone.level  # Fallback
    
    @property
    def target_2(self) -> float:
        """Calculate second target (opposite range edge)."""
        if self.direction == SetupDirection.LONG:
            return self.zone.level + (self.zone.range_size * 0.5)
        else:
            return self.zone.level - (self.zone.range_size * 0.5)


@dataclass
class MarketContext:
    """Market context and environment data."""
    is_trend_day: bool = False
    vwap_slope: float = 0.0
    value_area_overlap: bool = False
    market_balance: float = 0.5  # 0.0 = bearish, 0.5 = balanced, 1.0 = bullish
    volatility_regime: str = "normal"  # low, normal, high
    session_type: str = "regular"  # premarket, regular, afterhours
    
    @property
    def is_balanced(self) -> bool:
        """Check if market is balanced."""
        return not self.is_trend_day and abs(self.vwap_slope) < 0.001 and self.value_area_overlap


@dataclass
class Alert:
    """Trading alert data."""
    setup: ZoneFadeSetup
    alert_id: str
    created_at: datetime
    priority: str = "normal"  # low, normal, high, critical
    status: str = "active"  # active, acknowledged, dismissed, expired
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "symbol": self.setup.symbol,
            "direction": self.setup.direction.value,
            "zone_level": self.setup.zone.level,
            "zone_type": self.setup.zone.zone_type.value,
            "qrs_score": self.setup.qrs_score,
            "entry_price": self.setup.entry_price,
            "stop_loss": self.setup.stop_loss,
            "target_1": self.setup.target_1,
            "target_2": self.setup.target_2,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority,
            "status": self.status,
            "notes": self.notes
        }