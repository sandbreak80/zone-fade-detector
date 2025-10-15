"""
Enhanced Volume Spike Detection

This module implements enhanced volume spike detection with:
- Multiple confirmation methods
- Stricter 2.0x threshold (up from 1.8x)
- Volume cluster analysis
- Relative volume strength
- Exhaustion detection
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics


@dataclass
class VolumeSpike:
    """Volume spike detection result."""
    is_spike: bool
    spike_ratio: float
    avg_volume: float
    current_volume: float
    threshold_met: bool
    relative_strength: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    spike_type: str  # 'NORMAL', 'STRONG', 'EXTREME'
    supporting_metrics: Dict[str, float]


@dataclass
class VolumeConfirmation:
    """Complete volume confirmation result."""
    has_spike: bool
    spike_details: VolumeSpike
    volume_exhaustion: bool
    volume_profile: str  # 'INCREASING', 'DECREASING', 'STABLE'
    rejection_confirmed: bool
    confidence: float


class EnhancedVolumeDetector:
    """
    Enhanced volume spike detector with multiple confirmation methods.
    
    Features:
    - Stricter 2.0x threshold (was 1.8x)
    - Multiple lookback period analysis
    - Volume cluster detection
    - Relative strength calculation
    - Exhaustion detection
    """
    
    def __init__(self,
                 base_threshold: float = 2.0,
                 strong_threshold: float = 2.5,
                 extreme_threshold: float = 3.0,
                 primary_lookback: int = 20,
                 secondary_lookback: int = 50,
                 min_confidence: float = 0.7):
        """
        Initialize enhanced volume detector.
        
        Args:
            base_threshold: Base spike threshold (2.0x)
            strong_threshold: Strong spike threshold (2.5x)
            extreme_threshold: Extreme spike threshold (3.0x)
            primary_lookback: Primary lookback period
            secondary_lookback: Secondary lookback period
            min_confidence: Minimum confidence level
        """
        self.base_threshold = base_threshold
        self.strong_threshold = strong_threshold
        self.extreme_threshold = extreme_threshold
        self.primary_lookback = primary_lookback
        self.secondary_lookback = secondary_lookback
        self.min_confidence = min_confidence
        
        # Statistics
        self.total_analyzed = 0
        self.spikes_detected = 0
        self.strong_spikes = 0
        self.extreme_spikes = 0
    
    def detect_volume_spike(self,
                           bars: List,
                           current_index: int) -> VolumeSpike:
        """
        Detect volume spike with multiple confirmation methods.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            
        Returns:
            VolumeSpike with detailed analysis
        """
        self.total_analyzed += 1
        
        if current_index < self.primary_lookback:
            return self._create_no_spike_result()
        
        current_bar = bars[current_index]
        current_volume = current_bar.volume if hasattr(current_bar, 'volume') else current_bar['volume']
        
        # Calculate average volumes over different periods
        primary_bars = bars[max(0, current_index - self.primary_lookback):current_index]
        primary_volumes = [b.volume if hasattr(b, 'volume') else b['volume'] for b in primary_bars]
        primary_avg = statistics.mean(primary_volumes) if primary_volumes else 0
        
        secondary_bars = bars[max(0, current_index - self.secondary_lookback):current_index] if current_index >= self.secondary_lookback else primary_bars
        secondary_volumes = [b.volume if hasattr(b, 'volume') else b['volume'] for b in secondary_bars]
        secondary_avg = statistics.mean(secondary_volumes) if secondary_volumes else primary_avg
        
        # Calculate spike ratios
        primary_spike_ratio = current_volume / primary_avg if primary_avg > 0 else 0
        secondary_spike_ratio = current_volume / secondary_avg if secondary_avg > 0 else 0
        
        # Use primary ratio as main metric
        spike_ratio = primary_spike_ratio
        
        # Check threshold
        threshold_met = spike_ratio >= self.base_threshold
        
        # Determine spike type
        if spike_ratio >= self.extreme_threshold:
            spike_type = 'EXTREME'
            self.extreme_spikes += 1
        elif spike_ratio >= self.strong_threshold:
            spike_type = 'STRONG'
            self.strong_spikes += 1
        elif spike_ratio >= self.base_threshold:
            spike_type = 'NORMAL'
        else:
            spike_type = 'NONE'
        
        # Calculate relative strength (compared to recent max)
        recent_max_volume = max(primary_volumes) if primary_volumes else current_volume
        relative_strength = min(1.0, current_volume / recent_max_volume) if recent_max_volume > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            primary_spike_ratio,
            secondary_spike_ratio,
            relative_strength,
            spike_type
        )
        
        # Supporting metrics
        supporting_metrics = {
            'primary_spike_ratio': primary_spike_ratio,
            'secondary_spike_ratio': secondary_spike_ratio,
            'relative_strength': relative_strength,
            'median_volume': statistics.median(primary_volumes) if primary_volumes else 0,
            'max_recent_volume': recent_max_volume,
            'volume_percentile': self._calculate_percentile(current_volume, primary_volumes)
        }
        
        is_spike = threshold_met and confidence >= self.min_confidence
        
        if is_spike:
            self.spikes_detected += 1
        
        return VolumeSpike(
            is_spike=is_spike,
            spike_ratio=spike_ratio,
            avg_volume=primary_avg,
            current_volume=current_volume,
            threshold_met=threshold_met,
            relative_strength=relative_strength,
            confidence=confidence,
            spike_type=spike_type,
            supporting_metrics=supporting_metrics
        )
    
    def confirm_rejection_volume(self,
                                 bars: List,
                                 current_index: int,
                                 wick_ratio: float) -> VolumeConfirmation:
        """
        Confirm rejection with volume spike and additional analysis.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            wick_ratio: Wick ratio (upper or lower)
            
        Returns:
            VolumeConfirmation with complete analysis
        """
        # Detect volume spike
        spike_details = self.detect_volume_spike(bars, current_index)
        
        # Check for volume exhaustion
        volume_exhaustion = self._detect_exhaustion(bars, current_index)
        
        # Analyze volume profile
        volume_profile = self._analyze_volume_profile(bars, current_index)
        
        # Confirm rejection based on volume + wick
        rejection_confirmed = (
            spike_details.is_spike and
            wick_ratio > 0.40 and  # Strict wick requirement
            spike_details.confidence >= self.min_confidence
        )
        
        # Overall confidence
        confidence = spike_details.confidence * 0.7 + (wick_ratio * 0.3)
        
        return VolumeConfirmation(
            has_spike=spike_details.is_spike,
            spike_details=spike_details,
            volume_exhaustion=volume_exhaustion,
            volume_profile=volume_profile,
            rejection_confirmed=rejection_confirmed,
            confidence=confidence
        )
    
    def _calculate_confidence(self,
                             primary_ratio: float,
                             secondary_ratio: float,
                             relative_strength: float,
                             spike_type: str) -> float:
        """Calculate confidence level for spike detection."""
        confidence = 0.0
        
        # Primary ratio confidence (40%)
        if primary_ratio >= self.extreme_threshold:
            confidence += 0.40
        elif primary_ratio >= self.strong_threshold:
            confidence += 0.35
        elif primary_ratio >= self.base_threshold:
            confidence += 0.25
        
        # Secondary confirmation (30%)
        if secondary_ratio >= self.base_threshold:
            confidence += 0.30
        elif secondary_ratio >= self.base_threshold * 0.8:
            confidence += 0.20
        
        # Relative strength (30%)
        confidence += relative_strength * 0.30
        
        return min(1.0, confidence)
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of value in values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        count_below = sum(1 for v in sorted_values if v < value)
        percentile = (count_below / len(sorted_values)) * 100 if sorted_values else 0
        
        return percentile
    
    def _detect_exhaustion(self, bars: List, current_index: int, lookback: int = 5) -> bool:
        """Detect volume exhaustion pattern."""
        if current_index < lookback:
            return False
        
        recent_bars = bars[current_index - lookback:current_index + 1]
        volumes = [b.volume if hasattr(b, 'volume') else b['volume'] for b in recent_bars]
        
        # Check for decreasing volume trend
        if len(volumes) < 3:
            return False
        
        # Simple trend check: are volumes generally decreasing?
        decreasing_count = 0
        for i in range(1, len(volumes)):
            if volumes[i] < volumes[i-1]:
                decreasing_count += 1
        
        exhaustion = decreasing_count >= (len(volumes) - 1) * 0.6
        return exhaustion
    
    def _analyze_volume_profile(self, bars: List, current_index: int, lookback: int = 10) -> str:
        """Analyze recent volume profile."""
        if current_index < lookback:
            return 'STABLE'
        
        recent_bars = bars[current_index - lookback:current_index + 1]
        volumes = [b.volume if hasattr(b, 'volume') else b['volume'] for b in recent_bars]
        
        if len(volumes) < 5:
            return 'STABLE'
        
        # Calculate trend
        first_half = volumes[:len(volumes)//2]
        second_half = volumes[len(volumes)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.2:
            return 'INCREASING'
        elif second_avg < first_avg * 0.8:
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def _create_no_spike_result(self) -> VolumeSpike:
        """Create result for no spike detected."""
        return VolumeSpike(
            is_spike=False,
            spike_ratio=0.0,
            avg_volume=0.0,
            current_volume=0.0,
            threshold_met=False,
            relative_strength=0.0,
            confidence=0.0,
            spike_type='NONE',
            supporting_metrics={}
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get detection statistics."""
        return {
            'total_analyzed': self.total_analyzed,
            'spikes_detected': self.spikes_detected,
            'strong_spikes': self.strong_spikes,
            'extreme_spikes': self.extreme_spikes,
            'detection_rate': (self.spikes_detected / self.total_analyzed * 100) if self.total_analyzed > 0 else 0,
            'strong_spike_rate': (self.strong_spikes / self.spikes_detected * 100) if self.spikes_detected > 0 else 0,
            'extreme_spike_rate': (self.extreme_spikes / self.spikes_detected * 100) if self.spikes_detected > 0 else 0
        }
