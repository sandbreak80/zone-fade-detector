"""
Signal processor for Zone Fade setups.

This module provides signal processing, filtering, and coordination
for Zone Fade setups across multiple symbols.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
import uuid

from zone_fade_detector.core.models import OHLCVBar, ZoneFadeSetup, Alert
from zone_fade_detector.strategies.zone_fade_strategy import ZoneFadeStrategy


@dataclass
class SignalProcessorConfig:
    """Configuration for signal processor."""
    min_qrs_score: int = 7
    max_setups_per_symbol: int = 3
    setup_cooldown_minutes: int = 15
    alert_deduplication_minutes: int = 5
    enable_intermarket_filtering: bool = True
    enable_volume_filtering: bool = True
    min_volume_ratio: float = 1.2


class SignalProcessor:
    """
    Signal processor for Zone Fade setups.
    
    Provides signal processing, filtering, deduplication, and coordination
    for Zone Fade setups across multiple symbols.
    """
    
    def __init__(self, config: SignalProcessorConfig):
        """
        Initialize signal processor.
        
        Args:
            config: Signal processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy
        self.strategy = ZoneFadeStrategy(min_qrs_score=config.min_qrs_score)
        
        # Track recent setups and alerts
        self.recent_setups: Dict[str, List[ZoneFadeSetup]] = {}
        self.recent_alerts: Dict[str, datetime] = {}
        self.setup_history: List[ZoneFadeSetup] = []
        
        # Intermarket tracking
        self.intermarket_signals: Dict[str, Any] = {}
    
    def process_signals(
        self,
        symbol_data: Dict[str, List[OHLCVBar]],
        window_data: Optional[Dict[str, Dict[str, List[OHLCVBar]]]] = None
    ) -> List[Alert]:
        """
        Process signals for multiple symbols.
        
        Args:
            symbol_data: Dictionary mapping symbols to OHLCV bars
            window_data: Optional window data for additional analysis
            
        Returns:
            List of generated alerts
        """
        all_alerts = []
        
        # Detect setups for each symbol
        all_setups = self.strategy.analyze_multiple_symbols(symbol_data)
        
        # Process setups for each symbol
        for symbol, setups in all_setups.items():
            if not setups:
                continue
            
            # Filter setups
            filtered_setups = self._filter_setups(symbol, setups)
            
            # Generate alerts for filtered setups
            for setup in filtered_setups:
                if self._should_generate_alert(symbol, setup):
                    alert = self.strategy.generate_alert(setup)
                    all_alerts.append(alert)
                    
                    # Track the alert
                    self._track_alert(symbol, alert)
        
        # Update intermarket signals
        self._update_intermarket_signals(symbol_data)
        
        return all_alerts
    
    def _filter_setups(
        self,
        symbol: str,
        setups: List[ZoneFadeSetup]
    ) -> List[ZoneFadeSetup]:
        """
        Filter setups based on various criteria.
        
        Args:
            symbol: Stock symbol
            setups: List of setups to filter
            
        Returns:
            Filtered list of setups
        """
        filtered = []
        
        for setup in setups:
            # Check QRS score
            if setup.qrs_score < self.config.min_qrs_score:
                continue
            
            # Check volume filtering
            if self.config.enable_volume_filtering:
                if not self._passes_volume_filter(setup):
                    continue
            
            # Check cooldown period
            if not self._passes_cooldown_filter(symbol, setup):
                continue
            
            # Check intermarket filtering
            if self.config.enable_intermarket_filtering:
                if not self._passes_intermarket_filter(symbol, setup):
                    continue
            
            filtered.append(setup)
        
        # Limit setups per symbol
        if len(filtered) > self.config.max_setups_per_symbol:
            # Sort by QRS score and take the best ones
            filtered.sort(key=lambda x: x.qrs_score, reverse=True)
            filtered = filtered[:self.config.max_setups_per_symbol]
        
        return filtered
    
    def _passes_volume_filter(self, setup: ZoneFadeSetup) -> bool:
        """
        Check if setup passes volume filtering.
        
        Args:
            setup: ZoneFadeSetup to check
            
        Returns:
            True if setup passes volume filter
        """
        if not setup.volume_analysis:
            return True  # No volume data, allow setup
        
        # Check volume ratio
        if setup.volume_analysis.volume_ratio < self.config.min_volume_ratio:
            return False
        
        # Check for volume expansion
        if not setup.volume_analysis.is_expansion:
            return False
        
        return True
    
    def _passes_cooldown_filter(
        self,
        symbol: str,
        setup: ZoneFadeSetup
    ) -> bool:
        """
        Check if setup passes cooldown filter.
        
        Args:
            symbol: Stock symbol
            setup: ZoneFadeSetup to check
            
        Returns:
            True if setup passes cooldown filter
        """
        if symbol not in self.recent_setups:
            return True
        
        # Check if there's a recent setup for this symbol
        recent_setups = self.recent_setups[symbol]
        cooldown_time = setup.timestamp - timedelta(minutes=self.config.setup_cooldown_minutes)
        
        for recent_setup in recent_setups:
            if recent_setup.timestamp > cooldown_time:
                return False
        
        return True
    
    def _passes_intermarket_filter(
        self,
        symbol: str,
        setup: ZoneFadeSetup
    ) -> bool:
        """
        Check if setup passes intermarket filtering.
        
        Args:
            symbol: Stock symbol
            setup: ZoneFadeSetup to check
            
        Returns:
            True if setup passes intermarket filter
        """
        # This is a simplified implementation
        # In a real system, this would check for intermarket divergence
        
        # For now, allow all setups
        return True
    
    def _should_generate_alert(
        self,
        symbol: str,
        setup: ZoneFadeSetup
    ) -> bool:
        """
        Check if an alert should be generated for a setup.
        
        Args:
            symbol: Stock symbol
            setup: ZoneFadeSetup to check
            
        Returns:
            True if alert should be generated
        """
        # Check for duplicate alerts
        alert_key = f"{symbol}_{setup.zone.zone_type.value}_{setup.direction.value}"
        
        if alert_key in self.recent_alerts:
            last_alert_time = self.recent_alerts[alert_key]
            if setup.timestamp - last_alert_time < timedelta(minutes=self.config.alert_deduplication_minutes):
                return False
        
        return True
    
    def _track_alert(self, symbol: str, alert: Alert) -> None:
        """
        Track a generated alert.
        
        Args:
            symbol: Stock symbol
            alert: Generated alert
        """
        # Track in recent alerts
        alert_key = f"{symbol}_{alert.setup.zone.zone_type.value}_{alert.setup.direction.value}"
        self.recent_alerts[alert_key] = alert.created_at
        
        # Track in recent setups
        if symbol not in self.recent_setups:
            self.recent_setups[symbol] = []
        
        self.recent_setups[symbol].append(alert.setup)
        
        # Clean up old setups
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.recent_setups[symbol] = [
            setup for setup in self.recent_setups[symbol]
            if setup.timestamp.replace(tzinfo=None) > cutoff_time
        ]
        
        # Add to history
        self.setup_history.append(alert.setup)
        
        # Clean up old history
        self.setup_history = [
            setup for setup in self.setup_history
            if setup.timestamp.replace(tzinfo=None) > cutoff_time
        ]
    
    def _update_intermarket_signals(
        self,
        symbol_data: Dict[str, List[OHLCVBar]]
    ) -> None:
        """
        Update intermarket signals.
        
        Args:
            symbol_data: Dictionary mapping symbols to OHLCV bars
        """
        # This is a simplified implementation
        # In a real system, this would analyze intermarket relationships
        
        self.intermarket_signals = {
            'last_update': datetime.now(),
            'symbols_analyzed': list(symbol_data.keys()),
            'divergence_detected': False
        }
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        total_setups = len(self.setup_history)
        recent_setups = sum(len(setups) for setups in self.recent_setups.values())
        
        # Count setups by symbol
        symbol_counts = {}
        for setup in self.setup_history:
            symbol = setup.symbol
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Count setups by direction
        direction_counts = {}
        for setup in self.setup_history:
            direction = setup.direction.value
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Count setups by zone type
        zone_type_counts = {}
        for setup in self.setup_history:
            zone_type = setup.zone.zone_type.value
            zone_type_counts[zone_type] = zone_type_counts.get(zone_type, 0) + 1
        
        return {
            'total_setups_processed': total_setups,
            'recent_setups_active': recent_setups,
            'symbols_tracked': len(self.recent_setups),
            'alerts_generated': len(self.recent_alerts),
            'symbol_distribution': symbol_counts,
            'direction_distribution': direction_counts,
            'zone_type_distribution': zone_type_counts,
            'intermarket_signals': self.intermarket_signals
        }
    
    def clear_history(self) -> None:
        """Clear setup and alert history."""
        self.recent_setups.clear()
        self.recent_alerts.clear()
        self.setup_history.clear()
        self.intermarket_signals.clear()
        self.logger.info("Signal processor history cleared")
    
    def get_recent_setups(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[ZoneFadeSetup]:
        """
        Get recent setups.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of setups to return
            
        Returns:
            List of recent setups
        """
        if symbol:
            return self.recent_setups.get(symbol, [])[:limit]
        else:
            # Return all recent setups, sorted by timestamp
            all_setups = []
            for setups in self.recent_setups.values():
                all_setups.extend(setups)
            
            all_setups.sort(key=lambda x: x.timestamp, reverse=True)
            return all_setups[:limit]
    
    def validate_setup(
        self,
        setup: ZoneFadeSetup,
        additional_bars: List[OHLCVBar]
    ) -> bool:
        """
        Validate a setup with additional data.
        
        Args:
            setup: ZoneFadeSetup to validate
            additional_bars: Additional bars for validation
            
        Returns:
            True if setup is still valid
        """
        return self.strategy.validate_setup(setup, additional_bars)
    
    def get_setup_analysis(
        self,
        setup: ZoneFadeSetup
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of a setup.
        
        Args:
            setup: ZoneFadeSetup to analyze
            
        Returns:
            Dictionary with setup analysis
        """
        return self.strategy.get_setup_summary(setup)