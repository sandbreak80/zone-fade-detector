"""
Main Zone Fade Detector.

This module provides the main detector that coordinates all components
and runs the Zone Fade detection system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
import sys

from zone_fade_detector.core.models import OHLCVBar, Alert
from zone_fade_detector.data import DataManager, DataManagerConfig, AlpacaConfig, PolygonConfig
from zone_fade_detector.strategies import SignalProcessor, SignalProcessorConfig
from zone_fade_detector.utils.logging import setup_logging


class ZoneFadeDetector:
    """
    Main Zone Fade Detector.
    
    Coordinates all components to detect Zone Fade setups and generate alerts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Zone Fade Detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        setup_logging(config.get('logging', {}))
        
        # Initialize data manager
        self.data_manager = self._create_data_manager()
        
        # Initialize signal processor
        self.signal_processor = self._create_signal_processor()
        
        # Runtime state
        self.is_running = False
        self.symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM'])
        self.poll_interval = config.get('polling', {}).get('interval_seconds', 30)
        self.last_data_fetch = {}
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_alerts': 0,
            'total_setups': 0,
            'last_update': None,
            'errors': 0
        }
    
    def _create_data_manager(self) -> DataManager:
        """Create data manager from configuration."""
        # Get API credentials from environment or config
        alpaca_config = AlpacaConfig(
            api_key=self.config.get('alpaca', {}).get('api_key', ''),
            secret_key=self.config.get('alpaca', {}).get('secret_key', ''),
            base_url=self.config.get('alpaca', {}).get('base_url', 'https://paper-api.alpaca.markets')
        )
        
        polygon_config = PolygonConfig(
            api_key=self.config.get('polygon', {}).get('api_key', '')
        )
        
        data_config = DataManagerConfig(
            alpaca_config=alpaca_config,
            polygon_config=polygon_config,
            cache_dir=self.config.get('cache', {}).get('dir', 'cache'),
            cache_ttl=self.config.get('cache', {}).get('ttl', 3600),
            primary_source=self.config.get('data', {}).get('primary_source', 'alpaca')
        )
        
        return DataManager(data_config)
    
    def _create_signal_processor(self) -> SignalProcessor:
        """Create signal processor from configuration."""
        processor_config = SignalProcessorConfig(
            min_qrs_score=self.config.get('qrs', {}).get('a_setup_threshold', 7),
            max_setups_per_symbol=self.config.get('alerts', {}).get('max_setups_per_symbol', 3),
            setup_cooldown_minutes=self.config.get('alerts', {}).get('cooldown_minutes', 15),
            alert_deduplication_minutes=self.config.get('alerts', {}).get('deduplication_minutes', 5),
            enable_intermarket_filtering=self.config.get('filters', {}).get('intermarket', True),
            enable_volume_filtering=self.config.get('filters', {}).get('volume', True)
        )
        
        return SignalProcessor(processor_config)
    
    async def run(self) -> None:
        """Run the Zone Fade Detector."""
        self.logger.info("Starting Zone Fade Detector")
        self.stats['start_time'] = datetime.now()
        self.is_running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            async with self.data_manager:
                while self.is_running:
                    try:
                        await self._detection_cycle()
                        await asyncio.sleep(self.poll_interval)
                    except Exception as e:
                        self.logger.error(f"Error in detection cycle: {e}")
                        self.stats['errors'] += 1
                        await asyncio.sleep(5)  # Short delay before retry
        except Exception as e:
            self.logger.error(f"Fatal error in detector: {e}")
            raise
        finally:
            self.is_running = False
            self.logger.info("Zone Fade Detector stopped")
    
    async def _detection_cycle(self) -> None:
        """Run one detection cycle."""
        self.logger.debug("Starting detection cycle")
        
        # Fetch data for all symbols
        symbol_data = await self._fetch_symbol_data()
        
        if not symbol_data:
            self.logger.warning("No data available for detection")
            return
        
        # Process signals
        alerts = self.signal_processor.process_signals(symbol_data)
        
        # Handle alerts
        if alerts:
            await self._handle_alerts(alerts)
        
        # Update statistics
        self._update_stats(alerts)
        
        self.logger.debug(f"Detection cycle completed: {len(alerts)} alerts generated")
    
    async def _fetch_symbol_data(self) -> Dict[str, List[OHLCVBar]]:
        """Fetch data for all symbols."""
        symbol_data = {}
        
        for symbol in self.symbols:
            try:
                # Check if we need to fetch new data
                if self._should_fetch_data(symbol):
                    bars = await self.data_manager.get_latest_bars(symbol, limit=1000)
                    symbol_data[symbol] = bars
                    self.last_data_fetch[symbol] = datetime.now()
                    self.logger.debug(f"Fetched {len(bars)} bars for {symbol}")
                else:
                    # Use cached data
                    symbol_data[symbol] = []
                    self.logger.debug(f"Using cached data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                symbol_data[symbol] = []
        
        return symbol_data
    
    def _should_fetch_data(self, symbol: str) -> bool:
        """Check if we should fetch new data for a symbol."""
        if symbol not in self.last_data_fetch:
            return True
        
        last_fetch = self.last_data_fetch[symbol]
        time_since_fetch = datetime.now() - last_fetch
        
        # Fetch new data every poll interval
        return time_since_fetch.total_seconds() >= self.poll_interval
    
    async def _handle_alerts(self, alerts: List[Alert]) -> None:
        """Handle generated alerts."""
        for alert in alerts:
            try:
                await self._process_alert(alert)
                self.stats['total_alerts'] += 1
            except Exception as e:
                self.logger.error(f"Error processing alert {alert.alert_id}: {e}")
    
    async def _process_alert(self, alert: Alert) -> None:
        """Process a single alert."""
        self.logger.info(f"Zone Fade Alert: {alert.alert_id}")
        self.logger.info(f"Symbol: {alert.setup.symbol}")
        self.logger.info(f"Direction: {alert.setup.direction.value}")
        self.logger.info(f"Zone Level: {alert.setup.zone.level}")
        self.logger.info(f"QRS Score: {alert.setup.qrs_score}")
        self.logger.info(f"Priority: {alert.priority}")
        
        # Log alert details
        alert_dict = alert.to_dict()
        self.logger.info(f"Alert details: {alert_dict}")
        
        # Here you would typically send alerts to external systems
        # For now, we just log them
        self._log_alert_to_file(alert)
    
    def _log_alert_to_file(self, alert: Alert) -> None:
        """Log alert to file."""
        try:
            log_file = self.config.get('alerts', {}).get('log_file', 'logs/alerts.log')
            
            # Ensure log directory exists
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(f"{alert.created_at.isoformat()} - {alert.to_dict()}\n")
        except Exception as e:
            self.logger.error(f"Error logging alert to file: {e}")
    
    def _update_stats(self, alerts: List[Alert]) -> None:
        """Update detector statistics."""
        self.stats['total_setups'] += len(alerts)
        self.stats['last_update'] = datetime.now()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'symbols': self.symbols,
            'poll_interval': self.poll_interval,
            'stats': self.stats,
            'data_manager_stats': self.data_manager.get_cache_stats(),
            'signal_processor_stats': self.signal_processor.get_processor_stats()
        }
    
    def stop(self) -> None:
        """Stop the detector."""
        self.logger.info("Stopping Zone Fade Detector...")
        self.is_running = False
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components."""
        health_status = {}
        
        try:
            # Check data manager health
            data_health = await self.data_manager.health_check()
            health_status['data_manager'] = any(data_health.values())
            
            # Check signal processor
            health_status['signal_processor'] = True  # Always healthy if initialized
            
            # Overall health
            health_status['overall'] = all(health_status.values())
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_status['overall'] = False
        
        return health_status
    
    def get_recent_setups(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent setups."""
        setups = self.signal_processor.get_recent_setups(symbol, limit)
        return [self.signal_processor.get_setup_analysis(setup) for setup in setups]
    
    def clear_history(self) -> None:
        """Clear all history and statistics."""
        self.signal_processor.clear_history()
        self.data_manager.clear_cache()
        self.stats = {
            'start_time': self.stats['start_time'],
            'total_alerts': 0,
            'total_setups': 0,
            'last_update': None,
            'errors': 0
        }
        self.logger.info("History and statistics cleared")