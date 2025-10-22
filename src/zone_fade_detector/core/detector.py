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
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
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
        
        # Initialize rolling window manager
        self.window_manager = self._create_window_manager()
        
        # Initialize signal processor
        self.signal_processor = self._create_signal_processor()
        
        # Initialize alert system
        self.alert_system = self._create_alert_system()
        
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
        import os
        
        # Get API credentials from environment variables
        alpaca_config = AlpacaConfig(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        polygon_config = PolygonConfig(
            api_key=os.getenv('POLYGON_API_KEY', '')
        )
        
        from zone_fade_detector.data.data_manager import DataSource
        
        # Convert string to DataSource enum
        primary_source_str = self.config.get('data', {}).get('primary_source', 'alpaca')
        primary_source = DataSource(primary_source_str) if primary_source_str in [ds.value for ds in DataSource] else DataSource.ALPACA
        
        data_config = DataManagerConfig(
            alpaca_config=alpaca_config,
            polygon_config=polygon_config,
            cache_dir=self.config.get('cache', {}).get('dir', 'cache'),
            cache_ttl=self.config.get('cache', {}).get('ttl', 3600),
            primary_source=primary_source
        )
        
        return DataManager(data_config)
    
    def _create_window_manager(self) -> RollingWindowManager:
        """Create rolling window manager from configuration."""
        evaluation_cadence = self.config.get('polling', {}).get('interval_seconds', 30)
        memory_limit = self.config.get('performance', {}).get('memory_limit_mb', 500)
        
        return RollingWindowManager(
            evaluation_cadence_seconds=evaluation_cadence,
            memory_limit_mb=memory_limit
        )
    
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
    
    def _create_alert_system(self) -> AlertSystem:
        """Create alert system from configuration."""
        import os
        
        # Get alert configuration
        alert_config = self.config.get('alerts', {})
        
        # Create alert channel configuration
        channel_config = AlertChannelConfig(
            console_enabled=alert_config.get('console', {}).get('enabled', True),
            file_enabled=alert_config.get('file', {}).get('enabled', True),
            email_enabled=alert_config.get('email', {}).get('enabled', False),
            webhook_enabled=alert_config.get('webhook', {}).get('enabled', False),
            
            # File configuration
            file_path=alert_config.get('file', {}).get('log_file', '/app/logs/alerts.log'),
            file_max_size=alert_config.get('file', {}).get('max_file_size', 10485760),
            file_backup_count=alert_config.get('file', {}).get('backup_count', 5),
            
            # Email configuration
            email_smtp_server=alert_config.get('email', {}).get('smtp_server', 'smtp.gmail.com'),
            email_smtp_port=alert_config.get('email', {}).get('smtp_port', 587),
            email_username=alert_config.get('email', {}).get('username', ''),
            email_password=alert_config.get('email', {}).get('password', ''),
            email_from=alert_config.get('email', {}).get('from', ''),
            email_to=alert_config.get('email', {}).get('to', []),
            
            # Webhook configuration
            webhook_url=os.getenv('DISCORD_WEBHOOK_URL', alert_config.get('webhook', {}).get('url', '')),
            webhook_secret=alert_config.get('webhook', {}).get('secret', ''),
            webhook_timeout=alert_config.get('webhook', {}).get('timeout', 5)
        )
        
        return AlertSystem(channel_config)
    
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
        self.logger.info("🔍 Starting detection cycle...")
        
        # Fetch data for all symbols
        self.logger.info("📊 Fetching market data for all symbols...")
        symbol_data = await self._fetch_symbol_data()
        
        if not symbol_data:
            self.logger.warning("⚠️ No data available for detection")
            return
        
        # Update rolling windows with new data
        self.logger.info("🪟 Updating rolling windows with new data...")
        await self._update_rolling_windows(symbol_data)
        
        # Log data summary
        for symbol, bars in symbol_data.items():
            self.logger.info(f"📈 {symbol}: {len(bars)} bars available")
            if bars:
                latest_bar = bars[-1]
                self.logger.info(f"   Latest: {latest_bar.timestamp} - O:{latest_bar.open:.2f} H:{latest_bar.high:.2f} L:{latest_bar.low:.2f} C:{latest_bar.close:.2f} V:{latest_bar.volume}")
        
        # Process signals using rolling window data
        self.logger.info("🎯 Processing signals through Zone Fade strategy...")
        alerts = await self._process_signals_with_windows(symbol_data)
        
        # Log signal processing results
        if alerts:
            self.logger.info(f"🚨 Generated {len(alerts)} Zone Fade alerts!")
            for alert in alerts:
                self.logger.info(f"   Alert {alert.alert_id}: {alert.setup.symbol} {alert.setup.direction.value.upper()} - QRS: {alert.setup.qrs_score}/10")
        else:
            self.logger.info("ℹ️ No Zone Fade setups detected in current data")
        
        # Handle alerts
        if alerts:
            self.logger.info("📤 Sending alerts to configured channels...")
            await self._handle_alerts(alerts)
        
        # Update statistics
        self._update_stats(alerts)
        
        self.logger.info(f"✅ Detection cycle completed: {len(alerts)} alerts generated")
    
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
        if not alerts:
            return
            
        self.logger.info(f"Processing {len(alerts)} Zone Fade alerts...")
        
        for alert in alerts:
            try:
                await self._process_alert(alert)
                self.stats['total_alerts'] += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error processing alert {alert.alert_id}: {e}")
                self.stats['errors'] += 1
    
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
        
        # Send alerts through all configured channels (console, file, Discord, etc.)
        try:
            results = await self.alert_system.send_alert(alert)
            
            # Log results
            successful_channels = [name for name, success in results.items() if success]
            failed_channels = [name for name, success in results.items() if not success]
            
            if successful_channels:
                self.logger.info(f"Alert sent successfully through: {', '.join(successful_channels)}")
            
            if failed_channels:
                self.logger.warning(f"Alert failed to send through: {', '.join(failed_channels)}")
                
        except Exception as e:
            self.logger.error(f"Error sending alert through alert system: {e}")
            # Fallback to file logging
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
    
    async def test_alert_system(self) -> Dict[str, bool]:
        """Test the alert system by sending a test alert."""
        from zone_fade_detector.core.models import ZoneFadeSetup, Zone, ZoneType, SetupDirection, OHLCVBar, QRSFactors
        from datetime import datetime
        
        try:
            # Create a test alert
            test_bar = OHLCVBar(
                timestamp=datetime.now(),
                open=485.0,
                high=485.5,
                low=484.8,
                close=485.2,
                volume=1000000
            )
            
            test_zone = Zone(
                level=485.0,
                zone_type=ZoneType.PRIOR_DAY_HIGH,
                strength=0.8,
                quality=2
            )
            
            qrs_factors = QRSFactors()
            qrs_factors.zone_quality = 2
            qrs_factors.rejection_clarity = 2
            qrs_factors.structure_flip = 1
            qrs_factors.context = 2
            qrs_factors.intermarket_divergence = 1
            
            test_setup = ZoneFadeSetup(
                symbol='TEST',
                direction=SetupDirection.LONG,
                zone=test_zone,
                rejection_candle=test_bar,
                choch_confirmed=True,
                qrs_factors=qrs_factors,
                timestamp=datetime.now()
            )
            
            test_alert = Alert(
                alert_id='TEST_ALERT_001',
                setup=test_setup,
                priority='HIGH',
                created_at=datetime.now()
            )
            
            # Send test alert
            results = await self.alert_system.send_alert(test_alert)
            
            self.logger.info(f"Alert system test results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing alert system: {e}")
            return {'error': str(e)}
    
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
            'signal_processor_stats': self.signal_processor.get_processor_stats(),
            'alert_channels': len(self.alert_system.channels)
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
    
    async def _update_rolling_windows(self, symbol_data: Dict[str, List[OHLCVBar]]) -> None:
        """Update rolling windows with new data."""
        for symbol, bars in symbol_data.items():
            for bar in bars:
                # Add to different window types
                self.window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, symbol)
                self.window_manager.add_bar(WindowType.SESSION_CONTEXT, bar, symbol)
                self.window_manager.add_bar(WindowType.SWING_CHOCH, bar, symbol)
                self.window_manager.add_bar(WindowType.INITIATIVE_ANALYSIS, bar, symbol)
                self.window_manager.add_bar(WindowType.INTERMARKET, bar, symbol)
                self.window_manager.add_bar(WindowType.QRS_ACCUMULATOR, bar, symbol)
                
                # Add to opening range if within first 30 minutes of session
                if self._is_within_opening_range(bar.timestamp):
                    self.window_manager.add_bar(WindowType.OPENING_RANGE, bar, symbol)
                
                # Add to HTF zones if it's a significant bar
                if self._is_significant_for_htf(bar):
                    self.window_manager.add_bar(WindowType.HTF_ZONES, bar, symbol)
    
    async def _process_signals_with_windows(self, symbol_data: Dict[str, List[OHLCVBar]]) -> List[Alert]:
        """Process signals using rolling window data."""
        # Get window data for each symbol
        window_data = {}
        
        for symbol in symbol_data.keys():
            window_data[symbol] = {
                'vwap': self.window_manager.get_window_bars(WindowType.VWAP_COMPUTATION, symbol),
                'session_context': self.window_manager.get_window_bars(WindowType.SESSION_CONTEXT, symbol),
                'opening_range': self.window_manager.get_window_bars(WindowType.OPENING_RANGE, symbol),
                'swing_choch': self.window_manager.get_window_bars(WindowType.SWING_CHOCH, symbol),
                'initiative': self.window_manager.get_window_bars(WindowType.INITIATIVE_ANALYSIS, symbol),
                'intermarket': self.window_manager.get_window_bars(WindowType.INTERMARKET, symbol),
                'qrs_accumulator': self.window_manager.get_window_bars(WindowType.QRS_ACCUMULATOR, symbol),
                'htf_zones': self.window_manager.get_window_bars(WindowType.HTF_ZONES, symbol)
            }
        
        # Process signals with window data
        return self.signal_processor.process_signals(symbol_data, window_data)
    
    def _is_within_opening_range(self, timestamp: datetime) -> bool:
        """Check if timestamp is within opening range (first 30 minutes of RTH)."""
        # RTH starts at 9:30 AM ET
        rth_start = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        opening_range_end = rth_start + timedelta(minutes=30)
        
        return rth_start <= timestamp <= opening_range_end
    
    def _is_significant_for_htf(self, bar: OHLCVBar) -> bool:
        """Check if bar is significant for HTF zone analysis."""
        # Check for significant price movement or volume
        price_range = bar.high - bar.low
        avg_price = (bar.high + bar.low) / 2
        price_change_pct = price_range / avg_price if avg_price > 0 else 0
        
        # Consider significant if price change > 0.5% or high volume
        return price_change_pct > 0.005 or bar.volume > 100000
    
    def get_window_performance_stats(self) -> Dict[str, Any]:
        """Get rolling window performance statistics."""
        return self.window_manager.get_performance_stats()
    
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