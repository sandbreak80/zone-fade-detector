#!/usr/bin/env python3
"""
Bitcoin Zone Fade Detector - 24/7 Cryptocurrency Trading System

This module provides a specialized Zone Fade Detector for Bitcoin and other
cryptocurrencies that trade 24/7, using the same core strategy but adapted
for crypto market characteristics.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from zone_fade_detector.core.models import OHLCVBar, Alert, ZoneFadeSetup, Zone, ZoneType, SetupDirection, QRSFactors
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
from zone_fade_detector.data.bitcoin_data_manager import BitcoinDataManager, BitcoinDataManagerConfig, CryptoConfig
from zone_fade_detector.strategies import SignalProcessor, SignalProcessorConfig
from zone_fade_detector.utils.logging import setup_logging


class BitcoinZoneFadeDetector:
    """
    Bitcoin-specific Zone Fade Detector.
    
    Coordinates all components to detect Zone Fade setups in Bitcoin and other
    cryptocurrencies with 24/7 trading capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bitcoin Zone Fade Detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup logging with fallback for permission issues
        try:
            # Create a modified config that disables file logging if there are permission issues
            logging_config = config.get('logging', {}).copy()
            if 'file' in logging_config:
                # Try to create the log directory first
                import os
                log_file = logging_config['file']
                log_dir = os.path.dirname(log_file)
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    # Test if we can write to the directory
                    test_file = os.path.join(log_dir, 'test_write.tmp')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except (PermissionError, OSError):
                    # Disable file logging if we can't write
                    logging_config.pop('file', None)
                    self.logger.warning(f"Cannot write to log directory {log_dir}, disabling file logging")
            
            setup_logging(logging_config)
        except Exception as e:
            # Fallback to console-only logging if anything fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            self.logger.warning(f"Logging setup failed: {e}, using console logging only")
        
        # Initialize Bitcoin data manager
        self.data_manager = self._create_bitcoin_data_manager()
        
        # Initialize rolling window manager
        self.window_manager = self._create_window_manager()
        
        # Initialize signal processor
        self.signal_processor = self._create_signal_processor()
        
        # Initialize alert system
        self.alert_system = self._create_alert_system()
        
        # Runtime state
        self.is_running = False
        self.coins = config.get('coins', ['bitcoin', 'ethereum'])
        self.poll_interval = config.get('polling', {}).get('interval_seconds', 60)  # 1 minute for crypto
        self.last_data_fetch = {}
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_alerts': 0,
            'total_setups': 0,
            'last_update': None,
            'errors': 0
        }
    
    def _create_bitcoin_data_manager(self) -> BitcoinDataManager:
        """Create Bitcoin data manager from configuration."""
        import os
        
        # Get API credentials from environment variables
        crypto_config = CryptoConfig(
            api_key=os.getenv('COINGECKO_API_KEY', ''),  # Optional
            base_url=os.getenv('COINGECKO_BASE_URL', 'https://api.coingecko.com/api/v3'),
            timeout=self.config.get('data', {}).get('timeout', 30),
            rate_limit_delay=self.config.get('data', {}).get('rate_limit_delay', 1.0)
        )
        
        data_config = BitcoinDataManagerConfig(
            crypto_config=crypto_config,
            cache_dir=self.config.get('cache', {}).get('dir', 'cache/bitcoin'),
            cache_ttl=self.config.get('cache', {}).get('ttl', 300)  # 5 minutes for crypto
        )
        
        return BitcoinDataManager(data_config)
    
    def _create_window_manager(self) -> RollingWindowManager:
        """Create rolling window manager from configuration."""
        evaluation_cadence = self.config.get('polling', {}).get('interval_seconds', 60)
        memory_limit = self.config.get('performance', {}).get('memory_limit_mb', 500)
        
        # Use default configurations for Bitcoin trading
        return RollingWindowManager(
            configs=None,  # Use defaults
            evaluation_cadence_seconds=evaluation_cadence,
            memory_limit_mb=memory_limit
        )
    
    def _create_signal_processor(self) -> SignalProcessor:
        """Create signal processor from configuration."""
        processor_config = SignalProcessorConfig(
            min_qrs_score=self.config.get('strategy', {}).get('min_qrs_score', 6),  # Lower for crypto
            max_setups_per_symbol=self.config.get('strategy', {}).get('max_setups_per_symbol', 3),
            setup_cooldown_minutes=self.config.get('strategy', {}).get('setup_cooldown_minutes', 15),
            alert_deduplication_minutes=self.config.get('strategy', {}).get('alert_deduplication_minutes', 5),
            enable_intermarket_filtering=self.config.get('strategy', {}).get('enable_intermarket_filtering', True),
            enable_volume_filtering=self.config.get('strategy', {}).get('enable_volume_filtering', True),
            min_volume_ratio=self.config.get('strategy', {}).get('min_volume_ratio', 1.2)
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
            file_path=alert_config.get('file', {}).get('log_file', '/app/logs/bitcoin_alerts.log'),
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
        """Run the Bitcoin Zone Fade Detector."""
        self.logger.info("🚀 Starting Bitcoin Zone Fade Detector - 24/7 Mode")
        self.logger.info(f"📊 Monitoring coins: {', '.join(self.coins)}")
        self.logger.info(f"⏱️ Polling interval: {self.poll_interval} seconds")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Use async context manager for data manager
            async with self.data_manager:
                while self.is_running:
                    try:
                        await self._detection_cycle()
                        await asyncio.sleep(self.poll_interval)
                    except Exception as e:
                        self.logger.error(f"Error in detection cycle: {e}")
                        self.stats['errors'] += 1
                        await asyncio.sleep(5)  # Short delay before retry
                        
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown requested by user")
        finally:
            self.is_running = False
            self.logger.info("👋 Bitcoin Zone Fade Detector stopped")
    
    async def _detection_cycle(self) -> None:
        """Run a single detection cycle."""
        self.logger.debug("🔄 Running detection cycle...")
        
        # Fetch data for all coins
        symbol_data = {}
        for coin_id in self.coins:
            try:
                # Get latest data (last 24 hours)
                bars = await self.data_manager.get_latest_bars(coin_id, hours=24)
                
                if bars:
                    # Convert coin_id to symbol for processing
                    symbol = self.data_manager.get_coin_symbol(coin_id)
                    symbol_data[symbol] = bars
                    self.last_data_fetch[coin_id] = datetime.now()
                    self.logger.debug(f"📈 {coin_id}: {len(bars)} bars")
                else:
                    self.logger.warning(f"⚠️ No data received for {coin_id}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error fetching data for {coin_id}: {e}")
                self.stats['errors'] += 1
        
        if not symbol_data:
            self.logger.warning("⚠️ No data available for any coins")
            return
        
        # Process signals
        try:
            alerts = self.signal_processor.process_signals(symbol_data)
            
            if alerts:
                self.logger.info(f"🎯 Generated {len(alerts)} Bitcoin Zone Fade alerts")
                await self._handle_alerts(alerts)
            else:
                self.logger.debug("ℹ️ No Zone Fade setups detected")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing signals: {e}")
            self.stats['errors'] += 1
        
        # Update statistics
        self._update_stats(alerts if 'alerts' in locals() else [])
    
    async def _handle_alerts(self, alerts: List[Alert]) -> None:
        """Handle generated alerts."""
        if not alerts:
            return
            
        self.logger.info(f"📤 Processing {len(alerts)} Bitcoin Zone Fade alerts...")
        
        for alert in alerts:
            try:
                await self._process_alert(alert)
                self.stats['total_alerts'] += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"❌ Error processing alert {alert.alert_id}: {e}")
                self.stats['errors'] += 1
    
    async def _process_alert(self, alert: Alert) -> None:
        """Process a single alert."""
        self.logger.info(f"🚨 Bitcoin Zone Fade Alert: {alert.alert_id}")
        self.logger.info(f"💰 Symbol: {alert.setup.symbol}")
        self.logger.info(f"📈 Direction: {alert.setup.direction.value}")
        self.logger.info(f"🎯 Zone Level: ${alert.setup.zone.level:,.2f}")
        self.logger.info(f"⭐ QRS Score: {alert.setup.qrs_score}")
        self.logger.info(f"🔥 Priority: {alert.priority}")
        
        # Log alert details
        alert_dict = alert.to_dict()
        self.logger.info(f"📋 Alert details: {alert_dict}")
        
        # Send alerts through all configured channels (console, file, Discord, etc.)
        try:
            results = await self.alert_system.send_alert(alert)
            
            # Log results
            successful_channels = [name for name, success in results.items() if success]
            failed_channels = [name for name, success in results.items() if not success]
            
            if successful_channels:
                self.logger.info(f"✅ Alert sent successfully through: {', '.join(successful_channels)}")
            
            if failed_channels:
                self.logger.warning(f"⚠️ Alert failed to send through: {', '.join(failed_channels)}")
                
        except Exception as e:
            self.logger.error(f"❌ Error sending alert through alert system: {e}")
            # Fallback to file logging
            self._log_alert_to_file(alert)
    
    def _log_alert_to_file(self, alert: Alert) -> None:
        """Log alert to file."""
        try:
            log_file = self.config.get('alerts', {}).get('log_file', 'logs/bitcoin_alerts.log')
            
            # Ensure log directory exists
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(f"{alert.created_at.isoformat()} - {alert.to_dict()}\n")
        except Exception as e:
            self.logger.error(f"❌ Error logging alert to file: {e}")
    
    def _update_stats(self, alerts: List[Alert]) -> None:
        """Update detector statistics."""
        self.stats['total_setups'] += len(alerts)
        self.stats['last_update'] = datetime.now()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    async def test_alert_system(self) -> Dict[str, bool]:
        """Test the alert system by sending a test Bitcoin alert."""
        try:
            # Create a test Bitcoin alert
            test_bar = OHLCVBar(
                timestamp=datetime.now(),
                open=45000.0,
                high=45500.0,
                low=44800.0,
                close=45200.0,
                volume=1000000
            )
            
            test_zone = Zone(
                level=45000.0,
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
                symbol='BTC',
                direction=SetupDirection.LONG,
                zone=test_zone,
                rejection_candle=test_bar,
                choch_confirmed=True,
                qrs_factors=qrs_factors,
                timestamp=datetime.now()
            )
            
            test_alert = Alert(
                alert_id='BTC_TEST_ALERT_001',
                setup=test_setup,
                priority='HIGH',
                created_at=datetime.now()
            )
            
            # Send test alert
            results = await self.alert_system.send_alert(test_alert)
            
            self.logger.info(f"🧪 Bitcoin alert system test results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error testing Bitcoin alert system: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'coins': self.coins,
            'poll_interval': self.poll_interval,
            'stats': self.stats,
            'data_manager_stats': self.data_manager.get_cache_stats(),
            'signal_processor_stats': self.signal_processor.get_processor_stats(),
            'alert_channels': len(self.alert_system.channels)
        }
    
    def stop(self) -> None:
        """Stop the detector."""
        self.logger.info("🛑 Stopping Bitcoin Zone Fade Detector...")
        self.is_running = False
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components."""
        health_status = {}
        
        try:
            # Check data manager health
            data_health = await self.data_manager.health_check()
            health_status['data_manager'] = data_health
            
            # Check signal processor
            health_status['signal_processor'] = True  # Always available
            
            # Check alert system
            health_status['alert_system'] = len(self.alert_system.channels) > 0
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            health_status['error'] = str(e)
        
        return health_status