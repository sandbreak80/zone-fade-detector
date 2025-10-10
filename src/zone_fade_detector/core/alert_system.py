"""
Alert system for Zone Fade Detector.

This module provides multiple alert channels including console, file,
email, and webhook notifications.
"""

import logging
import smtplib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import asyncio

from zone_fade_detector.core.models import Alert


@dataclass
class AlertChannelConfig:
    """Configuration for alert channels."""
    console_enabled: bool = True
    file_enabled: bool = True
    email_enabled: bool = False
    webhook_enabled: bool = False
    
    # File configuration
    file_path: str = "logs/alerts.log"
    file_max_size: int = 10485760  # 10MB
    file_backup_count: int = 5
    
    # Email configuration
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = None
    
    # Webhook configuration
    webhook_url: str = ""
    webhook_secret: str = ""
    webhook_timeout: int = 5


class AlertChannel:
    """Base class for alert channels."""
    
    def __init__(self, config: AlertChannelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through this channel."""
        raise NotImplementedError


class ConsoleAlertChannel(AlertChannel):
    """Console alert channel."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to console."""
        try:
            alert_dict = alert.to_dict()
            
            print(f"\n{'='*60}")
            print(f"🚨 ZONE FADE ALERT - {alert.alert_id}")
            print(f"{'='*60}")
            print(f"Symbol: {alert_dict['symbol']}")
            print(f"Direction: {alert_dict['direction'].upper()}")
            print(f"Zone Level: ${alert_dict['zone_level']:.2f}")
            print(f"Zone Type: {alert_dict['zone_type']}")
            print(f"QRS Score: {alert_dict['qrs_score']}/10")
            print(f"Priority: {alert_dict['priority'].upper()}")
            print(f"Entry Price: ${alert_dict['entry_price']:.2f}")
            print(f"Stop Loss: ${alert_dict['stop_loss']:.2f}")
            print(f"Target 1: ${alert_dict['target_1']:.2f}")
            print(f"Target 2: ${alert_dict['target_2']:.2f}")
            print(f"Time: {alert_dict['created_at']}")
            print(f"{'='*60}\n")
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending console alert: {e}")
            return False


class FileAlertChannel(AlertChannel):
    """File alert channel."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to file."""
        try:
            import os
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.file_path), exist_ok=True)
            
            # Format alert for file
            alert_dict = alert.to_dict()
            alert_line = f"{alert.created_at.isoformat()} | {json.dumps(alert_dict)}\n"
            
            # Write to file
            with open(self.config.file_path, 'a') as f:
                f.write(alert_line)
            
            # Check file size and rotate if needed
            self._rotate_file_if_needed()
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending file alert: {e}")
            return False
    
    def _rotate_file_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        try:
            import os
            import shutil
            
            if not os.path.exists(self.config.file_path):
                return
            
            file_size = os.path.getsize(self.config.file_path)
            if file_size < self.config.file_max_size:
                return
            
            # Rotate files
            for i in range(self.config.file_backup_count - 1, 0, -1):
                old_file = f"{self.config.file_path}.{i}"
                new_file = f"{self.config.file_path}.{i + 1}"
                if os.path.exists(old_file):
                    if i == self.config.file_backup_count - 1:
                        os.remove(old_file)
                    else:
                        shutil.move(old_file, new_file)
            
            # Move current file to .1
            shutil.move(self.config.file_path, f"{self.config.file_path}.1")
            
        except Exception as e:
            self.logger.error(f"Error rotating log file: {e}")


class EmailAlertChannel(AlertChannel):
    """Email alert channel."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.config.email_enabled or not self.config.email_to:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"Zone Fade Alert - {alert.setup.symbol} {alert.setup.direction.value.upper()}"
            
            # Create email body
            alert_dict = alert.to_dict()
            body = self._format_email_body(alert_dict)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            return False
    
    def _format_email_body(self, alert_dict: Dict[str, Any]) -> str:
        """Format alert data as HTML email body."""
        return f"""
        <html>
        <body>
        <h2>🚨 Zone Fade Alert</h2>
        <table border="1" cellpadding="5" cellspacing="0">
        <tr><td><b>Alert ID</b></td><td>{alert_dict['alert_id']}</td></tr>
        <tr><td><b>Symbol</b></td><td>{alert_dict['symbol']}</td></tr>
        <tr><td><b>Direction</b></td><td>{alert_dict['direction'].upper()}</td></tr>
        <tr><td><b>Zone Level</b></td><td>${alert_dict['zone_level']:.2f}</td></tr>
        <tr><td><b>Zone Type</b></td><td>{alert_dict['zone_type']}</td></tr>
        <tr><td><b>QRS Score</b></td><td>{alert_dict['qrs_score']}/10</td></tr>
        <tr><td><b>Priority</b></td><td>{alert_dict['priority'].upper()}</td></tr>
        <tr><td><b>Entry Price</b></td><td>${alert_dict['entry_price']:.2f}</td></tr>
        <tr><td><b>Stop Loss</b></td><td>${alert_dict['stop_loss']:.2f}</td></tr>
        <tr><td><b>Target 1</b></td><td>${alert_dict['target_1']:.2f}</td></tr>
        <tr><td><b>Target 2</b></td><td>${alert_dict['target_2']:.2f}</td></tr>
        <tr><td><b>Time</b></td><td>{alert_dict['created_at']}</td></tr>
        </table>
        </body>
        </html>
        """


class WebhookAlertChannel(AlertChannel):
    """Webhook alert channel."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not self.config.webhook_enabled or not self.config.webhook_url:
            return False
        
        try:
            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'timestamp': alert.created_at.isoformat(),
                'setup': alert.to_dict(),
                'source': 'zone_fade_detector'
            }
            
            # Add signature if secret is provided
            if self.config.webhook_secret:
                import hmac
                import hashlib
                signature = hmac.new(
                    self.config.webhook_secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                payload['signature'] = signature
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.error(f"Webhook returned status {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
            return False


class AlertSystem:
    """
    Alert system that manages multiple alert channels.
    
    Provides a unified interface for sending alerts through
    multiple channels (console, file, email, webhook).
    """
    
    def __init__(self, config: AlertChannelConfig):
        """
        Initialize alert system.
        
        Args:
            config: Alert channel configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize channels
        self.channels = []
        
        if config.console_enabled:
            self.channels.append(ConsoleAlertChannel(config))
        
        if config.file_enabled:
            self.channels.append(FileAlertChannel(config))
        
        if config.email_enabled:
            self.channels.append(EmailAlertChannel(config))
        
        if config.webhook_enabled:
            self.channels.append(WebhookAlertChannel(config))
        
        self.logger.info(f"Alert system initialized with {len(self.channels)} channels")
    
    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert through all enabled channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            Dictionary mapping channel names to success status
        """
        results = {}
        
        # Send through all channels concurrently
        tasks = []
        for channel in self.channels:
            task = asyncio.create_task(self._send_through_channel(channel, alert))
            tasks.append((channel.__class__.__name__, task))
        
        # Wait for all tasks to complete
        for channel_name, task in tasks:
            try:
                success = await task
                results[channel_name] = success
            except Exception as e:
                self.logger.error(f"Error in {channel_name}: {e}")
                results[channel_name] = False
        
        # Log results
        successful_channels = [name for name, success in results.items() if success]
        failed_channels = [name for name, success in results.items() if not success]
        
        if successful_channels:
            self.logger.info(f"Alert sent successfully through: {', '.join(successful_channels)}")
        
        if failed_channels:
            self.logger.warning(f"Alert failed to send through: {', '.join(failed_channels)}")
        
        return results
    
    async def _send_through_channel(self, channel: AlertChannel, alert: Alert) -> bool:
        """Send alert through a specific channel."""
        try:
            return await channel.send_alert(alert)
        except Exception as e:
            self.logger.error(f"Error in {channel.__class__.__name__}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert system status."""
        return {
            'channels_enabled': len(self.channels),
            'console_enabled': self.config.console_enabled,
            'file_enabled': self.config.file_enabled,
            'email_enabled': self.config.email_enabled,
            'webhook_enabled': self.config.webhook_enabled,
            'file_path': self.config.file_path if self.config.file_enabled else None,
            'webhook_url': self.config.webhook_url if self.config.webhook_enabled else None
        }
    
    def test_channels(self) -> Dict[str, bool]:
        """Test all enabled channels with a sample alert."""
        from zone_fade_detector.core.models import Zone, ZoneType, OHLCVBar, ZoneFadeSetup, QRSFactors, SetupDirection
        
        # Create a test alert
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        qrs_factors = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="TEST",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        test_alert = Alert(
            setup=setup,
            alert_id="TEST_ALERT",
            created_at=datetime.now(),
            priority="normal"
        )
        
        # Test channels synchronously
        results = {}
        for channel in self.channels:
            try:
                success = asyncio.run(channel.send_alert(test_alert))
                results[channel.__class__.__name__] = success
            except Exception as e:
                self.logger.error(f"Test failed for {channel.__class__.__name__}: {e}")
                results[channel.__class__.__name__] = False
        
        return results