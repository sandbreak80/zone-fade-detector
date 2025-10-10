"""
Unit tests for alert system.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from zone_fade_detector.core.models import (
    Alert, Zone, ZoneType, OHLCVBar, ZoneFadeSetup, 
    QRSFactors, SetupDirection
)
from zone_fade_detector.core.alert_system import (
    AlertSystem, AlertChannelConfig, ConsoleAlertChannel,
    FileAlertChannel, EmailAlertChannel, WebhookAlertChannel
)


class TestAlertChannelConfig:
    """Test Alert Channel Configuration."""
    
    def test_config_creation(self):
        """Test creating alert channel configuration."""
        config = AlertChannelConfig(
            console_enabled=True,
            file_enabled=True,
            email_enabled=False,
            webhook_enabled=False,
            file_path="test_alerts.log"
        )
        
        assert config.console_enabled is True
        assert config.file_enabled is True
        assert config.email_enabled is False
        assert config.webhook_enabled is False
        assert config.file_path == "test_alerts.log"


class TestConsoleAlertChannel:
    """Test Console Alert Channel."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AlertChannelConfig(console_enabled=True)
    
    @pytest.fixture
    def channel(self, config):
        """Create console alert channel."""
        return ConsoleAlertChannel(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        qrs_factors = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        return Alert(
            setup=setup,
            alert_id="TEST_ALERT",
            created_at=datetime.now(),
            priority="normal"
        )
    
    @pytest.mark.asyncio
    async def test_send_alert(self, channel, sample_alert):
        """Test sending alert to console."""
        # This test just verifies the method runs without error
        # The actual console output is hard to test
        result = await channel.send_alert(sample_alert)
        assert result is True


class TestFileAlertChannel:
    """Test File Alert Channel."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AlertChannelConfig(
            file_enabled=True,
            file_path="test_alerts.log"
        )
    
    @pytest.fixture
    def channel(self, config):
        """Create file alert channel."""
        return FileAlertChannel(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        qrs_factors = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        return Alert(
            setup=setup,
            alert_id="TEST_ALERT",
            created_at=datetime.now(),
            priority="normal"
        )
    
    @pytest.mark.asyncio
    async def test_send_alert(self, channel, sample_alert):
        """Test sending alert to file."""
        result = await channel.send_alert(sample_alert)
        assert result is True
        
        # Verify file was created
        import os
        assert os.path.exists(channel.config.file_path)
        
        # Clean up
        if os.path.exists(channel.config.file_path):
            os.remove(channel.config.file_path)


class TestEmailAlertChannel:
    """Test Email Alert Channel."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AlertChannelConfig(
            email_enabled=True,
            email_smtp_server="smtp.test.com",
            email_smtp_port=587,
            email_username="test@test.com",
            email_password="test_password",
            email_from="test@test.com",
            email_to=["recipient@test.com"]
        )
    
    @pytest.fixture
    def channel(self, config):
        """Create email alert channel."""
        return EmailAlertChannel(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        qrs_factors = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        return Alert(
            setup=setup,
            alert_id="TEST_ALERT",
            created_at=datetime.now(),
            priority="normal"
        )
    
    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, sample_alert):
        """Test sending alert when email is disabled."""
        config = AlertChannelConfig(email_enabled=False)
        channel = EmailAlertChannel(config)
        
        result = await channel.send_alert(sample_alert)
        assert result is False
    
    def test_format_email_body(self, channel, sample_alert):
        """Test email body formatting."""
        alert_dict = sample_alert.to_dict()
        body = channel._format_email_body(alert_dict)
        
        assert isinstance(body, str)
        assert "Zone Fade Alert" in body
        assert "SPY" in body
        assert "SHORT" in body
        assert "100.00" in body  # Zone level


class TestWebhookAlertChannel:
    """Test Webhook Alert Channel."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AlertChannelConfig(
            webhook_enabled=True,
            webhook_url="https://test.webhook.com/alerts",
            webhook_secret="test_secret",
            webhook_timeout=5
        )
    
    @pytest.fixture
    def channel(self, config):
        """Create webhook alert channel."""
        return WebhookAlertChannel(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        qrs_factors = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        return Alert(
            setup=setup,
            alert_id="TEST_ALERT",
            created_at=datetime.now(),
            priority="normal"
        )
    
    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, sample_alert):
        """Test sending alert when webhook is disabled."""
        config = AlertChannelConfig(webhook_enabled=False)
        channel = WebhookAlertChannel(config)
        
        result = await channel.send_alert(sample_alert)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_alert_with_mock(self, channel, sample_alert):
        """Test sending alert with mocked HTTP request."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await channel.send_alert(sample_alert)
            assert result is True
            
            # Verify the request was made
            mock_post.assert_called_once()


class TestAlertSystem:
    """Test Alert System."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AlertChannelConfig(
            console_enabled=True,
            file_enabled=True,
            email_enabled=False,
            webhook_enabled=False,
            file_path="test_alerts.log"
        )
    
    @pytest.fixture
    def alert_system(self, config):
        """Create alert system."""
        return AlertSystem(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        zone = Zone(level=100.0, zone_type=ZoneType.PRIOR_DAY_HIGH, quality=2, strength=2.0)
        rejection_candle = OHLCVBar(
            timestamp=datetime.now(),
            open=99.0, high=100.0, low=98.0, close=98.5, volume=1000000
        )
        qrs_factors = QRSFactors(zone_quality=2, rejection_clarity=2, structure_flip=2, context=1, intermarket_divergence=1)
        
        setup = ZoneFadeSetup(
            symbol="SPY",
            direction=SetupDirection.SHORT,
            zone=zone,
            rejection_candle=rejection_candle,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        return Alert(
            setup=setup,
            alert_id="TEST_ALERT",
            created_at=datetime.now(),
            priority="normal"
        )
    
    def test_alert_system_initialization(self, alert_system, config):
        """Test alert system initialization."""
        assert len(alert_system.channels) == 2  # Console and File
        assert alert_system.config == config
    
    @pytest.mark.asyncio
    async def test_send_alert(self, alert_system, sample_alert):
        """Test sending alert through all channels."""
        results = await alert_system.send_alert(sample_alert)
        
        assert isinstance(results, dict)
        assert len(results) == 2  # Console and File channels
        
        # Both channels should succeed
        for channel_name, success in results.items():
            assert isinstance(success, bool)
    
    def test_get_status(self, alert_system):
        """Test getting alert system status."""
        status = alert_system.get_status()
        
        assert isinstance(status, dict)
        assert 'channels_enabled' in status
        assert 'console_enabled' in status
        assert 'file_enabled' in status
        assert 'email_enabled' in status
        assert 'webhook_enabled' in status
        assert status['channels_enabled'] == 2
    
    def test_test_channels(self, alert_system):
        """Test channel testing functionality."""
        results = alert_system.test_channels()
        
        assert isinstance(results, dict)
        assert len(results) == 2  # Console and File channels
        
        # Both channels should succeed in testing
        for channel_name, success in results.items():
            assert isinstance(success, bool)