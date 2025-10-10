"""
Unit tests for data clients.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig, AlpacaAPIError
from zone_fade_detector.data.polygon_client import PolygonClient, PolygonConfig, PolygonAPIError
from zone_fade_detector.data.data_manager import DataManager, DataManagerConfig, DataSource


class TestAlpacaConfig:
    """Test Alpaca configuration."""
    
    def test_config_creation(self):
        """Test creating Alpaca configuration."""
        config = AlpacaConfig(
            api_key="test_key",
            secret_key="test_secret",
            base_url="https://test.api.com"
        )
        
        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.base_url == "https://test.api.com"
        assert config.timeout == 30
        assert config.max_retries == 3


class TestAlpacaClient:
    """Test Alpaca client functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AlpacaConfig(
            api_key="test_key",
            secret_key="test_secret"
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return AlpacaClient(config)
    
    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client.client is not None
        assert client.logger is not None
    
    def test_validate_symbol(self, client):
        """Test symbol validation."""
        assert client.validate_symbol("SPY") is True
        assert client.validate_symbol("QQQ") is True
        assert client.validate_symbol("IWM") is True
        assert client.validate_symbol("AAPL") is False
        assert client.validate_symbol("invalid") is False
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        with patch.object(client, 'get_bars', return_value=[Mock()]) as mock_get_bars:
            result = await client.health_check()
            assert result is True
            mock_get_bars.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test failed health check."""
        with patch.object(client, 'get_bars', side_effect=Exception("API Error")):
            result = await client.health_check()
            assert result is False


class TestPolygonConfig:
    """Test Polygon configuration."""
    
    def test_config_creation(self):
        """Test creating Polygon configuration."""
        config = PolygonConfig(
            api_key="test_key",
            base_url="https://test.api.com"
        )
        
        assert config.api_key == "test_key"
        assert config.base_url == "https://test.api.com"
        assert config.timeout == 30
        assert config.max_retries == 3


class TestPolygonClient:
    """Test Polygon client functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PolygonConfig(api_key="test_key")
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return PolygonClient(config)
    
    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client.client is not None
        assert client.logger is not None
    
    def test_validate_symbol(self, client):
        """Test symbol validation."""
        assert client.validate_symbol("SPY") is True
        assert client.validate_symbol("QQQ") is True
        assert client.validate_symbol("IWM") is True
        assert client.validate_symbol("AAPL") is False
        assert client.validate_symbol("invalid") is False
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        with patch.object(client, 'get_market_status', return_value={'market': 'open'}) as mock_status:
            result = await client.health_check()
            assert result is True
            mock_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test failed health check."""
        with patch.object(client, 'get_market_status', side_effect=Exception("API Error")):
            result = await client.health_check()
            assert result is False


class TestDataManagerConfig:
    """Test DataManager configuration."""
    
    def test_config_creation(self):
        """Test creating DataManager configuration."""
        alpaca_config = AlpacaConfig("alpaca_key", "alpaca_secret")
        polygon_config = PolygonConfig("polygon_key")
        
        config = DataManagerConfig(
            alpaca_config=alpaca_config,
            polygon_config=polygon_config,
            cache_dir="test_cache",
            primary_source=DataSource.ALPACA
        )
        
        assert config.alpaca_config == alpaca_config
        assert config.polygon_config == polygon_config
        assert config.cache_dir == "test_cache"
        assert config.primary_source == DataSource.ALPACA
        assert config.cache_ttl == 3600


class TestDataManager:
    """Test DataManager functionality."""
    
    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca configuration."""
        return AlpacaConfig("alpaca_key", "alpaca_secret")
    
    @pytest.fixture
    def polygon_config(self):
        """Create Polygon configuration."""
        return PolygonConfig("polygon_key")
    
    @pytest.fixture
    def manager_config(self, alpaca_config, polygon_config):
        """Create DataManager configuration."""
        return DataManagerConfig(
            alpaca_config=alpaca_config,
            polygon_config=polygon_config,
            cache_dir="test_cache"
        )
    
    @pytest.fixture
    def data_manager(self, manager_config):
        """Create DataManager instance."""
        return DataManager(manager_config)
    
    def test_manager_initialization(self, data_manager, manager_config):
        """Test DataManager initialization."""
        assert data_manager.config == manager_config
        assert data_manager.cache is not None
        assert data_manager.alpaca_client is not None
        assert data_manager.polygon_client is not None
        assert data_manager.supported_symbols == ['SPY', 'QQQ', 'IWM']
    
    def test_validate_symbol(self, data_manager):
        """Test symbol validation."""
        assert data_manager.validate_symbol("SPY") is True
        assert data_manager.validate_symbol("QQQ") is True
        assert data_manager.validate_symbol("IWM") is True
        assert data_manager.validate_symbol("AAPL") is False
    
    def test_get_cache_key(self, data_manager):
        """Test cache key generation."""
        start = datetime(2024, 1, 1, 9, 30)
        end = datetime(2024, 1, 1, 16, 0)
        
        key = data_manager._get_cache_key("SPY", start, end, "alpaca")
        expected = "alpaca:SPY:2024-01-01T09:30:00:2024-01-01T16:00:00"
        assert key == expected
    
    @pytest.mark.asyncio
    async def test_health_check(self, data_manager):
        """Test health check for all sources."""
        with patch.object(data_manager.alpaca_client, 'health_check', return_value=True), \
             patch.object(data_manager.polygon_client, 'health_check', return_value=True):
            
            health = await data_manager.health_check()
            
            assert health['alpaca'] is True
            assert health['polygon'] is True
    
    def test_clear_cache(self, data_manager):
        """Test cache clearing."""
        # Add some data to cache
        data_manager.cache.set("test_key", "test_value")
        assert len(data_manager.cache) == 1
        
        # Clear cache
        data_manager.clear_cache()
        assert len(data_manager.cache) == 0
    
    def test_get_cache_stats(self, data_manager):
        """Test cache statistics."""
        stats = data_manager.get_cache_stats()
        
        assert 'size' in stats
        assert 'hit_count' in stats
        assert 'miss_count' in stats
        assert isinstance(stats['size'], int)
        assert isinstance(stats['hit_count'], int)
        assert isinstance(stats['miss_count'], int)


class TestDataSource:
    """Test DataSource enum."""
    
    def test_data_source_values(self):
        """Test DataSource enum values."""
        assert DataSource.ALPACA.value == "alpaca"
        assert DataSource.POLYGON.value == "polygon"
        assert DataSource.BOTH.value == "both"
    
    def test_data_source_enumeration(self):
        """Test DataSource enumeration."""
        sources = list(DataSource)
        assert len(sources) == 3
        assert DataSource.ALPACA in sources
        assert DataSource.POLYGON in sources
        assert DataSource.BOTH in sources