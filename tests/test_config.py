"""
Tests for configuration utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from zone_fade_detector.utils.config import load_config, validate_config, get_env_var, ConfigError


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.yaml"
        config_content = """
symbols:
  - SPY
  - QQQ
  - IWM
polling:
  interval_seconds: 30
indicators:
  vwap:
    enabled: true
zones:
  prior_day: true
alerts:
  channels: ['console']
"""
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        
        assert config['symbols'] == ['SPY', 'QQQ', 'IWM']
        assert config['polling']['interval_seconds'] == 30
        assert config['indicators']['vwap']['enabled'] is True
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config(Path("nonexistent.yaml"))
    
    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(config_file)
    
    def test_load_empty_file(self, tmp_path):
        """Test loading an empty configuration file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        
        with pytest.raises(ConfigError, match="Configuration file.*is empty"):
            load_config(config_file)


class TestValidateConfig:
    """Test configuration validation functionality."""
    
    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config = {
            'symbols': ['SPY', 'QQQ'],
            'polling': {'interval_seconds': 30},
            'indicators': {'vwap': {'enabled': True}},
            'zones': {'prior_day': True},
            'alerts': {'channels': ['console']}
        }
        
        # Should not raise any exception
        validate_config(config)
    
    def test_validate_missing_section(self):
        """Test validating config with missing required section."""
        config = {
            'symbols': ['SPY'],
            'polling': {'interval_seconds': 30}
            # Missing indicators, zones, alerts
        }
        
        with pytest.raises(ConfigError, match="Missing required configuration section"):
            validate_config(config)
    
    def test_validate_invalid_symbols(self):
        """Test validating config with invalid symbols."""
        config = {
            'symbols': [],  # Empty list
            'polling': {'interval_seconds': 30},
            'indicators': {'vwap': {'enabled': True}},
            'zones': {'prior_day': True},
            'alerts': {'channels': ['console']}
        }
        
        with pytest.raises(ConfigError, match="Symbols must be a non-empty list"):
            validate_config(config)
    
    def test_validate_invalid_polling_interval(self):
        """Test validating config with invalid polling interval."""
        config = {
            'symbols': ['SPY'],
            'polling': {'interval_seconds': -1},  # Invalid
            'indicators': {'vwap': {'enabled': True}},
            'zones': {'prior_day': True},
            'alerts': {'channels': ['console']}
        }
        
        with pytest.raises(ConfigError, match="Polling interval must be a positive integer"):
            validate_config(config)
    
    def test_validate_qrs_threshold(self):
        """Test validating QRS threshold."""
        config = {
            'symbols': ['SPY'],
            'polling': {'interval_seconds': 30},
            'indicators': {'vwap': {'enabled': True}},
            'zones': {'prior_day': True},
            'alerts': {'channels': ['console']},
            'qrs': {'a_setup_threshold': 15}  # Invalid range
        }
        
        with pytest.raises(ConfigError, match="QRS A-setup threshold must be between 0 and 10"):
            validate_config(config)


class TestGetEnvVar:
    """Test environment variable utilities."""
    
    def test_get_existing_env_var(self):
        """Test getting an existing environment variable."""
        with patch.dict('os.environ', {'TEST_VAR': 'test_value'}):
            result = get_env_var('TEST_VAR')
            assert result == 'test_value'
    
    def test_get_missing_env_var_with_default(self):
        """Test getting a missing environment variable with default."""
        result = get_env_var('MISSING_VAR', default='default_value')
        assert result == 'default_value'
    
    def test_get_missing_env_var_required(self):
        """Test getting a missing required environment variable."""
        with pytest.raises(ConfigError, match="Required environment variable not set"):
            get_env_var('MISSING_VAR', required=True)
    
    def test_get_missing_env_var_no_default(self):
        """Test getting a missing environment variable without default."""
        result = get_env_var('MISSING_VAR')
        assert result is None