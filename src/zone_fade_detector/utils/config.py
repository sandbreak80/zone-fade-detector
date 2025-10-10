"""
Configuration management utilities.

This module provides functions for loading and validating configuration
from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the loaded configuration
        
    Raises:
        ConfigError: If configuration file cannot be loaded or is invalid
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Substitute environment variables
        content = os.path.expandvars(content)
        
        # Load YAML
        config = yaml.safe_load(content)
        
        if not config:
            raise ConfigError(f"Configuration file {config_path} is empty or invalid")
        
        return config
        
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigError: If configuration is invalid
    """
    # Basic validation - can be extended with Pydantic models
    required_sections = ['symbols', 'polling', 'indicators', 'zones', 'alerts']
    
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")
    
    # Validate symbols
    if not isinstance(config['symbols'], list) or not config['symbols']:
        raise ConfigError("Symbols must be a non-empty list")
    
    # Validate polling configuration
    polling = config['polling']
    if not isinstance(polling.get('interval_seconds'), int) or polling['interval_seconds'] <= 0:
        raise ConfigError("Polling interval must be a positive integer")
    
    # Validate QRS threshold
    if 'qrs' in config and 'a_setup_threshold' in config['qrs']:
        threshold = config['qrs']['a_setup_threshold']
        if not isinstance(threshold, int) or not 0 <= threshold <= 10:
            raise ConfigError("QRS A-setup threshold must be between 0 and 10")


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get environment variable with optional default and required validation.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        required: Whether the variable is required
        
    Returns:
        Environment variable value
        
    Raises:
        ConfigError: If required variable is not set
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ConfigError(f"Required environment variable not set: {key}")
    
    return value