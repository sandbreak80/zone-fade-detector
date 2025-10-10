"""
Logging configuration utilities.

This module provides functions for setting up structured logging
with appropriate formatters and handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup structured logging based on configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    # Extract configuration
    level = config.get('level', 'INFO')
    log_file = config.get('file')
    console_output = config.get('console_output', True)
    format_string = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True
        )
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)