"""
Zone Fade Detector - Main Entry Point

This module provides the main entry point for the Zone Fade Detector application.
It handles command-line arguments, configuration loading, and application startup.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import structlog
from rich.console import Console
from rich.logging import RichHandler

from zone_fade_detector.core.detector import ZoneFadeDetector
from zone_fade_detector.utils.config import load_config
from zone_fade_detector.utils.logging import setup_logging


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="config/config.yaml",
    help="Path to configuration file",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default=None,
    help="Override log level from configuration",
)
@click.option(
    "--symbols",
    multiple=True,
    help="Override symbols from configuration (can be used multiple times)",
)
@click.option(
    "--poll-interval",
    type=int,
    help="Override polling interval in seconds",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run in dry-run mode (no actual API calls)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(
    config: Path,
    log_level: Optional[str],
    symbols: tuple,
    poll_interval: Optional[int],
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Zone Fade Detector - Identify high-probability reversal setups.

    This application monitors SPY, QQQ, and IWM for Zone Fade trading opportunities
    using 15-minute delayed data from Alpaca and Polygon APIs.
    """
    console = Console()
    
    try:
        # Load configuration
        config_data = load_config(config)
        
        # Override configuration with command-line arguments
        if log_level:
            config_data["logging"]["level"] = log_level
        if symbols:
            config_data["symbols"] = list(symbols)
        if poll_interval:
            config_data["polling"]["interval_seconds"] = poll_interval
        if dry_run:
            config_data["development"]["dry_run"] = True
        if verbose:
            config_data["development"]["verbose_logging"] = True
        
        # Setup logging
        setup_logging(config_data["logging"])
        logger = structlog.get_logger(__name__)
        
        # Display startup information
        console.print("[bold blue]Zone Fade Detector[/bold blue]")
        console.print(f"Configuration: {config}")
        console.print(f"Symbols: {', '.join(config_data['symbols'])}")
        console.print(f"Poll Interval: {config_data['polling']['interval_seconds']}s")
        console.print(f"Dry Run: {config_data['development'].get('dry_run', False)}")
        console.print()
        
        # Initialize and run detector
        detector = ZoneFadeDetector(config_data)
        
        # Run the detector
        asyncio.run(detector.run())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()