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
@click.option(
    "--test-alerts",
    is_flag=True,
    help="Test alert channels and exit",
)
@click.option(
    "--replay",
    is_flag=True,
    help="Run in replay mode for historical data",
)
@click.option(
    "--start-date",
    type=str,
    help="Start date for replay mode (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=str,
    help="End date for replay mode (YYYY-MM-DD)",
)
@click.option(
    "--provider",
    type=click.Choice(["alpaca", "polygon", "both"]),
    default="alpaca",
    help="Data provider for replay mode",
)
@click.option(
    "--live",
    is_flag=True,
    help="Run in live mode with real-time data and Discord alerts",
)
def main(
    config: Path,
    log_level: Optional[str],
    symbols: tuple,
    poll_interval: Optional[int],
    dry_run: bool,
    verbose: bool,
    test_alerts: bool,
    replay: bool,
    start_date: Optional[str],
    end_date: Optional[str],
    provider: str,
    live: bool,
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
        
        # Handle different modes
        if replay:
            console.print(f"[yellow]Replay Mode: {start_date} to {end_date} using {provider}[/yellow]")
        elif live:
            console.print("[green]Live Mode: Running during RTH[/green]")
        else:
            console.print("[blue]Standard Mode: Continuous monitoring[/blue]")
        
        console.print()
        
        # Initialize detector
        detector = ZoneFadeDetector(config_data)
        
        # Test alerts if requested
        if test_alerts:
            console.print("[yellow]Testing alert channels...[/yellow]")
            asyncio.run(test_alert_system(detector))
            return
        
        # Run in appropriate mode
        if replay:
            asyncio.run(run_replay_mode(detector, start_date, end_date, provider))
        elif live:
            asyncio.run(run_live_mode(detector))
        else:
            asyncio.run(detector.run())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


async def test_alert_system(detector: ZoneFadeDetector) -> None:
    """Test alert system with Discord webhook."""
    console = Console()
    
    try:
        console.print("[blue]Testing Zone Fade Detector Alert System...[/blue]")
        
        # Test the alert system
        results = await detector.test_alert_system()
        
        console.print("\n[bold]Alert System Test Results:[/bold]")
        
        if 'error' in results:
            console.print(f"[red]❌ Error: {results['error']}[/red]")
            return
        
        success_count = 0
        for channel, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            console.print(f"  {channel}: {status}")
            if success:
                success_count += 1
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total Channels: {len(results)}")
        console.print(f"  Successful: {success_count}")
        console.print(f"  Failed: {len(results) - success_count}")
        
        if success_count == len(results):
            console.print("[green]✅ All alert channels working![/green]")
            console.print("[green]Discord webhook is ready for Zone Fade alerts![/green]")
        else:
            console.print("[yellow]⚠️ Some alert channels failed. Check configuration.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Error testing alert system: {e}[/red]")
        console.print_exception()


async def run_replay_mode(detector: ZoneFadeDetector, start_date: str, end_date: str, provider: str) -> None:
    """Run in replay mode for historical data."""
    from datetime import datetime, timedelta
    import json
    import os
    
    console = Console()
    console.print(f"[yellow]Starting replay mode: {start_date} to {end_date}[/yellow]")
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create signals directory
    os.makedirs("signals", exist_ok=True)
    
    # Run replay for each day
    current_date = start_dt
    total_alerts = 0
    
    while current_date <= end_dt:
        console.print(f"[blue]Processing {current_date.strftime('%Y-%m-%d')}...[/blue]")
        
        try:
            # Fetch historical data for the day
            symbol_data = await detector.data_manager.get_multiple_symbols(
                detector.symbols,
                current_date,
                current_date + timedelta(days=1)
            )
            
            # Process signals
            alerts = detector.signal_processor.process_signals(symbol_data)
            
            # Save signals to file
            if alerts:
                signal_file = f"signals/{current_date.strftime('%Y-%m-%d')}.jsonl"
                with open(signal_file, 'a') as f:
                    for alert in alerts:
                        f.write(json.dumps(alert.to_dict()) + '\n')
                
                console.print(f"[green]Found {len(alerts)} signals for {current_date.strftime('%Y-%m-%d')}[/green]")
                total_alerts += len(alerts)
            else:
                console.print(f"[dim]No signals for {current_date.strftime('%Y-%m-%d')}[/dim]")
        
        except Exception as e:
            console.print(f"[red]Error processing {current_date.strftime('%Y-%m-%d')}: {e}[/red]")
        
        current_date += timedelta(days=1)
    
    console.print(f"[bold green]Replay complete! Total alerts: {total_alerts}[/bold green]")


async def run_live_mode(detector: ZoneFadeDetector) -> None:
    """Run in live mode during RTH."""
    from datetime import datetime, time
    
    console = Console()
    console.print("[green]Starting live mode...[/green]")
    
    # Check if we're in RTH
    now = datetime.now().time()
    rth_start = time(9, 30)  # 9:30 AM
    rth_end = time(16, 0)    # 4:00 PM
    
    if not (rth_start <= now <= rth_end):
        console.print("[yellow]Warning: Not in regular trading hours (9:30 AM - 4:00 PM ET)[/yellow]")
        console.print("[yellow]Continuing anyway...[/yellow]")
    
    # Run the detector
    await detector.run()


if __name__ == "__main__":
    main()