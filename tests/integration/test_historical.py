#!/usr/bin/env python3
"""
Test script for Zone Fade Detector with historical data.
This script tests the detection logic using data from a specific trading day.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.detector import ZoneFadeDetector
from zone_fade_detector.core.models import Alert
from zone_fade_detector.data.data_manager import DataManager, DataManagerConfig, DataSource
from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig
from zone_fade_detector.data.polygon_client import PolygonClient, PolygonConfig
from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
from zone_fade_detector.strategies.signal_processor import SignalProcessor, SignalProcessorConfig
from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneFadeSetup, SetupDirection, QRSFactors
import yaml
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_historical_detection():
    """Test Zone Fade detection with historical data."""
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up data manager
    alpaca_config = AlpacaConfig(
        api_key=os.getenv('ALPACA_API_KEY', ''),
        secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
        base_url='https://paper-api.alpaca.markets'
    )
    
    polygon_config = PolygonConfig(
        api_key=os.getenv('POLYGON_API_KEY', '')
    )
    
    data_config = DataManagerConfig(
        alpaca_config=alpaca_config,
        polygon_config=polygon_config,
        cache_dir='/tmp/cache',
        cache_ttl=3600,
        primary_source=DataSource.ALPACA
    )
    
    data_manager = DataManager(data_config)
    
    # Set up alert system with enhanced Discord messaging
    alert_config = AlertChannelConfig(
        console_enabled=True,
        file_enabled=True,
        webhook_enabled=True,
        webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
        webhook_secret=None,
        email_enabled=False,
        file_path='/tmp/alerts.log'
    )
    
    alert_system = AlertSystem(alert_config)
    
    # Set up signal processor
    processor_config = SignalProcessorConfig(
        min_qrs_score=7,
        max_setups_per_symbol=3,
        setup_cooldown_minutes=15,
        alert_deduplication_minutes=5,
        enable_intermarket_filtering=True,
        enable_volume_filtering=True
    )
    signal_processor = SignalProcessor(processor_config)
    
    # Test with historical data from multiple trading days
    # Let's use data from October 7-8, 2024 (Monday-Tuesday with good market activity)
    test_date = datetime(2024, 10, 7, 9, 30)  # Previous day market open
    end_date = datetime(2024, 10, 8, 16, 0)   # Current day market close
    
    logger.info(f"Testing with historical data from {test_date} to {end_date}")
    
    # Fetch historical data for SPY
    symbol = 'SPY'
    logger.info(f"Fetching historical data for {symbol}...")
    
    try:
        # Get bars for the entire trading day
        bars = await data_manager.get_bars(
            symbol=symbol,
            start=test_date,
            end=end_date
        )
        
        logger.info(f"Retrieved {len(bars)} bars for {symbol}")
        
        if not bars:
            logger.error("No historical data retrieved")
            return
        
        # Process the data through the signal processor
        logger.info("Processing signals...")
        alerts = signal_processor.process_signals({symbol: bars})
        
        logger.info(f"Generated {len(alerts)} alerts")
        
        # Send alerts to Discord with enhanced messaging
        if alerts:
            logger.info("Sending alerts to Discord...")
            for alert in alerts:
                success = await alert_system.send_alert(alert)
                if success:
                    logger.info(f"✅ Alert {alert.alert_id} sent successfully")
                else:
                    logger.error(f"❌ Failed to send alert {alert.alert_id}")
        else:
            logger.info("No Zone Fade setups detected in historical data")
            
            # Let's analyze why no setups were found
            logger.info("Analyzing data for potential issues...")
            
            # Check if we have enough data
            if len(bars) < 100:
                logger.warning(f"Only {len(bars)} bars available - may not be enough for analysis")
            
            # Check data quality
            recent_bars = bars[-10:] if len(bars) >= 10 else bars
            logger.info(f"Recent bars sample: {[f'{bar.timestamp}: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume}' for bar in recent_bars]}")
            
    except Exception as e:
        logger.error(f"Error during historical test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_historical_detection())
