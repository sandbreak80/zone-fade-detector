#!/usr/bin/env python3
"""
Test Zone Fade detection on 2024 historical data.
This script loads the cached 2024 data and runs detection logic.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import logging

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
from zone_fade_detector.strategies.signal_processor import SignalProcessor, SignalProcessorConfig
from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneFadeSetup, SetupDirection, QRSFactors

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_2024_detection():
    """Test Zone Fade detection on 2024 data."""
    
    # Load cached data (container path)
    data_dir = Path('/app/data/2024')
    combined_file = data_dir / "all_symbols_2024.pkl"
    
    if not combined_file.exists():
        logger.error(f"❌ Data file not found: {combined_file}")
        logger.error("Please run download_2024_data.py first")
        return
    
    logger.info(f"📂 Loading 2024 data from {combined_file}")
    
    with open(combined_file, 'rb') as f:
        all_data = pickle.load(f)
    
    logger.info(f"📊 Loaded data for {len(all_data)} symbols")
    for symbol, bars in all_data.items():
        logger.info(f"   {symbol}: {len(bars)} bars")
        if bars:
            logger.info(f"      Date range: {bars[0].timestamp} to {bars[-1].timestamp}")
    
    # Set up alert system with enhanced Discord messaging
    alert_config = AlertChannelConfig(
        console_enabled=True,
        file_enabled=True,
        webhook_enabled=True,
        webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
        webhook_secret=None,
        email_enabled=False,
        file_path='/tmp/alerts_2024.log'
    )
    
    alert_system = AlertSystem(alert_config)
    
    # Set up signal processor with very low threshold for testing
    processor_config = SignalProcessorConfig(
        min_qrs_score=1,  # Very low threshold to catch any setups
        max_setups_per_symbol=50,  # Increased for testing
        setup_cooldown_minutes=1,  # Minimal cooldown for testing
        alert_deduplication_minutes=1,  # Minimal deduplication for testing
        enable_intermarket_filtering=False,  # Disabled for testing
        enable_volume_filtering=False  # Disabled for testing
    )
    signal_processor = SignalProcessor(processor_config)
    
    logger.info("🎯 Starting Zone Fade detection on 2024 data...")
    logger.info(f"⚙️ QRS Threshold: {processor_config.min_qrs_score}")
    logger.info(f"⚙️ Max setups per symbol: {processor_config.max_setups_per_symbol}")
    
    # Process signals
    alerts = signal_processor.process_signals(all_data)
    
    logger.info(f"🚨 Generated {len(alerts)} Zone Fade alerts!")
    
    if alerts:
        logger.info("📤 Sending alerts to Discord...")
        
        for i, alert in enumerate(alerts, 1):
            logger.info(f"   Alert {i}/{len(alerts)}: {alert.setup.symbol} {alert.setup.direction.value.upper()} - QRS: {alert.setup.qrs_score}/10")
            
            # Send to Discord
            success = await alert_system.send_alert(alert)
            if success:
                logger.info(f"   ✅ Alert {alert.alert_id} sent successfully")
            else:
                logger.error(f"   ❌ Failed to send alert {alert.alert_id}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        # Summary
        logger.info(f"📊 Detection Summary:")
        logger.info(f"   Total alerts: {len(alerts)}")
        
        # Group by symbol
        by_symbol = {}
        for alert in alerts:
            symbol = alert.setup.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(alert)
        
        for symbol, symbol_alerts in by_symbol.items():
            logger.info(f"   {symbol}: {len(symbol_alerts)} alerts")
            for alert in symbol_alerts:
                logger.info(f"      {alert.setup.direction.value.upper()} - QRS: {alert.setup.qrs_score}/10 - {alert.created_at}")
        
        # Group by QRS score
        by_qrs = {}
        for alert in alerts:
            qrs = alert.setup.qrs_score
            if qrs not in by_qrs:
                by_qrs[qrs] = []
            by_qrs[qrs].append(alert)
        
        logger.info(f"   QRS Score distribution:")
        for qrs in sorted(by_qrs.keys(), reverse=True):
            logger.info(f"      {qrs}/10: {len(by_qrs[qrs])} alerts")
        
    else:
        logger.info("ℹ️ No Zone Fade setups detected in 2024 data")
        logger.info("💡 This could mean:")
        logger.info("   - Zone Fade setups are rare (as expected)")
        logger.info("   - QRS threshold too high (try lowering to 3-4)")
        logger.info("   - Need to check detection logic")

if __name__ == "__main__":
    asyncio.run(test_2024_detection())
