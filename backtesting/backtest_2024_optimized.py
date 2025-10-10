#!/usr/bin/env python3
"""
Optimized 2024 backtesting script with comprehensive Discord reporting.
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

async def send_discord_status(message: str, alert_system: AlertSystem):
    """Send status update to Discord."""
    try:
        from zone_fade_detector.core.models import Alert, ZoneFadeSetup, Zone, ZoneType, QRSFactors, OHLCVBar
        
        # Create minimal test setup for status message
        test_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=0, high=0, low=0, close=0, volume=0
        )
        test_zone = Zone(level=0, zone_type=ZoneType.PRIOR_DAY_HIGH, strength=0, quality=0)
        test_setup = ZoneFadeSetup(
            symbol='STATUS',
            direction=SetupDirection.LONG,
            zone=test_zone,
            rejection_candle=test_bar,
            choch_confirmed=False,
            qrs_factors=QRSFactors(),
            timestamp=datetime.now()
        )
        
        # Override the alert message
        test_alert = Alert(
            alert_id=f'STATUS_{datetime.now().strftime("%H%M%S")}',
            setup=test_setup,
            priority='INFO',
            created_at=datetime.now()
        )
        
        # Send to Discord only
        results = await alert_system.send_alert(test_alert)
        if results.get('WebhookAlertChannel', False):
            logger.info(f"✅ Discord status sent: {message}")
        else:
            logger.warning(f"⚠️ Discord status failed: {message}")
            
    except Exception as e:
        logger.error(f"❌ Discord status error: {e}")

async def backtest_2024_optimized():
    """Run optimized 2024 backtesting with comprehensive reporting."""
    
    # Load cached data
    data_dir = Path('/app/data/2024')
    combined_file = data_dir / "all_symbols_2024.pkl"
    
    if not combined_file.exists():
        logger.error(f"❌ Data file not found: {combined_file}")
        return
    
    logger.info(f"📂 Loading 2024 data from {combined_file}")
    
    with open(combined_file, 'rb') as f:
        all_data = pickle.load(f)
    
    logger.info(f"📊 Loaded data for {len(all_data)} symbols")
    for symbol, bars in all_data.items():
        logger.info(f"   {symbol}: {len(bars)} bars")
    
    # Set up Discord alerts for status updates
    alert_config = AlertChannelConfig(
        console_enabled=True,
        file_enabled=False,  # Disable file logging to avoid permission issues
        webhook_enabled=True,
        webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
        webhook_timeout=10
    )
    alert_system = AlertSystem(alert_config)
    
    # Send initial status
    await send_discord_status(f"🚀 Starting 2024 Zone Fade Backtesting\n📊 Symbols: {', '.join(all_data.keys())}\n📅 Data: {len(all_data[list(all_data.keys())[0]])} bars per symbol", alert_system)
    
    # Set up signal processor with optimized parameters
    processor_config = SignalProcessorConfig(
        min_qrs_score=4,  # Lower threshold for more setups
        max_setups_per_symbol=20,  # More setups per symbol
        setup_cooldown_minutes=30,  # 30-minute cooldown
        alert_deduplication_minutes=10,  # 10-minute deduplication
        enable_intermarket_filtering=False,  # Disabled for backtesting
        enable_volume_filtering=False  # Disabled for backtesting
    )
    signal_processor = SignalProcessor(processor_config)
    
    logger.info("🎯 Starting optimized Zone Fade detection on 2024 data...")
    logger.info(f"⚙️ QRS Threshold: {processor_config.min_qrs_score}")
    logger.info(f"⚙️ Max setups per symbol: {processor_config.max_setups_per_symbol}")
    
    # Process signals
    alerts = signal_processor.process_signals(all_data)
    
    logger.info(f"🚨 Generated {len(alerts)} Zone Fade alerts!")
    
    if alerts:
        # Send summary to Discord
        await send_discord_status(f"📊 Backtesting Complete!\n🎯 Total Setups: {len(alerts)}\n⚙️ QRS Threshold: {processor_config.min_qrs_score}", alert_system)
        
        # Group by symbol
        by_symbol = {}
        for alert in alerts:
            symbol = alert.setup.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(alert)
        
        # Send detailed breakdown to Discord
        breakdown_msg = "📈 **Setup Breakdown by Symbol:**\n"
        for symbol, symbol_alerts in by_symbol.items():
            breakdown_msg += f"• **{symbol}**: {len(symbol_alerts)} setups\n"
        
        await send_discord_status(breakdown_msg, alert_system)
        
        # Group by QRS score
        by_qrs = {}
        for alert in alerts:
            qrs = alert.setup.qrs_score
            if qrs not in by_qrs:
                by_qrs[qrs] = []
            by_qrs[qrs].append(alert)
        
        # Send QRS distribution to Discord
        qrs_msg = "⭐ **QRS Score Distribution:**\n"
        for qrs in sorted(by_qrs.keys(), reverse=True):
            qrs_msg += f"• **{qrs}/10**: {len(by_qrs[qrs])} setups\n"
        
        await send_discord_status(qrs_msg, alert_system)
        
        # Send top setups to Discord
        top_setups = sorted(alerts, key=lambda x: x.setup.qrs_score, reverse=True)[:5]
        top_msg = "🏆 **Top 5 Setups:**\n"
        for i, alert in enumerate(top_setups, 1):
            setup = alert.setup
            top_msg += f"{i}. **{setup.symbol}** {setup.direction.value.upper()} - QRS: {setup.qrs_score}/10 - Zone: {setup.zone.zone_type.value} @ ${setup.zone.level:.2f}\n"
        
        await send_discord_status(top_msg, alert_system)
        
        # Send alerts to Discord
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
            await asyncio.sleep(0.5)
        
        # Final summary
        logger.info(f"📊 Detection Summary:")
        logger.info(f"   Total alerts: {len(alerts)}")
        
        for symbol, symbol_alerts in by_symbol.items():
            logger.info(f"   {symbol}: {len(symbol_alerts)} alerts")
            for alert in symbol_alerts:
                logger.info(f"      {alert.setup.direction.value.upper()} - QRS: {alert.setup.qrs_score}/10 - {alert.created_at}")
        
        logger.info(f"   QRS Score distribution:")
        for qrs in sorted(by_qrs.keys(), reverse=True):
            logger.info(f"      {qrs}/10: {len(by_qrs[qrs])} alerts")
        
        # Send final completion message
        await send_discord_status(f"✅ **Backtesting Complete!**\n📊 Processed {len(alerts)} Zone Fade setups\n🎯 Average QRS: {sum(a.setup.qrs_score for a in alerts) / len(alerts):.1f}/10\n⏰ Completed at {datetime.now().strftime('%H:%M:%S')}", alert_system)
        
    else:
        logger.info("ℹ️ No Zone Fade setups detected in 2024 data")
        await send_discord_status("ℹ️ No Zone Fade setups detected in 2024 data\n💡 Consider lowering QRS threshold or checking detection parameters", alert_system)

if __name__ == "__main__":
    asyncio.run(backtest_2024_optimized())