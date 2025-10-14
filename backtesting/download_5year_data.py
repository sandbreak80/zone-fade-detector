#!/usr/bin/env python3
"""
Download 5 years of historical data (2020-2024) for Zone Fade Detector testing.
This script downloads and caches data for 5 years to get more comprehensive backtesting.
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

from zone_fade_detector.data.data_manager import DataManager, DataManagerConfig, DataSource
from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig
from zone_fade_detector.data.polygon_client import PolygonClient, PolygonConfig
from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def send_discord_status(message: str, alert_system: AlertSystem):
    """Send status update to Discord."""
    try:
        # Create a simple status alert
        from zone_fade_detector.core.models import Alert, ZoneFadeSetup, Zone, ZoneType, SetupDirection, QRSFactors, OHLCVBar
        from datetime import datetime
        
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

async def download_5year_data():
    """Download and cache 5 years of data (2020-2024) for all symbols."""
    
    # Set up Discord alerts for status updates
    alert_config = AlertChannelConfig(
        console_enabled=True,
        file_enabled=True,
        webhook_enabled=True,
        webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
        webhook_timeout=10
    )
    alert_system = AlertSystem(alert_config)
    
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
        cache_dir='/tmp/cache',  # Use /tmp which is always writable
        cache_ttl=86400 * 30,  # 30 days cache
        primary_source=DataSource.ALPACA
    )
    
    data_manager = DataManager(data_config)
    
    # Define symbols and date range for 5 years
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = datetime(2020, 1, 1, 9, 30)  # Start of 2020
    end_date = datetime(2024, 12, 31, 16, 0)  # End of 2024
    
    # Create data directory (inside container, mounted to host)
    data_dir = Path('/app/data/5year')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📅 Downloading 5-year data from {start_date} to {end_date}")
    logger.info(f"📊 Symbols: {', '.join(symbols)}")
    logger.info(f"💾 Saving to: {data_dir}")
    
    # Send initial status to Discord
    await send_discord_status(f"🚀 Starting 5-year data download (2020-2024) for {', '.join(symbols)}", alert_system)
    
    all_data = {}
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"📈 Downloading {symbol} data... ({i}/{len(symbols)})")
        
        # Send progress update to Discord
        await send_discord_status(f"📈 Downloading {symbol} data... ({i}/{len(symbols)})", alert_system)
        
        try:
            # Download data in chunks to avoid API limits
            bars = await data_manager.get_bars(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
            logger.info(f"✅ {symbol}: Retrieved {len(bars)} bars")
            
            if bars:
                # Save individual symbol data
                symbol_file = data_dir / f"{symbol}_5year.pkl"
                with open(symbol_file, 'wb') as f:
                    pickle.dump(bars, f)
                logger.info(f"💾 Saved {symbol} data to {symbol_file}")
                
                all_data[symbol] = bars
                
                # Log sample data
                if len(bars) >= 5:
                    logger.info(f"   Sample: {bars[0].timestamp} to {bars[-1].timestamp}")
                    logger.info(f"   Price range: ${min(bar.low for bar in bars):.2f} - ${max(bar.high for bar in bars):.2f}")
                
                # Send success update to Discord
                await send_discord_status(f"✅ {symbol}: {len(bars)} bars downloaded and saved", alert_system)
            else:
                logger.warning(f"⚠️ No data retrieved for {symbol}")
                await send_discord_status(f"⚠️ No data retrieved for {symbol}", alert_system)
                
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
            await send_discord_status(f"❌ Error downloading {symbol}: {str(e)[:100]}", alert_system)
            continue
    
    # Save combined data
    if all_data:
        combined_file = data_dir / "all_symbols_5year.pkl"
        with open(combined_file, 'wb') as f:
            pickle.dump(all_data, f)
        logger.info(f"💾 Saved combined data to {combined_file}")
        
        # Log summary
        total_bars = sum(len(bars) for bars in all_data.values())
        logger.info(f"📊 Summary: {total_bars} total bars across {len(all_data)} symbols")
        
        # Calculate file sizes
        total_size = sum(f.stat().st_size for f in data_dir.glob("*.pkl"))
        logger.info(f"💾 Total data size: {total_size / (1024*1024):.1f} MB")
        
        # Send final summary to Discord
        summary_msg = f"🎉 5-Year Download Complete!\n📊 {total_bars:,} total bars across {len(all_data)} symbols\n💾 {total_size / (1024*1024):.1f} MB saved\n📁 Location: {data_dir}"
        await send_discord_status(summary_msg, alert_system)
        
    else:
        logger.error("❌ No data was successfully downloaded")
        await send_discord_status("❌ 5-year download failed - no data retrieved", alert_system)

if __name__ == "__main__":
    asyncio.run(download_5year_data())