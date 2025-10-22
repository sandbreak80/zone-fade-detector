#!/usr/bin/env python3
"""
Simple test script for Bitcoin Zone Fade Detector.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def test_bitcoin_simple():
    """Simple test of Bitcoin detector components."""
    print("🚀 Simple Bitcoin Zone Fade Detector Test")
    print("=" * 50)
    
    try:
        # Test crypto client directly
        print("\n📊 Testing CoinGecko API Client...")
        from zone_fade_detector.data.crypto_client import CryptoClient, CryptoConfig
        
        crypto_config = CryptoConfig()
        async with CryptoClient(crypto_config) as client:
            # Test health check
            health = await client.health_check()
            print(f"✅ CoinGecko API Health: {health}")
            
            if health:
                # Test getting Bitcoin data
                print("📈 Fetching Bitcoin data...")
                bars = await client.get_ohlc('bitcoin', 'usd', 1)  # 1 day
                
                if bars:
                    print(f"✅ Successfully fetched {len(bars)} Bitcoin bars")
                    latest = bars[-1]
                    print(f"   Latest: {latest.timestamp} - O:${latest.open:,.2f} H:${latest.high:,.2f} L:${latest.low:,.2f} C:${latest.close:,.2f}")
                    
                    # Show price range
                    prices = [bar.close for bar in bars]
                    min_price = min(prices)
                    max_price = max(prices)
                    print(f"   Range: ${min_price:,.2f} - ${max_price:,.2f} (${max_price-min_price:,.2f} spread)")
                else:
                    print("❌ No Bitcoin data received")
                    return False
            else:
                print("❌ CoinGecko API not accessible")
                return False
        
        # Test Bitcoin data manager
        print("\n🪙 Testing Bitcoin Data Manager...")
        from zone_fade_detector.data.bitcoin_data_manager import BitcoinDataManager, BitcoinDataManagerConfig
        
        # Use a temp directory for cache to avoid permission issues
        import tempfile
        temp_cache_dir = tempfile.mkdtemp(prefix='bitcoin_test_cache_')
        data_config = BitcoinDataManagerConfig(
            crypto_config=crypto_config,
            cache_dir=temp_cache_dir
        )
        async with BitcoinDataManager(data_config) as data_manager:
            # Test health check
            health = await data_manager.health_check()
            print(f"✅ Bitcoin Data Manager Health: {health}")
            
            if health:
                # Test getting zones
                print("🎯 Testing zone detection...")
                zones = await data_manager.get_zones('bitcoin')
                if zones:
                    print(f"✅ Detected {len(zones)} zones for Bitcoin:")
                    for zone in zones:
                        print(f"   {zone.zone_type.value}: ${zone.level:,.2f} (quality: {zone.quality})")
                else:
                    print("ℹ️ No zones detected for Bitcoin")
        
        # Test alert system
        print("\n📱 Testing Alert System...")
        from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
        
        # Check if Discord webhook is configured
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        if webhook_url:
            print(f"✅ Discord webhook configured: {webhook_url[:50]}...")
            
            # Test alert system
            alert_config = AlertChannelConfig(
                console_enabled=True,
                file_enabled=False,  # Disable file logging for this test
                webhook_enabled=True,
                webhook_url=webhook_url,
                webhook_timeout=10
            )
            
            alert_system = AlertSystem(alert_config)
            
            # Create a test alert using real Bitcoin prices
            from zone_fade_detector.core.models import Alert, ZoneFadeSetup, Zone, ZoneType, SetupDirection, OHLCVBar, QRSFactors
            
            # Get real Bitcoin price for the test
            if bars:
                latest_bar = bars[-1]
                current_price = latest_bar.close
                # Create test zone slightly above current price
                zone_level = current_price * 1.002  # 0.2% above current price
                entry_price = current_price * 0.999  # 0.1% below current price
                stop_loss = current_price * 0.995    # 0.5% below current price
                target1 = zone_level
                target2 = current_price * 1.005      # 0.5% above current price
            else:
                # Fallback to current market price if no data
                current_price = 100000.0
                zone_level = 100200.0
                entry_price = 99900.0
                stop_loss = 99500.0
                target1 = 100200.0
                target2 = 100500.0
            
            test_bar = OHLCVBar(
                timestamp=datetime.now(),
                open=current_price * 0.999,
                high=current_price * 1.001,
                low=current_price * 0.998,
                close=current_price,
                volume=1000000
            )
            
            test_zone = Zone(
                level=zone_level,
                zone_type=ZoneType.PRIOR_DAY_HIGH,
                strength=0.8,
                quality=2
            )
            
            qrs_factors = QRSFactors()
            qrs_factors.zone_quality = 2
            qrs_factors.rejection_clarity = 2
            qrs_factors.structure_flip = 1
            qrs_factors.context = 2
            qrs_factors.intermarket_divergence = 1
            
            test_setup = ZoneFadeSetup(
                symbol='BTC',
                direction=SetupDirection.LONG,
                zone=test_zone,
                rejection_candle=test_bar,
                choch_confirmed=True,
                qrs_factors=qrs_factors,
                timestamp=datetime.now()
            )
            
            test_alert = Alert(
                alert_id='BTC_TEST_ALERT_001',
                setup=test_setup,
                priority='HIGH',
                created_at=datetime.now()
            )
            
            # Send test alert
            print("📤 Sending test Bitcoin alert...")
            results = await alert_system.send_alert(test_alert)
            print(f"Alert results: {results}")
            
            webhook_success = results.get('WebhookAlertChannel', False)
            if webhook_success:
                print("✅ Discord webhook test successful!")
            else:
                print("❌ Discord webhook test failed")
        else:
            print("⚠️ No Discord webhook configured (DISCORD_WEBHOOK_URL not set)")
        
        print("\n✅ Simple Bitcoin detector test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in simple test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bitcoin_simple())
    print(f"\n🏁 Simple Test Result: {'PASS' if success else 'FAIL'}")