#!/usr/bin/env python3
"""
Test script for Discord webhook functionality.
"""

import asyncio
import os
from datetime import datetime
from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
from zone_fade_detector.core.models import Alert, ZoneFadeSetup, Zone, ZoneType, SetupDirection, OHLCVBar, QRSFactors

async def test_discord():
    """Test Discord webhook message sending."""
    print("🔍 Testing Discord Webhook Functionality")
    print("=" * 50)
    
    try:
        # Get Discord webhook URL from environment variables
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        if not webhook_url:
            print("❌ Missing Discord webhook URL in environment variables")
            print("   Required: DISCORD_WEBHOOK_URL")
            return False
        
        print(f"🔗 Using Webhook URL: {webhook_url[:50]}...")
        
        # Create test alert configuration
        config = AlertChannelConfig(
            console_enabled=True,
            file_enabled=True,
            webhook_enabled=True,
            webhook_url=webhook_url,
            webhook_timeout=10  # Longer timeout for testing
        )
        
        print("\n📋 Alert Channel Configuration:")
        print(f"   Console: {'✅ Enabled' if config.console_enabled else '❌ Disabled'}")
        print(f"   File: {'✅ Enabled' if config.file_enabled else '❌ Disabled'}")
        print(f"   Webhook: {'✅ Enabled' if config.webhook_enabled else '❌ Disabled'}")
        
        # Create test setup data
        print("\n🏗️ Creating test Zone Fade setup...")
        
        # Create test bar data
        test_bar = OHLCVBar(
            timestamp=datetime.now(),
            open=485.0,
            high=485.5,
            low=484.8,
            close=485.2,
            volume=1000000
        )
        
        # Create test zone
        test_zone = Zone(
            level=485.0,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            strength=0.8,
            quality=2
        )
        
        # Create QRS factors with a high score
        qrs_factors = QRSFactors()
        qrs_factors.zone_quality = 2
        qrs_factors.rejection_clarity = 2
        qrs_factors.structure_flip = 2
        qrs_factors.context = 1
        qrs_factors.intermarket_divergence = 1
        
        # Create test setup
        test_setup = ZoneFadeSetup(
            symbol='SPY',
            direction=SetupDirection.LONG,
            zone=test_zone,
            rejection_candle=test_bar,
            choch_confirmed=True,
            qrs_factors=qrs_factors,
            timestamp=datetime.now()
        )
        
        # Create test alert
        test_alert = Alert(
            alert_id='TEST_ALERT_001',
            setup=test_setup,
            priority='HIGH',
            created_at=datetime.now()
        )
        
        print(f"   Symbol: {test_setup.symbol}")
        print(f"   Direction: {test_setup.direction.value}")
        print(f"   Zone Level: {test_setup.zone.level}")
        print(f"   QRS Score: {test_setup.qrs_score}")
        print(f"   Alert ID: {test_alert.alert_id}")
        
        # Test alert system
        print("\n🚀 Testing Alert System...")
        alert_system = AlertSystem(config)
        
        print("   Sending test alert through all channels...")
        results = await alert_system.send_alert(test_alert)
        
        print(f"\n📊 Alert Channel Results:")
        success_count = 0
        for channel, success in results.items():
            status = '✅ PASS' if success else '❌ FAIL'
            print(f"   {channel}: {status}")
            if success:
                success_count += 1
        
        print(f"\n📈 Summary:")
        print(f"   Total Channels: {len(results)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(results) - success_count}")
        
        # Check if webhook specifically worked
        webhook_success = results.get('WebhookAlertChannel', False)
        if webhook_success:
            print(f"\n🎉 Discord Webhook Test: SUCCESS!")
            print(f"   Check your Discord channel for the test message")
        else:
            print(f"\n❌ Discord Webhook Test: FAILED")
            print(f"   Check webhook URL and permissions")
        
        return webhook_success
        
    except Exception as e:
        print(f"❌ Discord Webhook Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_discord())
    print(f"\n🏁 Test Result: {'PASS' if result else 'FAIL'}")