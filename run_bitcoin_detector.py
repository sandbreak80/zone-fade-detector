#!/usr/bin/env python3
"""
Run the Bitcoin Zone Fade Detector in live 24/7 mode with Discord alerts.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from zone_fade_detector.utils.config import load_config
from bitcoin_zone_fade_detector import BitcoinZoneFadeDetector

async def run_bitcoin_detector():
    """Run the Bitcoin detector in live 24/7 mode."""
    print("🚀 Starting Bitcoin Zone Fade Detector - 24/7 Live Mode")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("🪙 Monitoring: Bitcoin, Ethereum, and other cryptocurrencies")
    print("⏰ Trading: 24/7 (cryptocurrency markets never close)")
    print("📱 Alerts: Discord webhook enabled")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Load Bitcoin configuration
        config = load_config("config/bitcoin_config.yaml")
        
        # Initialize detector
        detector = BitcoinZoneFadeDetector(config)
        
        # Test alert system before starting
        print("\n🧪 Testing Alert System...")
        alert_results = await detector.test_alert_system()
        
        webhook_success = alert_results.get('WebhookAlertChannel', False)
        if webhook_success:
            print("✅ Discord webhook is ready for Bitcoin alerts!")
        else:
            print("⚠️ Discord webhook not working - alerts will only go to console/file")
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down...")
            detector.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the detector
        print("\n🔄 Starting 24/7 detection loop...")
        print("💡 The detector will check for Zone Fade setups every minute")
        print("🎯 Bitcoin and crypto markets trade 24/7, so alerts can come anytime!")
        print("\n" + "="*60)
        
        await detector.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error running Bitcoin detector: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Bitcoin Zone Fade Detector stopped")

if __name__ == "__main__":
    asyncio.run(run_bitcoin_detector())