#!/usr/bin/env python3
"""
Test script for Alpaca API connectivity.
"""

import asyncio
from datetime import datetime, timedelta
from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig
from zone_fade_detector.utils.config import load_config

async def test_alpaca():
    """Test Alpaca API connectivity and data fetching."""
    print("🔍 Testing Alpaca API connectivity...")
    
    try:
        # Get API credentials from environment variables
        import os
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            print("❌ Missing Alpaca API credentials in environment variables")
            print("   Required: ALPACA_API_KEY, ALPACA_SECRET_KEY")
            return False
        
        print(f"🔑 Using API Key: {api_key[:8]}...")
        print(f"🌐 Using Base URL: {base_url}")
        
        # Create Alpaca client
        alpaca_config = AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        client = AlpacaClient(alpaca_config)
        
        async with client:
            # Test fetching SPY data for last 2 days
            end = datetime.now()
            start = end - timedelta(days=2)
            
            print(f"📊 Fetching SPY data from {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
            
            bars = await client.get_bars('SPY', start, end)
            
            print(f"✅ Alpaca API Test: Retrieved {len(bars)} bars for SPY")
            
            if bars:
                latest_bar = bars[-1]
                print(f"   📈 Latest bar: {latest_bar.timestamp}")
                print(f"   💰 OHLC: O:{latest_bar.open:.2f} H:{latest_bar.high:.2f} L:{latest_bar.low:.2f} C:{latest_bar.close:.2f}")
                print(f"   📊 Volume: {latest_bar.volume:,}")
                
                # Show a few recent bars
                print(f"\n📋 Recent bars (last 3):")
                for i, bar in enumerate(bars[-3:], 1):
                    print(f"   {i}. {bar.timestamp} - C:{bar.close:.2f} V:{bar.volume:,}")
                
                return True
            else:
                print("❌ No data returned from Alpaca API")
                return False
                
    except Exception as e:
        print(f"❌ Alpaca API Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_alpaca())
    print(f"\n🏁 Test Result: {'PASS' if result else 'FAIL'}")