#!/usr/bin/env python3
"""
Wide range test for Alpaca API connectivity.
"""

import asyncio
import os
from datetime import datetime, timedelta
from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig

async def test_alpaca_wide():
    """Test Alpaca API with a wide date range."""
    print("🔍 Testing Alpaca API with wide date range...")
    
    try:
        # Get API credentials from environment variables
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            print("❌ Missing Alpaca API credentials")
            return False
        
        print(f"🔑 Using API Key: {api_key[:8]}...")
        
        # Create Alpaca client
        alpaca_config = AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        client = AlpacaClient(alpaca_config)
        
        async with client:
            # Try a much wider date range - last 30 days
            end = datetime.now()
            start = end - timedelta(days=30)
            
            print(f"📊 Fetching SPY data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
            print(f"   (30-day range to catch any trading days)")
            
            bars = await client.get_bars('SPY', start, end)
            
            print(f"✅ Retrieved {len(bars)} bars for SPY")
            
            if bars:
                print(f"   📈 Date range: {bars[0].timestamp} to {bars[-1].timestamp}")
                print(f"   💰 Latest: O:{bars[-1].open:.2f} H:{bars[-1].high:.2f} L:{bars[-1].low:.2f} C:{bars[-1].close:.2f}")
                print(f"   📊 Volume: {bars[-1].volume:,}")
                
                # Show first and last few bars
                print(f"\n📋 First 3 bars:")
                for i, bar in enumerate(bars[:3], 1):
                    print(f"   {i}. {bar.timestamp} - C:{bar.close:.2f} V:{bar.volume:,}")
                
                print(f"\n📋 Last 3 bars:")
                for i, bar in enumerate(bars[-3:], 1):
                    print(f"   {i}. {bar.timestamp} - C:{bar.close:.2f} V:{bar.volume:,}")
                
                return True
            else:
                print("❌ Still no data - trying different approach...")
                
                # Try with a very recent date (today)
                today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
                tomorrow = today + timedelta(days=1)
                
                print(f"\n🔄 Trying today's data: {today.strftime('%Y-%m-%d %H:%M')} to {tomorrow.strftime('%Y-%m-%d %H:%M')}")
                bars_today = await client.get_bars('SPY', today, tomorrow)
                print(f"   Today's bars: {len(bars_today)}")
                
                if bars_today:
                    print("✅ Got today's data!")
                    return True
                else:
                    print("❌ No data for today either")
                    print("\n🔍 Possible issues:")
                    print("   - Paper trading account might not have historical data access")
                    print("   - API key might not have market data permissions")
                    print("   - Alpaca paper trading might be limited")
                    return False
                
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_alpaca_wide())
    print(f"\n🏁 Test Result: {'PASS' if result else 'FAIL'}")