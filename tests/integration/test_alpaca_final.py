#!/usr/bin/env python3
"""
Final test for Alpaca API with proper BarSet handling.
"""

import asyncio
import os
from datetime import datetime, timedelta
from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig

async def test_alpaca_final():
    """Test Alpaca API with proper BarSet handling."""
    print("🔍 Final Alpaca API Test with Proper BarSet Handling")
    print("=" * 60)
    
    try:
        # Get API credentials from environment variables
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            print("❌ Missing Alpaca API credentials")
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
            # Test with a known trading day (last Friday)
            now = datetime.now()
            
            # Go back to find the last Friday
            days_back = 0
            while True:
                test_date = now - timedelta(days=days_back)
                if test_date.weekday() == 4:  # Friday
                    break
                days_back += 1
                if days_back > 7:  # Safety check
                    test_date = now - timedelta(days=1)
                    break
            
            # Request data for that Friday
            start = test_date.replace(hour=9, minute=30, second=0, microsecond=0)  # Market open
            end = test_date.replace(hour=16, minute=0, second=0, microsecond=0)    # Market close
            
            print(f"📊 Fetching SPY data for {test_date.strftime('%Y-%m-%d (%A)')}")
            print(f"   Market Hours: {start.strftime('%H:%M')} to {end.strftime('%H:%M')}")
            
            # Let's also test with daily data to see if we get anything
            print(f"\n📊 Also testing daily data for last 7 days...")
            daily_start = now - timedelta(days=7)
            daily_end = now
            
            bars = await client.get_bars('SPY', daily_start, daily_end)
            
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
                print("   This could mean:")
                print("   - Market was closed on those days")
                print("   - SPY symbol not available in paper trading")
                print("   - API permissions issue")
                
                # Try with a different symbol
                print("\n🔄 Trying with QQQ...")
                bars_qqq = await client.get_bars('QQQ', daily_start, daily_end)
                print(f"   QQQ bars: {len(bars_qqq)}")
                
                if bars_qqq:
                    print("✅ QQQ data available - SPY might not be available in paper trading")
                    return True
                else:
                    print("❌ No data for QQQ either")
                    return False
                
    except Exception as e:
        print(f"❌ Alpaca API Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_alpaca_final())
    print(f"\n🏁 Test Result: {'PASS' if result else 'FAIL'}")