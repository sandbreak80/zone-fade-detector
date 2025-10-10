#!/usr/bin/env python3
"""
Test script for Polygon API connectivity.
"""

import asyncio
import os
from datetime import datetime, timedelta
from zone_fade_detector.data.polygon_client import PolygonClient, PolygonConfig

async def test_polygon():
    """Test Polygon API connectivity and data fetching."""
    print("🔍 Testing Polygon API connectivity...")
    
    try:
        # Get API credentials from environment variables
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            print("❌ Missing Polygon API credentials in environment variables")
            print("   Required: POLYGON_API_KEY")
            return False
        
        print(f"🔑 Using API Key: {api_key[:8]}...")
        
        # Create Polygon client
        polygon_config = PolygonConfig(
            api_key=api_key
        )
        
        client = PolygonClient(polygon_config)
        
        async with client:
            # Test fetching SPY aggregates for last 2 days
            end = datetime.now()
            start = end - timedelta(days=2)
            
            print(f"📊 Fetching SPY aggregates from {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
            
            bars = await client.get_aggregates('SPY', start, end)
            
            print(f"✅ Polygon API Test: Retrieved {len(bars)} aggregate bars for SPY")
            
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
                print("❌ No data returned from Polygon API")
                print("   This could mean:")
                print("   - Market was closed on those days")
                print("   - SPY symbol not available")
                print("   - API permissions issue")
                
                # Try with a different symbol
                print("\n🔄 Trying with QQQ...")
                bars_qqq = await client.get_aggregates('QQQ', start, end)
                print(f"   QQQ bars: {len(bars_qqq)}")
                
                if bars_qqq:
                    print("✅ QQQ data available - SPY might not be available")
                    return True
                else:
                    print("❌ No data for QQQ either")
                    return False
                
    except Exception as e:
        print(f"❌ Polygon API Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_polygon())
    print(f"\n🏁 Test Result: {'PASS' if result else 'FAIL'}")