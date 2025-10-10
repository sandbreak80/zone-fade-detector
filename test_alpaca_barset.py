#!/usr/bin/env python3
"""
Debug BarSet object structure from Alpaca API.
"""

import asyncio
import os
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

async def test_alpaca_barset():
    """Debug the BarSet object structure."""
    print("🔍 Debugging BarSet Object Structure")
    print("=" * 50)
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    # Create client
    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key
    )
    
    # Make a request
    now = datetime.now()
    start = now - timedelta(days=7)
    
    request = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Day,
        start=start,
        end=now,
        adjustment='raw'
    )
    
    print(f"📊 Making request for SPY from {start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
    
    try:
        bars_response = client.get_stock_bars(request)
        
        print(f"\n📋 BarSet Object Analysis:")
        print(f"   Type: {type(bars_response)}")
        print(f"   Dir: {[attr for attr in dir(bars_response) if not attr.startswith('_')]}")
        
        # Try different ways to access the data
        print(f"\n🔍 Trying different access methods:")
        
        # Method 1: Direct iteration
        try:
            print(f"   Method 1 - Direct iteration:")
            count = 0
            for item in bars_response:
                print(f"      Item {count}: {type(item)} - {item}")
                count += 1
                if count >= 3:  # Limit output
                    break
            if count == 0:
                print(f"      No items found")
        except Exception as e:
            print(f"      ❌ Direct iteration failed: {e}")
        
        # Method 2: Dictionary access
        try:
            print(f"   Method 2 - Dictionary access:")
            print(f"      Keys: {list(bars_response.keys())}")
            print(f"      Values: {list(bars_response.values())}")
            print(f"      Items: {list(bars_response.items())}")
        except Exception as e:
            print(f"      ❌ Dictionary access failed: {e}")
        
        # Method 3: Attribute access
        try:
            print(f"   Method 3 - Attribute access:")
            print(f"      data: {getattr(bars_response, 'data', 'No data attr')}")
            print(f"      symbols: {getattr(bars_response, 'symbols', 'No symbols attr')}")
            print(f"      symbol: {getattr(bars_response, 'symbol', 'No symbol attr')}")
        except Exception as e:
            print(f"      ❌ Attribute access failed: {e}")
        
        # Method 4: Convert to dict
        try:
            print(f"   Method 4 - Convert to dict:")
            bars_dict = dict(bars_response)
            print(f"      Dict keys: {list(bars_dict.keys())}")
            print(f"      Dict values: {list(bars_dict.values())}")
        except Exception as e:
            print(f"      ❌ Convert to dict failed: {e}")
        
        # Method 5: Check if it's iterable
        try:
            print(f"   Method 5 - Check iterability:")
            print(f"      Iterable: {hasattr(bars_response, '__iter__')}")
            print(f"      Length: {len(bars_response) if hasattr(bars_response, '__len__') else 'No length'}")
        except Exception as e:
            print(f"      ❌ Iterability check failed: {e}")
        
        # Method 6: Try to get SPY specifically
        try:
            print(f"   Method 6 - Get SPY specifically:")
            spy_bars = bars_response["SPY"]
            print(f"      SPY bars: {len(spy_bars)} items")
            if spy_bars:
                print(f"      First bar: {spy_bars[0]}")
        except Exception as e:
            print(f"      ❌ Get SPY failed: {e}")
        
        # Method 7: Try different symbol access
        try:
            print(f"   Method 7 - Try different symbol access:")
            for symbol in ["SPY", "spy", "SPY.US", "SPY_US"]:
                try:
                    bars = bars_response[symbol]
                    print(f"      {symbol}: {len(bars)} bars")
                    if bars:
                        print(f"         First: {bars[0].timestamp} - C:{bars[0].close:.2f}")
                        return True
                except KeyError:
                    print(f"      {symbol}: Not found")
                except Exception as e:
                    print(f"      {symbol}: Error - {e}")
        except Exception as e:
            print(f"      ❌ Symbol access failed: {e}")
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n❌ No data found with any method")
    return False

if __name__ == "__main__":
    result = asyncio.run(test_alpaca_barset())
    print(f"\n🏁 Debug Result: {'PASS' if result else 'FAIL'}")