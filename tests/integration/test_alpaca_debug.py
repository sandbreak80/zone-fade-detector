#!/usr/bin/env python3
"""
Debug test for Alpaca Python library implementation.
"""

import asyncio
import os
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

async def test_alpaca_debug():
    """Debug the Alpaca Python library implementation."""
    print("🔍 Debugging Alpaca Python Library Implementation")
    print("=" * 60)
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"🔑 Secret: {secret_key[:8]}...")
    
    # Test 1: Basic client initialization
    print("\n📋 Test 1: Client Initialization")
    try:
        client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False
    
    # Test 2: Check client configuration
    print("\n📋 Test 2: Client Configuration")
    print(f"   API Key: {client._api_key[:8]}...")
    print(f"   Secret Key: {client._secret_key[:8]}...")
    print(f"   Base URL: {client._base_url}")
    
    # Test 3: Try different request parameters
    print("\n📋 Test 3: Different Request Parameters")
    
    # Test with different timeframes
    timeframes_to_test = [
        (TimeFrame.Minute, "1 minute"),
        (TimeFrame.Hour, "1 hour"),
        (TimeFrame.Day, "1 day")
    ]
    
    # Test with different date ranges
    now = datetime.now()
    test_ranges = [
        (now - timedelta(days=7), now, "Last 7 days"),
        (now - timedelta(days=30), now, "Last 30 days"),
        (now - timedelta(days=365), now, "Last year")
    ]
    
    for start, end, desc in test_ranges:
        print(f"\n   📅 Testing {desc}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        for timeframe, tf_desc in timeframes_to_test:
            try:
                print(f"      ⏰ {tf_desc} bars...")
                
                request = StockBarsRequest(
                    symbol_or_symbols=["SPY"],
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    adjustment='raw'
                )
                
                # Make the request
                bars_response = client.get_stock_bars(request)
                
                print(f"         Response type: {type(bars_response)}")
                print(f"         Response keys: {list(bars_response.keys()) if hasattr(bars_response, 'keys') else 'No keys'}")
                
                if "SPY" in bars_response:
                    bars = bars_response["SPY"]
                    print(f"         ✅ SUCCESS: {len(bars)} bars for SPY")
                    
                    if bars:
                        print(f"         📈 First bar: {bars[0].timestamp} - C:{bars[0].close:.2f}")
                        print(f"         📈 Last bar: {bars[-1].timestamp} - C:{bars[-1].close:.2f}")
                        return True
                    else:
                        print(f"         ⚠️  Empty response for SPY")
                else:
                    print(f"         ❌ SPY not in response")
                    
            except Exception as e:
                print(f"         ❌ {tf_desc} failed: {str(e)[:100]}...")
                continue
    
    # Test 4: Try different symbols
    print("\n📋 Test 4: Different Symbols")
    symbols_to_test = ["SPY", "AAPL", "MSFT", "QQQ"]
    
    for symbol in symbols_to_test:
        try:
            print(f"   📊 Testing {symbol}...")
            
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=now - timedelta(days=30),
                end=now,
                adjustment='raw'
            )
            
            bars_response = client.get_stock_bars(request)
            
            if symbol in bars_response:
                bars = bars_response[symbol]
                print(f"      ✅ {symbol}: {len(bars)} bars")
                if bars:
                    print(f"         Latest: {bars[-1].timestamp} - C:{bars[-1].close:.2f}")
                    return True
            else:
                print(f"      ❌ {symbol}: Not in response")
                
        except Exception as e:
            print(f"      ❌ {symbol}: {str(e)[:100]}...")
    
    print("\n❌ All tests failed")
    return False

if __name__ == "__main__":
    result = asyncio.run(test_alpaca_debug())
    print(f"\n🏁 Debug Result: {'PASS' if result else 'FAIL'}")