#!/usr/bin/env python3
"""
Detailed diagnostic test for Alpaca API connectivity.
"""

import asyncio
import os
from datetime import datetime, timedelta
from zone_fade_detector.data.alpaca_client import AlpacaClient, AlpacaConfig

async def test_alpaca_detailed():
    """Detailed test of Alpaca API connectivity."""
    print("🔍 Detailed Alpaca API Diagnostic Test")
    print("=" * 50)
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    print(f"   ALPACA_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
    if api_key:
        print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
        print(f"   Key Length: {len(api_key)} characters")
    
    print(f"   ALPACA_SECRET_KEY: {'✅ Set' if secret_key else '❌ Missing'}")
    if secret_key:
        print(f"   Secret Key: {secret_key[:8]}...{secret_key[-4:]}")
        print(f"   Secret Length: {len(secret_key)} characters")
    
    print(f"   ALPACA_BASE_URL: {base_url}")
    
    if not api_key or not secret_key:
        print("\n❌ Missing required credentials")
        return False
    
    # Test different base URLs
    base_urls_to_test = [
        "https://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/v2",
        "https://data.alpaca.markets",
        "https://data.alpaca.markets/v2"
    ]
    
    for test_url in base_urls_to_test:
        print(f"\n🌐 Testing Base URL: {test_url}")
        
        try:
            # Create client with this URL
            config = AlpacaConfig(
                api_key=api_key,
                secret_key=secret_key,
                base_url=test_url
            )
            
            client = AlpacaClient(config)
            
            async with client:
                # Try to fetch a small amount of data
                end = datetime.now()
                start = end - timedelta(hours=1)  # Just 1 hour of data
                
                print(f"   📊 Fetching SPY data from {start.strftime('%H:%M')} to {end.strftime('%H:%M')}")
                
                bars = await client.get_bars('SPY', start, end)
                
                print(f"   ✅ SUCCESS! Retrieved {len(bars)} bars")
                if bars:
                    latest = bars[-1]
                    print(f"   📈 Latest: {latest.timestamp} - C:{latest.close:.2f} V:{latest.volume:,}")
                
                return True
                
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)[:100]}...")
            continue
    
    print("\n❌ All base URLs failed")
    return False

if __name__ == "__main__":
    result = asyncio.run(test_alpaca_detailed())
    print(f"\n🏁 Overall Result: {'PASS' if result else 'FAIL'}")