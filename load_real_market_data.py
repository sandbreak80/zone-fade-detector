#!/usr/bin/env python3
"""
Load Real Market Data using Alpaca API.

This script loads 1-hour candle data for all instruments from 2010-2025
using the Alpaca API and saves it for use in our validation framework.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import alpaca-py
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.models import Bar
    ALPACA_AVAILABLE = True
except ImportError:
    logger.error("alpaca-py not installed. Install with: pip install alpaca-py")
    ALPACA_AVAILABLE = False


class RealMarketDataLoader:
    """
    Load real market data using Alpaca API.
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize the data loader.
        
        Args:
            api_key: Alpaca API key (defaults to environment variable)
            secret_key: Alpaca secret key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py not available. Install with: pip install alpaca-py")
        
        # Initialize Alpaca client
        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        logger.info("✅ Alpaca client initialized successfully")
    
    async def load_instrument_data(self, 
                                 symbol: str, 
                                 start_date: datetime, 
                                 end_date: datetime) -> List[Dict[str, Any]]:
        """
        Load 1-hour candle data for a single instrument.
        
        Args:
            symbol: Stock symbol (e.g., 'QQQ', 'SPY')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            List of OHLCV bars
        """
        logger.info(f"📈 Loading data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        try:
            # Create request for 1-hour bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,  # 1-hour candles
                start=start_date,
                end=end_date,
                adjustment='all'  # Split and dividend adjusted
            )
            
            # Get the data
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to list of dictionaries
            bars_list = []
            for bar in bars[symbol]:
                bars_list.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'symbol': symbol
                })
            
            logger.info(f"✅ Loaded {len(bars_list)} bars for {symbol}")
            return bars_list
            
        except Exception as e:
            logger.error(f"❌ Error loading data for {symbol}: {e}")
            return []
    
    async def load_all_instruments(self, 
                                 symbols: List[str], 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 output_dir: str = "data/real_market_data") -> Dict[str, List[Dict[str, Any]]]:
        """
        Load data for all instruments and save to files.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            output_dir: Directory to save data files
            
        Returns:
            Dictionary mapping symbols to their data
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = {}
        
        for symbol in symbols:
            logger.info(f"🔄 Processing {symbol}...")
            
            # Load data for this symbol
            symbol_data = await self.load_instrument_data(symbol, start_date, end_date)
            
            if symbol_data:
                all_data[symbol] = symbol_data
                
                # Save to individual file
                df = pd.DataFrame(symbol_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                output_file = output_path / f"{symbol}_1h_bars.csv"
                df.to_csv(output_file)
                logger.info(f"💾 Saved {symbol} data to {output_file}")
            else:
                logger.warning(f"⚠️ No data loaded for {symbol}")
        
        # Save combined data
        combined_file = output_path / "all_instruments_1h_bars.csv"
        all_dataframes = []
        
        for symbol, data in all_data.items():
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df['symbol'] = symbol
                all_dataframes.append(df)
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=False)
            combined_df.to_csv(combined_file)
            logger.info(f"💾 Saved combined data to {combined_file}")
        
        # Save metadata
        metadata = {
            'loaded_at': datetime.now().isoformat(),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'timeframe': '1H',
            'instruments': symbols,
            'successful_loads': list(all_data.keys()),
            'failed_loads': [s for s in symbols if s not in all_data],
            'total_bars': sum(len(data) for data in all_data.values())
        }
        
        import json
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📊 Data loading complete!")
        logger.info(f"✅ Successfully loaded: {len(all_data)} instruments")
        logger.info(f"❌ Failed to load: {len(symbols) - len(all_data)} instruments")
        logger.info(f"📁 Data saved to: {output_path}")
        
        return all_data


async def main():
    """Main function to load real market data."""
    
    # Configuration
    SYMBOLS = ["QQQ", "SPY", "LLY", "AVGO", "AAPL", "CRM", "ORCL"]
    START_DATE = datetime(2010, 1, 1, 9, 30)  # Market open time
    END_DATE = datetime(2025, 1, 1, 16, 0)    # Market close time
    OUTPUT_DIR = "data/real_market_data"
    
    logger.info("🚀 Starting Real Market Data Loading")
    logger.info("=" * 60)
    logger.info(f"📅 Date Range: {START_DATE.date()} to {END_DATE.date()}")
    logger.info(f"⏰ Timeframe: 1 Hour")
    logger.info(f"📈 Instruments: {', '.join(SYMBOLS)}")
    logger.info(f"📁 Output Directory: {OUTPUT_DIR}")
    
    try:
        # Initialize data loader
        loader = RealMarketDataLoader()
        
        # Load all data
        all_data = await loader.load_all_instruments(
            symbols=SYMBOLS,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=OUTPUT_DIR
        )
        
        # Print summary
        logger.info("\n📊 Loading Summary:")
        for symbol in SYMBOLS:
            if symbol in all_data:
                bar_count = len(all_data[symbol])
                logger.info(f"  ✅ {symbol}: {bar_count:,} bars")
            else:
                logger.info(f"  ❌ {symbol}: Failed to load")
        
        logger.info(f"\n🎯 Ready for validation with real market data!")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
