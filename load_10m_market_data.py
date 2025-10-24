#!/usr/bin/env python3
"""
Load 10-minute market data from Alpaca API.

This script downloads 10-minute candle data for all instruments
and saves them for strategy testing.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataLoader10M:
    """10-minute market data loader."""
    
    def __init__(self):
        self.client = None
        self.instruments = ["QQQ", "SPY", "LLY", "AVGO", "AAPL", "CRM", "ORCL"]
        self.start_date = datetime(2016, 1, 1)
        self.end_date = datetime(2025, 1, 1)
        self.data_dir = Path("data/real_market_data_10m")
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📊 Initialized 10-minute Market Data Loader")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📅 Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"📈 Instruments: {', '.join(self.instruments)}")
    
    def initialize_client(self):
        """Initialize Alpaca client."""
        try:
            # Load credentials from .env file
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("Alpaca API credentials not found in environment variables")
            
            self.client = StockHistoricalDataClient(api_key, secret_key)
            logger.info("✅ Alpaca client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alpaca client: {e}")
            return False
    
    def download_instrument_data(self, symbol: str) -> pd.DataFrame:
        """Download 10-minute data for a specific instrument."""
        logger.info(f"📥 Downloading 10-minute data for {symbol}...")
        
        try:
            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute(10),  # 10-minute candles
                start=self.start_date,
                end=self.end_date
            )
            
            # Get data
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            df = df.reset_index()
            
            # Rename columns to match our format
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            logger.info(f"✅ Downloaded {len(df)} 10-minute bars for {symbol}")
            logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"   Volume range: {df['volume'].min():,.0f} to {df['volume'].max():,.0f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_instrument_data(self, symbol: str, df: pd.DataFrame):
        """Save instrument data to CSV."""
        if df.empty:
            logger.warning(f"⚠️ No data to save for {symbol}")
            return
        
        # Save individual file
        filename = f"{symbol}_10m_bars.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {symbol} data to {filepath}")
    
    def save_combined_data(self, all_data: list):
        """Save combined data from all instruments."""
        if not all_data:
            logger.warning("⚠️ No data to combine")
            return
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined file
        combined_file = self.data_dir / "all_instruments_10m_bars.csv"
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"💾 Saved combined data to {combined_file}")
        logger.info(f"📊 Total bars: {len(combined_df):,}")
        logger.info(f"📈 Instruments: {combined_df['symbol'].nunique()}")
    
    def save_metadata(self, all_data: list):
        """Save metadata about the data loading process."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': '10-minute',
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'instruments': self.instruments,
            'total_bars': sum(len(df) for df in all_data),
            'instrument_counts': {df['symbol'].iloc[0] if not df.empty else 'unknown': len(df) for df in all_data},
            'data_directory': str(self.data_dir),
            'description': '10-minute candle data for strategy testing'
        }
        
        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved metadata to {metadata_file}")
    
    def load_all_data(self):
        """Load data for all instruments."""
        logger.info("🚀 Starting 10-minute data loading process...")
        
        # Initialize client
        if not self.initialize_client():
            return False
        
        all_data = []
        successful_downloads = 0
        
        for symbol in self.instruments:
            try:
                # Download data
                df = self.download_instrument_data(symbol)
                
                if not df.empty:
                    # Save individual file
                    self.save_instrument_data(symbol, df)
                    all_data.append(df)
                    successful_downloads += 1
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.5)
                else:
                    logger.warning(f"⚠️ No data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        # Save combined data and metadata
        if all_data:
            self.save_combined_data(all_data)
            self.save_metadata(all_data)
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 10-MINUTE DATA LOADING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Successful downloads: {successful_downloads}/{len(self.instruments)}")
            logger.info(f"📊 Total bars downloaded: {sum(len(df) for df in all_data):,}")
            logger.info(f"📁 Data saved to: {self.data_dir}")
            
            for df in all_data:
                symbol = df['symbol'].iloc[0] if not df.empty else 'unknown'
                logger.info(f"   {symbol}: {len(df):,} bars")
            
            return True
        else:
            logger.error("❌ No data was successfully downloaded")
            return False


def main():
    """Main function to load 10-minute market data."""
    logger.info("🚀 Starting 10-minute Market Data Loading")
    
    # Initialize loader
    loader = MarketDataLoader10M()
    
    # Load all data
    success = loader.load_all_data()
    
    if success:
        logger.info("✅ 10-minute data loading completed successfully!")
        logger.info(f"📁 Data available in: {loader.data_dir}")
    else:
        logger.error("❌ 10-minute data loading failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
