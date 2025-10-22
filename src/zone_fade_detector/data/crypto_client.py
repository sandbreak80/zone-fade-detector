"""
Cryptocurrency data client for Bitcoin and other crypto assets.

This module provides a client for fetching real-time cryptocurrency data
from CoinGecko API for Bitcoin (BTC) and other major cryptocurrencies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import aiohttp
import pandas as pd

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class CryptoConfig:
    """Configuration for cryptocurrency data client."""
    api_key: str = ""  # CoinGecko API key (optional for free tier)
    base_url: str = "https://api.coingecko.com/api/v3"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 1.0  # Delay between requests to respect rate limits


class CryptoClient:
    """
    Client for fetching cryptocurrency data from CoinGecko API.
    
    Provides methods to fetch OHLCV data for Bitcoin and other cryptocurrencies
    with proper error handling, rate limiting, and data validation.
    """
    
    def __init__(self, config: CryptoConfig):
        """
        Initialize crypto client.
        
        Args:
            config: Crypto configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting to respect API limits."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            await asyncio.sleep(self.config.rate_limit_delay - time_since_last)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def get_ohlc(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 1
    ) -> List[OHLCVBar]:
        """
        Fetch OHLC data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of data (1, 7, 14, 30, 90, 180, 365, max)
            
        Returns:
            List of OHLCVBar objects
            
        Raises:
            CryptoAPIError: If API request fails
        """
        try:
            await self._rate_limit()
            
            self.logger.info(f"Fetching {days} days of OHLC data for {coin_id}")
            
            url = f"{self.config.base_url}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            if self.config.api_key:
                params['x_cg_demo_api_key'] = self.config.api_key
            
            async with self._session.get(url, params=params) as response:
                if response.status == 429:
                    # Rate limited, wait longer
                    await asyncio.sleep(5)
                    return await self.get_ohlc(coin_id, vs_currency, days)
                
                response.raise_for_status()
                data = await response.json()
            
            # Convert to OHLCVBar objects
            bars = []
            for item in data:
                # CoinGecko OHLC format: [timestamp, open, high, low, close]
                timestamp = datetime.fromtimestamp(item[0] / 1000)  # Convert from milliseconds
                ohlcv_bar = OHLCVBar(
                    timestamp=timestamp,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=0  # CoinGecko OHLC doesn't include volume
                )
                bars.append(ohlcv_bar)
            
            # Sort by timestamp (oldest first)
            bars.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Retrieved {len(bars)} OHLC bars for {coin_id}")
            return bars
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLC data for {coin_id}: {e}")
            raise CryptoAPIError(f"Failed to fetch OHLC data for {coin_id}: {e}")
    
    async def get_current_price(
        self,
        coin_id: str,
        vs_currency: str = "usd"
    ) -> float:
        """
        Get current price for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to compare against
            
        Returns:
            Current price
        """
        try:
            await self._rate_limit()
            
            url = f"{self.config.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': vs_currency
            }
            
            if self.config.api_key:
                params['x_cg_demo_api_key'] = self.config.api_key
            
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
            
            return float(data[coin_id][vs_currency])
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {coin_id}: {e}")
            raise CryptoAPIError(f"Failed to fetch current price for {coin_id}: {e}")
    
    async def get_historical_data(
        self,
        coin_id: str,
        start_date: datetime,
        end_date: datetime,
        vs_currency: str = "usd"
    ) -> List[OHLCVBar]:
        """
        Get historical data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            start_date: Start date
            end_date: End date
            vs_currency: Currency to compare against
            
        Returns:
            List of OHLCVBar objects
        """
        try:
            # Calculate days difference
            days_diff = (end_date - start_date).days
            
            # CoinGecko API limits: 1, 7, 14, 30, 90, 180, 365, max
            if days_diff <= 1:
                days = 1
            elif days_diff <= 7:
                days = 7
            elif days_diff <= 14:
                days = 14
            elif days_diff <= 30:
                days = 30
            elif days_diff <= 90:
                days = 90
            elif days_diff <= 180:
                days = 180
            elif days_diff <= 365:
                days = 365
            else:
                days = "max"
            
            bars = await self.get_ohlc(coin_id, vs_currency, days)
            
            # Filter by date range
            filtered_bars = [
                bar for bar in bars
                if start_date <= bar.timestamp <= end_date
            ]
            
            return filtered_bars
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {coin_id}: {e}")
            raise CryptoAPIError(f"Failed to fetch historical data for {coin_id}: {e}")
    
    async def get_multiple_coins(
        self,
        coin_ids: List[str],
        vs_currency: str = "usd",
        days: int = 1
    ) -> Dict[str, List[OHLCVBar]]:
        """
        Fetch OHLC data for multiple cryptocurrencies.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            vs_currency: Currency to compare against
            days: Number of days of data
            
        Returns:
            Dictionary mapping coin IDs to their OHLCVBar lists
        """
        tasks = []
        for coin_id in coin_ids:
            task = self.get_ohlc(coin_id, vs_currency, days)
            tasks.append((coin_id, task))
        
        results = {}
        for coin_id, task in tasks:
            try:
                results[coin_id] = await task
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {coin_id}: {e}")
                results[coin_id] = []
        
        return results
    
    def validate_coin_id(self, coin_id: str) -> bool:
        """
        Validate if a coin ID is supported.
        
        Args:
            coin_id: CoinGecko coin ID to validate
            
        Returns:
            True if coin ID is supported, False otherwise
        """
        supported_coins = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
        return coin_id.lower() in supported_coins
    
    async def health_check(self) -> bool:
        """
        Check if the CoinGecko API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try to fetch current Bitcoin price
            price = await self.get_current_price('bitcoin')
            return price > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


class CryptoAPIError(Exception):
    """Exception raised for crypto API errors."""
    pass


class CryptoRateLimitError(CryptoAPIError):
    """Exception raised when crypto API rate limit is exceeded."""
    pass


class CryptoDataError(CryptoAPIError):
    """Exception raised for crypto data validation errors."""
    pass