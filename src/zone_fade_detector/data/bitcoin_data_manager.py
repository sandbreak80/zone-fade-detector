"""
Bitcoin-specific data manager for cryptocurrency trading.

This module provides a specialized data manager for Bitcoin and other
cryptocurrencies with 24/7 trading capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from diskcache import Cache

from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType
from zone_fade_detector.data.crypto_client import CryptoClient, CryptoConfig, CryptoAPIError


@dataclass
class BitcoinDataManagerConfig:
    """Configuration for Bitcoin data manager."""
    crypto_config: CryptoConfig
    cache_dir: str = "cache/bitcoin"
    cache_ttl: int = 300  # 5 minutes for crypto (more frequent updates)
    max_retries: int = 3
    retry_delay: float = 1.0


class BitcoinDataManager:
    """
    Bitcoin-specific data manager for cryptocurrency trading.
    
    Provides a specialized interface for fetching Bitcoin data with
    24/7 availability and crypto-specific optimizations.
    """
    
    def __init__(self, config: BitcoinDataManagerConfig):
        """
        Initialize Bitcoin data manager.
        
        Args:
            config: Bitcoin data manager configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache with fallback for permission issues
        try:
            self.cache = Cache(config.cache_dir)
        except (PermissionError, OSError) as e:
            # Fallback to in-memory cache if disk cache fails
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='bitcoin_cache_')
            self.cache = Cache(temp_dir)
            self.logger.warning(f"Cannot create cache directory {config.cache_dir}, using temp directory: {temp_dir}")
        
        # Initialize crypto client
        self.crypto_client = CryptoClient(config.crypto_config)
        
        # Supported cryptocurrencies
        self.supported_coins = ['bitcoin', 'ethereum', 'binancecoin']
        self.coin_symbols = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.crypto_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.crypto_client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _get_cache_key(self, coin_id: str, days: int, vs_currency: str) -> str:
        """Generate cache key for data request."""
        return f"crypto:{coin_id}:{days}:{vs_currency}"
    
    async def get_bars(
        self,
        coin_id: str,
        days: int = 1,
        vs_currency: str = "usd"
    ) -> List[OHLCVBar]:
        """
        Fetch OHLCV bars for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            days: Number of days of data
            vs_currency: Currency to compare against
            
        Returns:
            List of OHLCVBar objects
        """
        # Check cache first
        cache_key = self._get_cache_key(coin_id, days, vs_currency)
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.logger.debug(f"Cache hit for {coin_id} ({days} days)")
            return cached_data
        
        # Fetch from API
        bars = await self.crypto_client.get_ohlc(coin_id, vs_currency, days)
        
        # Cache the result
        self.cache.set(cache_key, bars, expire=self.config.cache_ttl)
        
        return bars
    
    async def get_latest_bars(
        self,
        coin_id: str,
        hours: int = 24,
        vs_currency: str = "usd"
    ) -> List[OHLCVBar]:
        """
        Fetch the latest bars for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            hours: Number of hours of data to fetch
            vs_currency: Currency to compare against
            
        Returns:
            List of latest OHLCVBar objects
        """
        # Convert hours to days for CoinGecko API
        if hours <= 24:
            days = 1
        elif hours <= 168:  # 7 days
            days = 7
        else:
            days = 30  # Max for recent data
        
        bars = await self.get_bars(coin_id, days, vs_currency)
        
        # Filter to last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_bars = [bar for bar in bars if bar.timestamp >= cutoff_time]
        
        return recent_bars
    
    async def get_multiple_coins(
        self,
        coin_ids: List[str],
        days: int = 1,
        vs_currency: str = "usd"
    ) -> Dict[str, List[OHLCVBar]]:
        """
        Fetch bars for multiple cryptocurrencies concurrently.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            days: Number of days of data
            vs_currency: Currency to compare against
            
        Returns:
            Dictionary mapping coin IDs to their OHLCVBar lists
        """
        tasks = []
        for coin_id in coin_ids:
            if self.validate_coin_id(coin_id):
                task = self.get_bars(coin_id, days, vs_currency)
                tasks.append((coin_id, task))
            else:
                self.logger.warning(f"Unsupported coin: {coin_id}")
        
        results = {}
        for coin_id, task in tasks:
            try:
                results[coin_id] = await task
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {coin_id}: {e}")
                results[coin_id] = []
        
        return results
    
    async def get_prior_day_data(
        self,
        coin_id: str,
        date: Optional[datetime] = None,
        vs_currency: str = "usd"
    ) -> Optional[OHLCVBar]:
        """
        Get the prior day's OHLCV data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            date: Reference date (defaults to now)
            vs_currency: Currency to compare against
            
        Returns:
            Prior day OHLCVBar or None if not available
        """
        if date is None:
            date = datetime.now()
        
        # Get previous day
        prev_day = date - timedelta(days=1)
        
        # Fetch data for the previous day
        bars = await self.get_bars(coin_id, 1, vs_currency)
        
        if not bars:
            return None
        
        # Find the bar closest to the previous day
        target_date = prev_day.replace(hour=12, minute=0, second=0, microsecond=0)
        closest_bar = min(bars, key=lambda x: abs((x.timestamp - target_date).total_seconds()))
        
        return closest_bar
    
    async def get_weekly_data(
        self,
        coin_id: str,
        week_start: Optional[datetime] = None,
        vs_currency: str = "usd"
    ) -> Tuple[Optional[OHLCVBar], Optional[OHLCVBar]]:
        """
        Get weekly high and low data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            week_start: Start of the week (defaults to current week)
            vs_currency: Currency to compare against
            
        Returns:
            Tuple of (weekly_high, weekly_low) OHLCVBar objects
        """
        if week_start is None:
            # Get start of current week (Monday)
            today = datetime.now()
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)
        
        # Fetch 7 days of data
        bars = await self.get_bars(coin_id, 7, vs_currency)
        
        if not bars:
            return None, None
        
        # Find weekly high and low
        weekly_high = max(bars, key=lambda x: x.high)
        weekly_low = min(bars, key=lambda x: x.low)
        
        return weekly_high, weekly_low
    
    async def get_zones(
        self,
        coin_id: str,
        date: Optional[datetime] = None,
        vs_currency: str = "usd"
    ) -> List[Zone]:
        """
        Get higher-timeframe zones for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            date: Reference date (defaults to now)
            vs_currency: Currency to compare against
            
        Returns:
            List of Zone objects
        """
        if date is None:
            date = datetime.now()
        
        zones = []
        
        try:
            # Get prior day data
            prior_day = await self.get_prior_day_data(coin_id, date, vs_currency)
            if prior_day:
                zones.append(Zone(
                    level=prior_day.high,
                    zone_type=ZoneType.PRIOR_DAY_HIGH,
                    quality=2
                ))
                zones.append(Zone(
                    level=prior_day.low,
                    zone_type=ZoneType.PRIOR_DAY_LOW,
                    quality=2
                ))
            
            # Get weekly data
            weekly_high, weekly_low = await self.get_weekly_data(coin_id, date, vs_currency)
            if weekly_high:
                zones.append(Zone(
                    level=weekly_high.high,
                    zone_type=ZoneType.WEEKLY_HIGH,
                    quality=2
                ))
            if weekly_low:
                zones.append(Zone(
                    level=weekly_low.low,
                    zone_type=ZoneType.WEEKLY_LOW,
                    quality=2
                ))
            
        except Exception as e:
            self.logger.error(f"Error fetching zones for {coin_id}: {e}")
        
        return zones
    
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
        return await self.crypto_client.get_current_price(coin_id, vs_currency)
    
    def validate_coin_id(self, coin_id: str) -> bool:
        """
        Validate if a coin ID is supported.
        
        Args:
            coin_id: CoinGecko coin ID to validate
            
        Returns:
            True if coin ID is supported, False otherwise
        """
        return coin_id.lower() in self.supported_coins
    
    def get_coin_symbol(self, coin_id: str) -> str:
        """
        Get trading symbol for a coin ID.
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Trading symbol (e.g., 'BTC' for 'bitcoin')
        """
        return self.coin_symbols.get(coin_id.lower(), coin_id.upper())
    
    async def health_check(self) -> bool:
        """
        Check health of the crypto data source.
        
        Returns:
            True if API is accessible, False otherwise
        """
        return await self.crypto_client.health_check()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.logger.info("Bitcoin data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'hit_count': self.cache.stats().hits,
            'miss_count': self.cache.stats().misses
        }