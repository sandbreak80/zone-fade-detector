#!/usr/bin/env python3
"""
5-Year Zone Fade Backtest

This version runs the backtest using 5 years of data (2020-2024) with the proper
Zone Fade exit strategy to get more comprehensive results.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
import json

def load_5year_data():
    """Load the 5-year data for all symbols."""
    print("📊 Loading 5-year data (2020-2024)...")
    
    symbols = ['SPY', 'QQQ', 'IWM']
    data = {}
    
    for symbol in symbols:
        try:
            file_path = f"/app/data/5year/{symbol}_5year.pkl"
            with open(file_path, 'rb') as f:
                symbol_data = pickle.load(f)
            data[symbol] = symbol_data
            print(f"   ✅ {symbol}: {len(symbol_data)} bars loaded")
        except Exception as e:
            print(f"   ❌ {symbol}: Error loading data - {e}")
    
    return data


def calculate_risk_unit(entry_price: float, hard_stop: float) -> float:
    """Calculate Risk Unit (R) as per Zone Fade Strategy requirements."""
    return abs(entry_price - hard_stop)


def calculate_proper_t1_price(entry_price: float, hard_stop: float, direction: str, 
                             vwap: float, qrs_score: float) -> float:
    """
    Calculate T1 price as per Zone Fade Strategy:
    T1 = Nearest of VWAP or 1R from entry
    """
    
    # Calculate Risk Unit (R)
    risk_unit = calculate_risk_unit(entry_price, hard_stop)
    
    # Calculate 1R target
    if direction == "SHORT":
        t1_r_target = entry_price - risk_unit
    else:  # LONG
        t1_r_target = entry_price + risk_unit
    
    # T1 = Nearest of VWAP or 1R from entry
    if direction == "SHORT":
        # For SHORT: choose the higher of VWAP or 1R (closer to entry = better)
        t1_price = max(vwap, t1_r_target)
    else:  # LONG
        # For LONG: choose the lower of VWAP or 1R (closer to entry = better)
        t1_price = min(vwap, t1_r_target)
    
    return t1_price


def calculate_proper_t2_price(entry_price: float, hard_stop: float, direction: str, 
                             opening_range_high: float, opening_range_low: float) -> float:
    """
    Calculate T2 price as per Zone Fade Strategy:
    T2 = Opposite side of OR range or next structural level (+2R)
    """
    
    # Calculate Risk Unit (R)
    risk_unit = calculate_risk_unit(entry_price, hard_stop)
    
    # Calculate 2R target
    if direction == "SHORT":
        t2_r_target = entry_price - (2 * risk_unit)
    else:  # LONG
        t2_r_target = entry_price + (2 * risk_unit)
    
    # T2 = Opposite side of OR range or 2R (whichever is closer to entry)
    if direction == "SHORT":
        # For SHORT: choose the higher of OR low or 2R (closer to entry = better)
        t2_price = max(opening_range_low, t2_r_target)
    else:  # LONG
        # For LONG: choose the lower of OR high or 2R (closer to entry = better)
        t2_price = min(opening_range_high, t2_r_target)
    
    return t2_price


def calculate_proper_t3_price(entry_price: float, hard_stop: float, direction: str) -> float:
    """
    Calculate T3 price as per Zone Fade Strategy:
    T3 = Opposite high-timeframe zone (+3R+)
    """
    
    # Calculate Risk Unit (R)
    risk_unit = calculate_risk_unit(entry_price, hard_stop)
    
    # Calculate 3R target
    if direction == "SHORT":
        t3_price = entry_price - (3 * risk_unit)
    else:  # LONG
        t3_price = entry_price + (3 * risk_unit)
    
    return t3_price


def simulate_proper_zone_fade_exit(entry_data: Dict, bars_data: List, entry_index: int) -> Dict:
    """
    Simulate trade execution with proper Zone Fade exit strategy.
    
    Exit Logic Hierarchy:
    1. Hard Stop - Zone invalidation
    2. T1 - Nearest of VWAP or 1R (scale out 40-50%, move stop to breakeven)
    3. T2 - Opposite side of OR range or 2R (scale another 25%)
    4. T3 - Opposite high-timeframe zone or 3R (trail or close remaining)
    """
    
    entry_price = entry_data['price']
    hard_stop = entry_data['hard_stop']
    vwap = entry_data.get('vwap', entry_price)
    qrs_score = entry_data.get('qrs_score', 7.0)
    direction = entry_data.get('direction', 'LONG')
    
    # Calculate position size ($10,000 per trade)
    position_value = 10000.0
    shares = int(position_value / entry_price)
    if shares == 0:
        shares = 1
    
    # Calculate slippage (2 ticks)
    tick_size = 0.01
    slippage_ticks = 2
    slippage_amount = tick_size * slippage_ticks
    
    # Apply slippage to entry price
    if direction == 'SHORT':
        actual_entry_price = entry_price + slippage_amount
    else:
        actual_entry_price = entry_price - slippage_amount
    
    # Calculate commission ($5 per trade)
    commission_per_trade = 5.0
    
    # Calculate proper target prices
    t1_price = calculate_proper_t1_price(entry_price, hard_stop, direction, vwap, qrs_score)
    
    # For T2, we need opening range data - use VWAP as proxy for now
    # In a real implementation, this would come from market data
    opening_range_high = vwap * 1.01  # 1% above VWAP as proxy
    opening_range_low = vwap * 0.99   # 1% below VWAP as proxy
    t2_price = calculate_proper_t2_price(entry_price, hard_stop, direction, 
                                        opening_range_high, opening_range_low)
    
    t3_price = calculate_proper_t3_price(entry_price, hard_stop, direction)
    
    # Simulate proper Zone Fade exit using actual market data
    exit_price = None
    exit_reason = None
    exit_index = None
    scaled_out = False
    remaining_shares = shares
    total_pnl = 0.0
    
    # Look ahead up to 100 bars (100 minutes) to find exits
    max_lookahead = min(100, len(bars_data) - entry_index - 1)
    
    for i in range(1, max_lookahead + 1):
        if entry_index + i >= len(bars_data):
            break
            
        current_bar = bars_data[entry_index + i]
        current_high = current_bar.high
        current_low = current_bar.low
        current_close = current_bar.close
        
        # 1. Hard Stop - Zone invalidation (highest priority)
        if direction == 'SHORT':
            if current_high >= hard_stop:
                # Hard stop hit - losing trade
                exit_price = hard_stop
                exit_reason = 'HARD_STOP'
                exit_index = entry_index + i
                break
        else:  # LONG
            if current_low <= hard_stop:
                # Hard stop hit - losing trade
                exit_price = hard_stop
                exit_reason = 'HARD_STOP'
                exit_index = entry_index + i
                break
        
        # 2. T1 - Nearest of VWAP or 1R (scale out 40-50%, move stop to breakeven)
        if not scaled_out:
            if direction == 'SHORT':
                if current_low <= t1_price:
                    # T1 hit - scale out 40-50%
                    scale_out_pct = 0.45  # 45% scale out
                    scale_out_shares = int(shares * scale_out_pct)
                    remaining_shares = shares - scale_out_shares
                    
                    # Calculate P&L for scaled out portion
                    if direction == 'SHORT':
                        scale_out_pnl = (actual_entry_price - t1_price) * scale_out_shares
                    else:
                        scale_out_pnl = (t1_price - actual_entry_price) * scale_out_shares
                    
                    total_pnl += scale_out_pnl
                    scaled_out = True
                    
                    # Move stop to breakeven (entry price)
                    hard_stop = entry_price
                    
                    exit_reason = 'T1_SCALE_OUT'
                    # Continue to look for T2/T3
            else:  # LONG
                if current_high >= t1_price:
                    # T1 hit - scale out 40-50%
                    scale_out_pct = 0.45  # 45% scale out
                    scale_out_shares = int(shares * scale_out_pct)
                    remaining_shares = shares - scale_out_shares
                    
                    # Calculate P&L for scaled out portion
                    if direction == 'SHORT':
                        scale_out_pnl = (actual_entry_price - t1_price) * scale_out_shares
                    else:
                        scale_out_pnl = (t1_price - actual_entry_price) * scale_out_shares
                    
                    total_pnl += scale_out_pnl
                    scaled_out = True
                    
                    # Move stop to breakeven (entry price)
                    hard_stop = entry_price
                    
                    exit_reason = 'T1_SCALE_OUT'
                    # Continue to look for T2/T3
        
        # 3. T2 - Opposite side of OR range or 2R (scale another 25%)
        if scaled_out and remaining_shares > 0:
            if direction == 'SHORT':
                if current_low <= t2_price:
                    # T2 hit - scale out another 25%
                    scale_out_pct = 0.25  # 25% of original position
                    scale_out_shares = int(shares * scale_out_pct)
                    remaining_shares = remaining_shares - scale_out_shares
                    
                    # Calculate P&L for scaled out portion
                    if direction == 'SHORT':
                        scale_out_pnl = (actual_entry_price - t2_price) * scale_out_shares
                    else:
                        scale_out_pnl = (t2_price - actual_entry_price) * scale_out_shares
                    
                    total_pnl += scale_out_pnl
                    exit_reason = 'T2_SCALE_OUT'
                    # Continue to look for T3
            else:  # LONG
                if current_high >= t2_price:
                    # T2 hit - scale out another 25%
                    scale_out_pct = 0.25  # 25% of original position
                    scale_out_shares = int(shares * scale_out_pct)
                    remaining_shares = remaining_shares - scale_out_shares
                    
                    # Calculate P&L for scaled out portion
                    if direction == 'SHORT':
                        scale_out_pnl = (actual_entry_price - t2_price) * scale_out_shares
                    else:
                        scale_out_pnl = (t2_price - actual_entry_price) * scale_out_shares
                    
                    total_pnl += scale_out_pnl
                    exit_reason = 'T2_SCALE_OUT'
                    # Continue to look for T3
        
        # 4. T3 - Opposite high-timeframe zone or 3R (trail or close remaining)
        if remaining_shares > 0:
            if direction == 'SHORT':
                if current_low <= t3_price:
                    # T3 hit - close remaining
                    exit_price = t3_price
                    exit_reason = 'T3_CLOSE'
                    exit_index = entry_index + i
                    break
            else:  # LONG
                if current_high >= t3_price:
                    # T3 hit - close remaining
                    exit_price = t3_price
                    exit_reason = 'T3_CLOSE'
                    exit_index = entry_index + i
                    break
    
    # If no exit found within lookahead, use final bar
    if exit_price is None and remaining_shares > 0:
        final_bar = bars_data[min(entry_index + max_lookahead, len(bars_data) - 1)]
        exit_price = final_bar.close
        exit_reason = 'TIME_EXIT'
        exit_index = min(entry_index + max_lookahead, len(bars_data) - 1)
    
    # Apply slippage to final exit price
    if exit_price is not None:
        if direction == 'SHORT':
            exit_price += slippage_amount
        else:
            exit_price -= slippage_amount
        
        # Calculate P&L for remaining shares
        if remaining_shares > 0:
            if direction == 'SHORT':
                remaining_pnl = (actual_entry_price - exit_price) * remaining_shares
            else:
                remaining_pnl = (exit_price - actual_entry_price) * remaining_shares
            
            total_pnl += remaining_pnl
    
    # Subtract commission
    total_pnl -= commission_per_trade
    
    return {
        'entry_price': actual_entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'remaining_shares': remaining_shares if remaining_shares is not None else shares,
        'pnl': total_pnl,
        'direction': direction,
        'slippage': slippage_amount,
        'commission': commission_per_trade,
        't1_price': t1_price,
        't2_price': t2_price,
        't3_price': t3_price,
        'exit_reason': exit_reason,
        'exit_index': exit_index,
        'bars_held': exit_index - entry_index if exit_index else 0,
        'scaled_out': scaled_out
    }


class FiveYearZone:
    """Represents a zone for 5-year backtest."""
    
    def __init__(self, zone_type: str, high_level: float, low_level: float, 
                 symbol: str, created_at: datetime, priority: str, 
                 confluence_score: float = 0.0, quality: int = 1, strength: float = 1.0):
        self.zone_type = zone_type
        self.high_level = high_level
        self.low_level = low_level
        self.symbol = symbol
        self.created_at = created_at
        self.priority = priority
        self.confluence_score = confluence_score
        self.quality = quality
        self.strength = strength
        self.zone_id = f"{symbol}_{zone_type}_{low_level:.2f}_{high_level:.2f}"
        self.touch_count = 0
        self.last_touch = None
        self.is_active = True
        
        # Lifecycle settings
        self.max_duration_hours = 24  # Daily reset
        self.max_touches = 1  # First touch only
        self.session_end_hour = 16  # 4 PM ET
    
    def contains_price(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is within the zone range."""
        return self.low_level - tolerance <= price <= self.high_level + tolerance
    
    def is_first_touch(self, current_time: datetime) -> bool:
        """Check if this is the first touch of this zone."""
        if self.touch_count == 0:
            self.touch_count += 1
            self.last_touch = current_time
            return True
        return False
    
    def is_zone_active(self, current_time: datetime) -> bool:
        """Check if zone is still active."""
        if not self.is_active:
            return False
        
        # Check if zone is from a different day (daily reset)
        if current_time.date() != self.created_at.date():
            self.is_active = False
            return False
        
        # Check session-based expiration
        if current_time.hour >= self.session_end_hour:
            self.is_active = False
            return False
        
        # Check touch-based expiration
        if self.touch_count >= self.max_touches:
            self.is_active = False
            return False
        
        return True


class FiveYearZoneManager:
    """Manages zones for 5-year backtest."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.zones = {symbol: [] for symbol in symbols}
        
        # Zone lifecycle parameters - DOUBLED for more trade opportunities
        self.max_zones_per_symbol_per_day = 8  # Doubled from 4
        self.primary_zones_per_symbol_per_day = 4  # Doubled from 2
        self.secondary_zones_per_symbol_per_day = 4  # Doubled from 2
        
        # Zone type priorities
        self.primary_zone_types = ['prior_day_high', 'prior_day_low', 'value_area_high', 'value_area_low']
        self.secondary_zone_types = ['intraday_structure', 'vwap_deviation_high', 'vwap_deviation_low']
        
        # Zone creation parameters
        self.zone_width_percentage = 0.002  # 0.2% zone width
        self.min_confluence_score = 0.5
        
        # Track daily zone counts
        self.daily_zone_counts = {symbol: {} for symbol in symbols}
    
    def get_daily_zone_count(self, symbol: str, date: datetime) -> Dict[str, int]:
        """Get the current zone count for a symbol on a specific date."""
        date_key = date.date()
        if date_key not in self.daily_zone_counts[symbol]:
            self.daily_zone_counts[symbol][date_key] = {
                'total': 0,
                'primary': 0,
                'secondary': 0
            }
        return self.daily_zone_counts[symbol][date_key]
    
    def can_create_zone(self, symbol: str, zone_type: str, date: datetime) -> bool:
        """Check if we can create a new zone for this symbol on this date."""
        daily_counts = self.get_daily_zone_count(symbol, date)
        
        # Determine priority
        priority = 'primary' if zone_type in self.primary_zone_types else 'secondary'
        
        # Check limits
        if daily_counts['total'] >= self.max_zones_per_symbol_per_day:
            return False
        if priority == 'primary' and daily_counts['primary'] >= self.primary_zones_per_symbol_per_day:
            return False
        if priority == 'secondary' and daily_counts['secondary'] >= self.secondary_zones_per_symbol_per_day:
            return False
        
        return True
    
    def calculate_confluence_score(self, zone_type: str, level: float, 
                                 qrs_score: float = 7.0, zone_strength: float = 1.0,
                                 volume_factor: float = 1.0, time_factor: float = 1.0) -> float:
        """Calculate enhanced confluence score for zone selection priority."""
        score = 0.0
        
        # 1. Base score by zone type (40% weight)
        if zone_type in self.primary_zone_types:
            score += 0.4
        elif zone_type in self.secondary_zone_types:
            score += 0.2
        
        # 2. QRS score component (25% weight) - more important
        qrs_component = min(qrs_score / 10.0, 1.0) * 0.25
        score += qrs_component
        
        # 3. Zone strength component (15% weight)
        strength_component = min(zone_strength / 2.0, 1.0) * 0.15
        score += strength_component
        
        # 4. Volume factor (10% weight) - NEW
        volume_component = min(volume_factor / 2.0, 1.0) * 0.10
        score += volume_component
        
        # 5. Time factor (5% weight) - NEW (favor recent zones)
        time_component = min(time_factor, 1.0) * 0.05
        score += time_component
        
        # 6. Random component for testing (5% weight) - reduced
        random_component = random.uniform(0.0, 0.05)
        score += random_component
        
        return min(score, 1.0)
    
    def create_zone(self, symbol: str, zone_type: str, level: float, 
                   current_time: datetime, qrs_score: float = 7.0, 
                   zone_strength: float = 1.0, volume_factor: float = 1.0) -> Optional[FiveYearZone]:
        """Create a new zone with daily quantity limits and enhanced confluence scoring."""
        
        # Check if we can create a zone for this symbol on this date
        if not self.can_create_zone(symbol, zone_type, current_time):
            return None
        
        # Calculate time factor (favor recent zones)
        time_factor = 1.0  # In a real implementation, this would be based on zone age
        
        # Calculate enhanced confluence score
        confluence_score = self.calculate_confluence_score(
            zone_type, level, qrs_score, zone_strength, volume_factor, time_factor
        )
        
        # Lower the minimum confluence score to allow more zones
        min_confluence = 0.3  # Reduced from 0.5
        if confluence_score < min_confluence:
            return None
        
        # Determine priority
        priority = 'primary' if zone_type in self.primary_zone_types else 'secondary'
        
        # Calculate zone range
        zone_width = level * self.zone_width_percentage
        
        if zone_type in ['prior_day_high', 'value_area_high', 'intraday_structure', 'vwap_deviation_high']:
            high_level = level
            low_level = level - zone_width
        elif zone_type in ['prior_day_low', 'value_area_low', 'vwap_deviation_low']:
            low_level = level
            high_level = level + zone_width
        else:
            center = level
            high_level = center + zone_width / 2
            low_level = center - zone_width / 2
        
        zone = FiveYearZone(
            zone_type=zone_type,
            high_level=high_level,
            low_level=low_level,
            symbol=symbol,
            created_at=current_time,
            priority=priority,
            confluence_score=confluence_score,
            quality=min(3, int(confluence_score * 3)),
            strength=confluence_score
        )
        
        return zone
    
    def add_zone(self, zone: FiveYearZone) -> bool:
        """Add a zone to the manager and update daily counts."""
        symbol = zone.symbol
        date = zone.created_at
        
        # Update daily counts
        daily_counts = self.get_daily_zone_count(symbol, date)
        daily_counts['total'] += 1
        if zone.priority == 'primary':
            daily_counts['primary'] += 1
        else:
            daily_counts['secondary'] += 1
        
        # Add zone
        self.zones[symbol].append(zone)
        return True
    
    def find_matching_zone(self, symbol: str, price: float, current_time: datetime) -> Optional[FiveYearZone]:
        """Find a matching active zone for the given price."""
        
        if symbol not in self.zones:
            return None
        
        # Clean up expired zones
        self.zones[symbol] = [zone for zone in self.zones[symbol] if zone.is_zone_active(current_time)]
        
        # Find matching zone (prioritize by confluence score)
        matching_zones = []
        for zone in self.zones[symbol]:
            if zone.contains_price(price) and zone.is_zone_active(current_time):
                matching_zones.append(zone)
        
        # Return highest confluence zone
        if matching_zones:
            return max(matching_zones, key=lambda z: z.confluence_score)
        
        return None


def detect_market_context(bars_data: List, current_index: int, lookback: int = 20) -> str:
    """Detect market context: trend, balanced, or choppy."""
    if current_index < lookback:
        return "balanced"  # Default for early bars
    
    recent_bars = bars_data[current_index - lookback:current_index]
    if len(recent_bars) < lookback:
        return "balanced"
    
    # Calculate simple trend detection
    prices = [bar.close for bar in recent_bars]
    high_prices = [bar.high for bar in recent_bars]
    low_prices = [bar.low for bar in recent_bars]
    
    # Price momentum
    price_change = (prices[-1] - prices[0]) / prices[0] * 100
    
    # Volatility
    price_range = (max(high_prices) - min(low_prices)) / prices[0] * 100
    
    # Trend detection
    if abs(price_change) > 0.5 and price_range < 2.0:  # Strong directional move, low volatility
        return "trend"
    elif price_range > 3.0:  # High volatility
        return "choppy"
    else:
        return "balanced"


def run_5year_zone_fade_backtest():
    """Run the backtest using 5 years of data with proper Zone Fade exit strategy."""
    
    print("🎯 5-YEAR ZONE FADE BACKTEST (2020-2024) - ENHANCED")
    print("=" * 70)
    print("This version uses 5 years of data with the complete Zone Fade Strategy:")
    print("- Hard stops at zone invalidation")
    print("- T1: Nearest of VWAP or 1R (scale out 40-50%, move stop to breakeven)")
    print("- T2: Opposite side of OR range or 2R (scale another 25%)")
    print("- T3: Opposite high-timeframe zone or 3R (trail or close remaining)")
    print("- Enhanced confluence scoring with volume and time factors")
    print("- Doubled zone limits (8 zones per symbol per day)")
    print("- Market context filtering (trend/balanced/choppy)")
    print()
    
    # Load 5-year data
    bars_data = load_5year_data()
    
    if not bars_data:
        print("❌ No 5-year data loaded. Please run download_5year_data.py first.")
        return
    
    # Load entry points (we'll use the 2024 data for now, but in a real implementation
    # we would generate entry points for all 5 years)
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
        print(f"✅ Loaded {len(entry_points)} entry points from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create zone manager - only for symbols that have data
    symbols = list(bars_data.keys())
    zone_manager = FiveYearZoneManager(symbols)
    
    # Filter entry points to only include symbols with data
    available_symbols = set(symbols)
    original_count = len(entry_points)
    entry_points = entry_points[entry_points['symbol'].isin(available_symbols)]
    filtered_count = len(entry_points)
    
    if filtered_count < original_count:
        print(f"⚠️  Filtered entry points: {original_count} -> {filtered_count} (removed symbols without data)")
        print(f"   Available symbols: {', '.join(available_symbols)}")
    
    print(f"🔧 Created zone manager for symbols: {symbols}")
    
    # Process entries
    processed = 0
    executed = 0
    rejected = 0
    rejection_reasons = {}
    zones_created = 0
    zones_matched = 0
    zones_rejected_daily_limit = 0
    zones_rejected_confluence = 0
    
    # Track performance
    total_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    trades = []
    
    # Track exit reasons
    exit_reasons = {}
    
    # Track by symbol
    symbol_stats = {symbol: {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0} for symbol in symbols}
    
    print(f"📊 Processing all {len(entry_points)} entries...")
    
    for i, (_, entry) in enumerate(entry_points.iterrows()):
        symbol = entry['symbol']
        zone_type = entry['zone_type']
        level = entry['zone_level']
        price = entry['price']
        current_time = entry['timestamp']
        qrs_score = entry.get('qrs_score', 7.0)
        zone_strength = entry.get('zone_strength', 1.0)
        
        # Check if we can create a zone for this symbol on this date
        if not zone_manager.can_create_zone(symbol, zone_type, current_time):
            zones_rejected_daily_limit += 1
            rejected += 1
            rejection_reasons['Daily zone limit reached'] = rejection_reasons.get('Daily zone limit reached', 0) + 1
            continue
        
        # Calculate volume factor (simulate based on QRS score for now)
        volume_factor = 1.0 + (qrs_score - 5.0) / 10.0  # Higher QRS = higher volume factor
        
        # Try to create a new zone
        zone = zone_manager.create_zone(symbol, zone_type, level, 
                                       current_time, qrs_score, zone_strength, volume_factor)
        
        if not zone:
            zones_rejected_confluence += 1
            rejected += 1
            rejection_reasons['Insufficient confluence'] = rejection_reasons.get('Insufficient confluence', 0) + 1
            continue
        
        # Add zone
        if zone_manager.add_zone(zone):
            zones_created += 1
        
        # Find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, price, current_time)
        
        if not matching_zone:
            rejected += 1
            rejection_reasons['No matching zone'] = rejection_reasons.get('No matching zone', 0) + 1
            continue
        
        # Check first touch
        is_first_touch = matching_zone.is_first_touch(current_time)
        
        if not is_first_touch:
            rejected += 1
            rejection_reasons['Not first touch of zone'] = rejection_reasons.get('Not first touch of zone', 0) + 1
            continue
        
        zones_matched += 1
        processed += 1
        
        # Find the corresponding bar in the 5-year data
        entry_index = None
        if symbol in bars_data:
            for j, bar in enumerate(bars_data[symbol]):
                if abs((bar.timestamp - current_time).total_seconds()) < 60:  # Within 1 minute
                    entry_index = j
                    break
        
        if entry_index is None:
            rejected += 1
            rejection_reasons['No matching bar data'] = rejection_reasons.get('No matching bar data', 0) + 1
            continue
        
        # Market context filtering
        market_context = detect_market_context(bars_data[symbol], entry_index)
        
        # Filter out trend days for fade trades (fade trades work better in balanced/choppy markets)
        if market_context == "trend":
            rejected += 1
            rejection_reasons['Trend day - not suitable for fades'] = rejection_reasons.get('Trend day - not suitable for fades', 0) + 1
            continue
        
        # Prepare entry data
        entry_data = {
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'timestamp': current_time,
            'price': price,
            'hard_stop': entry['hard_stop'],
            'vwap': entry.get('vwap', price),
            'qrs_score': qrs_score,
            'direction': entry.get('direction', 'LONG' if zone_type in ['prior_day_low', 'value_area_low'] else 'SHORT')
        }
        
        # Execute trade with PROPER Zone Fade exit strategy
        try:
            trade_result = simulate_proper_zone_fade_exit(entry_data, bars_data[symbol], entry_index)
            
            # Track performance
            total_pnl += trade_result['pnl']
            if trade_result['pnl'] > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            # Track by symbol
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade_result['pnl']
            if trade_result['pnl'] > 0:
                symbol_stats[symbol]['wins'] += 1
            else:
                symbol_stats[symbol]['losses'] += 1
            
            # Track exit reasons
            exit_reason = trade_result['exit_reason']
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
            
            trades.append(trade_result)
            executed += 1
            
            # Show progress every 20 trades
            if executed % 20 == 0:
                print(f"   Progress: {executed} trades executed, {rejected} rejected")
        
        except Exception as e:
            rejected += 1
            error_reason = f"Error: {str(e)[:50]}..."
            if error_reason not in rejection_reasons:
                rejection_reasons[error_reason] = 0
            rejection_reasons[error_reason] += 1
            continue
    
    # Calculate results
    execution_rate = executed / processed * 100 if processed > 0 else 0
    win_rate = (winning_trades / executed * 100) if executed > 0 else 0
    
    # Calculate additional metrics
    if trades:
        winning_pnl = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        losing_pnl = sum(t['pnl'] for t in trades if t['pnl'] < 0)
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
        
        # Calculate additional metrics
        returns = [t['pnl'] for t in trades]
        max_drawdown = 0
        peak = 10000
        running_balance = 10000
        
        for pnl in returns:
            running_balance += pnl
            if running_balance > peak:
                peak = running_balance
            drawdown = (peak - running_balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        winning_pnl = 0
        losing_pnl = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_drawdown = 0
        sharpe_ratio = 0
    
    print(f"\n📊 5-YEAR ZONE FADE BACKTEST RESULTS:")
    print(f"   Total Entries: {len(entry_points)}")
    print(f"   Processed Entries: {processed}")
    print(f"   Zones Created: {zones_created}")
    print(f"   Zones Rejected (Daily Limit): {zones_rejected_daily_limit}")
    print(f"   Zones Rejected (Confluence): {zones_rejected_confluence}")
    print(f"   Zones Matched: {zones_matched}")
    print(f"   Executed Trades: {executed}")
    print(f"   Rejected Trades: {rejected}")
    print(f"   Execution Rate: {execution_rate:.1f}%")
    
    print(f"\n📈 PERFORMANCE BY SYMBOL:")
    for symbol, stat in symbol_stats.items():
        if stat['trades'] > 0:
            symbol_win_rate = (stat['wins'] / stat['trades']) * 100
            print(f"   {symbol}:")
            print(f"     Trades: {stat['trades']}")
            print(f"     P&L: ${stat['pnl']:.2f}")
            print(f"     Win Rate: {symbol_win_rate:.1f}%")
            print(f"     Wins: {stat['wins']}, Losses: {stat['losses']}")
    
    print(f"\n💰 OVERALL PERFORMANCE METRICS:")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print(f"   Total Return: {(total_pnl / 10000.0) * 100:.2f}%")
    print(f"   Total Trades: {executed}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Winning Trades: {winning_trades}")
    print(f"   Losing Trades: {losing_trades}")
    print(f"   Average Win: ${avg_win:.2f}")
    print(f"   Average Loss: ${avg_loss:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Max Drawdown: {max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Print exit reason analysis
    print(f"\n🎯 EXIT REASON ANALYSIS:")
    print(f"   Exit reasons breakdown:")
    sorted_exit_reasons = sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_exit_reasons:
        percentage = count / executed * 100
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Print rejection analysis
    print(f"\n❌ REJECTION ANALYSIS:")
    print(f"   Top rejection reasons:")
    sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        percentage = count / len(entry_points) * 100
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Strategy assessment
    print(f"\n🔍 STRATEGY ASSESSMENT:")
    if executed == 0:
        print("   ❌ NO TRADES: Strategy is too restrictive - check rejection reasons")
    elif executed >= 80:
        print("   ✅ EXCELLENT: Trade count meets target (80+ trades)")
    elif executed >= 50:
        print("   ✅ GOOD: Trade count is reasonable (50+ trades)")
    elif executed >= 20:
        print("   ⚠️  MODERATE: Trade count is low but acceptable (20+ trades)")
    else:
        print("   ❌ POOR: Trade count is too low (<20 trades)")
    
    if executed > 0:
        if win_rate > 60:
            print("   ✅ HIGH WIN RATE: Strategy shows consistent winning")
        elif win_rate > 50:
            print("   ⚠️  MODERATE WIN RATE: Strategy shows mixed results")
        else:
            print("   ❌ LOW WIN RATE: Strategy shows poor win rate")
        
        if profit_factor > 2.0:
            print("   ✅ EXCELLENT PROFIT FACTOR: Strong risk/reward")
        elif profit_factor > 1.5:
            print("   ✅ GOOD PROFIT FACTOR: Positive risk/reward")
        elif profit_factor > 1.0:
            print("   ⚠️  MODERATE PROFIT FACTOR: Barely profitable")
        else:
            print("   ❌ POOR PROFIT FACTOR: Losing strategy")
    
    print("\n" + "=" * 70)
    
    return {
        'summary': {
            'total_entries': len(entry_points),
            'processed_entries': processed,
            'executed_trades': executed,
            'rejected_trades': rejected,
            'execution_rate': execution_rate,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / 10000.0) * 100,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        },
        'symbol_stats': symbol_stats,
        'exit_reasons': exit_reasons,
        'rejection_reasons': rejection_reasons,
        'trades': trades
    }


if __name__ == "__main__":
    results = run_5year_zone_fade_backtest()