#!/usr/bin/env python3
"""
Full 1-Minute Data Backtest

This version runs the backtest using the full 1-minute data to investigate
the 0% win rate bug and properly track rejected setups.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
import json

def load_1min_data():
    """Load the full 1-minute data for all symbols."""
    print("📊 Loading 1-minute data...")
    
    symbols = ['SPY', 'QQQ', 'IWM']
    data = {}
    
    for symbol in symbols:
        try:
            file_path = f"/app/data/2024/{symbol}_2024.pkl"
            with open(file_path, 'rb') as f:
                symbol_data = pickle.load(f)
            data[symbol] = symbol_data
            print(f"   ✅ {symbol}: {len(symbol_data)} bars loaded")
        except Exception as e:
            print(f"   ❌ {symbol}: Error loading data - {e}")
    
    return data


def calculate_realistic_t1_price(entry_price: float, hard_stop: float, direction: str, 
                                vwap: float, qrs_score: float) -> float:
    """Calculate realistic T1 price that can sometimes fail."""
    
    # Calculate risk amount
    risk_amount = abs(entry_price - hard_stop)
    if risk_amount <= 0:
        risk_amount = entry_price * 0.01  # 1% fallback
    
    # Calculate T1 reward based on QRS quality
    if qrs_score >= 9.0:  # A-grade setup
        t1_reward = 1.0 * risk_amount
    elif qrs_score >= 7.0:  # B-grade setup
        t1_reward = 0.8 * risk_amount
    else:  # C-grade setup
        t1_reward = 0.6 * risk_amount
    
    # FIXED: Use realistic T1 calculation that can fail
    if direction == "SHORT":
        # For SHORT: T1 should be below entry price
        if vwap < entry_price:
            # VWAP is below entry - use it as target
            t1_price = vwap
        else:
            # VWAP is above entry - use calculated reward target
            t1_price = entry_price - t1_reward
    else:  # LONG
        # For LONG: T1 should be above entry price
        if vwap > entry_price:
            # VWAP is above entry - use it as target
            t1_price = vwap
        else:
            # VWAP is below entry - use calculated reward target
            t1_price = entry_price + t1_reward
    
    return t1_price


def simulate_realistic_trade_execution(entry_data: Dict, bars_data: List, entry_index: int) -> Dict:
    """Simulate trade execution with realistic P&L calculation using actual market data."""
    
    entry_price = entry_data['price']
    hard_stop = entry_data['hard_stop']
    vwap = entry_data.get('vwap', entry_price)
    qrs_score = entry_data.get('qrs_score', 7.0)
    direction = entry_data.get('direction', 'LONG')
    
    # Calculate REALISTIC target prices
    t1_price = calculate_realistic_t1_price(entry_price, hard_stop, direction, vwap, qrs_score)
    
    # Calculate position size (10% of $10,000 = $1,000)
    account_balance = 10000.0
    max_equity_per_trade = 0.10
    position_value = account_balance * max_equity_per_trade
    
    # Calculate shares/contracts
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
    
    # Calculate commission
    commission_per_trade = 5.0
    
    # Simulate realistic exit using actual market data
    # Look ahead in the bars to see what actually happened
    exit_price = None
    exit_reason = None
    exit_index = None
    
    # Look ahead up to 100 bars (100 minutes) to find exit
    max_lookahead = min(100, len(bars_data) - entry_index - 1)
    
    for i in range(1, max_lookahead + 1):
        if entry_index + i >= len(bars_data):
            break
            
        current_bar = bars_data[entry_index + i]
        current_high = current_bar.high
        current_low = current_bar.low
        current_close = current_bar.close
        
        if direction == 'SHORT':
            # For SHORT trades, check if price went below T1 (profit) or above stop (loss)
            if current_low <= t1_price:
                # T1 hit - winning trade
                exit_price = t1_price
                exit_reason = 'T1_HIT'
                exit_index = entry_index + i
                break
            elif current_high >= hard_stop:
                # Stop loss hit - losing trade
                exit_price = hard_stop
                exit_reason = 'STOP_LOSS'
                exit_index = entry_index + i
                break
        else:  # LONG
            # For LONG trades, check if price went above T1 (profit) or below stop (loss)
            if current_high >= t1_price:
                # T1 hit - winning trade
                exit_price = t1_price
                exit_reason = 'T1_HIT'
                exit_index = entry_index + i
                break
            elif current_low <= hard_stop:
                # Stop loss hit - losing trade
                exit_price = hard_stop
                exit_reason = 'STOP_LOSS'
                exit_index = entry_index + i
                break
    
    # If no exit found within lookahead, use final bar
    if exit_price is None:
        final_bar = bars_data[min(entry_index + max_lookahead, len(bars_data) - 1)]
        exit_price = final_bar.close
        exit_reason = 'TIME_EXIT'
        exit_index = min(entry_index + max_lookahead, len(bars_data) - 1)
    
    # Apply slippage to exit price
    if direction == 'SHORT':
        exit_price += slippage_amount
    else:
        exit_price -= slippage_amount
    
    # Calculate P&L
    if direction == 'SHORT':
        pnl = (actual_entry_price - exit_price) * shares - commission_per_trade
    else:
        pnl = (exit_price - actual_entry_price) * shares - commission_per_trade
    
    return {
        'entry_price': actual_entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'pnl': pnl,
        'direction': direction,
        'slippage': slippage_amount,
        'commission': commission_per_trade,
        't1_price': t1_price,
        'exit_reason': exit_reason,
        'exit_index': exit_index,
        'bars_held': exit_index - entry_index if exit_index else 0
    }


class FullDataZone:
    """Represents a zone for full data backtest."""
    
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


class FullDataZoneManager:
    """Manages zones for full data backtest."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.zones = {symbol: [] for symbol in symbols}
        
        # Zone lifecycle parameters
        self.max_zones_per_symbol_per_day = 4
        self.primary_zones_per_symbol_per_day = 2
        self.secondary_zones_per_symbol_per_day = 2
        
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
                                 qrs_score: float = 7.0, zone_strength: float = 1.0) -> float:
        """Calculate confluence score for zone selection priority."""
        score = 0.0
        
        # Base score by zone type
        if zone_type in self.primary_zone_types:
            score += 0.4
        elif zone_type in self.secondary_zone_types:
            score += 0.2
        
        # Add QRS score component (normalized to 0-1)
        qrs_component = min(qrs_score / 10.0, 1.0) * 0.3
        score += qrs_component
        
        # Add zone strength component
        strength_component = min(zone_strength / 2.0, 1.0) * 0.2
        score += strength_component
        
        # Add random component for testing
        random_component = random.uniform(0.0, 0.1)
        score += random_component
        
        return min(score, 1.0)
    
    def create_zone(self, symbol: str, zone_type: str, level: float, 
                   current_time: datetime, qrs_score: float = 7.0, 
                   zone_strength: float = 1.0) -> Optional[FullDataZone]:
        """Create a new zone with daily quantity limits."""
        
        # Check if we can create a zone for this symbol on this date
        if not self.can_create_zone(symbol, zone_type, current_time):
            return None
        
        # Calculate confluence score
        confluence_score = self.calculate_confluence_score(zone_type, level, qrs_score, zone_strength)
        
        # Only create zones with sufficient confluence
        if confluence_score < self.min_confluence_score:
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
        
        zone = FullDataZone(
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
    
    def add_zone(self, zone: FullDataZone) -> bool:
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
    
    def find_matching_zone(self, symbol: str, price: float, current_time: datetime) -> Optional[FullDataZone]:
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


def run_full_1min_data_backtest():
    """Run the backtest using full 1-minute data."""
    
    print("🚀 FULL 1-MINUTE DATA BACKTEST")
    print("=" * 80)
    print("This version uses actual 1-minute market data to investigate the 0% win rate bug.")
    print("We will track all rejected setups for analysis.")
    print()
    
    # Load 1-minute data
    bars_data = load_1min_data()
    
    if not bars_data:
        print("❌ No 1-minute data loaded. Exiting.")
        return
    
    # Load entry points
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
        print(f"✅ Loaded {len(entry_points)} entry points from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create zone manager
    symbols = list(bars_data.keys())
    zone_manager = FullDataZoneManager(symbols)
    
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
    
    # Track rejected setups
    rejected_setups = []
    
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
            
            # Track rejected setup
            rejected_setups.append({
                'entry_id': entry['entry_id'],
                'symbol': symbol,
                'timestamp': current_time,
                'price': price,
                'zone_type': zone_type,
                'zone_level': level,
                'qrs_score': qrs_score,
                'rejection_reason': 'Daily zone limit reached',
                'rejection_details': f"Daily limit: {zone_manager.get_daily_zone_count(symbol, current_time)}"
            })
            continue
        
        # Try to create a new zone
        zone = zone_manager.create_zone(symbol, zone_type, level, 
                                       current_time, qrs_score, zone_strength)
        
        if not zone:
            zones_rejected_confluence += 1
            rejected += 1
            rejection_reasons['Insufficient confluence'] = rejection_reasons.get('Insufficient confluence', 0) + 1
            
            # Track rejected setup
            rejected_setups.append({
                'entry_id': entry['entry_id'],
                'symbol': symbol,
                'timestamp': current_time,
                'price': price,
                'zone_type': zone_type,
                'zone_level': level,
                'qrs_score': qrs_score,
                'rejection_reason': 'Insufficient confluence',
                'rejection_details': f"Confluence score too low"
            })
            continue
        
        # Add zone
        if zone_manager.add_zone(zone):
            zones_created += 1
        
        # Find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, price, current_time)
        
        if not matching_zone:
            rejected += 1
            rejection_reasons['No matching zone'] = rejection_reasons.get('No matching zone', 0) + 1
            
            # Track rejected setup
            rejected_setups.append({
                'entry_id': entry['entry_id'],
                'symbol': symbol,
                'timestamp': current_time,
                'price': price,
                'zone_type': zone_type,
                'zone_level': level,
                'qrs_score': qrs_score,
                'rejection_reason': 'No matching zone',
                'rejection_details': f"No active zone found for price {price}"
            })
            continue
        
        # Check first touch
        is_first_touch = matching_zone.is_first_touch(current_time)
        
        if not is_first_touch:
            rejected += 1
            rejection_reasons['Not first touch of zone'] = rejection_reasons.get('Not first touch of zone', 0) + 1
            
            # Track rejected setup
            rejected_setups.append({
                'entry_id': entry['entry_id'],
                'symbol': symbol,
                'timestamp': current_time,
                'price': price,
                'zone_type': zone_type,
                'zone_level': level,
                'qrs_score': qrs_score,
                'rejection_reason': 'Not first touch of zone',
                'rejection_details': f"Zone already touched: {matching_zone.zone_id}"
            })
            continue
        
        zones_matched += 1
        processed += 1
        
        # Find the corresponding bar in the 1-minute data
        entry_index = None
        if symbol in bars_data:
            for j, bar in enumerate(bars_data[symbol]):
                if abs((bar.timestamp - current_time).total_seconds()) < 60:  # Within 1 minute
                    entry_index = j
                    break
        
        if entry_index is None:
            rejected += 1
            rejection_reasons['No matching bar data'] = rejection_reasons.get('No matching bar data', 0) + 1
            
            # Track rejected setup
            rejected_setups.append({
                'entry_id': entry['entry_id'],
                'symbol': symbol,
                'timestamp': current_time,
                'price': price,
                'zone_type': zone_type,
                'zone_level': level,
                'qrs_score': qrs_score,
                'rejection_reason': 'No matching bar data',
                'rejection_details': f"No 1-minute bar found for timestamp {current_time}"
            })
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
        
        # Execute trade with REAL market data
        try:
            trade_result = simulate_realistic_trade_execution(entry_data, bars_data[symbol], entry_index)
            
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
            
            # Track rejected setup
            rejected_setups.append({
                'entry_id': entry['entry_id'],
                'symbol': symbol,
                'timestamp': current_time,
                'price': price,
                'zone_type': zone_type,
                'zone_level': level,
                'qrs_score': qrs_score,
                'rejection_reason': 'Execution error',
                'rejection_details': str(e)
            })
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
    
    print(f"\n📊 FULL 1-MINUTE DATA BACKTEST RESULTS:")
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
    
    # Print rejection analysis
    print(f"\n❌ REJECTION ANALYSIS:")
    print(f"   Top rejection reasons:")
    sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        percentage = count / len(entry_points) * 100
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Print rejected setups summary
    print(f"\n📋 REJECTED SETUPS SUMMARY:")
    print(f"   Total Rejected Setups: {len(rejected_setups)}")
    
    # Group by rejection reason
    rejection_groups = {}
    for setup in rejected_setups:
        reason = setup['rejection_reason']
        if reason not in rejection_groups:
            rejection_groups[reason] = []
        rejection_groups[reason].append(setup)
    
    for reason, setups in rejection_groups.items():
        print(f"   {reason}: {len(setups)} setups")
        if len(setups) <= 5:  # Show details for small groups
            for setup in setups[:3]:  # Show first 3
                print(f"     - {setup['entry_id']}: {setup['symbol']} at {setup['price']} ({setup['timestamp']})")
        else:
            print(f"     - Showing first 3 of {len(setups)} setups")
            for setup in setups[:3]:
                print(f"     - {setup['entry_id']}: {setup['symbol']} at {setup['price']} ({setup['timestamp']})")
    
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
    
    # Save detailed results
    results = {
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
        'rejection_reasons': rejection_reasons,
        'rejected_setups': rejected_setups,
        'trades': trades
    }
    
    # Save to JSON file
    with open('/app/results/2024/corrected/full_1min_data_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: /app/results/2024/corrected/full_1min_data_backtest_results.json")
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_full_1min_data_backtest()