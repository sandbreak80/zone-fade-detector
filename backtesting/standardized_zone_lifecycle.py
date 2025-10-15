#!/usr/bin/env python3
"""
Standardized Zone Lifecycle Implementation

Implements the exact zone lifecycle specification with 3-4 zones per symbol per day.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from unified_trading_framework import UnifiedTradingFramework, create_unified_framework


class StandardizedZone:
    """Represents a standardized zone following the lifecycle specification."""
    
    def __init__(self, zone_type: str, high_level: float, low_level: float, 
                 symbol: str, created_at: datetime, priority: str, 
                 confluence_score: float = 0.0, quality: int = 1, strength: float = 1.0):
        self.zone_type = zone_type
        self.high_level = high_level
        self.low_level = low_level
        self.symbol = symbol
        self.created_at = created_at
        self.priority = priority  # 'primary' or 'secondary'
        self.confluence_score = confluence_score
        self.quality = quality
        self.strength = strength
        self.zone_id = f"{symbol}_{zone_type}_{low_level:.2f}_{high_level:.2f}"
        self.touch_count = 0
        self.last_touch = None
        self.is_active = True
        
        # Lifecycle settings
        self.max_duration_hours = 24  # Zones expire after 24 hours (daily reset)
        self.max_touches = 1  # First touch only
        self.session_end_hour = 16  # 4 PM ET (end of regular session)
    
    def contains_price(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is within the zone range."""
        return self.low_level - tolerance <= price <= self.high_level + tolerance
    
    def is_first_touch(self, current_time: datetime) -> bool:
        """Check if this is the first touch of this zone (BattleCard principle)."""
        if self.touch_count == 0:
            self.touch_count += 1
            self.last_touch = current_time
            return True
        return False
    
    def is_zone_active(self, current_time: datetime) -> bool:
        """Check if zone is still active based on lifecycle rules."""
        if not self.is_active:
            return False
        
        # Check time-based expiration (daily reset)
        hours_since_creation = (current_time - self.created_at).total_seconds() / 3600
        if hours_since_creation > self.max_duration_hours:
            self.is_active = False
            return False
        
        # Check session-based expiration
        if current_time.hour >= self.session_end_hour:
            self.is_active = False
            return False
        
        # Check touch-based expiration (first touch only)
        if self.touch_count >= self.max_touches:
            self.is_active = False
            return False
        
        return True
    
    def get_zone_center(self) -> float:
        """Get the center of the zone range."""
        return (self.high_level + self.low_level) / 2
    
    def get_zone_width(self) -> float:
        """Get the width of the zone range."""
        return self.high_level - self.low_level


class StandardizedZoneManager:
    """Manages zones according to the standardized lifecycle specification."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.zones = {symbol: [] for symbol in symbols}
        
        # Zone lifecycle parameters
        self.max_zones_per_symbol = 4  # 3-4 total zones per symbol per day
        self.primary_zones_per_symbol = 2  # 2 primary zones
        self.secondary_zones_per_symbol = 2  # 1-2 secondary zones
        
        # Zone type priorities
        self.primary_zone_types = ['prior_day_high', 'prior_day_low', 'value_area_high', 'value_area_low']
        self.secondary_zone_types = ['intraday_structure', 'vwap_deviation_high', 'vwap_deviation_low']
        
        # Zone creation parameters
        self.zone_width_percentage = 0.002  # 0.2% zone width
        self.min_confluence_score = 0.6  # Minimum confluence score for zone creation
    
    def calculate_confluence_score(self, zone_type: str, level: float, 
                                 bars: List, current_index: int) -> float:
        """Calculate confluence score for zone selection priority."""
        score = 0.0
        
        # Base score by zone type
        if zone_type in self.primary_zone_types:
            score += 0.4
        elif zone_type in self.secondary_zone_types:
            score += 0.2
        
        # Add structural swing score
        if self._has_structural_swing(bars, current_index, level):
            score += 0.3
        
        # Add VWAP proximity score
        if self._is_near_vwap(bars, current_index, level):
            score += 0.2
        
        # Add volume edge score
        if self._has_volume_edge(bars, current_index, level):
            score += 0.1
        
        return min(score, 1.0)
    
    def _has_structural_swing(self, bars: List, current_index: int, level: float) -> bool:
        """Check if level has structural swing significance."""
        if current_index < 10 or current_index >= len(bars) - 10:
            return False
        
        # Look for swing highs/lows in recent bars
        recent_bars = bars[max(0, current_index-10):current_index+10]
        highs = [bar.high for bar in recent_bars]
        lows = [bar.low for bar in recent_bars]
        
        # Check if level is near a swing high or low
        tolerance = level * 0.001  # 0.1% tolerance
        for high in highs:
            if abs(high - level) <= tolerance:
                return True
        for low in lows:
            if abs(low - level) <= tolerance:
                return True
        
        return False
    
    def _is_near_vwap(self, bars: List, current_index: int, level: float) -> bool:
        """Check if level is near VWAP."""
        if current_index < 20:
            return False
        
        # Calculate VWAP for last 20 bars
        recent_bars = bars[max(0, current_index-20):current_index]
        total_volume = sum(bar.volume for bar in recent_bars)
        if total_volume == 0:
            return False
        
        vwap = sum(bar.volume * (bar.high + bar.low + bar.close) / 3 for bar in recent_bars) / total_volume
        
        # Check if level is within 0.5% of VWAP
        tolerance = vwap * 0.005
        return abs(level - vwap) <= tolerance
    
    def _has_volume_edge(self, bars: List, current_index: int, level: float) -> bool:
        """Check if level has volume edge."""
        if current_index < 5:
            return False
        
        # Check if recent volume is above average
        recent_bars = bars[max(0, current_index-5):current_index]
        recent_volume = sum(bar.volume for bar in recent_bars) / len(recent_bars)
        
        # Calculate average volume for longer period
        if current_index < 20:
            return False
        
        longer_bars = bars[max(0, current_index-20):current_index]
        avg_volume = sum(bar.volume for bar in longer_bars) / len(longer_bars)
        
        # Volume edge if recent volume is 1.5x average
        return recent_volume >= avg_volume * 1.5
    
    def create_zone(self, symbol: str, zone_type: str, level: float, 
                   current_time: datetime, bars: List, current_index: int) -> Optional[StandardizedZone]:
        """Create a new standardized zone with confluence scoring."""
        
        # Calculate confluence score
        confluence_score = self.calculate_confluence_score(zone_type, level, bars, current_index)
        
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
        
        zone = StandardizedZone(
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
    
    def add_zone(self, zone: StandardizedZone) -> bool:
        """Add a zone to the manager, respecting quantity limits."""
        symbol = zone.symbol
        
        # Check if we can add more zones for this symbol
        current_zones = [z for z in self.zones[symbol] if z.is_active]
        primary_count = len([z for z in current_zones if z.priority == 'primary'])
        secondary_count = len([z for z in current_zones if z.priority == 'secondary'])
        
        # Check limits
        if zone.priority == 'primary' and primary_count >= self.primary_zones_per_symbol:
            return False
        if zone.priority == 'secondary' and secondary_count >= self.secondary_zones_per_symbol:
            return False
        if len(current_zones) >= self.max_zones_per_symbol:
            return False
        
        # Add zone
        self.zones[symbol].append(zone)
        return True
    
    def find_matching_zone(self, symbol: str, price: float, current_time: datetime) -> Optional[StandardizedZone]:
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
    
    def get_zone_stats(self) -> Dict:
        """Get statistics about current zones."""
        stats = {}
        for symbol in self.symbols:
            active_zones = [zone for zone in self.zones[symbol] if zone.is_active]
            primary_zones = [zone for zone in active_zones if zone.priority == 'primary']
            secondary_zones = [zone for zone in active_zones if zone.priority == 'secondary']
            
            stats[symbol] = {
                'total_zones': len(self.zones[symbol]),
                'active_zones': len(active_zones),
                'primary_zones': len(primary_zones),
                'secondary_zones': len(secondary_zones),
                'avg_confluence': np.mean([zone.confluence_score for zone in active_zones]) if active_zones else 0,
                'avg_touches': np.mean([zone.touch_count for zone in active_zones]) if active_zones else 0
            }
        return stats


def test_standardized_zone_lifecycle():
    """Test the standardized zone lifecycle implementation."""
    
    print("🚀 TESTING STANDARDIZED ZONE LIFECYCLE")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
        print(f"✅ Loaded {len(entry_points)} first-touch entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create standardized zone manager
    symbols = entry_points['symbol'].unique().tolist()
    zone_manager = StandardizedZoneManager(symbols)
    
    print(f"🔧 Created standardized zone manager for symbols: {symbols}")
    print(f"   Max zones per symbol: {zone_manager.max_zones_per_symbol}")
    print(f"   Primary zones per symbol: {zone_manager.primary_zones_per_symbol}")
    print(f"   Secondary zones per symbol: {zone_manager.secondary_zones_per_symbol}")
    print(f"   Zone width: {zone_manager.zone_width_percentage*100:.1f}%")
    print(f"   Min confluence score: {zone_manager.min_confluence_score}")
    
    # Create adequate bars data for confluence calculation
    bars_data = {}
    for symbol in entry_points['symbol'].unique():
        symbol_entries = entry_points[entry_points['symbol'] == symbol].sort_values('bar_index')
        max_bar_index = symbol_entries['bar_index'].max()
        
        bars = []
        base_price = 100.0
        total_bars = max_bar_index + 100
        
        for i in range(total_bars):
            trend = i * 0.001
            volatility = np.random.normal(0, 0.2)
            price = base_price + trend + volatility
            
            bar = type('Bar', (), {
                'open': price - 0.05,
                'high': price + 0.1,
                'low': price - 0.1,
                'close': price,
                'volume': 1000 + np.random.randint(-100, 100),
                'timestamp': datetime.now() + timedelta(minutes=i)
            })()
            bars.append(bar)
        
        bars_data[symbol] = bars
    
    # Process entries and create zones
    processed = 0
    zones_created = 0
    zones_matched = 0
    zones_rejected = 0
    
    for _, entry in entry_points.iterrows():
        symbol = entry['symbol']
        zone_type = entry['zone_type']
        level = entry['zone_level']
        price = entry['price']
        current_time = entry['timestamp']
        bar_index = entry['bar_index']
        
        processed += 1
        
        # Get bars for confluence calculation
        bars = bars_data.get(symbol, [])
        if bar_index >= len(bars):
            continue
        
        # Try to create a new zone
        zone = zone_manager.create_zone(symbol, zone_type, level, current_time, bars, bar_index)
        
        if zone:
            # Try to add zone (respects quantity limits)
            if zone_manager.add_zone(zone):
                zones_created += 1
                print(f"   Created {zone.priority} zone: {zone.zone_id} (confluence: {zone.confluence_score:.2f})")
            else:
                zones_rejected += 1
                print(f"   Rejected zone: {zone.zone_id} (quantity limit reached)")
        
        # Try to find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, price, current_time)
        if matching_zone:
            zones_matched += 1
            is_first_touch = matching_zone.is_first_touch(current_time)
            print(f"   Matched zone: {matching_zone.zone_id} (first touch: {is_first_touch}, confluence: {matching_zone.confluence_score:.2f})")
    
    # Get zone statistics
    stats = zone_manager.get_zone_stats()
    
    print(f"\n📊 STANDARDIZED ZONE LIFECYCLE RESULTS:")
    print(f"   Processed Entries: {processed}")
    print(f"   Zones Created: {zones_created}")
    print(f"   Zones Rejected: {zones_rejected}")
    print(f"   Zones Matched: {zones_matched}")
    print(f"   Match Rate: {zones_matched/processed*100:.1f}%")
    
    print(f"\n📈 ZONE STATISTICS BY SYMBOL:")
    for symbol, stat in stats.items():
        print(f"   {symbol}:")
        print(f"     Total Zones: {stat['total_zones']}")
        print(f"     Active Zones: {stat['active_zones']}")
        print(f"     Primary Zones: {stat['primary_zones']}")
        print(f"     Secondary Zones: {stat['secondary_zones']}")
        print(f"     Avg Confluence: {stat['avg_confluence']:.2f}")
        print(f"     Avg Touches: {stat['avg_touches']:.1f}")
    
    return zone_manager


def run_standardized_framework_backtest(zone_manager: StandardizedZoneManager):
    """Run backtest with standardized zone lifecycle."""
    
    print(f"\n🚀 RUNNING STANDARDIZED FRAMEWORK BACKTEST")
    print("=" * 60)
    
    # Load first touch data
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    entry_points = pd.read_csv(csv_file)
    entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
    
    # Create adequate bars data
    bars_data = {}
    for symbol in entry_points['symbol'].unique():
        symbol_entries = entry_points[entry_points['symbol'] == symbol].sort_values('bar_index')
        max_bar_index = symbol_entries['bar_index'].max()
        
        bars = []
        base_price = 100.0
        total_bars = max_bar_index + 100
        
        for i in range(total_bars):
            trend = i * 0.001
            volatility = np.random.normal(0, 0.2)
            price = base_price + trend + volatility
            
            bar = type('Bar', (), {
                'open': price - 0.05,
                'high': price + 0.1,
                'low': price - 0.1,
                'close': price,
                'volume': 1000 + np.random.randint(-100, 100),
                'timestamp': datetime.now() + timedelta(minutes=i)
            })()
            bars.append(bar)
        
        bars_data[symbol] = bars
    
    # Create unified framework
    framework = create_unified_framework(
        initial_balance=10000.0,
        max_equity_per_trade=0.10,
        slippage_ticks=2,
        commission_per_trade=5.0,
        min_confidence=0.5
    )
    
    print(f"🔄 Processing entries with standardized zones...")
    
    # Process entries
    processed = 0
    executed = 0
    rejected = 0
    rejection_reasons = {}
    
    for _, entry in entry_points.iterrows():
        symbol = entry['symbol']
        if symbol not in bars_data:
            continue
        
        bars = bars_data[symbol]
        bar_index = entry['bar_index']
        
        if bar_index >= len(bars):
            continue
        
        # Find matching zone
        matching_zone = zone_manager.find_matching_zone(symbol, entry['price'], entry['timestamp'])
        
        if not matching_zone:
            rejected += 1
            rejection_reasons['No matching zone'] = rejection_reasons.get('No matching zone', 0) + 1
            continue
        
        # Check first touch (BattleCard principle)
        is_first_touch = matching_zone.is_first_touch(entry['timestamp'])
        
        if not is_first_touch:
            rejected += 1
            rejection_reasons['Not first touch of zone'] = rejection_reasons.get('Not first touch of zone', 0) + 1
            continue
        
        # Prepare zone data for framework
        zone_data = {
            'zone_type': matching_zone.zone_type,
            'zone_level': matching_zone.get_zone_center(),
            'zone_high': matching_zone.high_level,
            'zone_low': matching_zone.low_level,
            'zone_width': matching_zone.get_zone_width(),
            'priority': matching_zone.priority,
            'confluence_score': matching_zone.confluence_score
        }
        
        # Prepare entry data
        entry_data = {
            'entry_id': entry['entry_id'],
            'symbol': entry['symbol'],
            'timestamp': entry['timestamp'],
            'price': entry['price'],
            'hard_stop': entry['hard_stop'],
            't1_price': entry['t1_price'],
            't2_price': entry['t2_price'],
            't3_price': entry['t3_price']
        }
        
        try:
            # Evaluate trade opportunity
            trade_decision, choch_signal = framework.evaluate_trade_opportunity(
                bars, bar_index, zone_data, entry['price']
            )
            
            processed += 1
            
            # Track rejection reasons
            reason = trade_decision.reason
            if reason not in rejection_reasons:
                rejection_reasons[reason] = 0
            rejection_reasons[reason] += 1
            
            print(f"\n   Entry {processed}: {entry['entry_id']}")
            print(f"   Zone: {matching_zone.zone_type} ({matching_zone.priority}) range {matching_zone.low_level:.2f}-{matching_zone.high_level:.2f}")
            print(f"   Price: {entry['price']} (in zone: {matching_zone.contains_price(entry['price'])})")
            print(f"   Confluence: {matching_zone.confluence_score:.2f}")
            print(f"   Decision: {trade_decision.trade_type.value}")
            print(f"   Reason: {trade_decision.reason}")
            print(f"   Confidence: {trade_decision.confidence:.2f}")
            print(f"   Is First Touch: {is_first_touch}")
            print(f"   Should Execute: {framework.should_execute_trade(trade_decision)}")
            
            # Check if trade should be executed
            if framework.should_execute_trade(trade_decision):
                # Execute trade
                trade_execution = framework.execute_trade(
                    trade_decision, choch_signal, entry_data, bars, bar_index, simulation_mode=True
                )
                executed += 1
                print(f"   ✅ EXECUTED: P&L = ${trade_execution.pnl:.2f}")
            else:
                rejected += 1
                print(f"   ❌ REJECTED: {trade_decision.reason}")
        
        except Exception as e:
            rejected += 1
            error_reason = f"Error: {str(e)[:50]}..."
            if error_reason not in rejection_reasons:
                rejection_reasons[error_reason] = 0
            rejection_reasons[error_reason] += 1
            print(f"   ⚠️  ERROR: {e}")
            continue
    
    # Calculate results
    execution_rate = executed / processed * 100 if processed > 0 else 0
    metrics = framework.get_performance_metrics()
    
    print(f"\n📊 STANDARDIZED FRAMEWORK RESULTS:")
    print(f"   Processed Entries: {processed}")
    print(f"   Executed Trades: {executed}")
    print(f"   Rejected Trades: {rejected}")
    print(f"   Execution Rate: {execution_rate:.1f}%")
    
    # Print performance metrics
    print(f"\n💰 PERFORMANCE METRICS:")
    print(f"   Final Balance: ${metrics['final_balance']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.2f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Max Drawdown: ${metrics['max_drawdown']:,.2f}")
    
    # Print framework compliance
    if metrics['total_trades'] > 0:
        print(f"\n🎯 FRAMEWORK COMPLIANCE:")
        framework_compliance = metrics['framework_compliant_trades'] / metrics['total_trades'] * 100
        print(f"   Framework Compliant: {metrics['framework_compliant_trades']}/{metrics['total_trades']} ({framework_compliance:.1f}%)")
        
        if metrics['choch_required_trades'] > 0:
            choch_compliance = metrics['choch_aligned_trades'] / metrics['choch_required_trades'] * 100
            print(f"   CHoCH Compliance: {metrics['choch_aligned_trades']}/{metrics['choch_required_trades']} ({choch_compliance:.1f}%)")
        
        print(f"\n📊 TRADE BREAKDOWN:")
        print(f"   Fade Trades: {metrics['fade_trades']} ({metrics['fade_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Continuation Trades: {metrics['continuation_trades']} ({metrics['continuation_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Bullish Bias: {metrics['bullish_trades']} ({metrics['bullish_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Bearish Bias: {metrics['bearish_trades']} ({metrics['bearish_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Trend Day: {metrics['trend_day_trades']} ({metrics['trend_day_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Balanced Day: {metrics['balanced_day_trades']} ({metrics['balanced_day_trades']/metrics['total_trades']*100:.1f}%)")
    
    # Print rejection analysis
    print(f"\n❌ REJECTION ANALYSIS:")
    print(f"   Top rejection reasons:")
    sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        percentage = count / processed * 100 if processed > 0 else 0
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Strategy assessment
    print(f"\n🔍 STRATEGY ASSESSMENT:")
    if metrics['total_trades'] == 0:
        print("   ❌ NO TRADES: Strategy is too restrictive - check rejection reasons")
    elif metrics['total_return'] > 20:
        print("   ✅ EXCELLENT: Strategy shows strong profitability")
    elif metrics['total_return'] > 10:
        print("   ✅ GOOD: Strategy shows good profitability")
    elif metrics['total_return'] > 0:
        print("   ⚠️  MODERATE: Strategy shows modest profitability")
    else:
        print("   ❌ POOR: Strategy shows negative returns")
    
    if metrics['total_trades'] > 0:
        if metrics['win_rate'] > 60:
            print("   ✅ HIGH WIN RATE: Strategy shows consistent winning")
        elif metrics['win_rate'] > 50:
            print("   ⚠️  MODERATE WIN RATE: Strategy shows mixed results")
        else:
            print("   ❌ LOW WIN RATE: Strategy shows poor win rate")
        
        if metrics['max_drawdown'] > -1000:
            print("   ✅ LOW RISK: Strategy shows controlled drawdowns")
        elif metrics['max_drawdown'] > -5000:
            print("   ⚠️  MODERATE RISK: Strategy shows acceptable drawdowns")
        else:
            print("   ❌ HIGH RISK: Strategy shows concerning drawdowns")
    
    print("\n" + "=" * 60)
    
    return {
        'metrics': metrics,
        'processed': processed,
        'executed': executed,
        'rejected': rejected,
        'execution_rate': execution_rate,
        'rejection_reasons': rejection_reasons
    }


if __name__ == "__main__":
    # Test standardized zone lifecycle
    zone_manager = test_standardized_zone_lifecycle()
    
    # Run standardized framework backtest
    results = run_standardized_framework_backtest(zone_manager)