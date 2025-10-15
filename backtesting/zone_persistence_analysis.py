#!/usr/bin/env python3
"""
Zone Persistence Analysis

Analyze current zone detection and propose persistence strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def analyze_zone_persistence():
    """Analyze current zone detection and persistence patterns."""
    
    print("🔍 ZONE PERSISTENCE ANALYSIS")
    print("=" * 60)
    
    # Load the full dataset to understand zone patterns
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Loaded {len(df)} total entry points")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    # Create zone_id from zone_type and zone_level
    df['zone_id'] = df['zone_type'] + '_' + df['zone_level'].round(2).astype(str)
    
    # Analyze zone distribution
    print(f"\n📊 ZONE DISTRIBUTION ANALYSIS:")
    print(f"   Total Entries: {len(df)}")
    print(f"   Unique Zones: {df['zone_id'].nunique()}")
    print(f"   Zone Types: {df['zone_type'].nunique()}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    
    # Analyze touches per zone
    zone_touches = df.groupby('zone_id').agg({
        'entry_id': 'count',
        'timestamp': ['min', 'max'],
        'zone_level': 'first',
        'zone_type': 'first',
        'symbol': 'first'
    }).round(2)
    
    zone_touches.columns = ['touch_count', 'first_touch', 'last_touch', 'zone_level', 'zone_type', 'symbol']
    zone_touches['duration_hours'] = (zone_touches['last_touch'] - zone_touches['first_touch']).dt.total_seconds() / 3600
    
    print(f"\n📈 ZONE TOUCH STATISTICS:")
    print(f"   Average Touches per Zone: {zone_touches['touch_count'].mean():.1f}")
    print(f"   Median Touches per Zone: {zone_touches['touch_count'].median():.1f}")
    print(f"   Max Touches per Zone: {zone_touches['touch_count'].max()}")
    print(f"   Min Touches per Zone: {zone_touches['touch_count'].min()}")
    
    print(f"\n⏰ ZONE DURATION STATISTICS:")
    print(f"   Average Duration: {zone_touches['duration_hours'].mean():.1f} hours")
    print(f"   Median Duration: {zone_touches['duration_hours'].median():.1f} hours")
    print(f"   Max Duration: {zone_touches['duration_hours'].max():.1f} hours")
    print(f"   Min Duration: {zone_touches['duration_hours'].min():.1f} hours")
    
    # Analyze by zone type
    print(f"\n🏷️  ZONE TYPE ANALYSIS:")
    for zone_type in df['zone_type'].unique():
        type_data = zone_touches[zone_touches['zone_type'] == zone_type]
        print(f"   {zone_type}:")
        print(f"     Count: {len(type_data)} zones")
        print(f"     Avg Touches: {type_data['touch_count'].mean():.1f}")
        print(f"     Avg Duration: {type_data['duration_hours'].mean():.1f} hours")
    
    # Analyze by symbol
    print(f"\n📈 SYMBOL ANALYSIS:")
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        print(f"   {symbol}:")
        print(f"     Total Entries: {len(symbol_data)}")
        print(f"     Unique Zones: {symbol_data['zone_id'].nunique()}")
        print(f"     Avg Touches per Zone: {len(symbol_data) / symbol_data['zone_id'].nunique():.1f}")
    
    # Show top zones by touch count
    print(f"\n🔥 TOP ZONES BY TOUCH COUNT:")
    top_zones = zone_touches.nlargest(10, 'touch_count')
    for i, (zone_id, row) in enumerate(top_zones.iterrows(), 1):
        print(f"   {i:2d}. {zone_id}")
        print(f"       Touches: {row['touch_count']}, Duration: {row['duration_hours']:.1f}h")
        print(f"       Symbol: {row['symbol']}, Type: {row['zone_type']}")
        print(f"       Level: {row['zone_level']:.2f}")
    
    return zone_touches


def propose_zone_persistence_strategies(zone_touches):
    """Propose different zone persistence strategies."""
    
    print(f"\n💡 PROPOSED ZONE PERSISTENCE STRATEGIES")
    print("=" * 60)
    
    # Strategy 1: Time-based persistence
    print(f"🕐 STRATEGY 1: TIME-BASED PERSISTENCE")
    print(f"   - Zones expire after X hours")
    print(f"   - Current average duration: {zone_touches['duration_hours'].mean():.1f} hours")
    print(f"   - Suggested: 4-8 hours (intraday only)")
    print(f"   - Pros: Simple, prevents stale zones")
    print(f"   - Cons: May miss valid setups")
    
    # Strategy 2: Touch-based persistence
    print(f"\n👆 STRATEGY 2: TOUCH-BASED PERSISTENCE")
    print(f"   - Zones expire after X touches")
    print(f"   - Current average touches: {zone_touches['touch_count'].mean():.1f}")
    print(f"   - Suggested: 3-5 touches")
    print(f"   - Pros: Based on market interaction")
    print(f"   - Cons: May be too restrictive")
    
    # Strategy 3: Session-based persistence
    print(f"\n📅 STRATEGY 3: SESSION-BASED PERSISTENCE")
    print(f"   - Zones expire at end of trading session")
    print(f"   - New zones created each day")
    print(f"   - Pros: Fresh zones daily, no stale data")
    print(f"   - Cons: May miss multi-day setups")
    
    # Strategy 4: Quality-based persistence
    print(f"\n⭐ STRATEGY 4: QUALITY-BASED PERSISTENCE")
    print(f"   - Zones expire based on quality score")
    print(f"   - High-quality zones persist longer")
    print(f"   - Pros: Adaptive to market conditions")
    print(f"   - Cons: Complex to implement")
    
    # Strategy 5: Hybrid persistence
    print(f"\n🔄 STRATEGY 5: HYBRID PERSISTENCE")
    print(f"   - Combine time + touches + quality")
    print(f"   - Zones expire when ANY condition met")
    print(f"   - Pros: Most flexible and robust")
    print(f"   - Cons: Most complex to implement")


def propose_zone_quantity_strategies(zone_touches):
    """Propose different zone quantity strategies."""
    
    print(f"\n📊 PROPOSED ZONE QUANTITY STRATEGIES")
    print("=" * 60)
    
    # Current state
    total_zones = len(zone_touches)
    avg_touches = zone_touches['touch_count'].mean()
    
    print(f"📈 CURRENT STATE:")
    print(f"   Total Zones: {total_zones}")
    print(f"   Average Touches: {avg_touches:.1f}")
    print(f"   Zones per Symbol: {total_zones / 3:.1f}")  # 3 symbols
    
    # Strategy 1: More zones, shorter persistence
    print(f"\n🔢 STRATEGY 1: MORE ZONES, SHORTER PERSISTENCE")
    print(f"   - Create zones every 0.1% price movement")
    print(f"   - Expire after 2-4 hours")
    print(f"   - Expected: 50-100 zones per symbol per day")
    print(f"   - Pros: More opportunities, fresh zones")
    print(f"   - Cons: More noise, higher computational cost")
    
    # Strategy 2: Fewer zones, longer persistence
    print(f"\n🎯 STRATEGY 2: FEWER ZONES, LONGER PERSISTENCE")
    print(f"   - Create zones only at significant levels")
    print(f"   - Expire after 1-2 days")
    print(f"   - Expected: 5-10 zones per symbol per day")
    print(f"   - Pros: Higher quality, less noise")
    print(f"   - Cons: Fewer opportunities, may miss setups")
    
    # Strategy 3: Adaptive zones
    print(f"\n🧠 STRATEGY 3: ADAPTIVE ZONES")
    print(f"   - Adjust zone density based on volatility")
    print(f"   - High volatility = more zones")
    print(f"   - Low volatility = fewer zones")
    print(f"   - Pros: Adapts to market conditions")
    print(f"   - Cons: Complex to implement")
    
    # Strategy 4: Multi-timeframe zones
    print(f"\n⏰ STRATEGY 4: MULTI-TIMEFRAME ZONES")
    print(f"   - Different persistence for different timeframes")
    print(f"   - 1min zones: 1 hour")
    print(f"   - 5min zones: 4 hours")
    print(f"   - 15min zones: 1 day")
    print(f"   - Pros: Captures different market rhythms")
    print(f"   - Cons: Complex to manage")


def recommend_implementation():
    """Recommend specific implementation strategy."""
    
    print(f"\n🎯 RECOMMENDED IMPLEMENTATION")
    print("=" * 60)
    
    print(f"📋 RECOMMENDED STRATEGY:")
    print(f"   Zone Persistence: 4-6 hours (intraday only)")
    print(f"   Zone Quantity: 10-20 zones per symbol per day")
    print(f"   Zone Width: 0.2-0.5% of price")
    print(f"   First Touch Rule: Only first touch per zone per session")
    
    print(f"\n🔧 IMPLEMENTATION STEPS:")
    print(f"   1. Modify zone detection to create more zones")
    print(f"   2. Add time-based expiration (4-6 hours)")
    print(f"   3. Implement session-based zone reset")
    print(f"   4. Add zone quality scoring")
    print(f"   5. Test with different persistence parameters")
    
    print(f"\n📊 EXPECTED RESULTS:")
    print(f"   - 30-60 zones per symbol per day")
    print(f"   - 90-180 total zones per day")
    print(f"   - 1-3 trades per zone (first touch only)")
    print(f"   - 100-300 trades per day total")
    print(f"   - Much higher than current 0 trades")


if __name__ == "__main__":
    # Analyze current zone persistence
    zone_touches = analyze_zone_persistence()
    
    # Propose strategies
    propose_zone_persistence_strategies(zone_touches)
    propose_zone_quantity_strategies(zone_touches)
    recommend_implementation()