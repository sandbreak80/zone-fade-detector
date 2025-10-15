#!/usr/bin/env python3
"""
Zone Analysis

Analyze the zone data to understand the first touch issue.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


def analyze_zone_touches(csv_file_path: str):
    """Analyze zone touches to understand the first touch issue."""
    
    print("🔍 ZONE TOUCH ANALYSIS")
    print("=" * 50)
    
    # Load entry points
    df = pd.read_csv(csv_file_path)
    print(f"✅ Loaded {len(df)} entry points")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create zone_id
    df['zone_id'] = df['zone_type'] + '_' + df['zone_level'].round(2).astype(str)
    
    print(f"\n📊 ZONE STATISTICS:")
    print(f"   Total Entries: {len(df)}")
    print(f"   Unique Zones: {df['zone_id'].nunique()}")
    print(f"   Zone Types: {df['zone_type'].nunique()}")
    print(f"   Unique Levels: {df['zone_level'].nunique()}")
    
    # Analyze touches per zone
    zone_touches = df.groupby('zone_id').agg({
        'entry_id': 'count',
        'timestamp': ['min', 'max'],
        'symbol': 'nunique'
    }).round(2)
    
    zone_touches.columns = ['touch_count', 'first_touch', 'last_touch', 'symbols']
    zone_touches['duration_hours'] = (zone_touches['last_touch'] - zone_touches['first_touch']).dt.total_seconds() / 3600
    
    print(f"\n📈 TOUCHES PER ZONE:")
    print(f"   Average touches per zone: {zone_touches['touch_count'].mean():.1f}")
    print(f"   Max touches per zone: {zone_touches['touch_count'].max()}")
    print(f"   Min touches per zone: {zone_touches['touch_count'].min()}")
    print(f"   Average duration: {zone_touches['duration_hours'].mean():.1f} hours")
    
    # Show zones with most touches
    print(f"\n🔥 ZONES WITH MOST TOUCHES:")
    top_zones = zone_touches.sort_values('touch_count', ascending=False).head(10)
    for zone_id, row in top_zones.iterrows():
        print(f"   {zone_id}: {row['touch_count']} touches over {row['duration_hours']:.1f}h")
    
    # Analyze by symbol
    print(f"\n📊 TOUCHES BY SYMBOL:")
    symbol_touches = df.groupby('symbol').agg({
        'entry_id': 'count',
        'zone_id': 'nunique'
    })
    symbol_touches.columns = ['total_touches', 'unique_zones']
    symbol_touches['avg_touches_per_zone'] = symbol_touches['total_touches'] / symbol_touches['unique_zones']
    
    for symbol, row in symbol_touches.iterrows():
        print(f"   {symbol}: {row['total_touches']} touches, {row['unique_zones']} zones, {row['avg_touches_per_zone']:.1f} avg per zone")
    
    # Find first touches only
    print(f"\n🎯 FIRST TOUCH ANALYSIS:")
    first_touches = df.sort_values(['zone_id', 'timestamp']).groupby('zone_id').first().reset_index()
    print(f"   First touches only: {len(first_touches)}")
    print(f"   Reduction: {len(df) - len(first_touches)} touches removed ({((len(df) - len(first_touches)) / len(df) * 100):.1f}%)")
    
    # Show first touch distribution by symbol
    first_touch_by_symbol = first_touches.groupby('symbol').size()
    print(f"\n   First touches by symbol:")
    for symbol, count in first_touch_by_symbol.items():
        print(f"     {symbol}: {count}")
    
    # Analyze time gaps between touches
    print(f"\n⏰ TIME GAP ANALYSIS:")
    df_sorted = df.sort_values(['zone_id', 'timestamp'])
    df_sorted['prev_timestamp'] = df_sorted.groupby('zone_id')['timestamp'].shift(1)
    df_sorted['time_gap_minutes'] = (df_sorted['timestamp'] - df_sorted['prev_timestamp']).dt.total_seconds() / 60
    
    gaps = df_sorted[df_sorted['time_gap_minutes'].notna()]['time_gap_minutes']
    print(f"   Average gap between touches: {gaps.mean():.1f} minutes")
    print(f"   Median gap: {gaps.median():.1f} minutes")
    print(f"   Min gap: {gaps.min():.1f} minutes")
    print(f"   Max gap: {gaps.max():.1f} minutes")
    
    # Show gaps < 1 hour (current threshold)
    gaps_under_hour = gaps[gaps < 60]
    print(f"   Gaps under 1 hour: {len(gaps_under_hour)} ({len(gaps_under_hour)/len(gaps)*100:.1f}%)")
    
    return first_touches, zone_touches


def create_first_touch_dataset(original_df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataset with only first touches of each zone."""
    
    # Sort by zone and timestamp
    df_sorted = original_df.sort_values(['zone_id', 'timestamp'])
    
    # Get first touch of each zone
    first_touches = df_sorted.groupby('zone_id').first().reset_index()
    
    # Add touch sequence info
    df_sorted['touch_sequence'] = df_sorted.groupby('zone_id').cumcount() + 1
    first_touches = df_sorted[df_sorted['touch_sequence'] == 1].copy()
    
    print(f"\n✅ CREATED FIRST TOUCH DATASET:")
    print(f"   Original entries: {len(original_df)}")
    print(f"   First touches: {len(first_touches)}")
    print(f"   Reduction: {len(original_df) - len(first_touches)} ({((len(original_df) - len(first_touches)) / len(original_df) * 100):.1f}%)")
    
    return first_touches


if __name__ == "__main__":
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    
    # Analyze zones
    first_touches, zone_touches = analyze_zone_touches(csv_file)
    
    # Create first touch dataset
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['zone_id'] = df['zone_type'] + '_' + df['zone_level'].round(2).astype(str)
    
    first_touch_df = create_first_touch_dataset(df)
    
    # Save first touch dataset
    output_file = "/app/results/2024/corrected/zone_fade_entry_points_first_touches_only.csv"
    first_touch_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved first touch dataset to: {output_file}")