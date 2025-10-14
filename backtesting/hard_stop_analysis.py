#!/usr/bin/env python3
"""
Hard Stop Analysis

This script analyzes why 70.6% of trades are hitting hard stops
to identify patterns and improve zone quality.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_5year_data():
    """Load the 5-year data for analysis."""
    print("📊 Loading 5-year data for hard stop analysis...")
    
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

def analyze_hard_stop_patterns():
    """Analyze patterns in hard stop hits."""
    
    print("🔍 HARD STOP ANALYSIS")
    print("=" * 50)
    
    # Load data
    bars_data = load_5year_data()
    
    if not bars_data:
        print("❌ No data available for analysis")
        return
    
    # Load entry points
    csv_file = "/app/results/2024/corrected/zone_fade_entry_points_2024_corrected.csv"
    try:
        entry_points = pd.read_csv(csv_file)
        entry_points['timestamp'] = pd.to_datetime(entry_points['timestamp'])
        print(f"✅ Loaded {len(entry_points)} entry points")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Filter for available symbols
    available_symbols = set(bars_data.keys())
    entry_points = entry_points[entry_points['symbol'].isin(available_symbols)]
    print(f"📊 Analyzing {len(entry_points)} entries for symbols: {', '.join(available_symbols)}")
    
    # Analysis results
    hard_stop_analysis = {
        'total_entries': len(entry_points),
        'hard_stops': 0,
        'successful_trades': 0,
        'zone_breakdown': {},
        'time_analysis': {},
        'price_analysis': {},
        'qrs_analysis': {}
    }
    
    # Process each entry
    for _, entry in entry_points.iterrows():
        symbol = entry['symbol']
        zone_type = entry['zone_type']
        level = entry['zone_level']
        price = entry['price']
        hard_stop = entry['hard_stop']
        qrs_score = entry.get('qrs_score', 7.0)
        current_time = entry['timestamp']
        
        # Find corresponding bar
        entry_index = None
        if symbol in bars_data:
            for j, bar in enumerate(bars_data[symbol]):
                if abs((bar.timestamp - current_time).total_seconds()) < 60:
                    entry_index = j
                    break
        
        if entry_index is None:
            continue
        
        # Simulate trade to see if it hits hard stop
        direction = 'LONG' if zone_type in ['prior_day_low', 'value_area_low'] else 'SHORT'
        hit_hard_stop = False
        bars_until_stop = 0
        
        # Look ahead up to 100 bars
        max_lookahead = min(100, len(bars_data[symbol]) - entry_index - 1)
        
        for i in range(1, max_lookahead + 1):
            if entry_index + i >= len(bars_data[symbol]):
                break
                
            current_bar = bars_data[symbol][entry_index + i]
            current_high = current_bar.high
            current_low = current_bar.low
            
            # Check hard stop
            if direction == 'SHORT':
                if current_high >= hard_stop:
                    hit_hard_stop = True
                    bars_until_stop = i
                    break
            else:  # LONG
                if current_low <= hard_stop:
                    hit_hard_stop = True
                    bars_until_stop = i
                    break
        
        # Record analysis
        if hit_hard_stop:
            hard_stop_analysis['hard_stops'] += 1
        else:
            hard_stop_analysis['successful_trades'] += 1
        
        # Zone type breakdown
        if zone_type not in hard_stop_analysis['zone_breakdown']:
            hard_stop_analysis['zone_breakdown'][zone_type] = {'total': 0, 'hard_stops': 0}
        hard_stop_analysis['zone_breakdown'][zone_type]['total'] += 1
        if hit_hard_stop:
            hard_stop_analysis['zone_breakdown'][zone_type]['hard_stops'] += 1
        
        # Time analysis (hour of day)
        hour = current_time.hour
        if hour not in hard_stop_analysis['time_analysis']:
            hard_stop_analysis['time_analysis'][hour] = {'total': 0, 'hard_stops': 0}
        hard_stop_analysis['time_analysis'][hour]['total'] += 1
        if hit_hard_stop:
            hard_stop_analysis['time_analysis'][hour]['hard_stops'] += 1
        
        # Price analysis (distance to hard stop)
        if direction == 'SHORT':
            distance_to_stop = hard_stop - price
        else:
            distance_to_stop = price - hard_stop
        
        price_range = f"{distance_to_stop:.2f}"
        if price_range not in hard_stop_analysis['price_analysis']:
            hard_stop_analysis['price_analysis'][price_range] = {'total': 0, 'hard_stops': 0}
        hard_stop_analysis['price_analysis'][price_range]['total'] += 1
        if hit_hard_stop:
            hard_stop_analysis['price_analysis'][price_range]['hard_stops'] += 1
        
        # QRS analysis
        qrs_range = f"{int(qrs_score)}"
        if qrs_range not in hard_stop_analysis['qrs_analysis']:
            hard_stop_analysis['qrs_analysis'][qrs_range] = {'total': 0, 'hard_stops': 0}
        hard_stop_analysis['qrs_analysis'][qrs_range]['total'] += 1
        if hit_hard_stop:
            hard_stop_analysis['qrs_analysis'][qrs_range]['hard_stops'] += 1
    
    # Print analysis results
    print(f"\n📊 HARD STOP ANALYSIS RESULTS:")
    print(f"   Total Entries Analyzed: {hard_stop_analysis['total_entries']}")
    print(f"   Hard Stops: {hard_stop_analysis['hard_stops']}")
    print(f"   Successful Trades: {hard_stop_analysis['successful_trades']}")
    print(f"   Hard Stop Rate: {hard_stop_analysis['hard_stops'] / hard_stop_analysis['total_entries'] * 100:.1f}%")
    
    print(f"\n🎯 ZONE TYPE BREAKDOWN:")
    for zone_type, stats in hard_stop_analysis['zone_breakdown'].items():
        if stats['total'] > 0:
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            print(f"   {zone_type}: {stats['hard_stops']}/{stats['total']} ({hard_stop_rate:.1f}%)")
    
    print(f"\n⏰ TIME ANALYSIS (Hour of Day):")
    for hour in sorted(hard_stop_analysis['time_analysis'].keys()):
        stats = hard_stop_analysis['time_analysis'][hour]
        if stats['total'] > 0:
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            print(f"   {hour:02d}:00: {stats['hard_stops']}/{stats['total']} ({hard_stop_rate:.1f}%)")
    
    print(f"\n💰 PRICE ANALYSIS (Distance to Hard Stop):")
    for price_range in sorted(hard_stop_analysis['price_analysis'].keys(), key=float):
        stats = hard_stop_analysis['price_analysis'][price_range]
        if stats['total'] > 0:
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            print(f"   ${price_range}: {stats['hard_stops']}/{stats['total']} ({hard_stop_rate:.1f}%)")
    
    print(f"\n⭐ QRS SCORE ANALYSIS:")
    for qrs_range in sorted(hard_stop_analysis['qrs_analysis'].keys(), key=int):
        stats = hard_stop_analysis['qrs_analysis'][qrs_range]
        if stats['total'] > 0:
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            print(f"   QRS {qrs_range}: {stats['hard_stops']}/{stats['total']} ({hard_stop_rate:.1f}%)")
    
    # Identify patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    
    # Find worst performing zone types
    worst_zones = []
    for zone_type, stats in hard_stop_analysis['zone_breakdown'].items():
        if stats['total'] >= 5:  # Minimum sample size
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            worst_zones.append((zone_type, hard_stop_rate, stats['total']))
    
    worst_zones.sort(key=lambda x: x[1], reverse=True)
    print(f"   Worst Zone Types (by hard stop rate):")
    for zone_type, rate, count in worst_zones[:3]:
        print(f"     {zone_type}: {rate:.1f}% ({count} trades)")
    
    # Find worst time periods
    worst_times = []
    for hour, stats in hard_stop_analysis['time_analysis'].items():
        if stats['total'] >= 3:  # Minimum sample size
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            worst_times.append((hour, hard_stop_rate, stats['total']))
    
    worst_times.sort(key=lambda x: x[1], reverse=True)
    print(f"   Worst Time Periods (by hard stop rate):")
    for hour, rate, count in worst_times[:3]:
        print(f"     {hour:02d}:00: {rate:.1f}% ({count} trades)")
    
    # Find QRS patterns
    qrs_rates = []
    for qrs_range, stats in hard_stop_analysis['qrs_analysis'].items():
        if stats['total'] >= 3:  # Minimum sample size
            hard_stop_rate = stats['hard_stops'] / stats['total'] * 100
            qrs_rates.append((int(qrs_range), hard_stop_rate, stats['total']))
    
    qrs_rates.sort(key=lambda x: x[0])
    print(f"   QRS Score vs Hard Stop Rate:")
    for qrs, rate, count in qrs_rates:
        print(f"     QRS {qrs}: {rate:.1f}% ({count} trades)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if worst_zones:
        worst_zone = worst_zones[0]
        print(f"   1. Avoid {worst_zone[0]} zones (hard stop rate: {worst_zone[1]:.1f}%)")
    
    if worst_times:
        worst_time = worst_times[0]
        print(f"   2. Avoid trading at {worst_time[0]:02d}:00 (hard stop rate: {worst_time[1]:.1f}%)")
    
    # Check if QRS is working
    if len(qrs_rates) >= 2:
        high_qrs = [rate for qrs, rate, count in qrs_rates if qrs >= 8]
        low_qrs = [rate for qrs, rate, count in qrs_rates if qrs <= 6]
        if high_qrs and low_qrs:
            avg_high_qrs = np.mean(high_qrs)
            avg_low_qrs = np.mean(low_qrs)
            if avg_high_qrs < avg_low_qrs:
                print(f"   3. QRS scoring is working - higher QRS = lower hard stop rate")
            else:
                print(f"   3. QRS scoring needs improvement - no clear correlation with hard stop rate")
    
    print(f"   4. Consider increasing minimum distance to hard stop")
    print(f"   5. Add volume confirmation before entry")
    print(f"   6. Implement dynamic stop placement based on volatility")
    
    return hard_stop_analysis

if __name__ == "__main__":
    analyze_hard_stop_patterns()