#!/usr/bin/env python3
"""
Efficient 2024 Zone Fade Entry Points with Validation and Duration Tracking.

This script runs efficiently on the 2024 dataset with:
- Chunked processing for memory efficiency
- Real-time progress updates
- Entry window duration tracking
- CSV export for manual validation
- Optimized for speed while maintaining accuracy
"""

import asyncio
import sys
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import concurrent.futures
from threading import Lock
import numpy as np
import csv

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType


def load_2024_data_sample(sample_size: int = 50000):
    """Load a sample of 2024 data for efficient processing."""
    print(f"📊 Loading 2024 Data Sample (last {sample_size:,} bars per symbol)...")
    
    data_dir = Path("data/2024")
    symbols_data = {}
    
    # Load individual symbol data
    for symbol in ["SPY", "QQQ", "IWM"]:
        file_path = data_dir / f"{symbol}_2024.pkl"
        if file_path.exists():
            print(f"   Loading {symbol} data...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # Take the last sample_size bars for efficient processing
                symbols_data[symbol] = data[-sample_size:] if len(data) > sample_size else data
                print(f"     ✅ {symbol}: {len(symbols_data[symbol])} bars (sample)")
        else:
            print(f"     ❌ {symbol}: File not found")
    
    return symbols_data


def create_efficient_zones(symbol: str, bars: List[OHLCVBar]) -> List[Zone]:
    """Create efficient zones for validation."""
    zones = []
    
    if not bars:
        return zones
    
    # Use numpy for efficient calculations
    high_prices = np.array([bar.high for bar in bars])
    low_prices = np.array([bar.low for bar in bars])
    
    # Find significant highs and lows
    max_high = np.max(high_prices)
    min_low = np.min(low_prices)
    
    # Create supply zones (resistance levels)
    supply_levels = [
        max_high * 0.98,  # 2% below high
        max_high * 0.95,  # 5% below high
        max_high * 0.90   # 10% below high
    ]
    
    for i, level in enumerate(supply_levels):
        zone = Zone(
            level=level,
            zone_type=ZoneType.PRIOR_DAY_HIGH,
            quality=2,
            strength=1.0 - i * 0.2,
            touches=0,
            created_at=datetime.now(),
            last_touch=None
        )
        zones.append(zone)
    
    # Create demand zones (support levels)
    demand_levels = [
        min_low * 1.02,  # 2% above low
        min_low * 1.05,  # 5% above low
        min_low * 1.10   # 10% above low
    ]
    
    for i, level in enumerate(demand_levels):
        zone = Zone(
            level=level,
            zone_type=ZoneType.PRIOR_DAY_LOW,
            quality=2,
            strength=1.0 - i * 0.2,
            touches=0,
            created_at=datetime.now(),
            last_touch=None
        )
        zones.append(zone)
    
    return zones


def is_zone_touched(bar: OHLCVBar, zone: Zone) -> bool:
    """Check if a bar touches a zone."""
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        return bar.low <= zone.level <= bar.high
    else:
        return bar.low <= zone.level <= bar.high


def is_rejection_candle(bar: OHLCVBar, zone: Zone) -> bool:
    """Check if a bar is a rejection candle with 30% wick ratio."""
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        if bar.high > zone.level and bar.close < zone.level:
            upper_wick = bar.high - max(bar.open, bar.close)
            body_size = abs(bar.close - bar.open)
            if body_size > 0 and upper_wick / body_size > 0.3:
                return True
    else:
        if bar.low < zone.level and bar.close > zone.level:
            lower_wick = min(bar.open, bar.close) - bar.low
            body_size = abs(bar.close - bar.open)
            if body_size > 0 and lower_wick / body_size > 0.3:
                return True
    return False


def calculate_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar]) -> float:
    """Calculate QRS score efficiently."""
    score = 0.0
    
    # Zone strength bonus
    score += zone.strength * 2.0
    
    # Zone quality bonus
    score += zone.quality * 1.0
    
    # Rejection candle bonus
    if is_rejection_candle(bar, zone):
        score += 2.0
    
    # Volume analysis (1.8x threshold)
    if len(bars) >= 20:
        recent_volume = [b.volume for b in bars[-20:]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        if bar.volume > avg_volume * 1.8:
            score += 1.0
    
    # Trend context
    if len(bars) >= 10:
        recent_prices = [b.close for b in bars[-10:]]
        if recent_prices[-1] > recent_prices[0]:  # Uptrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:
                score += 1.0
        else:  # Downtrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW]:
                score += 1.0
    
    return min(score, 10.0)


def track_entry_window(entry_point: Dict, bars: List[OHLCVBar], entry_index: int) -> Dict:
    """Track entry window duration efficiently."""
    entry_price = entry_point["price"]
    zone_level = entry_point["zone_level"]
    zone_type = entry_point["zone_type"]
    
    window_duration_minutes = 0
    window_bars = 0
    max_price_deviation = 0.0
    min_price_deviation = 0.0
    
    # Check up to 30 bars ahead for efficiency
    for i in range(entry_index + 1, min(entry_index + 30, len(bars))):
        bar = bars[i]
        window_bars += 1
        window_duration_minutes += 1
        
        # Calculate price deviation
        price_deviation = (bar.close - entry_price) / entry_price * 100
        max_price_deviation = max(max_price_deviation, price_deviation)
        min_price_deviation = min(min_price_deviation, price_deviation)
        
        # Check if entry window is still valid
        if zone_type in ["prior_day_high", "weekly_high", "value_area_high"]:
            if bar.high > zone_level * 1.01:  # 1% buffer
                break
        else:
            if bar.low < zone_level * 0.99:  # 1% buffer
                break
    
    return {
        "window_duration_minutes": window_duration_minutes,
        "window_bars": window_bars,
        "max_price_deviation": max_price_deviation,
        "min_price_deviation": min_price_deviation,
        "entry_window_ended": window_bars < 30
    }


def process_symbol_efficient(symbol: str, bars: List[OHLCVBar], progress_lock: Lock, progress_counter: Dict[str, int]) -> Dict[str, Any]:
    """Process a single symbol efficiently."""
    print(f"🚀 Starting {symbol} efficient processing...")
    
    results = {
        "symbol": symbol,
        "total_bars": len(bars),
        "zones_created": 0,
        "zone_touches": 0,
        "rejection_candles": 0,
        "volume_spikes": 0,
        "qrs_scores": [],
        "entry_points": 0,
        "entry_details": [],
        "entry_windows": [],
        "performance_metrics": {}
    }
    
    # Create zones
    zones = create_efficient_zones(symbol, bars)
    results["zones_created"] = len(zones)
    
    # Process bars efficiently
    for i, bar in enumerate(bars):
        historical_bars = bars[:i + 1]
        
        # Check for zone touches
        for zone in zones:
            if is_zone_touched(bar, zone):
                results["zone_touches"] += 1
                
                if is_rejection_candle(bar, zone):
                    results["rejection_candles"] += 1
                    
                    # Check for volume spike
                    if len(historical_bars) >= 20:
                        recent_volume = [b.volume for b in historical_bars[-20:]]
                        avg_volume = sum(recent_volume) / len(recent_volume)
                        if bar.volume > avg_volume * 1.8:
                            results["volume_spikes"] += 1
                    
                    # Calculate QRS score
                    qrs_score = calculate_qrs_score(bar, zone, historical_bars)
                    results["qrs_scores"].append(qrs_score)
                    
                    # Check if valid entry point
                    if qrs_score >= 7.0:
                        results["entry_points"] += 1
                        
                        # Create entry detail
                        entry_detail = {
                            "entry_id": f"{symbol}_{results['entry_points']}",
                            "timestamp": bar.timestamp,
                            "price": bar.close,
                            "zone_level": zone.level,
                            "zone_type": zone.zone_type.value,
                            "qrs_score": qrs_score,
                            "rejection_candle": True,
                            "volume_spike": bar.volume > avg_volume * 1.8 if len(historical_bars) >= 20 else False,
                            "zone_strength": zone.strength,
                            "zone_quality": zone.quality,
                            "bar_index": i
                        }
                        results["entry_details"].append(entry_detail)
                        
                        # Track entry window
                        window_analysis = track_entry_window(entry_detail, bars, i)
                        entry_detail.update(window_analysis)
                        results["entry_windows"].append(entry_detail)
        
        # Progress update
        if i % 10000 == 0 and i > 0:
            with progress_lock:
                progress_counter[symbol] = i
                total_processed = sum(progress_counter.values())
                print(f"   📊 {symbol}: {i:,}/{len(bars):,} bars processed...")
    
    # Calculate metrics
    results["performance_metrics"] = {
        "zone_touch_rate": results["zone_touches"] / len(bars) if bars else 0,
        "rejection_rate": results["rejection_candles"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "volume_spike_rate": results["volume_spikes"] / results["rejection_candles"] if results["rejection_candles"] > 0 else 0,
        "entry_point_rate": results["entry_points"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "avg_qrs_score": sum(results["qrs_scores"]) / len(results["qrs_scores"]) if results["qrs_scores"] else 0,
        "avg_window_duration": sum(w["window_duration_minutes"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0,
        "avg_window_bars": sum(w["window_bars"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0
    }
    
    print(f"✅ {symbol} processing complete: {results['entry_points']} entry points found")
    return results


def run_efficient_validation(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run efficient validation backtesting."""
    print("\n🚀 Starting Efficient 2024 Zone Fade Validation")
    print("=" * 60)
    print("🔄 Using 3 threads with efficient processing")
    print("📊 Original values: 30% wick ratio, 1.8x volume, strict QRS scoring")
    
    # Shared progress tracking
    progress_lock = Lock()
    progress_counter = {symbol: 0 for symbol in symbols_data.keys()}
    
    # Process symbols in parallel
    all_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol_efficient, symbol, bars, progress_lock, progress_counter): symbol
            for symbol, bars in symbols_data.items()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                all_results[symbol] = result
            except Exception as exc:
                print(f"❌ {symbol} generated an exception: {exc}")
                all_results[symbol] = {
                    "symbol": symbol,
                    "total_bars": 0,
                    "zones_created": 0,
                    "zone_touches": 0,
                    "rejection_candles": 0,
                    "volume_spikes": 0,
                    "qrs_scores": [],
                    "entry_points": 0,
                    "entry_details": [],
                    "entry_windows": [],
                    "performance_metrics": {}
                }
    
    # Generate report
    generate_efficient_report(all_results)
    
    # Export data
    export_validation_data(all_results)
    
    return all_results


def generate_efficient_report(all_results: Dict[str, Any]):
    """Generate efficient validation report."""
    print("\n📊 EFFICIENT 2024 ZONE FADE VALIDATION REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_bars = sum(results["total_bars"] for results in all_results.values())
    total_entry_points = sum(results["entry_points"] for results in all_results.values())
    
    print(f"📈 Overall 2024 Statistics:")
    print(f"   Total Bars Processed: {total_bars:,}")
    print(f"   🎯 TOTAL ENTRY POINTS: {total_entry_points}")
    
    # Per-symbol results
    print(f"\n📊 Per-Symbol Entry Points:")
    for symbol, results in all_results.items():
        print(f"   {symbol}:")
        print(f"     Bars: {results['total_bars']:,}")
        print(f"     🎯 ENTRY POINTS: {results['entry_points']}")
        
        # Performance metrics
        metrics = results["performance_metrics"]
        print(f"     Zone Touch Rate: {metrics['zone_touch_rate']:.4f}")
        print(f"     Rejection Rate: {metrics['rejection_rate']:.4f}")
        print(f"     Volume Spike Rate: {metrics['volume_spike_rate']:.4f}")
        print(f"     Entry Point Rate: {metrics['entry_point_rate']:.4f}")
        print(f"     Average QRS Score: {metrics['avg_qrs_score']:.2f}")
        print(f"     Average Window Duration: {metrics['avg_window_duration']:.1f} minutes")
        print(f"     Average Window Bars: {metrics['avg_window_bars']:.1f} bars")
    
    # Entry window analysis
    print(f"\n⏱️  Entry Window Duration Analysis:")
    all_windows = []
    for results in all_results.values():
        all_windows.extend(results["entry_windows"])
    
    if all_windows:
        durations = [w["window_duration_minutes"] for w in all_windows]
        bars = [w["window_bars"] for w in all_windows]
        
        print(f"   Total Entry Windows: {len(all_windows)}")
        print(f"   Average Duration: {np.mean(durations):.1f} minutes")
        print(f"   Median Duration: {np.median(durations):.1f} minutes")
        print(f"   Min Duration: {np.min(durations):.1f} minutes")
        print(f"   Max Duration: {np.max(durations):.1f} minutes")
        print(f"   Average Bars: {np.mean(bars):.1f} bars")
        
        # Duration distribution
        short_windows = len([d for d in durations if d <= 5])
        medium_windows = len([d for d in durations if 5 < d <= 15])
        long_windows = len([d for d in durations if d > 15])
        
        print(f"   Duration Distribution:")
        print(f"     Short (≤5 min): {short_windows} ({short_windows/len(durations)*100:.1f}%)")
        print(f"     Medium (5-15 min): {medium_windows} ({medium_windows/len(durations)*100:.1f}%)")
        print(f"     Long (>15 min): {long_windows} ({long_windows/len(durations)*100:.1f}%)")
    
    print(f"\n🎯 2024 ENTRY POINTS SUMMARY:")
    print(f"   Total Entry Points Found: {total_entry_points}")
    
    if total_entry_points > 0:
        print(f"\n✅ SUCCESS: Found {total_entry_points} valid entry points!")
        print(f"   Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}")
        
        # Calculate entry points per day
        trading_days_2024 = 252
        entry_points_per_day = total_entry_points / trading_days_2024
        print(f"   Entry Points per Trading Day: {entry_points_per_day:.1f}")
    
    print(f"\n🎉 Efficient 2024 Zone Fade Validation Complete!")


def export_validation_data(all_results: Dict[str, Any]):
    """Export validation data to CSV."""
    print("\n📁 Exporting validation data to CSV...")
    
    # Create output directory
    output_dir = Path("/app/results/2024/efficient")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export entry points
    csv_file = output_dir / "zone_fade_entry_points_2024_efficient.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'entry_id', 'symbol', 'timestamp', 'price', 'zone_level', 'zone_type',
            'qrs_score', 'rejection_candle', 'volume_spike', 'zone_strength', 'zone_quality',
            'window_duration_minutes', 'window_bars', 'max_price_deviation', 'min_price_deviation',
            'entry_window_ended', 'bar_index'
        ])
        
        # Write data
        for symbol, results in all_results.items():
            for entry in results["entry_windows"]:
                writer.writerow([
                    entry.get('entry_id', ''),
                    symbol,
                    entry['timestamp'],
                    entry['price'],
                    entry['zone_level'],
                    entry['zone_type'],
                    entry['qrs_score'],
                    entry['rejection_candle'],
                    entry['volume_spike'],
                    entry['zone_strength'],
                    entry['zone_quality'],
                    entry['window_duration_minutes'],
                    entry['window_bars'],
                    entry['max_price_deviation'],
                    entry['min_price_deviation'],
                    entry['entry_window_ended'],
                    entry.get('bar_index', 0)
                ])
    
    print(f"   ✅ Entry points exported to: {csv_file}")
    print(f"   📁 Validation data saved to: {output_dir.absolute()}")
    
    # Also create a summary file
    summary_file = output_dir / "backtesting_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Zone Fade Detector - 2024 Backtesting Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Entry Points: {sum(results['entry_points'] for results in all_results.values())}\n")
        f.write(f"Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}\n")
        f.write(f"Average Window Duration: {sum(sum(w['window_duration_minutes'] for w in results['entry_windows']) for results in all_results.values()) / sum(len(results['entry_windows']) for results in all_results.values()):.1f} minutes\n\n")
        f.write("Per-Symbol Results:\n")
        for symbol, results in all_results.items():
            f.write(f"  {symbol}: {results['entry_points']} entry points\n")
    
    print(f"   ✅ Summary exported to: {summary_file}")


def main():
    """Main backtesting function."""
    print("🚀 Zone Fade Detector - Efficient 2024 Validation Backtesting")
    print("=" * 60)
    
    # Load data sample for efficiency
    symbols_data = load_2024_data_sample(sample_size=50000)  # Last 50k bars per symbol
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run backtesting
    start_time = time.time()
    results = run_efficient_validation(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 Efficient 2024 validation backtesting finished successfully!")
    print("\n📋 Manual Validation Instructions:")
    print("   1. Open validation_output/zone_fade_entry_points_2024_efficient.csv")
    print("   2. Use timestamp column to find entry points on your charts")
    print("   3. Check window_duration_minutes to see how long opportunities last")
    print("   4. Use bar_index to locate exact candles in your data")
    print("   5. Validate rejection candles and volume spikes manually")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run backtesting
    main()