#!/usr/bin/env python3
"""
Full 2024 Zone Fade Entry Points with Manual Validation and Duration Tracking.

This script runs the complete 2024 dataset with original values and provides:
- Entry point detection with duration tracking
- Manual validation capabilities
- Entry window analysis (how long opportunities last)
- Frequency and timing analysis
- Export to CSV for external analysis
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


def load_2024_data():
    """Load the 2024 data we have downloaded."""
    print("📊 Loading 2024 Data...")
    
    data_dir = Path("data/2024")
    symbols_data = {}
    
    # Load individual symbol data
    for symbol in ["SPY", "QQQ", "IWM"]:
        file_path = data_dir / f"{symbol}_2024.pkl"
        if file_path.exists():
            print(f"   Loading {symbol} data...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                symbols_data[symbol] = data
                print(f"     ✅ {symbol}: {len(data)} bars")
        else:
            print(f"     ❌ {symbol}: File not found")
    
    return symbols_data


def create_validation_zones(symbol: str, bars: List[OHLCVBar]) -> List[Zone]:
    """Create validation zones for 2024 backtesting."""
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
            strength=1.0 - i * 0.2,  # Decreasing strength
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
            strength=1.0 - i * 0.2,  # Decreasing strength
            touches=0,
            created_at=datetime.now(),
            last_touch=None
        )
        zones.append(zone)
    
    return zones


def is_zone_touched(bar: OHLCVBar, zone: Zone) -> bool:
    """Check if a bar touches a zone."""
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        # Supply zone - check if low touches the level
        return bar.low <= zone.level <= bar.high
    else:
        # Demand zone - check if high touches the level
        return bar.low <= zone.level <= bar.high


def is_rejection_candle(bar: OHLCVBar, zone: Zone) -> bool:
    """Check if a bar is a rejection candle with original 30% wick ratio."""
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        # Supply zone - look for rejection at resistance
        if bar.high > zone.level and bar.close < zone.level:
            # Upper wick rejection
            upper_wick = bar.high - max(bar.open, bar.close)
            body_size = abs(bar.close - bar.open)
            if body_size > 0 and upper_wick / body_size > 0.3:  # 30% wick ratio (original)
                return True
    else:
        # Demand zone - look for rejection at support
        if bar.low < zone.level and bar.close > zone.level:
            # Lower wick rejection
            lower_wick = min(bar.open, bar.close) - bar.low
            body_size = abs(bar.close - bar.open)
            if body_size > 0 and lower_wick / body_size > 0.3:  # 30% wick ratio (original)
                return True
    
    return False


def calculate_validation_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar]) -> float:
    """Calculate QRS score with original strict scoring."""
    score = 0.0
    
    # Zone strength bonus
    score += zone.strength * 2.0
    
    # Zone quality bonus
    score += zone.quality * 1.0
    
    # Rejection candle bonus
    if is_rejection_candle(bar, zone):
        score += 2.0
    
    # Volume analysis (original 1.8x threshold)
    if len(bars) >= 20:
        recent_volume = [b.volume for b in bars[-20:]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        if bar.volume > avg_volume * 1.8:  # Volume spike (original threshold)
            score += 1.0
    
    # Trend context
    if len(bars) >= 10:
        recent_prices = [b.close for b in bars[-10:]]
        if recent_prices[-1] > recent_prices[0]:  # Uptrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:  # Supply zone
                score += 1.0  # Fade uptrend at resistance
        else:  # Downtrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW]:  # Demand zone
                score += 1.0  # Fade downtrend at support
    
    return min(score, 10.0)  # Cap at 10


def track_entry_window_duration(entry_point: Dict, bars: List[OHLCVBar], entry_index: int) -> Dict:
    """Track how long an entry window remains valid."""
    entry_timestamp = entry_point["timestamp"]
    entry_price = entry_point["price"]
    zone_level = entry_point["zone_level"]
    zone_type = entry_point["zone_type"]
    
    # Track entry window duration
    window_duration_minutes = 0
    window_bars = 0
    max_price_deviation = 0.0
    min_price_deviation = 0.0
    
    # Look ahead to see how long the entry remains valid
    for i in range(entry_index + 1, min(entry_index + 60, len(bars))):  # Check up to 60 bars ahead
        bar = bars[i]
        window_bars += 1
        window_duration_minutes += 1  # Assuming 1-minute bars
        
        # Calculate price deviation from entry
        price_deviation = (bar.close - entry_price) / entry_price * 100
        
        if price_deviation > max_price_deviation:
            max_price_deviation = price_deviation
        if price_deviation < min_price_deviation:
            min_price_deviation = price_deviation
        
        # Check if entry window is still valid
        # For supply zones: price should stay below zone level
        # For demand zones: price should stay above zone level
        if zone_type in ["prior_day_high", "weekly_high", "value_area_high"]:
            # Supply zone - entry invalid if price breaks above zone
            if bar.high > zone_level * 1.01:  # 1% buffer
                break
        else:
            # Demand zone - entry invalid if price breaks below zone
            if bar.low < zone_level * 0.99:  # 1% buffer
                break
    
    return {
        "window_duration_minutes": window_duration_minutes,
        "window_bars": window_bars,
        "max_price_deviation": max_price_deviation,
        "min_price_deviation": min_price_deviation,
        "entry_window_ended": window_bars < 60  # True if window ended before 60 bars
    }


def process_symbol_validation(symbol: str, bars: List[OHLCVBar], progress_lock: Lock, progress_counter: Dict[str, int]) -> Dict[str, Any]:
    """Process a single symbol with validation and duration tracking."""
    print(f"🚀 Starting {symbol} validation processing...")
    
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
    
    # Create validation zones
    zones = create_validation_zones(symbol, bars)
    results["zones_created"] = len(zones)
    
    # Process bars in batches for progress updates
    batch_size = 50000
    for i in range(0, len(bars), batch_size):
        batch = bars[i:i + batch_size]
        
        for bar in batch:
            # Get current bar index
            bar_index = bars.index(bar)
            historical_bars = bars[:bar_index + 1]
            
            # Check for zone touches
            for zone in zones:
                if is_zone_touched(bar, zone):
                    results["zone_touches"] += 1
                    
                    # Check for rejection candle
                    if is_rejection_candle(bar, zone):
                        results["rejection_candles"] += 1
                        
                        # Check for volume spike
                        if len(historical_bars) >= 20:
                            recent_volume = [b.volume for b in historical_bars[-20:]]
                            avg_volume = sum(recent_volume) / len(recent_volume)
                            if bar.volume > avg_volume * 1.8:
                                results["volume_spikes"] += 1
                        
                        # Calculate QRS score
                        qrs_score = calculate_validation_qrs_score(bar, zone, historical_bars)
                        results["qrs_scores"].append(qrs_score)
                        
                        # Check if this is a valid entry point (QRS >= 7)
                        if qrs_score >= 7.0:
                            results["entry_points"] += 1
                            
                            # Store entry details
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
                                "bar_index": bar_index
                            }
                            results["entry_details"].append(entry_detail)
                            
                            # Track entry window duration
                            window_analysis = track_entry_window_duration(entry_detail, bars, bar_index)
                            entry_detail.update(window_analysis)
                            results["entry_windows"].append(entry_detail)
        
        # Update progress
        with progress_lock:
            progress_counter[symbol] = i + batch_size
            total_processed = sum(progress_counter.values())
            total_bars = sum(len(data) for data in [bars])
            print(f"   📊 Progress: {total_processed:,} bars processed across all symbols...")
    
    # Calculate performance metrics
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


def run_validation_backtest(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run Zone Fade validation backtesting with 3 parallel threads."""
    print("\n🚀 Starting Full 2024 Zone Fade Validation Backtest")
    print("=" * 60)
    print("🔄 Using 3 threads (one per asset): SPY, QQQ, IWM")
    print("📊 Original values: 30% wick ratio, 1.8x volume, strict QRS scoring")
    print("⏱️  Entry window duration tracking enabled")
    
    # Shared progress tracking
    progress_lock = Lock()
    progress_counter = {symbol: 0 for symbol in symbols_data.keys()}
    
    # Process symbols in parallel
    all_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol_validation, symbol, bars, progress_lock, progress_counter): symbol
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
    
    # Generate comprehensive report
    generate_validation_report(all_results)
    
    # Export to CSV for manual validation
    export_validation_data(all_results)
    
    return all_results


def generate_validation_report(all_results: Dict[str, Any]):
    """Generate comprehensive validation report."""
    print("\n📊 FULL 2024 ZONE FADE VALIDATION REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_bars = sum(results["total_bars"] for results in all_results.values())
    total_zones = sum(results["zones_created"] for results in all_results.values())
    total_zone_touches = sum(results["zone_touches"] for results in all_results.values())
    total_rejection_candles = sum(results["rejection_candles"] for results in all_results.values())
    total_volume_spikes = sum(results["volume_spikes"] for results in all_results.values())
    total_entry_points = sum(results["entry_points"] for results in all_results.values())
    
    print(f"📈 Overall 2024 Statistics:")
    print(f"   Total Bars Processed: {total_bars:,}")
    print(f"   Total Zones Created: {total_zones}")
    print(f"   Total Zone Touches: {total_zone_touches}")
    print(f"   Total Rejection Candles: {total_rejection_candles}")
    print(f"   Total Volume Spikes: {total_volume_spikes}")
    print(f"   🎯 TOTAL ENTRY POINTS: {total_entry_points}")
    
    # Per-symbol results
    print(f"\n📊 Per-Symbol Entry Points:")
    for symbol, results in all_results.items():
        print(f"   {symbol}:")
        print(f"     Bars: {results['total_bars']:,}")
        print(f"     Zones: {results['zones_created']}")
        print(f"     Zone Touches: {results['zone_touches']}")
        print(f"     Rejection Candles: {results['rejection_candles']}")
        print(f"     Volume Spikes: {results['volume_spikes']}")
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
    
    # Summary
    print(f"\n🎯 2024 ENTRY POINTS SUMMARY:")
    print(f"   Total Entry Points Found: {total_entry_points}")
    print(f"   Entry Points per Symbol:")
    for symbol, results in all_results.items():
        print(f"     {symbol}: {results['entry_points']} entry points")
    
    if total_entry_points > 0:
        print(f"\n✅ SUCCESS: Found {total_entry_points} valid entry points in 2024!")
        print(f"   Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}")
        
        # Calculate entry points per day
        trading_days_2024 = 252
        entry_points_per_day = total_entry_points / trading_days_2024
        print(f"   Entry Points per Trading Day: {entry_points_per_day:.1f}")
        
        # Calculate entry points per symbol per day
        for symbol, results in all_results.items():
            symbol_entry_points_per_day = results['entry_points'] / trading_days_2024
            print(f"   {symbol} Entry Points per Day: {symbol_entry_points_per_day:.1f}")
    
    print(f"\n🎉 Full 2024 Zone Fade Validation Backtesting Complete!")


def export_validation_data(all_results: Dict[str, Any]):
    """Export validation data to CSV for manual analysis."""
    print("\n📁 Exporting validation data to CSV...")
    
    # Create output directory
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export entry points with window analysis
    csv_file = output_dir / "zone_fade_entry_points_2024.csv"
    
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
    
    # Export summary statistics
    summary_file = output_dir / "zone_fade_summary_2024.csv"
    
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'symbol', 'total_bars', 'zones_created', 'zone_touches', 'rejection_candles',
            'volume_spikes', 'entry_points', 'zone_touch_rate', 'rejection_rate',
            'volume_spike_rate', 'entry_point_rate', 'avg_qrs_score',
            'avg_window_duration', 'avg_window_bars'
        ])
        
        # Write data
        for symbol, results in all_results.items():
            metrics = results["performance_metrics"]
            writer.writerow([
                symbol,
                results['total_bars'],
                results['zones_created'],
                results['zone_touches'],
                results['rejection_candles'],
                results['volume_spikes'],
                results['entry_points'],
                metrics['zone_touch_rate'],
                metrics['rejection_rate'],
                metrics['volume_spike_rate'],
                metrics['entry_point_rate'],
                metrics['avg_qrs_score'],
                metrics['avg_window_duration'],
                metrics['avg_window_bars']
            ])
    
    print(f"   ✅ Summary exported to: {summary_file}")
    print(f"   📁 All validation data saved to: {output_dir.absolute()}")


def main():
    """Main backtesting function."""
    print("🚀 Zone Fade Detector - Full 2024 Validation Backtesting")
    print("=" * 60)
    
    # Load data
    symbols_data = load_2024_data()
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run backtesting
    start_time = time.time()
    results = run_validation_backtest(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 Full 2024 validation backtesting finished successfully!")
    print("\n📋 Next Steps for Manual Validation:")
    print("   1. Review the CSV files in validation_output/")
    print("   2. Check entry_point_timestamps for manual chart verification")
    print("   3. Analyze window_duration_minutes for entry opportunity timing")
    print("   4. Use bar_index to locate exact candles in your data")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run backtesting
    main()