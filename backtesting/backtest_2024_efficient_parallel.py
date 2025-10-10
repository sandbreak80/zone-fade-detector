#!/usr/bin/env python3
"""
Efficient Parallel Zone Fade Entry Points Backtesting - 2024 Data.

This script runs Zone Fade strategy efficiently using 3 threads:
- Processes data in smaller chunks for better memory management
- Optimized zone detection and QRS scoring
- Real-time progress tracking
"""

import asyncio
import sys
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import time
import concurrent.futures
from threading import Lock
import numpy as np

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


def create_efficient_zones(symbol: str, bars: List[OHLCVBar]) -> List[Zone]:
    """Create efficient test zones for backtesting."""
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
    """Check if a bar is a rejection candle."""
    # Basic rejection candle logic
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        # Supply zone - look for rejection at resistance
        if bar.high > zone.level and bar.close < zone.level:
            # Upper wick rejection
            upper_wick = bar.high - max(bar.open, bar.close)
            body_size = abs(bar.close - bar.open)
            if body_size > 0 and upper_wick / body_size > 0.1:  # 10% wick ratio
                return True
    else:
        # Demand zone - look for rejection at support
        if bar.low < zone.level and bar.close > zone.level:
            # Lower wick rejection
            lower_wick = min(bar.open, bar.close) - bar.low
            body_size = abs(bar.close - bar.open)
            if body_size > 0 and lower_wick / body_size > 0.1:  # 10% wick ratio
                return True
    
    return False


def calculate_efficient_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar]) -> float:
    """Calculate an efficient QRS score."""
    score = 0.0
    
    # Base score for zone touch
    score += 1.0
    
    # Zone strength bonus
    score += zone.strength * 2.0
    
    # Zone quality bonus
    score += zone.quality * 1.0
    
    # Rejection candle bonus
    if is_rejection_candle(bar, zone):
        score += 2.0
    
    # Volume analysis (efficient)
    if len(bars) >= 20:
        recent_volume = [b.volume for b in bars[-20:]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        if bar.volume > avg_volume * 1.5:  # Volume spike
            score += 1.0
    
    # Trend context (efficient)
    if len(bars) >= 10:
        recent_prices = [b.close for b in bars[-10:]]
        if recent_prices[-1] > recent_prices[0]:  # Uptrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:  # Supply zone
                score += 1.0  # Fade uptrend at resistance
        else:  # Downtrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW]:  # Demand zone
                score += 1.0  # Fade downtrend at support
    
    return min(score, 10.0)  # Cap at 10


def process_symbol_efficient(symbol: str, bars: List[OHLCVBar], progress_lock: Lock, progress_counter: Dict[str, int]) -> Dict[str, Any]:
    """Process a single symbol efficiently."""
    print(f"🚀 Starting {symbol} processing...")
    
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
        "performance_metrics": {}
    }
    
    # Create test zones
    zones = create_efficient_zones(symbol, bars)
    results["zones_created"] = len(zones)
    
    # Process bars in larger batches for efficiency
    batch_size = 50000  # Larger batches
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
                            if bar.volume > avg_volume * 1.5:
                                results["volume_spikes"] += 1
                        
                        # Calculate QRS score
                        qrs_score = calculate_efficient_qrs_score(bar, zone, historical_bars)
                        results["qrs_scores"].append(qrs_score)
                        
                        # Check if this is a valid entry point (QRS >= 7)
                        if qrs_score >= 7.0:
                            results["entry_points"] += 1
                            
                            # Store entry details (limit to first 100 to save memory)
                            if len(results["entry_details"]) < 100:
                                entry_detail = {
                                    "timestamp": bar.timestamp,
                                    "price": bar.close,
                                    "zone_level": zone.level,
                                    "zone_type": zone.zone_type.value,
                                    "qrs_score": qrs_score,
                                    "rejection_candle": True,
                                    "volume_spike": bar.volume > avg_volume * 1.5 if len(historical_bars) >= 20 else False,
                                    "zone_strength": zone.strength,
                                    "zone_quality": zone.quality
                                }
                                results["entry_details"].append(entry_detail)
        
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
        "avg_qrs_score": sum(results["qrs_scores"]) / len(results["qrs_scores"]) if results["qrs_scores"] else 0
    }
    
    print(f"✅ {symbol} processing complete: {results['entry_points']} entry points found")
    return results


def run_efficient_parallel_backtest(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run Zone Fade strategy efficiently in parallel using 3 threads."""
    print("\n🚀 Starting Efficient Parallel Zone Fade Entry Points Backtest")
    print("=" * 60)
    print("🔄 Using 3 threads (one per asset): SPY, QQQ, IWM")
    print("⚡ Optimized for speed and memory efficiency")
    
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
                    "performance_metrics": {}
                }
    
    # Generate comprehensive report
    generate_efficient_parallel_report(all_results)
    
    return all_results


def generate_efficient_parallel_report(all_results: Dict[str, Any]):
    """Generate comprehensive entry points report."""
    print("\n📊 EFFICIENT PARALLEL ZONE FADE ENTRY POINTS REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_bars = sum(results["total_bars"] for results in all_results.values())
    total_zones = sum(results["zones_created"] for results in all_results.values())
    total_zone_touches = sum(results["zone_touches"] for results in all_results.values())
    total_rejection_candles = sum(results["rejection_candles"] for results in all_results.values())
    total_volume_spikes = sum(results["volume_spikes"] for results in all_results.values())
    total_entry_points = sum(results["entry_points"] for results in all_results.values())
    
    print(f"📈 Overall Entry Points Statistics:")
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
        
        # Show entry details for first few entries
        if results["entry_details"]:
            print(f"     📋 Entry Details (first 3):")
            for i, entry in enumerate(results["entry_details"][:3]):
                print(f"       {i+1}. {entry['timestamp']} - Price: ${entry['price']:.2f} - QRS: {entry['qrs_score']:.1f} - Zone: {entry['zone_type']} @ ${entry['zone_level']:.2f}")
    
    # Summary
    print(f"\n🎯 ENTRY POINTS SUMMARY:")
    print(f"   Total Entry Points Found: {total_entry_points}")
    print(f"   Entry Points per Symbol:")
    for symbol, results in all_results.items():
        print(f"     {symbol}: {results['entry_points']} entry points")
    
    if total_entry_points > 0:
        print(f"\n✅ SUCCESS: Found {total_entry_points} valid entry points!")
        print(f"   Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}")
        
        # Calculate entry points per day (approximate)
        total_days = 252  # Trading days in 2024
        entry_points_per_day = total_entry_points / total_days
        print(f"   Entry Points per Trading Day: {entry_points_per_day:.1f}")
        
        # Calculate entry points per symbol per day
        for symbol, results in all_results.items():
            symbol_entry_points_per_day = results['entry_points'] / total_days
            print(f"   {symbol} Entry Points per Day: {symbol_entry_points_per_day:.1f}")
    else:
        print(f"\n⚠️  WARNING: No entry points found!")
        print(f"   This could indicate:")
        print(f"   - QRS threshold too high (current: 7)")
        print(f"   - Zone detection issues")
        print(f"   - Rejection candle criteria too strict")
        print(f"   - Need to lower QRS threshold for testing")
    
    print(f"\n🎉 Efficient Parallel Zone Fade Entry Points Backtesting Complete!")


def main():
    """Main backtesting function."""
    print("🚀 Zone Fade Detector - Efficient Parallel Entry Points Backtesting")
    print("=" * 60)
    
    # Load data
    symbols_data = load_2024_data()
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run backtesting
    start_time = time.time()
    results = run_efficient_parallel_backtest(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 Efficient parallel entry points backtesting finished successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run backtesting
    main()