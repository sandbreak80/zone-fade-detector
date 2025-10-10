#!/usr/bin/env python3
"""
December 2024 Zone Fade Entry Points Debugging - Single Month.

This script runs Zone Fade strategy on December 2024 data for debugging:
- Single month processing (much faster)
- Detailed logging and debugging output
- Zone Detection, Rejection Candle Detection, QRS Scoring
"""

import asyncio
import sys
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import time
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType


def load_december_2024_data():
    """Load December 2024 data for debugging."""
    print("📊 Loading December 2024 Data for Debugging...")
    
    data_dir = Path("data/2024")
    symbols_data = {}
    
    # Load individual symbol data
    for symbol in ["SPY", "QQQ", "IWM"]:
        file_path = data_dir / f"{symbol}_2024.pkl"
        if file_path.exists():
            print(f"   Loading {symbol} data...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                # Filter for December 2024 only
                december_data = []
                for bar in data:
                    if bar.timestamp.month == 12 and bar.timestamp.year == 2024:
                        december_data.append(bar)
                
                symbols_data[symbol] = december_data
                print(f"     ✅ {symbol}: {len(december_data)} bars (December 2024)")
        else:
            print(f"     ❌ {symbol}: File not found")
    
    return symbols_data


def create_debug_zones(symbol: str, bars: List[OHLCVBar]) -> List[Zone]:
    """Create debug zones for December 2024."""
    zones = []
    
    if not bars:
        return zones
    
    # Use numpy for efficient calculations
    high_prices = np.array([bar.high for bar in bars])
    low_prices = np.array([bar.low for bar in bars])
    
    # Find significant highs and lows
    max_high = np.max(high_prices)
    min_low = np.min(low_prices)
    
    print(f"   📊 {symbol} Price Range: ${min_low:.2f} - ${max_high:.2f}")
    
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
        print(f"   📈 Supply Zone {i+1}: ${level:.2f} (strength: {zone.strength:.2f})")
    
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
        print(f"   📉 Demand Zone {i+1}: ${level:.2f} (strength: {zone.strength:.2f})")
    
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
    # Basic rejection candle logic with original 30% wick ratio
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


def calculate_debug_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar]) -> float:
    """Calculate a debug QRS score with detailed logging."""
    score = 0.0
    debug_details = []
    
    # Start with 0, earn points for quality (original approach)
    debug_details.append(f"Base zone touch: +0.0 (original strict scoring)")
    
    # Zone strength bonus
    strength_bonus = zone.strength * 2.0
    score += strength_bonus
    debug_details.append(f"Zone strength ({zone.strength:.2f}): +{strength_bonus:.2f}")
    
    # Zone quality bonus
    quality_bonus = zone.quality * 1.0
    score += quality_bonus
    debug_details.append(f"Zone quality ({zone.quality}): +{quality_bonus:.2f}")
    
    # Rejection candle bonus
    if is_rejection_candle(bar, zone):
        score += 2.0
        debug_details.append(f"Rejection candle: +2.0")
    
    # Volume analysis
    if len(bars) >= 20:
        recent_volume = [b.volume for b in bars[-20:]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        volume_ratio = bar.volume / avg_volume if avg_volume > 0 else 1.0
        if bar.volume > avg_volume * 1.8:  # Volume spike (original threshold)
            score += 1.0
            debug_details.append(f"Volume spike ({volume_ratio:.2f}x): +1.0")
        else:
            debug_details.append(f"Volume normal ({volume_ratio:.2f}x): +0.0")
    
    # Trend context
    if len(bars) >= 10:
        recent_prices = [b.close for b in bars[-10:]]
        trend = "up" if recent_prices[-1] > recent_prices[0] else "down"
        if trend == "up" and zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:
            score += 1.0
            debug_details.append(f"Uptrend fade at resistance: +1.0")
        elif trend == "down" and zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW]:
            score += 1.0
            debug_details.append(f"Downtrend fade at support: +1.0")
        else:
            debug_details.append(f"Trend context ({trend}): +0.0")
    
    final_score = min(score, 10.0)  # Cap at 10
    
    # Debug logging for high-scoring setups
    if final_score >= 7.0:
        print(f"   🔍 High QRS Score: {final_score:.2f} - {', '.join(debug_details)}")
    
    return final_score


def process_symbol_debug(symbol: str, bars: List[OHLCVBar]) -> Dict[str, Any]:
    """Process a single symbol with debug output."""
    print(f"\n🚀 Processing {symbol} with Debug Output...")
    print(f"   Bars: {len(bars)}")
    
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
    zones = create_debug_zones(symbol, bars)
    results["zones_created"] = len(zones)
    
    # Process bars
    for i, bar in enumerate(bars):
        # Get current bar index
        historical_bars = bars[:i + 1]
        
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
                    qrs_score = calculate_debug_qrs_score(bar, zone, historical_bars)
                    results["qrs_scores"].append(qrs_score)
                    
                    # Check if this is a valid entry point (QRS >= 7)
                    if qrs_score >= 7.0:
                        results["entry_points"] += 1
                        
                        # Store entry details
                        entry_detail = {
                            "timestamp": bar.timestamp,
                            "price": bar.close,
                            "zone_level": zone.level,
                            "zone_type": zone.zone_type.value,
                            "qrs_score": qrs_score,
                            "rejection_candle": True,
                            "volume_spike": bar.volume > avg_volume * 1.8 if len(historical_bars) >= 20 else False,
                            "zone_strength": zone.strength,
                            "zone_quality": zone.quality
                        }
                        results["entry_details"].append(entry_detail)
                        
                        print(f"   🎯 ENTRY POINT: {bar.timestamp} - Price: ${bar.close:.2f} - QRS: {qrs_score:.1f} - Zone: {zone.zone_type.value} @ ${zone.level:.2f}")
        
        # Progress update
        if i % 1000 == 0 and i > 0:
            print(f"     Processed {i}/{len(bars)} bars...")
    
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


def run_december_debug_backtest(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run December 2024 debug backtesting."""
    print("\n🚀 Starting December 2024 Debug Backtest")
    print("=" * 60)
    print("🔍 Debug mode: Detailed logging and analysis")
    
    # Process each symbol
    all_results = {}
    
    for symbol, bars in symbols_data.items():
        if bars:  # Only process if we have data
            results = process_symbol_debug(symbol, bars)
            all_results[symbol] = results
        else:
            print(f"⚠️  Skipping {symbol}: No data")
    
    # Generate comprehensive report
    generate_debug_report(all_results)
    
    return all_results


def generate_debug_report(all_results: Dict[str, Any]):
    """Generate comprehensive debug report."""
    print("\n📊 DECEMBER 2024 DEBUG REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_bars = sum(results["total_bars"] for results in all_results.values())
    total_zones = sum(results["zones_created"] for results in all_results.values())
    total_zone_touches = sum(results["zone_touches"] for results in all_results.values())
    total_rejection_candles = sum(results["rejection_candles"] for results in all_results.values())
    total_volume_spikes = sum(results["volume_spikes"] for results in all_results.values())
    total_entry_points = sum(results["entry_points"] for results in all_results.values())
    
    print(f"📈 December 2024 Statistics:")
    print(f"   Total Bars Processed: {total_bars:,}")
    print(f"   Total Zones Created: {total_zones}")
    print(f"   Total Zone Touches: {total_zone_touches}")
    print(f"   Total Rejection Candles: {total_rejection_candles}")
    print(f"   Total Volume Spikes: {total_volume_spikes}")
    print(f"   🎯 TOTAL ENTRY POINTS: {total_entry_points}")
    
    # Per-symbol results
    print(f"\n📊 Per-Symbol Results:")
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
        
        # Show all entry details for debugging
        if results["entry_details"]:
            print(f"     📋 All Entry Details:")
            for i, entry in enumerate(results["entry_details"]):
                print(f"       {i+1}. {entry['timestamp']} - Price: ${entry['price']:.2f} - QRS: {entry['qrs_score']:.1f} - Zone: {entry['zone_type']} @ ${entry['zone_level']:.2f}")
    
    # Summary
    print(f"\n🎯 DECEMBER 2024 SUMMARY:")
    print(f"   Total Entry Points Found: {total_entry_points}")
    print(f"   Entry Points per Symbol:")
    for symbol, results in all_results.items():
        print(f"     {symbol}: {results['entry_points']} entry points")
    
    if total_entry_points > 0:
        print(f"\n✅ SUCCESS: Found {total_entry_points} valid entry points in December 2024!")
        print(f"   Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}")
        
        # Calculate entry points per day
        trading_days_december = 22  # Approximate trading days in December
        entry_points_per_day = total_entry_points / trading_days_december
        print(f"   Entry Points per Trading Day: {entry_points_per_day:.1f}")
    else:
        print(f"\n⚠️  WARNING: No entry points found in December 2024!")
        print(f"   This could indicate:")
        print(f"   - QRS threshold too high (current: 7)")
        print(f"   - Zone detection issues")
        print(f"   - Rejection candle criteria too strict")
        print(f"   - December 2024 data issues")
    
    print(f"\n🎉 December 2024 Debug Backtesting Complete!")


def main():
    """Main backtesting function."""
    print("🚀 Zone Fade Detector - December 2024 Debug Backtesting")
    print("=" * 60)
    
    # Load data
    symbols_data = load_december_2024_data()
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run backtesting
    start_time = time.time()
    results = run_december_debug_backtest(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 December 2024 debug backtesting finished successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run backtesting
    main()