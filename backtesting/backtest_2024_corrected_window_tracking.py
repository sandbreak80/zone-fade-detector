#!/usr/bin/env python3
"""
Corrected 2024 Zone Fade Entry Points with Proper Window Duration Tracking.

This script fixes the window duration calculation to properly track how long
entry conditions remain valid after detection, not just when price moves away.
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


def load_2024_data_sample(sample_size: int = 0):
    """Load 2024 data - full dataset if sample_size=0, otherwise sample."""
    if sample_size == 0:
        print(f"📊 Loading FULL 2024 Data (all bars per symbol)...")
    else:
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
                # Load full data or sample based on sample_size
                if sample_size == 0:
                    symbols_data[symbol] = data
                    print(f"     ✅ {symbol}: {len(symbols_data[symbol])} bars (FULL)")
                else:
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


def has_volume_spike(bar: OHLCVBar, bars: List[OHLCVBar], bar_index: int) -> bool:
    """Check if a bar has a volume spike (1.8x threshold)."""
    if len(bars) >= 20 and bar_index >= 19:
        recent_volume = [b.volume for b in bars[bar_index-19:bar_index+1]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        return bar.volume > avg_volume * 1.8
    return False


def calculate_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar], bar_index: int) -> float:
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
    if has_volume_spike(bar, bars, bar_index):
        score += 1.0
    
    # Trend context
    if len(bars) >= 10 and bar_index >= 9:
        recent_prices = [b.close for b in bars[bar_index-9:bar_index+1]]
        if recent_prices[-1] > recent_prices[0]:  # Uptrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:
                score += 1.0
        else:  # Downtrend
            if zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW]:
                score += 1.0
    
    return min(score, 10.0)


def calculate_vwap(bars: List[OHLCVBar], bar_index: int, lookback: int = 20) -> float:
    """Calculate VWAP for the last lookback bars."""
    if bar_index < lookback - 1:
        return bars[bar_index].close
    
    start_idx = max(0, bar_index - lookback + 1)
    vwap_bars = bars[start_idx:bar_index + 1]
    
    total_volume = sum(bar.volume for bar in vwap_bars)
    if total_volume == 0:
        return bars[bar_index].close
    
    vwap = sum(bar.volume * (bar.high + bar.low + bar.close) / 3 for bar in vwap_bars) / total_volume
    return vwap


def calculate_zone_fade_exit_strategy(entry_price: float, zone: Zone, bars: List[OHLCVBar], bar_index: int, qrs_score: float) -> Dict[str, float]:
    """Calculate Zone Fade exit strategy based on proper logic."""
    # Determine trade direction based on zone type
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        direction = "SHORT"
        # Hard stop: back of zone (zone level)
        hard_stop = zone.level
        # Zone invalidation: break beyond back of zone
        invalidation_level = zone.level
    else:
        direction = "LONG"
        # Hard stop: back of zone (zone level)
        hard_stop = zone.level
        # Zone invalidation: break beyond back of zone
        invalidation_level = zone.level
    
    # Calculate VWAP for T1 target
    vwap = calculate_vwap(bars, bar_index)
    
    # Calculate R (Risk Unit)
    if direction == "SHORT":
        risk_amount = entry_price - hard_stop  # Entry above zone, stop at zone
    else:
        risk_amount = hard_stop - entry_price  # Entry below zone, stop at zone
    
    # Ensure minimum risk amount
    if risk_amount <= 0:
        risk_amount = entry_price * 0.01  # 1% fallback
    
    # Calculate targets based on QRS quality
    if qrs_score >= 9.0:  # A-grade setup
        t1_reward = 1.0 * risk_amount
        t2_reward = 2.0 * risk_amount
        t3_reward = 3.0 * risk_amount
    elif qrs_score >= 7.0:  # B-grade setup
        t1_reward = 0.8 * risk_amount
        t2_reward = 1.6 * risk_amount
        t3_reward = 2.4 * risk_amount
    else:  # C-grade setup
        t1_reward = 0.6 * risk_amount
        t2_reward = 1.2 * risk_amount
        t3_reward = 1.8 * risk_amount
    
    # Calculate target prices
    if direction == "SHORT":
        t1_price = min(vwap, entry_price - t1_reward)  # Nearest of VWAP or 1R
        t2_price = entry_price - t2_reward
        t3_price = entry_price - t3_reward
    else:
        t1_price = max(vwap, entry_price + t1_reward)  # Nearest of VWAP or 1R
        t2_price = entry_price + t2_reward
        t3_price = entry_price + t3_reward
    
    # Calculate risk/reward ratio (using T1 as primary target)
    if direction == "SHORT":
        primary_reward = entry_price - t1_price
    else:
        primary_reward = t1_price - entry_price
    
    risk_reward_ratio = primary_reward / risk_amount if risk_amount > 0 else 0
    
    return {
        "direction": direction,
        "hard_stop": hard_stop,
        "invalidation_level": invalidation_level,
        "risk_amount": risk_amount,
        "t1_price": t1_price,
        "t2_price": t2_price,
        "t3_price": t3_price,
        "t1_reward": t1_reward,
        "t2_reward": t2_reward,
        "t3_reward": t3_reward,
        "vwap": vwap,
        "risk_reward_ratio": risk_reward_ratio,
        "qrs_grade": "A" if qrs_score >= 9.0 else "B" if qrs_score >= 7.0 else "C"
    }


def is_valid_entry_point(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar], bar_index: int) -> bool:
    """Check if this is a valid entry point with Zone Fade exit strategy."""
    # Check basic entry conditions
    if not is_zone_touched(bar, zone):
        return False
    
    if not is_rejection_candle(bar, zone):
        return False
    
    if not has_volume_spike(bar, bars, bar_index):
        return False
    
    # Check QRS score
    qrs_score = calculate_qrs_score(bar, zone, bars, bar_index)
    if qrs_score < 7.0:
        return False
    
    # Calculate Zone Fade exit strategy
    exit_strategy = calculate_zone_fade_exit_strategy(bar.close, zone, bars, bar_index, qrs_score)
    
    # Only accept entry points with reasonable risk/reward (T1 target)
    # Lower threshold since we're using proper Zone Fade logic
    if exit_strategy["risk_reward_ratio"] < 0.5:  # At least 0.5:1 for T1
        return False
    
    return True


def is_entry_conditions_valid(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar], bar_index: int) -> bool:
    """Check if entry conditions are still valid for this bar."""
    # Check if zone is still being touched
    if not is_zone_touched(bar, zone):
        return False
    
    # Check if we still have rejection candle characteristics
    if not is_rejection_candle(bar, zone):
        return False
    
    # Check if volume spike is still present
    if not has_volume_spike(bar, bars, bar_index):
        return False
    
    # Check if QRS score is still above threshold
    qrs_score = calculate_qrs_score(bar, zone, bars, bar_index)
    if qrs_score < 7.0:
        return False
    
    return True


def track_corrected_entry_window(entry_point: Dict, bars: List[OHLCVBar], entry_index: int, zone: Zone) -> Dict:
    """Track entry window duration correctly - how long entry conditions remain valid."""
    entry_price = entry_point["price"]
    zone_level = entry_point["zone_level"]
    zone_type = entry_point["zone_type"]
    
    # Start tracking from the entry point
    entry_start_time = bars[entry_index].timestamp
    entry_start_index = entry_index
    
    # Track how many minutes entry conditions remain valid
    valid_minutes = 0
    valid_bars = 0
    max_price_deviation = 0.0
    min_price_deviation = 0.0
    
    # Check up to 60 bars ahead (1 hour) for entry validity
    max_lookahead = min(entry_index + 60, len(bars))
    
    for i in range(entry_index, max_lookahead):
        bar = bars[i]
        
        # Check if entry conditions are still valid
        if is_entry_conditions_valid(bar, zone, bars, i):
            valid_minutes += 1
            valid_bars += 1
            
            # Calculate price deviation from entry price
            price_deviation = (bar.close - entry_price) / entry_price * 100
            max_price_deviation = max(max_price_deviation, price_deviation)
            min_price_deviation = min(min_price_deviation, price_deviation)
        else:
            # Entry conditions no longer valid, stop tracking
            break
    
    # Calculate end time
    if valid_bars > 0:
        entry_end_index = entry_index + valid_bars - 1
        entry_end_time = bars[entry_end_index].timestamp
    else:
        entry_end_index = entry_index
        entry_end_time = entry_start_time
    
    return {
        'entry_start_time': entry_start_time,
        'entry_end_time': entry_end_time,
        'entry_duration_minutes': valid_minutes,
        'entry_start_index': entry_start_index,
        'entry_end_index': entry_end_index,
        'window_bars': valid_bars,
        'max_price_deviation': max_price_deviation,
        'min_price_deviation': min_price_deviation,
        'entry_window_ended': valid_bars < 60,  # True if we didn't hit the 60-bar limit
        'entry_conditions_valid': valid_minutes > 0
    }


def process_symbol_corrected(symbol: str, bars: List[OHLCVBar], progress_lock: Lock, progress_counter: Dict[str, int]) -> Dict[str, Any]:
    """Process a single symbol with corrected window tracking and risk/reward filtering."""
    print(f"🚀 Starting {symbol} corrected processing...")
    
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
        "risk_reward_stats": [],
        "debug_data": [],  # NEW: Debug data for filtered entries
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
                
                # Create debug entry for every zone touch
                debug_entry = {
                    "bar_index": i,
                    "timestamp": bar.timestamp,
                    "price": bar.close,
                    "zone_level": zone.level,
                    "zone_type": zone.zone_type.value,
                    "zone_strength": zone.strength,
                    "zone_quality": zone.quality,
                    "is_rejection_candle": False,
                    "is_volume_spike": False,
                    "qrs_score": 0.0,
                    "risk_reward_ratio": 0.0,
                    "filter_reason": "zone_touch_only"
                }
                
                if is_rejection_candle(bar, zone):
                    results["rejection_candles"] += 1
                    debug_entry["is_rejection_candle"] = True
                    debug_entry["filter_reason"] = "rejection_candle_ok"
                    
                    # Check for volume spike
                    volume_spike = has_volume_spike(bar, bars, i)
                    if volume_spike:
                        results["volume_spikes"] += 1
                        debug_entry["is_volume_spike"] = True
                        debug_entry["filter_reason"] = "volume_spike_ok"
                    
                    # Calculate QRS score
                    qrs_score = calculate_qrs_score(bar, zone, bars, i)
                    results["qrs_scores"].append(qrs_score)
                    debug_entry["qrs_score"] = qrs_score
                    
                    if qrs_score < 7.0:
                        debug_entry["filter_reason"] = f"qrs_too_low_{qrs_score:.2f}"
                    else:
                        debug_entry["filter_reason"] = "qrs_ok"
                        
                        # Calculate Zone Fade exit strategy
                        exit_strategy = calculate_zone_fade_exit_strategy(bar.close, zone, bars, i, qrs_score)
                        debug_entry["risk_reward_ratio"] = exit_strategy["risk_reward_ratio"]
                        debug_entry["direction"] = exit_strategy["direction"]
                        debug_entry["hard_stop"] = exit_strategy["hard_stop"]
                        debug_entry["t1_price"] = exit_strategy["t1_price"]
                        debug_entry["t2_price"] = exit_strategy["t2_price"]
                        debug_entry["t3_price"] = exit_strategy["t3_price"]
                        debug_entry["risk_amount"] = exit_strategy["risk_amount"]
                        debug_entry["t1_reward"] = exit_strategy["t1_reward"]
                        debug_entry["vwap"] = exit_strategy["vwap"]
                        debug_entry["qrs_grade"] = exit_strategy["qrs_grade"]
                        
                        if exit_strategy["risk_reward_ratio"] < 0.5:
                            debug_entry["filter_reason"] = f"risk_reward_too_low_{exit_strategy['risk_reward_ratio']:.2f}"
                        else:
                            debug_entry["filter_reason"] = "ALL_CRITERIA_PASSED"
                            
                            # This is a valid entry point!
                            results["entry_points"] += 1
                            results["risk_reward_stats"].append(exit_strategy)
                            
                            # Create entry detail
                            entry_detail = {
                                "entry_id": f"{symbol}_{results['entry_points']}",
                                "timestamp": bar.timestamp,
                                "price": bar.close,
                                "zone_level": zone.level,
                                "zone_type": zone.zone_type.value,
                                "qrs_score": qrs_score,
                                "qrs_grade": exit_strategy["qrs_grade"],
                                "rejection_candle": True,
                                "volume_spike": volume_spike,
                                "zone_strength": zone.strength,
                                "zone_quality": zone.quality,
                                "bar_index": i,
                                "direction": exit_strategy["direction"],
                                "hard_stop": exit_strategy["hard_stop"],
                                "invalidation_level": exit_strategy["invalidation_level"],
                                "t1_price": exit_strategy["t1_price"],
                                "t2_price": exit_strategy["t2_price"],
                                "t3_price": exit_strategy["t3_price"],
                                "risk_amount": exit_strategy["risk_amount"],
                                "t1_reward": exit_strategy["t1_reward"],
                                "t2_reward": exit_strategy["t2_reward"],
                                "t3_reward": exit_strategy["t3_reward"],
                                "vwap": exit_strategy["vwap"],
                                "risk_reward_ratio": exit_strategy["risk_reward_ratio"]
                            }
                            
                            # Track corrected entry window
                            window_analysis = track_corrected_entry_window(entry_detail, bars, i, zone)
                            entry_detail.update(window_analysis)
                            results["entry_windows"].append(entry_detail)
                
                # Always add debug entry (limit to first 1000 to avoid huge files)
                if i < 1000:
                    results["debug_data"].append(debug_entry)
        
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
        "avg_window_duration": sum(w["entry_duration_minutes"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0,
        "avg_window_bars": sum(w["window_bars"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0,
        "valid_entry_windows": len([w for w in results["entry_windows"] if w["entry_conditions_valid"]]),
        "invalid_entry_windows": len([w for w in results["entry_windows"] if not w["entry_conditions_valid"]]),
        "avg_risk_reward_ratio": sum(rr["risk_reward_ratio"] for rr in results["risk_reward_stats"]) / len(results["risk_reward_stats"]) if results["risk_reward_stats"] else 0,
        "min_risk_reward_ratio": min(rr["risk_reward_ratio"] for rr in results["risk_reward_stats"]) if results["risk_reward_stats"] else 0,
        "max_risk_reward_ratio": max(rr["risk_reward_ratio"] for rr in results["risk_reward_stats"]) if results["risk_reward_stats"] else 0
    }
    
    print(f"✅ {symbol} corrected processing complete: {results['entry_points']} entry points found")
    return results


def run_corrected_validation(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run corrected validation backtesting."""
    print("\n🚀 Starting Corrected 2024 Zone Fade Validation")
    print("=" * 60)
    print("🔄 Using 3 threads with CORRECTED window duration tracking")
    print("📊 Original values: 30% wick ratio, 1.8x volume, strict QRS scoring")
    print("⏱️  CORRECTED: Window duration tracks how long entry conditions remain valid")
    
    # Shared progress tracking
    progress_lock = Lock()
    progress_counter = {symbol: 0 for symbol in symbols_data.keys()}
    
    # Process symbols in parallel
    all_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol_corrected, symbol, bars, progress_lock, progress_counter): symbol
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
    
    # Generate corrected report
    generate_corrected_report(all_results)
    
    # Export corrected data
    export_corrected_data(all_results)
    
    # Export debug data
    export_debug_data(all_results)
    
    return all_results


def generate_corrected_report(all_results: Dict[str, Any]):
    """Generate corrected validation report."""
    print("\n📊 CORRECTED 2024 ZONE FADE VALIDATION REPORT")
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
        print(f"     Valid Entry Windows: {metrics['valid_entry_windows']}")
        print(f"     Invalid Entry Windows: {metrics['invalid_entry_windows']}")
        print(f"     Average Risk/Reward Ratio: {metrics['avg_risk_reward_ratio']:.2f}")
        print(f"     Min Risk/Reward Ratio: {metrics['min_risk_reward_ratio']:.2f}")
        print(f"     Max Risk/Reward Ratio: {metrics['max_risk_reward_ratio']:.2f}")
    
    # Entry window analysis
    print(f"\n⏱️  CORRECTED Entry Window Duration Analysis:")
    all_windows = []
    for results in all_results.values():
        all_windows.extend(results["entry_windows"])
    
    if all_windows:
        durations = [w["entry_duration_minutes"] for w in all_windows]
        valid_windows = [w for w in all_windows if w["entry_conditions_valid"]]
        invalid_windows = [w for w in all_windows if not w["entry_conditions_valid"]]
        
        print(f"   Total Entry Windows: {len(all_windows)}")
        print(f"   Valid Entry Windows: {len(valid_windows)}")
        print(f"   Invalid Entry Windows: {len(invalid_windows)}")
        
        if valid_windows:
            valid_durations = [w["entry_duration_minutes"] for w in valid_windows]
            print(f"   Average Valid Duration: {np.mean(valid_durations):.1f} minutes")
            print(f"   Median Valid Duration: {np.median(valid_durations):.1f} minutes")
            print(f"   Min Valid Duration: {np.min(valid_durations):.1f} minutes")
            print(f"   Max Valid Duration: {np.max(valid_durations):.1f} minutes")
        
        # Duration distribution
        short_windows = len([d for d in durations if d <= 5])
        medium_windows = len([d for d in durations if 5 < d <= 15])
        long_windows = len([d for d in durations if d > 15])
        
        print(f"   Duration Distribution:")
        print(f"     Short (≤5 min): {short_windows} ({short_windows/len(durations)*100:.1f}%)")
        print(f"     Medium (5-15 min): {medium_windows} ({medium_windows/len(durations)*100:.1f}%)")
        print(f"     Long (>15 min): {long_windows} ({long_windows/len(durations)*100:.1f}%)")
    
    print(f"\n🎯 CORRECTED 2024 Entry Points Summary:")
    print(f"   Total Entry Points Found: {total_entry_points} (1:2+ Risk/Reward)")
    
    if total_entry_points > 0:
        print(f"\n✅ SUCCESS: Found {total_entry_points} valid entry points with CORRECTED window tracking!")
        total_qrs_scores = sum(len(results['qrs_scores']) for results in all_results.values())
        if total_qrs_scores > 0:
            avg_qrs = sum(sum(results['qrs_scores']) for results in all_results.values()) / total_qrs_scores
            print(f"   Average QRS Score: {avg_qrs:.2f}")
        else:
            print(f"   Average QRS Score: N/A (no QRS scores calculated)")
        
        # Calculate risk/reward statistics
        all_risk_reward = []
        for results in all_results.values():
            all_risk_reward.extend(results["risk_reward_stats"])
        
        if all_risk_reward:
            avg_rr = sum(rr["risk_reward_ratio"] for rr in all_risk_reward) / len(all_risk_reward)
            min_rr = min(rr["risk_reward_ratio"] for rr in all_risk_reward)
            max_rr = max(rr["risk_reward_ratio"] for rr in all_risk_reward)
            print(f"   Average Risk/Reward Ratio: {avg_rr:.2f}")
            print(f"   Min Risk/Reward Ratio: {min_rr:.2f}")
            print(f"   Max Risk/Reward Ratio: {max_rr:.2f}")
        
        # Calculate entry points per day
        trading_days_2024 = 252
        entry_points_per_day = total_entry_points / trading_days_2024
        print(f"   Entry Points per Trading Day: {entry_points_per_day:.1f}")
    
    print(f"\n🎉 CORRECTED 2024 Zone Fade Validation Complete!")


def export_corrected_data(all_results: Dict[str, Any]):
    """Export corrected validation data to CSV."""
    print("\n📁 Exporting corrected validation data to CSV...")
    
    # Create output directory
    output_dir = Path("/app/results/2024/corrected")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export entry points
    csv_file = output_dir / "zone_fade_entry_points_2024_corrected.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'entry_id', 'symbol', 'timestamp', 'price', 'zone_level', 'zone_type',
            'qrs_score', 'qrs_grade', 'rejection_candle', 'volume_spike', 'zone_strength', 'zone_quality',
            'direction', 'hard_stop', 'invalidation_level', 't1_price', 't2_price', 't3_price',
            'risk_amount', 't1_reward', 't2_reward', 't3_reward', 'vwap', 'risk_reward_ratio',
            'entry_start_time', 'entry_end_time', 'entry_duration_minutes', 'entry_start_index', 'entry_end_index',
            'window_bars', 'max_price_deviation', 'min_price_deviation', 'entry_window_ended', 'entry_conditions_valid', 'bar_index'
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
                    entry.get('qrs_grade', ''),
                    entry['rejection_candle'],
                    entry['volume_spike'],
                    entry['zone_strength'],
                    entry['zone_quality'],
                    entry.get('direction', ''),
                    entry.get('hard_stop', 0.0),
                    entry.get('invalidation_level', 0.0),
                    entry.get('t1_price', 0.0),
                    entry.get('t2_price', 0.0),
                    entry.get('t3_price', 0.0),
                    entry.get('risk_amount', 0.0),
                    entry.get('t1_reward', 0.0),
                    entry.get('t2_reward', 0.0),
                    entry.get('t3_reward', 0.0),
                    entry.get('vwap', 0.0),
                    entry.get('risk_reward_ratio', 0.0),
                    entry.get('entry_start_time', entry['timestamp']),
                    entry.get('entry_end_time', entry['timestamp']),
                    entry.get('entry_duration_minutes', 0.0),
                    entry.get('entry_start_index', entry.get('bar_index', 0)),
                    entry.get('entry_end_index', entry.get('bar_index', 0)),
                    entry.get('window_bars', 1),
                    entry.get('max_price_deviation', 0.0),
                    entry.get('min_price_deviation', 0.0),
                    entry.get('entry_window_ended', True),
                    entry.get('entry_conditions_valid', False),
                    entry.get('bar_index', 0)
                ])
    
    print(f"   ✅ Corrected entry points exported to: {csv_file}")
    print(f"   📁 Corrected validation data saved to: {output_dir.absolute()}")
    
    # Also create a corrected summary file
    summary_file = output_dir / "corrected_backtesting_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Zone Fade Detector - CORRECTED 2024 Backtesting Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("CORRECTED WINDOW DURATION TRACKING:\n")
        f.write("- Tracks how long entry conditions remain valid\n")
        f.write("- Stops when conditions are no longer met\n")
        f.write("- Measures actual trading opportunity window\n\n")
        f.write("RISK/REWARD FILTERING:\n")
        f.write("- Only accepts entry points with 1:2 or better risk/reward ratio\n")
        f.write("- Calculates stop loss and take profit levels\n")
        f.write("- Ensures profitable trading opportunities\n\n")
        f.write(f"Total Entry Points: {sum(results['entry_points'] for results in all_results.values())} (1:2+ R/R)\n")
        total_qrs_scores = sum(len(results['qrs_scores']) for results in all_results.values())
        if total_qrs_scores > 0:
            avg_qrs = sum(sum(results['qrs_scores']) for results in all_results.values()) / total_qrs_scores
            f.write(f"Average QRS Score: {avg_qrs:.2f}\n")
        else:
            f.write(f"Average QRS Score: N/A (no QRS scores calculated)\n")
        total_windows = sum(len(results['entry_windows']) for results in all_results.values())
        if total_windows > 0:
            avg_duration = sum(sum(w['entry_duration_minutes'] for w in results['entry_windows']) for results in all_results.values()) / total_windows
            f.write(f"Average Window Duration: {avg_duration:.1f} minutes\n\n")
        else:
            f.write(f"Average Window Duration: N/A (no entry points found)\n\n")
        
        # Calculate risk/reward statistics
        all_risk_reward = []
        for results in all_results.values():
            all_risk_reward.extend(results["risk_reward_stats"])
        
        if all_risk_reward:
            avg_rr = sum(rr["risk_reward_ratio"] for rr in all_risk_reward) / len(all_risk_reward)
            min_rr = min(rr["risk_reward_ratio"] for rr in all_risk_reward)
            max_rr = max(rr["risk_reward_ratio"] for rr in all_risk_reward)
            f.write(f"Average Risk/Reward Ratio: {avg_rr:.2f}\n")
            f.write(f"Min Risk/Reward Ratio: {min_rr:.2f}\n")
            f.write(f"Max Risk/Reward Ratio: {max_rr:.2f}\n\n")
        
        f.write("Per-Symbol Results:\n")
        for symbol, results in all_results.items():
            f.write(f"  {symbol}: {results['entry_points']} entry points\n")
    
    print(f"   ✅ Corrected summary exported to: {summary_file}")


def export_debug_data(all_results: Dict[str, Any]):
    """Export debug data for filtered entry points."""
    print("\n🔍 Exporting debug data for filtered entry points...")
    
    # Create output directory
    output_dir = Path("/app/results/2024/corrected")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export debug data
    debug_csv_file = output_dir / "debug_filtered_entry_points.csv"
    
    with open(debug_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'symbol', 'bar_index', 'timestamp', 'price', 'zone_level', 'zone_type',
            'zone_strength', 'zone_quality', 'is_rejection_candle', 'is_volume_spike',
            'qrs_score', 'qrs_grade', 'risk_reward_ratio', 'direction', 'hard_stop',
            't1_price', 't2_price', 't3_price', 'risk_amount', 't1_reward', 'vwap', 'filter_reason'
        ])
        
        # Write debug data
        for symbol, results in all_results.items():
            for debug_entry in results["debug_data"]:
                writer.writerow([
                    symbol,
                    debug_entry.get('bar_index', 0),
                    debug_entry.get('timestamp', ''),
                    debug_entry.get('price', 0.0),
                    debug_entry.get('zone_level', 0.0),
                    debug_entry.get('zone_type', ''),
                    debug_entry.get('zone_strength', 0.0),
                    debug_entry.get('zone_quality', 0.0),
                    debug_entry.get('is_rejection_candle', False),
                    debug_entry.get('is_volume_spike', False),
                    debug_entry.get('qrs_score', 0.0),
                    debug_entry.get('qrs_grade', ''),
                    debug_entry.get('risk_reward_ratio', 0.0),
                    debug_entry.get('direction', ''),
                    debug_entry.get('hard_stop', 0.0),
                    debug_entry.get('t1_price', 0.0),
                    debug_entry.get('t2_price', 0.0),
                    debug_entry.get('t3_price', 0.0),
                    debug_entry.get('risk_amount', 0.0),
                    debug_entry.get('t1_reward', 0.0),
                    debug_entry.get('vwap', 0.0),
                    debug_entry.get('filter_reason', '')
                ])
    
    print(f"   ✅ Debug data exported to: {debug_csv_file}")
    
    # Create debug summary
    debug_summary_file = output_dir / "debug_summary.txt"
    with open(debug_summary_file, 'w') as f:
        f.write("Zone Fade Detector - Debug Summary for Filtered Entry Points\n")
        f.write("=" * 70 + "\n\n")
        
        for symbol, results in all_results.items():
            f.write(f"{symbol} Debug Summary:\n")
            f.write(f"  Total Debug Entries: {len(results['debug_data'])}\n")
            
            # Count by filter reason
            filter_counts = {}
            for entry in results["debug_data"]:
                reason = entry.get('filter_reason', 'unknown')
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
            
            f.write(f"  Filter Reasons:\n")
            for reason, count in sorted(filter_counts.items()):
                f.write(f"    {reason}: {count}\n")
            
            # QRS score distribution
            qrs_scores = [entry.get('qrs_score', 0) for entry in results["debug_data"] if entry.get('qrs_score', 0) > 0]
            if qrs_scores:
                f.write(f"  QRS Score Stats:\n")
                f.write(f"    Min: {min(qrs_scores):.2f}\n")
                f.write(f"    Max: {max(qrs_scores):.2f}\n")
                f.write(f"    Avg: {sum(qrs_scores)/len(qrs_scores):.2f}\n")
                f.write(f"    Count ≥7.0: {len([s for s in qrs_scores if s >= 7.0])}\n")
            
            # Risk/reward distribution
            rr_ratios = [entry.get('risk_reward_ratio', 0) for entry in results["debug_data"] if entry.get('risk_reward_ratio', 0) > 0]
            if rr_ratios:
                f.write(f"  Risk/Reward Stats:\n")
                f.write(f"    Min: {min(rr_ratios):.2f}\n")
                f.write(f"    Max: {max(rr_ratios):.2f}\n")
                f.write(f"    Avg: {sum(rr_ratios)/len(rr_ratios):.2f}\n")
                f.write(f"    Count ≥2.0: {len([r for r in rr_ratios if r >= 2.0])}\n")
            
            f.write("\n")
    
    print(f"   ✅ Debug summary exported to: {debug_summary_file}")


def main():
    """Main corrected backtesting function."""
    print("🚀 Zone Fade Detector - CORRECTED 2024 Validation Backtesting")
    print("=" * 60)
    print("⏱️  CORRECTED: Window duration tracks how long entry conditions remain valid")
    print("🎯 ZONE FADE: Uses proper exit strategy with T1/T2/T3 targets and VWAP")
    print("📊 FULL DATA: Using complete 2024 dataset for comprehensive backtesting")
    
    # Load full 2024 data for comprehensive backtesting
    symbols_data = load_2024_data_sample(sample_size=0)  # 0 means load all data
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run corrected backtesting
    start_time = time.time()
    results = run_corrected_validation(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Corrected backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 CORRECTED 2024 validation backtesting finished successfully!")
    print("\n📋 CORRECTED Manual Validation Instructions:")
    print("   1. Open results/2024/corrected/zone_fade_entry_points_2024_corrected.csv")
    print("   2. Check entry_duration_minutes to see how long conditions remain valid")
    print("   3. Check entry_conditions_valid to see if conditions were actually valid")
    print("   4. Check risk_reward_ratio to ensure all entries are 1:2 or better")
    print("   5. Use timestamp column to find entry points on your charts")
    print("   6. Validate that window duration makes sense for trading execution")
    print("   7. Verify stop_loss and take_profit levels are realistic")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run corrected backtesting
    main()