#!/usr/bin/env python3
"""
Enhanced 2024 Zone Fade Entry Points with Comprehensive Metrics.

This enhanced script includes all the detailed calculations shown in the visualization:
- Risk management metrics (stop loss, take profit, risk/reward)
- Price analysis (range, zone position, high/low)
- Volume analysis (ratio, VWAP distance)
- Entry window details (start/end times, duration)
- Comprehensive QRS breakdown
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


def create_enhanced_zones(symbol: str, bars: List[OHLCVBar]) -> List[Zone]:
    """Create enhanced zones with proper high/low tracking."""
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


def calculate_enhanced_qrs_score(bar: OHLCVBar, zone: Zone, bars: List[OHLCVBar]) -> Dict[str, float]:
    """Calculate enhanced QRS score with detailed breakdown."""
    qrs_breakdown = {
        'quality_score': 0.0,
        'risk_score': 0.0,
        'setup_score': 0.0,
        'overall_qrs': 0.0
    }
    
    # Quality Score (0-10): Zone strength and quality
    qrs_breakdown['quality_score'] = (zone.strength * 5.0) + (zone.quality * 2.5)
    
    # Risk Score (0-10): Volume and market context
    if len(bars) >= 20:
        recent_volume = [b.volume for b in bars[-20:]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        volume_ratio = bar.volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike scoring
        if volume_ratio >= 2.5:
            qrs_breakdown['risk_score'] += 4.0
        elif volume_ratio >= 1.8:
            qrs_breakdown['risk_score'] += 3.0
        elif volume_ratio >= 1.5:
            qrs_breakdown['risk_score'] += 2.0
        else:
            qrs_breakdown['risk_score'] += 1.0
        
        # Market context scoring
        if len(bars) >= 10:
            recent_prices = [b.close for b in bars[-10:]]
            if recent_prices[-1] > recent_prices[0]:  # Uptrend
                if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:
                    qrs_breakdown['risk_score'] += 3.0
            else:  # Downtrend
                if zone.zone_type in [ZoneType.PRIOR_DAY_LOW, ZoneType.WEEKLY_LOW]:
                    qrs_breakdown['risk_score'] += 3.0
    
    # Setup Score (0-10): Rejection candle and pattern quality
    if is_rejection_candle(bar, zone):
        qrs_breakdown['setup_score'] += 5.0
        
        # Additional setup quality based on wick ratio
        if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH]:
            upper_wick = bar.high - max(bar.open, bar.close)
            body_size = abs(bar.close - bar.open)
            if body_size > 0:
                wick_ratio = upper_wick / body_size
                if wick_ratio >= 0.5:
                    qrs_breakdown['setup_score'] += 3.0
                elif wick_ratio >= 0.3:
                    qrs_breakdown['setup_score'] += 2.0
        else:
            lower_wick = min(bar.open, bar.close) - bar.low
            body_size = abs(bar.close - bar.open)
            if body_size > 0:
                wick_ratio = lower_wick / body_size
                if wick_ratio >= 0.5:
                    qrs_breakdown['setup_score'] += 3.0
                elif wick_ratio >= 0.3:
                    qrs_breakdown['setup_score'] += 2.0
    
    # Overall QRS (weighted average)
    qrs_breakdown['overall_qrs'] = (
        qrs_breakdown['quality_score'] * 0.3 +
        qrs_breakdown['risk_score'] * 0.4 +
        qrs_breakdown['setup_score'] * 0.3
    )
    
    # Cap all scores at 10
    for key in qrs_breakdown:
        qrs_breakdown[key] = min(qrs_breakdown[key], 10.0)
    
    return qrs_breakdown


def calculate_risk_management(entry_price: float, zone: Zone, bars: List[OHLCVBar], entry_index: int) -> Dict[str, float]:
    """Calculate comprehensive risk management metrics."""
    risk_metrics = {}
    
    # Determine stop loss and take profit based on zone type
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        # Supply zone - short setup
        stop_loss = zone.level * 1.01  # 1% above zone
        take_profit = zone.level * 0.99  # 1% below zone
        direction = "SHORT"
    else:
        # Demand zone - long setup
        stop_loss = zone.level * 0.99  # 1% below zone
        take_profit = zone.level * 1.01  # 1% above zone
        direction = "LONG"
    
    # Calculate risk and reward amounts
    if direction == "LONG":
        risk_amount = entry_price - stop_loss
        reward_amount = take_profit - entry_price
    else:
        risk_amount = stop_loss - entry_price
        reward_amount = entry_price - take_profit
    
    # Calculate risk/reward ratio
    risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0
    
    risk_metrics.update({
        'direction': direction,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_amount': risk_amount,
        'reward_amount': reward_amount,
        'risk_reward_ratio': risk_reward_ratio
    })
    
    return risk_metrics


def calculate_price_analysis(entry_price: float, zone: Zone, bars: List[OHLCVBar], entry_index: int) -> Dict[str, float]:
    """Calculate comprehensive price analysis metrics."""
    price_metrics = {}
    
    # Get price range for the analysis window (20 bars around entry)
    start_idx = max(0, entry_index - 10)
    end_idx = min(len(bars), entry_index + 10)
    analysis_bars = bars[start_idx:end_idx]
    
    if analysis_bars:
        high_prices = [bar.high for bar in analysis_bars]
        low_prices = [bar.low for bar in analysis_bars]
        
        price_high = max(high_prices)
        price_low = min(low_prices)
        price_range = price_high - price_low
        price_range_pct = (price_range / entry_price) * 100
        
        # Calculate zone metrics
        zone_high = zone.level * 1.01  # 1% buffer
        zone_low = zone.level * 0.99   # 1% buffer
        zone_range = zone_high - zone_low
        zone_mid = (zone_high + zone_low) / 2
        
        # Calculate entry position within zone
        if zone_range > 0:
            if entry_price >= zone_mid:
                entry_zone_position = ((entry_price - zone_mid) / (zone_high - zone_mid)) * 50 + 50
            else:
                entry_zone_position = ((entry_price - zone_low) / (zone_mid - zone_low)) * 50
        else:
            entry_zone_position = 50.0
        
        price_metrics.update({
            'current_price': entry_price,
            'price_high': price_high,
            'price_low': price_low,
            'price_range': price_range,
            'price_range_pct': price_range_pct,
            'zone_high': zone_high,
            'zone_low': zone_low,
            'zone_range': zone_range,
            'zone_mid': zone_mid,
            'entry_zone_position': entry_zone_position
        })
    
    return price_metrics


def calculate_volume_analysis(bar: OHLCVBar, bars: List[OHLCVBar], entry_index: int) -> Dict[str, float]:
    """Calculate comprehensive volume analysis metrics."""
    volume_metrics = {}
    
    # Calculate average volume (20-bar lookback)
    if len(bars) >= 20:
        recent_volume = [b.volume for b in bars[max(0, entry_index-20):entry_index]]
        avg_volume = sum(recent_volume) / len(recent_volume)
        volume_ratio = bar.volume / avg_volume if avg_volume > 0 else 1.0
    else:
        avg_volume = bar.volume
        volume_ratio = 1.0
    
    # Calculate VWAP for the session
    session_bars = bars[:entry_index + 1]
    if session_bars:
        typical_prices = [(b.high + b.low + b.close) / 3 for b in session_bars]
        volumes = [b.volume for b in session_bars]
        
        cum_pv = sum(tp * vol for tp, vol in zip(typical_prices, volumes))
        cum_volume = sum(volumes)
        vwap = cum_pv / cum_volume if cum_volume > 0 else bar.close
        
        vwap_distance = ((bar.close - vwap) / vwap) * 100
    else:
        vwap = bar.close
        vwap_distance = 0.0
    
    volume_metrics.update({
        'entry_volume': bar.volume,
        'average_volume': avg_volume,
        'volume_ratio': volume_ratio,
        'entry_vwap': vwap,
        'vwap_distance_pct': vwap_distance
    })
    
    return volume_metrics


def track_enhanced_entry_window(entry_point: Dict, bars: List[OHLCVBar], entry_index: int) -> Dict:
    """Track enhanced entry window with detailed timing."""
    entry_price = entry_point["price"]
    zone_level = entry_point["zone_level"]
    zone_type = entry_point["zone_type"]
    
    # Find entry window start (when price first approached zone)
    entry_start_index = entry_index
    entry_start_time = bars[entry_index].timestamp
    
    # Look back up to 10 bars to find when price first approached zone
    for i in range(max(0, entry_index - 10), entry_index):
        bar = bars[i]
        if zone_type in ["prior_day_high", "weekly_high", "value_area_high"]:
            if bar.high >= zone_level * 0.99:  # Within 1% of zone
                entry_start_index = i
                entry_start_time = bar.timestamp
                break
        else:
            if bar.low <= zone_level * 1.01:  # Within 1% of zone
                entry_start_index = i
                entry_start_time = bar.timestamp
                break
    
    # Find entry window end (when price moved away from zone)
    entry_end_index = entry_index
    entry_end_time = bars[entry_index].timestamp
    
    # Look forward up to 30 bars to find when price moved away
    for i in range(entry_index + 1, min(entry_index + 30, len(bars))):
        bar = bars[i]
        if zone_type in ["prior_day_high", "weekly_high", "value_area_high"]:
            if bar.high > zone_level * 1.01:  # Moved 1% above zone
                entry_end_index = i
                entry_end_time = bar.timestamp
                break
        else:
            if bar.low < zone_level * 0.99:  # Moved 1% below zone
                entry_end_index = i
                entry_end_time = bar.timestamp
                break
    
    # Calculate duration
    duration_minutes = (entry_end_time - entry_start_time).total_seconds() / 60
    window_bars = entry_end_index - entry_start_index + 1
    
    # Calculate price deviation during window
    window_bars_data = bars[entry_start_index:entry_end_index + 1]
    if window_bars_data:
        window_prices = [bar.close for bar in window_bars_data]
        max_price = max(window_prices)
        min_price = min(window_prices)
        max_deviation = ((max_price - entry_price) / entry_price) * 100
        min_deviation = ((min_price - entry_price) / entry_price) * 100
    else:
        max_deviation = 0.0
        min_deviation = 0.0
    
    return {
        'entry_start_time': entry_start_time,
        'entry_end_time': entry_end_time,
        'entry_duration_minutes': duration_minutes,
        'entry_start_index': entry_start_index,
        'entry_end_index': entry_end_index,
        'window_bars': window_bars,
        'max_price_deviation': max_deviation,
        'min_price_deviation': min_deviation,
        'entry_window_ended': entry_end_index < entry_index + 30
    }


def process_symbol_enhanced(symbol: str, bars: List[OHLCVBar], progress_lock: Lock, progress_counter: Dict[str, int]) -> Dict[str, Any]:
    """Process a single symbol with enhanced metrics."""
    print(f"🚀 Starting {symbol} enhanced processing...")
    
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
    zones = create_enhanced_zones(symbol, bars)
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
                    volume_spike = False
                    if len(historical_bars) >= 20:
                        recent_volume = [b.volume for b in historical_bars[-20:]]
                        avg_volume = sum(recent_volume) / len(recent_volume)
                        if bar.volume > avg_volume * 1.8:
                            results["volume_spikes"] += 1
                            volume_spike = True
                    
                    # Calculate enhanced QRS score
                    qrs_breakdown = calculate_enhanced_qrs_score(bar, zone, historical_bars)
                    results["qrs_scores"].append(qrs_breakdown['overall_qrs'])
                    
                    # Check if valid entry point
                    if qrs_breakdown['overall_qrs'] >= 7.0:
                        results["entry_points"] += 1
                        
                        # Calculate all enhanced metrics
                        risk_metrics = calculate_risk_management(bar.close, zone, bars, i)
                        price_metrics = calculate_price_analysis(bar.close, zone, bars, i)
                        volume_metrics = calculate_volume_analysis(bar, bars, i)
                        
                        # Create comprehensive entry detail
                        entry_detail = {
                            "entry_id": f"{symbol}_{results['entry_points']}",
                            "timestamp": bar.timestamp,
                            "price": bar.close,
                            "zone_level": zone.level,
                            "zone_type": zone.zone_type.value,
                            "qrs_score": qrs_breakdown['overall_qrs'],
                            "rejection_candle": True,
                            "volume_spike": volume_spike,
                            "zone_strength": zone.strength,
                            "zone_quality": zone.quality,
                            "bar_index": i,
                            
                            # QRS Breakdown
                            "quality_score": qrs_breakdown['quality_score'],
                            "risk_score": qrs_breakdown['risk_score'],
                            "setup_score": qrs_breakdown['setup_score'],
                            
                            # Risk Management
                            "direction": risk_metrics['direction'],
                            "stop_loss": risk_metrics['stop_loss'],
                            "take_profit": risk_metrics['take_profit'],
                            "risk_amount": risk_metrics['risk_amount'],
                            "reward_amount": risk_metrics['reward_amount'],
                            "risk_reward_ratio": risk_metrics['risk_reward_ratio'],
                            
                            # Price Analysis
                            "current_price": price_metrics.get('current_price', bar.close),
                            "price_high": price_metrics.get('price_high', bar.high),
                            "price_low": price_metrics.get('price_low', bar.low),
                            "price_range": price_metrics.get('price_range', 0.0),
                            "price_range_pct": price_metrics.get('price_range_pct', 0.0),
                            "zone_high": price_metrics.get('zone_high', zone.level),
                            "zone_low": price_metrics.get('zone_low', zone.level),
                            "zone_range": price_metrics.get('zone_range', 0.0),
                            "zone_mid": price_metrics.get('zone_mid', zone.level),
                            "entry_zone_position": price_metrics.get('entry_zone_position', 50.0),
                            
                            # Volume Analysis
                            "entry_volume": volume_metrics['entry_volume'],
                            "average_volume": volume_metrics['average_volume'],
                            "volume_ratio": volume_metrics['volume_ratio'],
                            "entry_vwap": volume_metrics['entry_vwap'],
                            "vwap_distance_pct": volume_metrics['vwap_distance_pct']
                        }
                        
                        # Track enhanced entry window
                        window_analysis = track_enhanced_entry_window(entry_detail, bars, i)
                        entry_detail.update(window_analysis)
                        results["entry_windows"].append(entry_detail)
        
        # Progress update
        if i % 10000 == 0 and i > 0:
            with progress_lock:
                progress_counter[symbol] = i
                total_processed = sum(progress_counter.values())
                print(f"   📊 {symbol}: {i:,}/{len(bars):,} bars processed...")
    
    # Calculate enhanced metrics
    results["performance_metrics"] = {
        "zone_touch_rate": results["zone_touches"] / len(bars) if bars else 0,
        "rejection_rate": results["rejection_candles"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "volume_spike_rate": results["volume_spikes"] / results["rejection_candles"] if results["rejection_candles"] > 0 else 0,
        "entry_point_rate": results["entry_points"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "avg_qrs_score": sum(results["qrs_scores"]) / len(results["qrs_scores"]) if results["qrs_scores"] else 0,
        "avg_window_duration": sum(w["entry_duration_minutes"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0,
        "avg_window_bars": sum(w["window_bars"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0,
        "avg_risk_reward_ratio": sum(w["risk_reward_ratio"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0,
        "avg_volume_ratio": sum(w["volume_ratio"] for w in results["entry_windows"]) / len(results["entry_windows"]) if results["entry_windows"] else 0
    }
    
    print(f"✅ {symbol} enhanced processing complete: {results['entry_points']} entry points found")
    return results


def run_enhanced_validation(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run enhanced validation backtesting."""
    print("\n🚀 Starting Enhanced 2024 Zone Fade Validation")
    print("=" * 60)
    print("🔄 Using 3 threads with enhanced metrics calculation")
    print("📊 Original values: 30% wick ratio, 1.8x volume, strict QRS scoring")
    print("📈 Enhanced metrics: Risk management, price analysis, volume analysis")
    
    # Shared progress tracking
    progress_lock = Lock()
    progress_counter = {symbol: 0 for symbol in symbols_data.keys()}
    
    # Process symbols in parallel
    all_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol_enhanced, symbol, bars, progress_lock, progress_counter): symbol
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
    
    # Generate enhanced report
    generate_enhanced_report(all_results)
    
    # Export enhanced data
    export_enhanced_data(all_results)
    
    return all_results


def generate_enhanced_report(all_results: Dict[str, Any]):
    """Generate enhanced validation report."""
    print("\n📊 ENHANCED 2024 ZONE FADE VALIDATION REPORT")
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
        print(f"     Average Risk/Reward Ratio: {metrics['avg_risk_reward_ratio']:.2f}")
        print(f"     Average Volume Ratio: {metrics['avg_volume_ratio']:.2f}x")
    
    # Enhanced analysis
    print(f"\n💰 Risk Management Analysis:")
    all_windows = []
    for results in all_results.values():
        all_windows.extend(results["entry_windows"])
    
    if all_windows:
        risk_ratios = [w["risk_reward_ratio"] for w in all_windows]
        volume_ratios = [w["volume_ratio"] for w in all_windows]
        qrs_scores = [w["qrs_score"] for w in all_windows]
        
        print(f"   Total Entry Windows: {len(all_windows)}")
        print(f"   Average Risk/Reward Ratio: {np.mean(risk_ratios):.2f}")
        print(f"   Average Volume Ratio: {np.mean(volume_ratios):.2f}x")
        print(f"   Average QRS Score: {np.mean(qrs_scores):.2f}")
        
        # Quality distribution
        high_quality = len([r for r in risk_ratios if r >= 1.0])
        medium_quality = len([r for r in risk_ratios if 0.5 <= r < 1.0])
        low_quality = len([r for r in risk_ratios if r < 0.5])
        
        print(f"   Risk/Reward Distribution:")
        print(f"     Favorable (≥1.0): {high_quality} ({high_quality/len(risk_ratios)*100:.1f}%)")
        print(f"     Moderate (0.5-1.0): {medium_quality} ({medium_quality/len(risk_ratios)*100:.1f}%)")
        print(f"     Unfavorable (<0.5): {low_quality} ({low_quality/len(risk_ratios)*100:.1f}%)")
    
    print(f"\n🎯 Enhanced 2024 Entry Points Summary:")
    print(f"   Total Entry Points Found: {total_entry_points}")
    
    if total_entry_points > 0:
        print(f"\n✅ SUCCESS: Found {total_entry_points} valid entry points with enhanced metrics!")
        print(f"   Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}")
        
        # Calculate entry points per day
        trading_days_2024 = 252
        entry_points_per_day = total_entry_points / trading_days_2024
        print(f"   Entry Points per Trading Day: {entry_points_per_day:.1f}")
    
    print(f"\n🎉 Enhanced 2024 Zone Fade Validation Complete!")


def export_enhanced_data(all_results: Dict[str, Any]):
    """Export enhanced validation data to CSV."""
    print("\n📁 Exporting enhanced validation data to CSV...")
    
    # Create output directory
    output_dir = Path("/app/results/2024/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export entry points with all enhanced metrics
    csv_file = output_dir / "zone_fade_entry_points_2024_enhanced.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write comprehensive header
        writer.writerow([
            # Basic Info
            'entry_id', 'symbol', 'timestamp', 'price', 'zone_level', 'zone_type',
            'qrs_score', 'rejection_candle', 'volume_spike', 'zone_strength', 'zone_quality',
            
            # QRS Breakdown
            'quality_score', 'risk_score', 'setup_score',
            
            # Risk Management
            'direction', 'stop_loss', 'take_profit', 'risk_amount', 'reward_amount', 'risk_reward_ratio',
            
            # Price Analysis
            'current_price', 'price_high', 'price_low', 'price_range', 'price_range_pct',
            'zone_high', 'zone_low', 'zone_range', 'zone_mid', 'entry_zone_position',
            
            # Volume Analysis
            'entry_volume', 'average_volume', 'volume_ratio', 'entry_vwap', 'vwap_distance_pct',
            
            # Entry Window
            'entry_start_time', 'entry_end_time', 'entry_duration_minutes', 'entry_start_index', 'entry_end_index',
            'window_bars', 'max_price_deviation', 'min_price_deviation', 'entry_window_ended', 'bar_index'
        ])
        
        # Write data
        for symbol, results in all_results.items():
            for entry in results["entry_windows"]:
                writer.writerow([
                    # Basic Info
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
                    
                    # QRS Breakdown
                    entry.get('quality_score', 0.0),
                    entry.get('risk_score', 0.0),
                    entry.get('setup_score', 0.0),
                    
                    # Risk Management
                    entry.get('direction', ''),
                    entry.get('stop_loss', 0.0),
                    entry.get('take_profit', 0.0),
                    entry.get('risk_amount', 0.0),
                    entry.get('reward_amount', 0.0),
                    entry.get('risk_reward_ratio', 0.0),
                    
                    # Price Analysis
                    entry.get('current_price', entry['price']),
                    entry.get('price_high', entry['price']),
                    entry.get('price_low', entry['price']),
                    entry.get('price_range', 0.0),
                    entry.get('price_range_pct', 0.0),
                    entry.get('zone_high', entry['zone_level']),
                    entry.get('zone_low', entry['zone_level']),
                    entry.get('zone_range', 0.0),
                    entry.get('zone_mid', entry['zone_level']),
                    entry.get('entry_zone_position', 50.0),
                    
                    # Volume Analysis
                    entry.get('entry_volume', 0),
                    entry.get('average_volume', 0),
                    entry.get('volume_ratio', 1.0),
                    entry.get('entry_vwap', entry['price']),
                    entry.get('vwap_distance_pct', 0.0),
                    
                    # Entry Window
                    entry.get('entry_start_time', entry['timestamp']),
                    entry.get('entry_end_time', entry['timestamp']),
                    entry.get('entry_duration_minutes', 0.0),
                    entry.get('entry_start_index', entry.get('bar_index', 0)),
                    entry.get('entry_end_index', entry.get('bar_index', 0)),
                    entry.get('window_bars', 1),
                    entry.get('max_price_deviation', 0.0),
                    entry.get('min_price_deviation', 0.0),
                    entry.get('entry_window_ended', True),
                    entry.get('bar_index', 0)
                ])
    
    print(f"   ✅ Enhanced entry points exported to: {csv_file}")
    print(f"   📁 Enhanced validation data saved to: {output_dir.absolute()}")
    
    # Also create an enhanced summary file
    summary_file = output_dir / "enhanced_backtesting_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Zone Fade Detector - Enhanced 2024 Backtesting Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Entry Points: {sum(results['entry_points'] for results in all_results.values())}\n")
        f.write(f"Average QRS Score: {sum(sum(results['qrs_scores']) for results in all_results.values()) / sum(len(results['qrs_scores']) for results in all_results.values()):.2f}\n")
        f.write(f"Average Window Duration: {sum(sum(w['entry_duration_minutes'] for w in results['entry_windows']) for results in all_results.values()) / sum(len(results['entry_windows']) for results in all_results.values()):.1f} minutes\n")
        f.write(f"Average Risk/Reward Ratio: {sum(sum(w['risk_reward_ratio'] for w in results['entry_windows']) for results in all_results.values()) / sum(len(results['entry_windows']) for results in all_results.values()):.2f}\n")
        f.write(f"Average Volume Ratio: {sum(sum(w['volume_ratio'] for w in results['entry_windows']) for results in all_results.values()) / sum(len(results['entry_windows']) for results in all_results.values()):.2f}x\n\n")
        f.write("Per-Symbol Results:\n")
        for symbol, results in all_results.items():
            f.write(f"  {symbol}: {results['entry_points']} entry points\n")
    
    print(f"   ✅ Enhanced summary exported to: {summary_file}")


def main():
    """Main enhanced backtesting function."""
    print("🚀 Zone Fade Detector - Enhanced 2024 Validation Backtesting")
    print("=" * 60)
    
    # Load data sample for efficiency
    symbols_data = load_2024_data_sample(sample_size=50000)  # Last 50k bars per symbol
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run enhanced backtesting
    start_time = time.time()
    results = run_enhanced_validation(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Enhanced backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 Enhanced 2024 validation backtesting finished successfully!")
    print("\n📋 Enhanced Manual Validation Instructions:")
    print("   1. Open results/2024/enhanced/zone_fade_entry_points_2024_enhanced.csv")
    print("   2. Review all enhanced metrics: QRS breakdown, risk management, price analysis")
    print("   3. Use timestamp column to find entry points on your charts")
    print("   4. Check risk/reward ratios and volume analysis for quality assessment")
    print("   5. Validate entry window timing and duration metrics")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run enhanced backtesting
    main()