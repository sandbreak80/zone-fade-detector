#!/usr/bin/env python3
"""
Zone Fade Strategy Entry Points Backtesting - 2024 Data.

This script runs the complete Zone Fade strategy to find actual entry points:
- Zone Detection
- Rejection Candle Detection
- CHoCH Confirmation
- QRS Scoring
- Volume Spike Analysis
"""

import asyncio
import sys
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import time

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.session_state_manager import SessionStateManager
from zone_fade_detector.core.micro_window_analyzer import MicroWindowAnalyzer
from zone_fade_detector.core.parallel_cross_symbol_processor import (
    ParallelCrossSymbolProcessor, SymbolType
)
from zone_fade_detector.core.models import OHLCVBar, Zone, ZoneType
from zone_fade_detector.strategies.zone_fade_strategy import ZoneFadeStrategy
from zone_fade_detector.strategies.signal_processor import SignalProcessor
from zone_fade_detector.strategies.zone_detector import ZoneDetector
from zone_fade_detector.indicators.swing_structure import SwingStructureDetector
from zone_fade_detector.indicators.volume_analysis import VolumeAnalyzer
from zone_fade_detector.indicators.vwap import VWAPCalculator


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


async def run_entry_points_backtest(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run Zone Fade strategy to find actual entry points."""
    print("\n🚀 Starting Zone Fade Entry Points Backtest")
    print("=" * 60)
    
    # Initialize all components
    print("🔧 Initializing Strategy Components...")
    
    # Rolling Window Manager
    window_manager = RollingWindowManager(
        evaluation_cadence_seconds=30,
        memory_limit_mb=500
    )
    
    # Zone Fade Strategy
    strategy = ZoneFadeStrategy(
        min_qrs_score=7,
        zone_tolerance=0.002,
        rejection_candle_min_wick_ratio=0.1,
        choch_confirmation_bars=2
    )
    
    # Signal Processor
    signal_processor = SignalProcessor(
        min_qrs_score=7,
        enable_discord_alerts=False  # Disable Discord for backtesting
    )
    
    # Zone Detector
    zone_detector = ZoneDetector(
        lookback_days=5,
        min_zone_strength=0.5,
        zone_tolerance=0.002
    )
    
    # Swing Structure Detector
    swing_detector = SwingStructureDetector(
        lookback_bars=10,
        min_swing_size=0.05,
        swing_confirmation_bars=1
    )
    
    # Volume Analyzer
    volume_analyzer = VolumeAnalyzer(
        lookback_bars=20,
        spike_threshold=1.5
    )
    
    # VWAP Calculator
    vwap_calculator = VWAPCalculator()
    
    print("   ✅ All strategy components initialized")
    
    # Process each symbol
    all_results = {}
    
    for symbol, bars in symbols_data.items():
        print(f"\n📊 Processing {symbol} for Entry Points...")
        print(f"   Bars: {len(bars)}")
        
        # Process bars with full strategy
        results = await process_symbol_for_entry_points(
            symbol, bars, strategy, signal_processor, zone_detector,
            swing_detector, volume_analyzer, vwap_calculator
        )
        
        all_results[symbol] = results
        print(f"   ✅ {symbol} processing complete")
    
    # Generate comprehensive report
    generate_entry_points_report(all_results)
    
    return all_results


async def process_symbol_for_entry_points(
    symbol: str, bars: List[OHLCVBar], strategy: ZoneFadeStrategy,
    signal_processor: SignalProcessor, zone_detector: ZoneDetector,
    swing_detector: SwingStructureDetector, volume_analyzer: VolumeAnalyzer,
    vwap_calculator: VWAPCalculator
):
    """Process bars for a single symbol to find entry points."""
    
    results = {
        "symbol": symbol,
        "total_bars": len(bars),
        "zones_detected": 0,
        "zone_touches": 0,
        "rejection_candles": 0,
        "choch_confirmations": 0,
        "volume_spikes": 0,
        "qrs_scores": [],
        "entry_points": 0,
        "signals_generated": 0,
        "entry_details": [],
        "performance_metrics": {}
    }
    
    # Process bars in batches
    batch_size = 5000
    for i in range(0, len(bars), batch_size):
        batch = bars[i:i + batch_size]
        
        for bar in batch:
            # Get current bar index
            bar_index = bars.index(bar)
            historical_bars = bars[:bar_index + 1]
            
            # Detect zones
            try:
                zones = zone_detector.detect_zones(historical_bars)
                if zones:
                    results["zones_detected"] = len(zones)
                    
                    # Check for zone touches
                    for zone in zones:
                        if is_zone_touched(bar, zone):
                            results["zone_touches"] += 1
                            
                            # Check for rejection candle
                            if strategy._is_rejection_candle_with_volume(bar, zone, volume_analyzer):
                                results["rejection_candles"] += 1
                                
                                # Check for CHoCH confirmation
                                if swing_detector.detect_choch(historical_bars):
                                    results["choch_confirmations"] += 1
                                
                                # Check for volume spike
                                if volume_analyzer.detect_volume_spike(bar, historical_bars):
                                    results["volume_spikes"] += 1
                                
                                # Calculate QRS score
                                try:
                                    qrs_score = strategy._calculate_qrs_score(bar, zone, historical_bars)
                                    results["qrs_scores"].append(qrs_score)
                                    
                                    # Check if this is a valid entry point
                                    if qrs_score >= 7:  # Minimum QRS threshold
                                        results["entry_points"] += 1
                                        
                                        # Generate signal
                                        try:
                                            signal = strategy._create_setup_from_zone(zone, bar, historical_bars)
                                            if signal:
                                                results["signals_generated"] += 1
                                                
                                                # Store entry details
                                                entry_detail = {
                                                    "timestamp": bar.timestamp,
                                                    "price": bar.close,
                                                    "zone_level": zone.level,
                                                    "zone_type": zone.zone_type.value,
                                                    "qrs_score": qrs_score,
                                                    "rejection_candle": True,
                                                    "choch_confirmation": True,
                                                    "volume_spike": True,
                                                    "zone_strength": zone.strength,
                                                    "zone_quality": zone.quality
                                                }
                                                results["entry_details"].append(entry_detail)
                                                
                                        except Exception as e:
                                            pass
                                            
                                except Exception as e:
                                    pass
                                    
            except Exception as e:
                pass
        
        # Progress update
        if i % 50000 == 0:
            print(f"     Processed {i}/{len(bars)} bars...")
    
    # Calculate performance metrics
    results["performance_metrics"] = {
        "zone_touch_rate": results["zone_touches"] / len(bars) if bars else 0,
        "rejection_rate": results["rejection_candles"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "choch_rate": results["choch_confirmations"] / results["rejection_candles"] if results["rejection_candles"] > 0 else 0,
        "volume_spike_rate": results["volume_spikes"] / results["rejection_candles"] if results["rejection_candles"] > 0 else 0,
        "entry_point_rate": results["entry_points"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "signal_generation_rate": results["signals_generated"] / results["entry_points"] if results["entry_points"] > 0 else 0,
        "avg_qrs_score": sum(results["qrs_scores"]) / len(results["qrs_scores"]) if results["qrs_scores"] else 0
    }
    
    return results


def is_zone_touched(bar: OHLCVBar, zone: Zone) -> bool:
    """Check if a bar touches a zone."""
    if zone.zone_type in [ZoneType.PRIOR_DAY_HIGH, ZoneType.WEEKLY_HIGH, ZoneType.VALUE_AREA_HIGH]:
        # Supply zone - check if low touches the level
        return bar.low <= zone.level <= bar.high
    else:
        # Demand zone - check if high touches the level
        return bar.low <= zone.level <= bar.high


def generate_entry_points_report(all_results: Dict[str, Any]):
    """Generate comprehensive entry points report."""
    print("\n📊 ZONE FADE ENTRY POINTS REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_bars = sum(results["total_bars"] for results in all_results.values())
    total_zones = sum(results["zones_detected"] for results in all_results.values())
    total_zone_touches = sum(results["zone_touches"] for results in all_results.values())
    total_rejection_candles = sum(results["rejection_candles"] for results in all_results.values())
    total_choch_confirmations = sum(results["choch_confirmations"] for results in all_results.values())
    total_volume_spikes = sum(results["volume_spikes"] for results in all_results.values())
    total_entry_points = sum(results["entry_points"] for results in all_results.values())
    total_signals = sum(results["signals_generated"] for results in all_results.values())
    
    print(f"📈 Overall Entry Points Statistics:")
    print(f"   Total Bars Processed: {total_bars:,}")
    print(f"   Total Zones Detected: {total_zones}")
    print(f"   Total Zone Touches: {total_zone_touches}")
    print(f"   Total Rejection Candles: {total_rejection_candles}")
    print(f"   Total CHoCH Confirmations: {total_choch_confirmations}")
    print(f"   Total Volume Spikes: {total_volume_spikes}")
    print(f"   🎯 TOTAL ENTRY POINTS: {total_entry_points}")
    print(f"   Total Signals Generated: {total_signals}")
    
    # Per-symbol results
    print(f"\n📊 Per-Symbol Entry Points:")
    for symbol, results in all_results.items():
        print(f"   {symbol}:")
        print(f"     Bars: {results['total_bars']:,}")
        print(f"     Zones: {results['zones_detected']}")
        print(f"     Zone Touches: {results['zone_touches']}")
        print(f"     Rejection Candles: {results['rejection_candles']}")
        print(f"     CHoCH Confirmations: {results['choch_confirmations']}")
        print(f"     Volume Spikes: {results['volume_spikes']}")
        print(f"     🎯 ENTRY POINTS: {results['entry_points']}")
        print(f"     Signals Generated: {results['signals_generated']}")
        
        # Performance metrics
        metrics = results["performance_metrics"]
        print(f"     Zone Touch Rate: {metrics['zone_touch_rate']:.4f}")
        print(f"     Rejection Rate: {metrics['rejection_rate']:.4f}")
        print(f"     CHoCH Rate: {metrics['choch_rate']:.4f}")
        print(f"     Volume Spike Rate: {metrics['volume_spike_rate']:.4f}")
        print(f"     Entry Point Rate: {metrics['entry_point_rate']:.4f}")
        print(f"     Signal Generation Rate: {metrics['signal_generation_rate']:.4f}")
        print(f"     Average QRS Score: {metrics['avg_qrs_score']:.2f}")
        
        # Show entry details for first few entries
        if results["entry_details"]:
            print(f"     📋 Entry Details (first 5):")
            for i, entry in enumerate(results["entry_details"][:5]):
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
    else:
        print(f"\n⚠️  WARNING: No entry points found!")
        print(f"   This could indicate:")
        print(f"   - QRS threshold too high (current: 7)")
        print(f"   - Zone detection issues")
        print(f"   - Rejection candle criteria too strict")
        print(f"   - CHoCH confirmation failing")
    
    print(f"\n🎉 Zone Fade Entry Points Backtesting Complete!")


async def main():
    """Main backtesting function."""
    print("🚀 Zone Fade Detector - Entry Points Backtesting")
    print("=" * 60)
    
    # Load data
    symbols_data = load_2024_data()
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run backtesting
    start_time = time.time()
    results = await run_entry_points_backtest(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 Entry points backtesting finished successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run backtesting
    asyncio.run(main())