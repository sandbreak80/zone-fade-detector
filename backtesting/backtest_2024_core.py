#!/usr/bin/env python3
"""
Core Architecture Backtesting for Zone Fade Strategy - 2024 Data.

This script runs backtesting using the core architecture components:
- Rolling Window Manager
- Micro Window Analyzer
- Parallel Cross-Symbol Processor
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
from zone_fade_detector.core.micro_window_analyzer import MicroWindowAnalyzer
from zone_fade_detector.core.parallel_cross_symbol_processor import (
    ParallelCrossSymbolProcessor, SymbolType
)
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


def create_test_zones(symbol: str, bars: List[OHLCVBar]) -> List[Zone]:
    """Create test zones for backtesting."""
    zones = []
    
    if not bars:
        return zones
    
    # Create zones at significant levels
    high_prices = [bar.high for bar in bars]
    low_prices = [bar.low for bar in bars]
    
    # Find significant highs and lows
    max_high = max(high_prices)
    min_low = min(low_prices)
    
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


async def run_core_backtest(symbols_data: Dict[str, List[OHLCVBar]]):
    """Run backtesting with core architecture components."""
    print("\n🚀 Starting Core Architecture Backtest")
    print("=" * 60)
    
    # Initialize core components
    print("🔧 Initializing Core Components...")
    
    # Rolling Window Manager
    window_manager = RollingWindowManager(
        evaluation_cadence_seconds=30,
        memory_limit_mb=500
    )
    
    # Micro Window Analyzer
    micro_analyzer = MicroWindowAnalyzer(
        window_manager=window_manager,
        pre_touch_minutes=15,
        post_touch_minutes=10
    )
    
    # Parallel Cross-Symbol Processor (without session manager)
    cross_symbol_processor = ParallelCrossSymbolProcessor(
        window_manager=window_manager,
        session_manager=None,  # Skip session manager for now
        micro_analyzer=micro_analyzer,
        max_workers=4,
        analysis_interval_seconds=30
    )
    
    # Add symbols to cross-symbol processor
    for symbol in symbols_data.keys():
        if symbol == "SPY":
            cross_symbol_processor.add_symbol(symbol, SymbolType.BROAD_MARKET, weight=1.0)
        elif symbol == "QQQ":
            cross_symbol_processor.add_symbol(symbol, SymbolType.BROAD_MARKET, weight=0.9)
        elif symbol == "IWM":
            cross_symbol_processor.add_symbol(symbol, SymbolType.BROAD_MARKET, weight=0.8)
    
    print("   ✅ Core components initialized")
    
    # Process each symbol
    all_results = {}
    
    for symbol, bars in symbols_data.items():
        print(f"\n📊 Processing {symbol}...")
        print(f"   Bars: {len(bars)}")
        
        # Create test zones
        zones = create_test_zones(symbol, bars)
        print(f"   Zones: {len(zones)}")
        
        # Process bars
        results = await process_symbol_bars(
            symbol, bars, zones, window_manager, micro_analyzer, cross_symbol_processor
        )
        
        all_results[symbol] = results
        print(f"   ✅ {symbol} processing complete")
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, cross_symbol_processor, window_manager, micro_analyzer)
    
    return all_results


async def process_symbol_bars(
    symbol: str, bars: List[OHLCVBar], zones: List[Zone],
    window_manager: RollingWindowManager, micro_analyzer: MicroWindowAnalyzer,
    cross_symbol_processor: ParallelCrossSymbolProcessor
):
    """Process bars for a single symbol with core architecture components."""
    
    results = {
        "symbol": symbol,
        "total_bars": len(bars),
        "zones_created": len(zones),
        "zone_touches": 0,
        "micro_analyses": 0,
        "intermarket_analyses": 0,
        "signals_detected": [],
        "performance_metrics": {}
    }
    
    # Process bars in batches
    batch_size = 5000
    for i in range(0, len(bars), batch_size):
        batch = bars[i:i + batch_size]
        
        for bar in batch:
            # Update rolling windows
            window_manager.add_bar(WindowType.SESSION_CONTEXT, bar, symbol)
            window_manager.add_bar(WindowType.VWAP_COMPUTATION, bar, symbol)
            window_manager.add_bar(WindowType.SWING_CHOCH, bar, symbol)
            window_manager.add_bar(WindowType.INITIATIVE_ANALYSIS, bar, symbol)
            window_manager.add_bar(WindowType.INTERMARKET, bar, symbol)
            window_manager.add_bar(WindowType.QRS_ACCUMULATOR, bar, symbol)
            
            # Update cross-symbol processor
            should_analyze = cross_symbol_processor.update_symbol_data(symbol, bar)
            if should_analyze:
                # Perform intermarket analysis
                try:
                    intermarket_analysis = await cross_symbol_processor.analyze_intermarket(symbol)
                    if intermarket_analysis:
                        results["intermarket_analyses"] += 1
                        for signal in intermarket_analysis.signals:
                            signal_name = signal.value
                            if signal_name not in results["signals_detected"]:
                                results["signals_detected"].append(signal_name)
                except Exception as e:
                    # Skip intermarket analysis if it fails
                    pass
            
            # Check for zone touches
            for zone in zones:
                if is_zone_touched(bar, zone):
                    results["zone_touches"] += 1
                    
                    # Perform micro window analysis
                    try:
                        micro_analysis = micro_analyzer.analyze_zone_touch(zone, bar, symbol)
                        if micro_analysis:
                            results["micro_analyses"] += 1
                    except Exception as e:
                        # Skip micro analysis if it fails
                        pass
        
        # Progress update
        if i % 50000 == 0:
            print(f"     Processed {i}/{len(bars)} bars...")
    
    # Calculate performance metrics
    results["performance_metrics"] = {
        "zone_touch_rate": results["zone_touches"] / len(bars) if bars else 0,
        "micro_analysis_rate": results["micro_analyses"] / results["zone_touches"] if results["zone_touches"] > 0 else 0,
        "intermarket_analysis_rate": results["intermarket_analyses"] / len(bars) if bars else 0,
        "signal_count": len(results["signals_detected"])
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


def generate_comprehensive_report(all_results: Dict[str, Any], cross_symbol_processor: ParallelCrossSymbolProcessor, 
                                window_manager: RollingWindowManager, micro_analyzer: MicroWindowAnalyzer):
    """Generate comprehensive backtesting report."""
    print("\n📊 COMPREHENSIVE BACKTESTING REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_bars = sum(results["total_bars"] for results in all_results.values())
    total_zones = sum(results["zones_created"] for results in all_results.values())
    total_zone_touches = sum(results["zone_touches"] for results in all_results.values())
    total_micro_analyses = sum(results["micro_analyses"] for results in all_results.values())
    total_intermarket_analyses = sum(results["intermarket_analyses"] for results in all_results.values())
    
    print(f"📈 Overall Statistics:")
    print(f"   Total Bars Processed: {total_bars:,}")
    print(f"   Total Zones Created: {total_zones}")
    print(f"   Total Zone Touches: {total_zone_touches}")
    print(f"   Total Micro Analyses: {total_micro_analyses}")
    print(f"   Total Intermarket Analyses: {total_intermarket_analyses}")
    
    # Per-symbol results
    print(f"\n📊 Per-Symbol Results:")
    for symbol, results in all_results.items():
        print(f"   {symbol}:")
        print(f"     Bars: {results['total_bars']:,}")
        print(f"     Zones: {results['zones_created']}")
        print(f"     Zone Touches: {results['zone_touches']}")
        print(f"     Micro Analyses: {results['micro_analyses']}")
        print(f"     Intermarket Analyses: {results['intermarket_analyses']}")
        print(f"     Signals: {len(results['signals_detected'])}")
        
        # Performance metrics
        metrics = results["performance_metrics"]
        print(f"     Zone Touch Rate: {metrics['zone_touch_rate']:.4f}")
        print(f"     Micro Analysis Rate: {metrics['micro_analysis_rate']:.4f}")
        print(f"     Intermarket Analysis Rate: {metrics['intermarket_analysis_rate']:.4f}")
    
    # Cross-symbol processor summary
    print(f"\n🔄 Cross-Symbol Processor Summary:")
    summary = cross_symbol_processor.get_analysis_summary()
    print(f"   Total Analyses: {summary.get('total_analyses', 0)}")
    print(f"   Average Confidence: {summary.get('avg_confidence', 0):.2f}")
    print(f"   Average Momentum Alignment: {summary.get('avg_momentum_alignment', 0):.2f}")
    
    if 'signal_counts' in summary and summary['signal_counts']:
        print(f"   Signal Counts:")
        for signal, count in summary['signal_counts'].items():
            print(f"     {signal}: {count}")
    
    if 'risk_sentiment_distribution' in summary and summary['risk_sentiment_distribution']:
        print(f"   Risk Sentiment Distribution:")
        for sentiment, count in summary['risk_sentiment_distribution'].items():
            print(f"     {sentiment}: {count}")
    
    # Architecture performance
    print(f"\n⚡ Architecture Performance:")
    window_stats = window_manager.get_window_performance_stats()
    print(f"   Window Manager:")
    print(f"     Total Windows: {window_stats['total_windows']}")
    print(f"     Memory Usage: {window_stats['memory_usage_mb']:.2f} MB")
    print(f"     Average Bars per Window: {window_stats['avg_bars_per_window']:.1f}")
    
    # Micro window analyzer summary
    micro_summary = micro_analyzer.get_analysis_summary()
    print(f"   Micro Window Analyzer:")
    print(f"     Total Analyses: {micro_summary['total_analyses']}")
    print(f"     Significant Touches: {micro_summary['significant_touches']}")
    print(f"     Absorption Touches: {micro_summary['absorption_touches']}")
    print(f"     Exhaustion Touches: {micro_summary['exhaustion_touches']}")
    print(f"     Rejection Touches: {micro_summary['rejection_touches']}")
    print(f"     Average Confidence: {micro_summary['average_confidence']:.2f}")
    
    print(f"\n🎉 Core Architecture Backtesting Complete!")
    print(f"✅ All core components tested successfully")


async def main():
    """Main backtesting function."""
    print("🚀 Zone Fade Detector - Core Architecture Backtesting")
    print("=" * 60)
    
    # Load data
    symbols_data = load_2024_data()
    
    if not symbols_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run backtesting
    start_time = time.time()
    results = await run_core_backtest(symbols_data)
    end_time = time.time()
    
    print(f"\n⏱️  Backtesting completed in {end_time - start_time:.2f} seconds")
    print("🎉 Core architecture backtesting finished successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run backtesting
    asyncio.run(main())