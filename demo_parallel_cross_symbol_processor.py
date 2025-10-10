#!/usr/bin/env python3
"""
Demonstration of Parallel Cross-Symbol Processor for Zone Fade Strategy.

This script demonstrates how the Parallel Cross-Symbol Processor provides
real-time intermarket analysis by processing multiple symbols in parallel,
enabling detection of divergences, correlations, and market-wide patterns.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.session_state_manager import SessionStateManager
from zone_fade_detector.core.micro_window_analyzer import MicroWindowAnalyzer
from zone_fade_detector.core.parallel_cross_symbol_processor import (
    ParallelCrossSymbolProcessor, IntermarketSignal, SymbolType
)
from zone_fade_detector.core.models import OHLCVBar


def create_intermarket_scenarios():
    """Create different intermarket scenarios for demonstration."""
    print("📊 Creating Intermarket Scenarios...")
    
    scenarios = {}
    base_time = datetime(2024, 1, 2, 9, 30)
    
    # Scenario 1: Bullish Divergence (SPY up, QQQ down)
    spy_bars = []
    qqq_bars = []
    for i in range(50):
        # SPY trending up
        spy_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 0.02,
            high=100.5 + i * 0.02,
            low=99.5 + i * 0.02,
            close=100.2 + i * 0.02,
            volume=1000 + i * 20
        )
        spy_bars.append(spy_bar)
        
        # QQQ trending down (divergence)
        qqq_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=200.0 - i * 0.01,
            high=200.3 - i * 0.01,
            low=199.7 - i * 0.01,
            close=200.1 - i * 0.01,
            volume=800 + i * 15
        )
        qqq_bars.append(qqq_bar)
    
    scenarios["Bullish Divergence"] = {
        "SPY": spy_bars,
        "QQQ": qqq_bars
    }
    
    # Scenario 2: Risk Off (Bonds up, Stocks down)
    base_time = datetime(2024, 1, 2, 10, 0)
    spy_bars = []
    tlt_bars = []
    for i in range(50):
        # SPY trending down
        spy_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=102.0 - i * 0.015,
            high=102.3 - i * 0.015,
            low=101.7 - i * 0.015,
            close=102.1 - i * 0.015,
            volume=1200 + i * 25
        )
        spy_bars.append(spy_bar)
        
        # TLT trending up (risk off)
        tlt_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=120.0 + i * 0.02,
            high=120.3 + i * 0.02,
            low=119.7 + i * 0.02,
            close=120.1 + i * 0.02,
            volume=600 + i * 12
        )
        tlt_bars.append(tlt_bar)
    
    scenarios["Risk Off"] = {
        "SPY": spy_bars,
        "TLT": tlt_bars
    }
    
    # Scenario 3: Volatility Spike
    base_time = datetime(2024, 1, 2, 11, 0)
    spy_bars = []
    vix_bars = []
    for i in range(50):
        # SPY with high volatility
        volatility = 0.05 if 20 <= i <= 30 else 0.02
        spy_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.0 + i * 0.01 + volatility,
            low=100.0 + i * 0.01 - volatility,
            close=100.0 + i * 0.01 + volatility * 0.5,
            volume=1500 + i * 30
        )
        spy_bars.append(spy_bar)
        
        # VIX spiking
        vix_level = 20.0 + (i - 25) ** 2 * 0.1 if 20 <= i <= 30 else 20.0
        vix_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=vix_level,
            high=vix_level + 0.5,
            low=vix_level - 0.5,
            close=vix_level + 0.2,
            volume=400 + i * 8
        )
        vix_bars.append(vix_bar)
    
    scenarios["Volatility Spike"] = {
        "SPY": spy_bars,
        "VIX": vix_bars
    }
    
    # Scenario 4: Sector Rotation
    base_time = datetime(2024, 1, 2, 12, 0)
    xlk_bars = []
    xlf_bars = []
    for i in range(50):
        # XLK (Tech) strong
        xlk_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=150.0 + i * 0.03,
            high=150.5 + i * 0.03,
            low=149.5 + i * 0.03,
            close=150.2 + i * 0.03,
            volume=800 + i * 16
        )
        xlk_bars.append(xlk_bar)
        
        # XLF (Financials) weak
        xlf_bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=40.0 + i * 0.005,
            high=40.2 + i * 0.005,
            low=39.8 + i * 0.005,
            close=40.1 + i * 0.005,
            volume=600 + i * 12
        )
        xlf_bars.append(xlf_bar)
    
    scenarios["Sector Rotation"] = {
        "XLK": xlk_bars,
        "XLF": xlf_bars
    }
    
    print(f"   ✅ Created {len(scenarios)} intermarket scenarios")
    for scenario_name, symbols_data in scenarios.items():
        print(f"     {scenario_name}: {list(symbols_data.keys())}")
    
    return scenarios


def demonstrate_parallel_processing(processor, scenarios):
    """Demonstrate parallel processing of multiple symbols."""
    print("\n🔄 Demonstrating Parallel Processing")
    print("=" * 60)
    
    # Add symbols to processor
    symbols_to_add = [
        ("SPY", SymbolType.BROAD_MARKET, 1.0),
        ("QQQ", SymbolType.BROAD_MARKET, 0.9),
        ("IWM", SymbolType.BROAD_MARKET, 0.8),
        ("VIX", SymbolType.VOLATILITY, 0.7),
        ("TLT", SymbolType.BOND, 0.6),
        ("XLK", SymbolType.SECTOR, 0.5),
        ("XLF", SymbolType.SECTOR, 0.4)
    ]
    
    for symbol, symbol_type, weight in symbols_to_add:
        processor.add_symbol(symbol, symbol_type, weight)
    
    print(f"   Added {len(processor.symbol_configs)} symbols to processor")
    
    # Process each scenario
    for scenario_name, symbols_data in scenarios.items():
        print(f"\n   📊 Processing {scenario_name} scenario:")
        
        # Update symbols with data
        for symbol, bars in symbols_data.items():
            for bar in bars:
                processor.update_symbol_data(symbol, bar)
        
        # Check ready symbols
        ready_symbols = [s for s, w in processor.cross_symbol_windows.items() if w.is_ready]
        print(f"     Ready symbols: {ready_symbols}")
        
        # Show symbol metrics
        for symbol in ready_symbols:
            window = processor.cross_symbol_windows[symbol]
            if window.metrics:
                metrics = window.metrics
                print(f"       {symbol}: {metrics.trend_direction}, "
                      f"change: {metrics.price_change:.2%}, "
                      f"volatility: {metrics.volatility:.2%}")


async def demonstrate_intermarket_analysis(processor, scenarios):
    """Demonstrate intermarket analysis for different scenarios."""
    print("\n🎯 Demonstrating Intermarket Analysis")
    print("=" * 60)
    
    analyses = []
    
    for scenario_name, symbols_data in scenarios.items():
        print(f"\n   📊 Analyzing {scenario_name} scenario:")
        
        # Update all symbols with data
        for symbol, bars in symbols_data.items():
            for bar in bars:
                processor.update_symbol_data(symbol, bar)
        
        # Perform analysis
        analysis = await processor.analyze_intermarket("SPY")
        
        if analysis:
            analyses.append(analysis)
            
            print(f"     Analysis completed:")
            print(f"       Primary Symbol: {analysis.primary_symbol}")
            print(f"       Timestamp: {analysis.timestamp}")
            print(f"       Signals: {len(analysis.signals)}")
            print(f"       Confidence: {analysis.confidence_score:.2f}")
            print(f"       Momentum Alignment: {analysis.momentum_alignment:.2f}")
            print(f"       Risk Sentiment: {analysis.risk_sentiment}")
            print(f"       Volatility Regime: {analysis.volatility_regime}")
            
            # Show signals
            if analysis.signals:
                print(f"       Signals detected:")
                for signal in analysis.signals:
                    print(f"         - {signal.value}")
            
            # Show correlations
            if analysis.correlations:
                print(f"       Correlations:")
                for pair, correlation in list(analysis.correlations.items())[:3]:
                    print(f"         {pair}: {correlation:.2f}")
            
            # Show divergences
            if analysis.divergences:
                print(f"       Divergences:")
                for div in analysis.divergences:
                    print(f"         {div[0]} vs {div[1]}: {div[2]}")
            
            # Show sector rotation
            if analysis.sector_rotation:
                print(f"       Sector Rotation:")
                for sector, strength in analysis.sector_rotation.items():
                    print(f"         {sector}: {strength:.2f}")
            
            # Market context
            context = analysis.market_context
            print(f"       Market Context:")
            print(f"         Trend Day: {context.is_trend_day}")
            print(f"         VWAP Slope: {context.vwap_slope:.4f}")
            print(f"         Market Balance: {context.market_balance:.2f}")
            print(f"         Volatility Regime: {context.volatility_regime}")
        else:
            print(f"     ❌ Analysis failed for {scenario_name}")
    
    return analyses


def demonstrate_signal_detection(processor):
    """Demonstrate specific signal detection capabilities."""
    print("\n📊 Demonstrating Signal Detection")
    print("=" * 60)
    
    # Test different signal types
    signal_tests = [
        ("Bullish Divergence", {
            "SPY": {"price_change": 0.03, "trend_direction": "bullish"},
            "QQQ": {"price_change": -0.01, "trend_direction": "bearish"}
        }),
        ("Bearish Divergence", {
            "SPY": {"price_change": -0.02, "trend_direction": "bearish"},
            "QQQ": {"price_change": 0.02, "trend_direction": "bullish"}
        }),
        ("Risk Off", {
            "TLT": {"price_change": 0.025, "trend_direction": "bullish"},
            "SPY": {"price_change": -0.02, "trend_direction": "bearish"}
        }),
        ("Risk On", {
            "TLT": {"price_change": -0.015, "trend_direction": "bearish"},
            "SPY": {"price_change": 0.025, "trend_direction": "bullish"}
        }),
        ("Volatility Spike", {
            "VIX": {"volatility": 0.08, "price_change": 0.15}
        }),
        ("Volatility Suppression", {
            "VIX": {"volatility": 0.005, "price_change": -0.02}
        })
    ]
    
    for test_name, symbol_data in signal_tests:
        print(f"\n   🔍 Testing {test_name}:")
        
        # Create mock metrics
        symbol_metrics = {}
        for symbol, data in symbol_data.items():
            metrics = type('MockMetrics', (), {
                'symbol': symbol,
                'price_change': data.get('price_change', 0.01),
                'volume_ratio': 1.0,
                'momentum': data.get('price_change', 0.01) * 0.5,
                'volatility': data.get('volatility', 0.02),
                'relative_strength': 0.5,
                'trend_direction': data.get('trend_direction', 'neutral'),
                'is_outlier': data.get('volatility', 0.02) > 0.05,
                'correlation_score': 0.7
            })()
            symbol_metrics[symbol] = metrics
        
        # Test specific signal detection
        if "Divergence" in test_name:
            divergences = processor._detect_divergences(symbol_metrics)
            print(f"     Divergences: {len(divergences)}")
            for div in divergences:
                print(f"       {div[0]} vs {div[1]}: {div[2]}")
        
        elif "Risk" in test_name:
            bonds = {k: v for k, v in symbol_metrics.items() if k == "TLT"}
            broad_market = {k: v for k, v in symbol_metrics.items() if k == "SPY"}
            signals = processor._detect_risk_sentiment_signals(bonds, broad_market)
            print(f"     Risk signals: {[s.value for s in signals]}")
        
        elif "Volatility" in test_name:
            volatility = {k: v for k, v in symbol_metrics.items() if k == "VIX"}
            signals = processor._detect_volatility_signals(volatility)
            print(f"     Volatility signals: {[s.value for s in signals]}")


def demonstrate_analysis_filtering(processor):
    """Demonstrate analysis filtering and summary capabilities."""
    print("\n🔍 Demonstrating Analysis Filtering")
    print("=" * 60)
    
    # Get recent analyses
    recent = processor.get_recent_analyses(limit=10)
    print(f"   Recent analyses: {len(recent)}")
    
    if recent:
        print(f"   Latest analysis:")
        latest = recent[-1]
        print(f"     Primary Symbol: {latest.primary_symbol}")
        print(f"     Signals: {len(latest.signals)}")
        print(f"     Confidence: {latest.confidence_score:.2f}")
        print(f"     Risk Sentiment: {latest.risk_sentiment}")
        print(f"     Volatility Regime: {latest.volatility_regime}")
    
    # Get signal frequencies
    print(f"\n   Signal Frequencies (last 24h):")
    signal_frequencies = {}
    for signal in IntermarketSignal:
        frequency = processor.get_signal_frequency(signal, hours=24)
        if frequency > 0:
            signal_frequencies[signal.value] = frequency
    
    if signal_frequencies:
        for signal, frequency in signal_frequencies.items():
            print(f"     {signal}: {frequency}")
    else:
        print(f"     No signals detected in last 24h")
    
    # Get analysis summary
    summary = processor.get_analysis_summary()
    print(f"\n   Analysis Summary:")
    print(f"     Total Analyses: {summary.get('total_analyses', 0)}")
    print(f"     Average Confidence: {summary.get('avg_confidence', 0):.2f}")
    print(f"     Average Momentum Alignment: {summary.get('avg_momentum_alignment', 0):.2f}")
    
    if 'signal_counts' in summary and summary['signal_counts']:
        print(f"     Signal Counts:")
        for signal, count in summary['signal_counts'].items():
            print(f"       {signal}: {count}")
    
    if 'risk_sentiment_distribution' in summary and summary['risk_sentiment_distribution']:
        print(f"     Risk Sentiment Distribution:")
        for sentiment, count in summary['risk_sentiment_distribution'].items():
            print(f"       {sentiment}: {count}")


async def demonstrate_async_operations(processor, scenarios):
    """Demonstrate async operations and continuous analysis."""
    print("\n🔄 Demonstrating Async Operations")
    print("=" * 60)
    
    # Test continuous analysis loop
    print("   Testing continuous analysis loop...")
    
    # Start analysis loop
    analysis_task = asyncio.create_task(processor.start_analysis_loop())
    
    # Let it run for a short time
    await asyncio.sleep(3)
    
    # Stop the loop
    processor.stop_analysis_loop()
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(analysis_task, timeout=1)
    except asyncio.TimeoutError:
        analysis_task.cancel()
    
    print("   ✅ Continuous analysis loop completed")
    
    # Test parallel analysis
    print("   Testing parallel analysis...")
    
    # Create multiple analysis tasks
    tasks = []
    for i in range(5):
        task = asyncio.create_task(processor.analyze_intermarket("SPY"))
        tasks.append(task)
    
    # Wait for all analyses to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_analyses = [r for r in results if not isinstance(r, Exception)]
    print(f"   ✅ Completed {len(successful_analyses)} parallel analyses")
    
    # Test concurrent symbol updates
    print("   Testing concurrent symbol updates...")
    
    # Update multiple symbols concurrently
    update_tasks = []
    for scenario_name, symbols_data in scenarios.items():
        for symbol, bars in symbols_data.items():
            for bar in bars[:10]:  # Use first 10 bars
                task = asyncio.create_task(
                    update_symbol_async(processor, symbol, bar)
                )
                update_tasks.append(task)
    
    # Wait for all updates to complete
    await asyncio.gather(*update_tasks)
    
    print("   ✅ Concurrent symbol updates completed")


async def update_symbol_async(processor, symbol, bar):
    """Async wrapper for symbol update."""
    await asyncio.sleep(0.001)  # Simulate processing time
    processor.update_symbol_data(symbol, bar)


def demonstrate_performance_metrics(processor):
    """Demonstrate performance metrics and monitoring."""
    print("\n📊 Demonstrating Performance Metrics")
    print("=" * 60)
    
    # Get processor configuration
    print(f"   Processor Configuration:")
    print(f"     Max Workers: {processor.max_workers}")
    print(f"     Analysis Interval: {processor.analysis_interval}s")
    print(f"     Correlation Threshold: {processor.correlation_threshold}")
    print(f"     Divergence Threshold: {processor.divergence_threshold}")
    
    # Get symbol statistics
    print(f"\n   Symbol Statistics:")
    print(f"     Total Symbols: {len(processor.symbol_configs)}")
    print(f"     Ready Symbols: {len([s for s, w in processor.cross_symbol_windows.items() if w.is_ready])}")
    
    for symbol, config in processor.symbol_configs.items():
        window = processor.cross_symbol_windows[symbol]
        print(f"       {symbol} ({config.symbol_type.value}):")
        print(f"         Weight: {config.weight}")
        print(f"         Enabled: {config.enabled}")
        print(f"         Ready: {window.is_ready}")
        print(f"         Bars: {len(window.bars)}")
        if window.last_update:
            print(f"         Last Update: {window.last_update}")
    
    # Get analysis statistics
    summary = processor.get_analysis_summary()
    print(f"\n   Analysis Statistics:")
    print(f"     Total Analyses: {summary.get('total_analyses', 0)}")
    print(f"     Average Confidence: {summary.get('avg_confidence', 0):.2f}")
    print(f"     Average Momentum Alignment: {summary.get('avg_momentum_alignment', 0):.2f}")
    
    if summary.get('last_analysis_time'):
        print(f"     Last Analysis: {summary['last_analysis_time']}")


async def main():
    """Main demonstration function."""
    print("🚀 Parallel Cross-Symbol Processor Demonstration")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing components...")
    window_manager = RollingWindowManager(
        evaluation_cadence_seconds=5,
        memory_limit_mb=200
    )
    session_manager = SessionStateManager(window_manager, timezone_offset_hours=-5)
    micro_analyzer = MicroWindowAnalyzer(window_manager)
    processor = ParallelCrossSymbolProcessor(
        window_manager=window_manager,
        session_manager=session_manager,
        micro_analyzer=micro_analyzer,
        max_workers=4,
        analysis_interval_seconds=10
    )
    print("   ✅ Components initialized")
    
    # Create test scenarios
    scenarios = create_intermarket_scenarios()
    
    # Run demonstrations
    demonstrate_parallel_processing(processor, scenarios)
    analyses = await demonstrate_intermarket_analysis(processor, scenarios)
    demonstrate_signal_detection(processor)
    demonstrate_analysis_filtering(processor)
    await demonstrate_async_operations(processor, scenarios)
    demonstrate_performance_metrics(processor)
    
    # Final statistics
    print("\n📊 Final Statistics")
    print("=" * 60)
    
    summary = processor.get_analysis_summary()
    print(f"Total Analyses: {summary.get('total_analyses', 0)}")
    print(f"Average Confidence: {summary.get('avg_confidence', 0):.2f}")
    print(f"Average Momentum Alignment: {summary.get('avg_momentum_alignment', 0):.2f}")
    
    if 'signal_counts' in summary and summary['signal_counts']:
        print(f"Signal Counts:")
        for signal, count in summary['signal_counts'].items():
            print(f"  {signal}: {count}")
    
    if 'risk_sentiment_distribution' in summary and summary['risk_sentiment_distribution']:
        print(f"Risk Sentiment Distribution:")
        for sentiment, count in summary['risk_sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")
    
    print("\n🎉 Parallel Cross-Symbol Processor Demonstration Complete!")
    print("✅ All demonstrations completed successfully")


if __name__ == "__main__":
    asyncio.run(main())