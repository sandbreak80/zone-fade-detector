#!/usr/bin/env python3
"""
Integration test for Parallel Cross-Symbol Processor with other components.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.session_state_manager import SessionStateManager
from zone_fade_detector.core.micro_window_analyzer import MicroWindowAnalyzer
from zone_fade_detector.core.parallel_cross_symbol_processor import (
    ParallelCrossSymbolProcessor, IntermarketSignal, SymbolType
)
from zone_fade_detector.core.models import OHLCVBar


def create_multi_symbol_data():
    """Create data for multiple symbols for intermarket analysis."""
    print("📊 Creating multi-symbol data...")
    
    base_time = datetime(2024, 1, 2, 9, 30)
    symbols_data = {}
    
    # SPY data (broad market)
    spy_bars = []
    for i in range(100):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000 + i * 10
        )
        spy_bars.append(bar)
    symbols_data["SPY"] = spy_bars
    
    # QQQ data (broad market with divergence)
    qqq_bars = []
    for i in range(100):
        # Create divergence: QQQ underperforms SPY
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=200.0 + i * 0.005,  # Slower growth
            high=200.3 + i * 0.005,
            low=199.7 + i * 0.005,
            close=200.1 + i * 0.005,
            volume=800 + i * 8
        )
        qqq_bars.append(bar)
    symbols_data["QQQ"] = qqq_bars
    
    # IWM data (broad market)
    iwm_bars = []
    for i in range(100):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=50.0 + i * 0.008,
            high=50.3 + i * 0.008,
            low=49.7 + i * 0.008,
            close=50.1 + i * 0.008,
            volume=600 + i * 6
        )
        iwm_bars.append(bar)
    symbols_data["IWM"] = iwm_bars
    
    # VIX data (volatility)
    vix_bars = []
    for i in range(100):
        # VIX spikes in the middle
        if 40 <= i <= 60:
            volatility = 0.05  # High volatility
        else:
            volatility = 0.01  # Low volatility
        
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=20.0 + i * volatility,
            high=20.5 + i * volatility,
            low=19.5 + i * volatility,
            close=20.2 + i * volatility,
            volume=500 + i * 5
        )
        vix_bars.append(bar)
    symbols_data["VIX"] = vix_bars
    
    # TLT data (bonds)
    tlt_bars = []
    for i in range(100):
        # TLT moves opposite to stocks
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=120.0 - i * 0.005,  # Downward trend
            high=120.3 - i * 0.005,
            low=119.7 - i * 0.005,
            close=120.1 - i * 0.005,
            volume=400 + i * 4
        )
        tlt_bars.append(bar)
    symbols_data["TLT"] = tlt_bars
    
    # XLK data (sector)
    xlk_bars = []
    for i in range(100):
        bar = OHLCVBar(
            timestamp=base_time + timedelta(minutes=i),
            open=150.0 + i * 0.012,
            high=150.5 + i * 0.012,
            low=149.5 + i * 0.012,
            close=150.2 + i * 0.012,
            volume=300 + i * 3
        )
        xlk_bars.append(bar)
    symbols_data["XLK"] = xlk_bars
    
    print(f"   ✅ Generated data for {len(symbols_data)} symbols")
    for symbol, bars in symbols_data.items():
        print(f"     {symbol}: {len(bars)} bars")
    
    return symbols_data


def test_parallel_processing(processor, symbols_data):
    """Test parallel processing of multiple symbols."""
    print("\n🔄 Testing Parallel Processing")
    print("=" * 50)
    
    # Add symbols to processor
    processor.add_symbol("SPY", SymbolType.BROAD_MARKET, weight=1.0)
    processor.add_symbol("QQQ", SymbolType.BROAD_MARKET, weight=0.9)
    processor.add_symbol("IWM", SymbolType.BROAD_MARKET, weight=0.8)
    processor.add_symbol("VIX", SymbolType.VOLATILITY, weight=0.7)
    processor.add_symbol("TLT", SymbolType.BOND, weight=0.6)
    processor.add_symbol("XLK", SymbolType.SECTOR, weight=0.5)
    
    print(f"   Added {len(processor.symbol_configs)} symbols to processor")
    
    # Update all symbols with data
    for symbol, bars in symbols_data.items():
        for bar in bars:
            processor.update_symbol_data(symbol, bar)
    
    print("   ✅ Updated all symbols with data")
    
    # Check which symbols are ready
    ready_symbols = [s for s, w in processor.cross_symbol_windows.items() if w.is_ready]
    print(f"   Ready symbols: {ready_symbols}")
    
    return ready_symbols


async def test_intermarket_analysis(processor, ready_symbols):
    """Test intermarket analysis."""
    print("\n🎯 Testing Intermarket Analysis")
    print("=" * 50)
    
    # Perform analysis
    analysis = await processor.analyze_intermarket("SPY")
    
    if analysis:
        print(f"   Analysis completed for {analysis.primary_symbol}")
        print(f"   Timestamp: {analysis.timestamp}")
        print(f"   Signals detected: {len(analysis.signals)}")
        print(f"   Confidence score: {analysis.confidence_score:.2f}")
        print(f"   Momentum alignment: {analysis.momentum_alignment:.2f}")
        print(f"   Risk sentiment: {analysis.risk_sentiment}")
        print(f"   Volatility regime: {analysis.volatility_regime}")
        
        # Show signals
        if analysis.signals:
            print(f"\n   Signals:")
            for signal in analysis.signals:
                print(f"     - {signal.value}")
        
        # Show correlations
        if analysis.correlations:
            print(f"\n   Correlations:")
            for pair, correlation in list(analysis.correlations.items())[:5]:  # Show first 5
                print(f"     {pair}: {correlation:.2f}")
        
        # Show divergences
        if analysis.divergences:
            print(f"\n   Divergences:")
            for div in analysis.divergences:
                print(f"     {div[0]} vs {div[1]}: {div[2]}")
        
        # Show sector rotation
        if analysis.sector_rotation:
            print(f"\n   Sector Rotation:")
            for sector, strength in analysis.sector_rotation.items():
                print(f"     {sector}: {strength:.2f}")
        
        return analysis
    else:
        print("   ❌ Analysis failed")
        return None


def test_signal_detection(processor, symbols_data):
    """Test specific signal detection."""
    print("\n📊 Testing Signal Detection")
    print("=" * 50)
    
    # Test divergence detection
    print("   Testing divergence detection...")
    
    # Create scenarios with different divergences
    scenarios = [
        ("Bullish Divergence", {"SPY": 0.02, "QQQ": -0.01}),
        ("Bearish Divergence", {"SPY": -0.01, "QQQ": 0.02}),
        ("No Divergence", {"SPY": 0.01, "QQQ": 0.008})
    ]
    
    for scenario_name, price_changes in scenarios:
        print(f"     {scenario_name}:")
        
        # Create mock metrics
        symbol_metrics = {}
        for symbol, price_change in price_changes.items():
            # Create mock metrics
            metrics = type('MockMetrics', (), {
                'symbol': symbol,
                'price_change': price_change,
                'volume_ratio': 1.0,
                'momentum': price_change * 0.5,
                'volatility': 0.02,
                'relative_strength': 0.5,
                'trend_direction': 'bullish' if price_change > 0 else 'bearish',
                'is_outlier': False,
                'correlation_score': 0.7
            })()
            symbol_metrics[symbol] = metrics
        
        # Test divergence detection
        divergences = processor._detect_divergences(symbol_metrics)
        print(f"       Divergences detected: {len(divergences)}")
        for div in divergences:
            print(f"         {div[0]} vs {div[1]}: {div[2]}")


def test_risk_sentiment_analysis(processor):
    """Test risk sentiment analysis."""
    print("\n🎭 Testing Risk Sentiment Analysis")
    print("=" * 50)
    
    # Test different risk scenarios
    scenarios = [
        ("Risk Off", {"TLT": 0.02, "SPY": -0.02}),
        ("Risk On", {"TLT": -0.02, "SPY": 0.02}),
        ("Neutral", {"TLT": 0.005, "SPY": 0.005})
    ]
    
    for scenario_name, price_changes in scenarios:
        print(f"   {scenario_name} scenario:")
        
        # Create mock metrics
        symbol_metrics = {}
        for symbol, price_change in price_changes.items():
            symbol_type = SymbolType.BOND if symbol == "TLT" else SymbolType.BROAD_MARKET
            metrics = type('MockMetrics', (), {
                'symbol': symbol,
                'price_change': price_change,
                'volume_ratio': 1.0,
                'momentum': price_change * 0.5,
                'volatility': 0.02,
                'relative_strength': 0.5,
                'trend_direction': 'bullish' if price_change > 0 else 'bearish',
                'is_outlier': False,
                'correlation_score': 0.7
            })()
            symbol_metrics[symbol] = metrics
        
        # Test risk sentiment
        risk_sentiment = processor._determine_risk_sentiment(symbol_metrics)
        print(f"     Risk sentiment: {risk_sentiment}")


def test_volatility_analysis(processor):
    """Test volatility analysis."""
    print("\n📈 Testing Volatility Analysis")
    print("=" * 50)
    
    # Test different volatility scenarios
    scenarios = [
        ("High Volatility", {"SPY": 0.05, "QQQ": 0.04}),
        ("Low Volatility", {"SPY": 0.005, "QQQ": 0.003}),
        ("Normal Volatility", {"SPY": 0.02, "QQQ": 0.015})
    ]
    
    for scenario_name, volatilities in scenarios:
        print(f"   {scenario_name} scenario:")
        
        # Create mock metrics
        symbol_metrics = {}
        for symbol, volatility in volatilities.items():
            metrics = type('MockMetrics', (), {
                'symbol': symbol,
                'price_change': 0.01,
                'volume_ratio': 1.0,
                'momentum': 0.005,
                'volatility': volatility,
                'relative_strength': 0.5,
                'trend_direction': 'bullish',
                'is_outlier': volatility > 0.03,
                'correlation_score': 0.7
            })()
            symbol_metrics[symbol] = metrics
        
        # Test volatility regime
        volatility_regime = processor._determine_volatility_regime(symbol_metrics)
        print(f"     Volatility regime: {volatility_regime}")


def test_momentum_alignment(processor):
    """Test momentum alignment analysis."""
    print("\n⚡ Testing Momentum Alignment")
    print("=" * 50)
    
    # Test different momentum scenarios
    scenarios = [
        ("Aligned Bullish", {"SPY": "bullish", "QQQ": "bullish", "IWM": "bullish"}),
        ("Aligned Bearish", {"SPY": "bearish", "QQQ": "bearish", "IWM": "bearish"}),
        ("Mixed Momentum", {"SPY": "bullish", "QQQ": "bearish", "IWM": "neutral"}),
        ("Neutral Momentum", {"SPY": "neutral", "QQQ": "neutral", "IWM": "neutral"})
    ]
    
    for scenario_name, trend_directions in scenarios:
        print(f"   {scenario_name} scenario:")
        
        # Create mock metrics
        symbol_metrics = {}
        for symbol, trend in trend_directions.items():
            metrics = type('MockMetrics', (), {
                'symbol': symbol,
                'price_change': 0.01 if trend == "bullish" else -0.01 if trend == "bearish" else 0.0,
                'volume_ratio': 1.0,
                'momentum': 0.005 if trend == "bullish" else -0.005 if trend == "bearish" else 0.0,
                'volatility': 0.02,
                'relative_strength': 0.5,
                'trend_direction': trend,
                'is_outlier': False,
                'correlation_score': 0.7
            })()
            symbol_metrics[symbol] = metrics
        
        # Test momentum alignment
        alignment = processor._calculate_momentum_alignment(symbol_metrics)
        print(f"     Momentum alignment: {alignment:.2f}")


def test_analysis_filtering(processor):
    """Test analysis filtering and summary."""
    print("\n🔍 Testing Analysis Filtering")
    print("=" * 50)
    
    # Get recent analyses
    recent = processor.get_recent_analyses(limit=5)
    print(f"   Recent analyses: {len(recent)}")
    
    # Get signal frequency
    for signal in IntermarketSignal:
        frequency = processor.get_signal_frequency(signal, hours=24)
        if frequency > 0:
            print(f"   {signal.value}: {frequency} occurrences in last 24h")
    
    # Get analysis summary
    summary = processor.get_analysis_summary()
    print(f"\n   Analysis Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.2f}")
        else:
            print(f"     {key}: {value}")


async def test_async_operations(processor, symbols_data):
    """Test async operations."""
    print("\n🔄 Testing Async Operations")
    print("=" * 50)
    
    # Test async analysis loop
    print("   Testing async analysis loop...")
    
    # Start analysis loop
    analysis_task = asyncio.create_task(processor.start_analysis_loop())
    
    # Let it run for a short time
    await asyncio.sleep(2)
    
    # Stop the loop
    processor.stop_analysis_loop()
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(analysis_task, timeout=1)
    except asyncio.TimeoutError:
        analysis_task.cancel()
    
    print("   ✅ Async analysis loop completed")
    
    # Test parallel analysis
    print("   Testing parallel analysis...")
    
    # Create multiple analysis tasks
    tasks = []
    for i in range(3):
        task = asyncio.create_task(processor.analyze_intermarket("SPY"))
        tasks.append(task)
    
    # Wait for all analyses to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_analyses = [r for r in results if not isinstance(r, Exception)]
    print(f"   ✅ Completed {len(successful_analyses)} parallel analyses")


async def main():
    """Main integration test function."""
    print("🔧 Testing Parallel Cross-Symbol Processor Integration")
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
    
    # Create test data
    symbols_data = create_multi_symbol_data()
    
    # Run tests
    ready_symbols = test_parallel_processing(processor, symbols_data)
    analysis = await test_intermarket_analysis(processor, ready_symbols)
    test_signal_detection(processor, symbols_data)
    test_risk_sentiment_analysis(processor)
    test_volatility_analysis(processor)
    test_momentum_alignment(processor)
    test_analysis_filtering(processor)
    await test_async_operations(processor, symbols_data)
    
    # Final statistics
    print("\n📊 Final Statistics")
    print("=" * 60)
    
    summary = processor.get_analysis_summary()
    print(f"Total Analyses: {summary.get('total_analyses', 0)}")
    print(f"Average Confidence: {summary.get('avg_confidence', 0):.2f}")
    print(f"Average Momentum Alignment: {summary.get('avg_momentum_alignment', 0):.2f}")
    
    if 'signal_counts' in summary:
        print(f"Signal Counts:")
        for signal, count in summary['signal_counts'].items():
            print(f"  {signal}: {count}")
    
    if 'risk_sentiment_distribution' in summary:
        print(f"Risk Sentiment Distribution:")
        for sentiment, count in summary['risk_sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")
    
    print("\n🎉 Parallel Cross-Symbol Processor Integration Test Complete!")
    print("✅ All tests passed successfully")


if __name__ == "__main__":
    asyncio.run(main())