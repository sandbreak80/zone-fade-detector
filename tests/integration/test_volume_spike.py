#!/usr/bin/env python3
"""
Test volume spike detection for rejection candles.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import logging

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zone_fade_detector.core.alert_system import AlertSystem, AlertChannelConfig
from zone_fade_detector.strategies.signal_processor import SignalProcessor, SignalProcessorConfig
from zone_fade_detector.indicators.volume_analysis import VolumeAnalyzer
from zone_fade_detector.core.models import OHLCVBar

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_volume_spike_detection():
    """Test volume spike detection functionality."""
    
    # Load cached data
    data_dir = Path('/app/data/2024')
    combined_file = data_dir / "all_symbols_2024.pkl"
    
    if not combined_file.exists():
        logger.error(f"❌ Data file not found: {combined_file}")
        return
    
    logger.info(f"📂 Loading 2024 data from {combined_file}")
    
    with open(combined_file, 'rb') as f:
        all_data = pickle.load(f)
    
    logger.info(f"📊 Loaded data for {len(all_data)} symbols")
    
    # Initialize volume analyzer
    volume_analyzer = VolumeAnalyzer()
    
    # Test volume spike detection on each symbol
    for symbol, bars in all_data.items():
        logger.info(f"\n🔍 Testing volume spike detection for {symbol}...")
        
        # Test on last 100 bars
        test_bars = bars[-100:] if len(bars) >= 100 else bars
        
        volume_spikes = []
        rejection_candidates = []
        
        for i in range(15, len(test_bars)):  # Start after lookback period
            current_bar = test_bars[i]
            
            # Test basic volume spike detection
            is_spike, spike_ratio = volume_analyzer.detect_volume_spike(
                test_bars, i, spike_threshold=1.8, lookback_bars=15
            )
            
            # Test rejection volume spike detection
            is_rejection_spike, rejection_ratio, volume_metrics = volume_analyzer.detect_rejection_volume_spike(
                test_bars, i, spike_threshold=1.8, lookback_bars=15
            )
            
            if is_spike:
                volume_spikes.append({
                    'index': i,
                    'timestamp': current_bar.timestamp,
                    'volume': current_bar.volume,
                    'spike_ratio': spike_ratio,
                    'is_rejection_spike': is_rejection_spike,
                    'rejection_ratio': rejection_ratio
                })
            
            # Check for potential rejection candles (basic wick analysis)
            total_range = current_bar.total_range
            if total_range > 0:
                upper_wick_ratio = current_bar.upper_wick / total_range
                lower_wick_ratio = current_bar.lower_wick / total_range
                
                if upper_wick_ratio >= 0.1 or lower_wick_ratio >= 0.1:
                    rejection_candidates.append({
                        'index': i,
                        'timestamp': current_bar.timestamp,
                        'upper_wick_ratio': upper_wick_ratio,
                        'lower_wick_ratio': lower_wick_ratio,
                        'volume': current_bar.volume,
                        'is_volume_spike': is_spike,
                        'spike_ratio': spike_ratio
                    })
        
        logger.info(f"   📈 Volume spikes found: {len(volume_spikes)}")
        logger.info(f"   🕯️ Rejection candidates: {len(rejection_candidates)}")
        
        # Show top volume spikes
        if volume_spikes:
            top_spikes = sorted(volume_spikes, key=lambda x: x['spike_ratio'], reverse=True)[:5]
            logger.info(f"   🏆 Top 5 volume spikes:")
            for spike in top_spikes:
                logger.info(f"      {spike['timestamp']}: {spike['spike_ratio']:.2f}x volume ({spike['volume']:,})")
        
        # Show rejection candidates with volume spikes
        rejection_with_volume = [r for r in rejection_candidates if r['is_volume_spike']]
        if rejection_with_volume:
            logger.info(f"   🎯 Rejection candles with volume spikes: {len(rejection_with_volume)}")
            for candidate in rejection_with_volume[:3]:  # Show top 3
                logger.info(f"      {candidate['timestamp']}: {candidate['spike_ratio']:.2f}x volume, wick: {max(candidate['upper_wick_ratio'], candidate['lower_wick_ratio']):.2f}")

async def test_enhanced_detection():
    """Test enhanced detection with volume spike integration."""
    
    # Load cached data
    data_dir = Path('/app/data/2024')
    combined_file = data_dir / "all_symbols_2024.pkl"
    
    if not combined_file.exists():
        logger.error(f"❌ Data file not found: {combined_file}")
        return
    
    logger.info(f"📂 Loading 2024 data from {combined_file}")
    
    with open(combined_file, 'rb') as f:
        all_data = pickle.load(f)
    
    # Set up Discord alerts for status updates
    alert_config = AlertChannelConfig(
        console_enabled=True,
        file_enabled=False,
        webhook_enabled=True,
        webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
        webhook_timeout=10
    )
    alert_system = AlertSystem(alert_config)
    
    # Set up signal processor with volume spike integration
    processor_config = SignalProcessorConfig(
        min_qrs_score=3,  # Lower threshold to see more setups
        max_setups_per_symbol=10,
        setup_cooldown_minutes=30,
        alert_deduplication_minutes=10,
        enable_intermarket_filtering=False,
        enable_volume_filtering=False
    )
    signal_processor = SignalProcessor(processor_config)
    
    logger.info("🎯 Testing enhanced Zone Fade detection with volume spike integration...")
    
    # Process signals
    alerts = signal_processor.process_signals(all_data)
    
    logger.info(f"🚨 Generated {len(alerts)} Zone Fade alerts with volume spike integration!")
    
    if alerts:
        # Send summary to Discord
        await alert_system.send_alert(alerts[0])  # Send first alert as test
        
        # Show detailed analysis
        for i, alert in enumerate(alerts[:3]):  # Show first 3 alerts
            setup = alert.setup
            logger.info(f"\n📊 Alert {i+1}: {setup.symbol} {setup.direction.value.upper()}")
            logger.info(f"   QRS Score: {setup.qrs_score}/10")
            logger.info(f"   Zone: {setup.zone.zone_type.value} @ ${setup.zone.level:.2f}")
            logger.info(f"   Entry: ${setup.entry_price:.2f}")
            logger.info(f"   Stop: ${setup.stop_loss:.2f}")
            logger.info(f"   Target 1: ${setup.target_1:.2f}")
            logger.info(f"   Target 2: ${setup.target_2:.2f}")
    else:
        logger.info("ℹ️ No Zone Fade setups detected with volume spike integration")

if __name__ == "__main__":
    print("🔍 Testing Volume Spike Detection")
    print("=" * 50)
    
    # Test basic volume spike detection
    test_volume_spike_detection()
    
    print("\n" + "=" * 50)
    print("🎯 Testing Enhanced Detection with Volume Spikes")
    print("=" * 50)
    
    # Test enhanced detection
    asyncio.run(test_enhanced_detection())