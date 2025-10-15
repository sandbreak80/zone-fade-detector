#!/usr/bin/env python3
"""
Framework Test

Simple test of the framework components without full backtest.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

from directional_bias_detector import DirectionalBiasDetector, DirectionalBias, SessionType
from session_aware_trading import SessionAwareTradingRules, TradeType, TradeDirection, TradeDecision
from choch_confirmation_system import CHoCHConfirmationSystem


def test_framework_components():
    """Test all framework components together."""
    
    print("🧪 Testing Framework Components")
    print("=" * 50)
    
    # Initialize components
    bias_detector = DirectionalBiasDetector()
    trading_rules = SessionAwareTradingRules(bias_detector)
    choch_system = CHoCHConfirmationSystem()
    
    # Create sample bars with clear structure
    bars = []
    base_price = 100.0
    
    # Create bullish structure
    for i in range(50):
        price = base_price + i * 0.2 + np.random.normal(0, 0.3)
        bar = type('Bar', (), {
            'open': price - 0.1,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1000 + np.random.randint(-100, 200),
            'timestamp': datetime.now() + timedelta(minutes=i)
        })()
        bars.append(bar)
    
    print(f"Created {len(bars)} sample bars")
    
    # Test 1: Directional Bias Detection
    print("\n1. Testing Directional Bias Detection:")
    bias_analysis = bias_detector.detect_directional_bias(bars, len(bars) - 1)
    print(f"   Bias: {bias_analysis.bias.value}")
    print(f"   Confidence: {bias_analysis.confidence:.2f}")
    print(f"   Session Type: {bias_analysis.session_type.value}")
    print(f"   CHoCH Confirmed: {bias_analysis.choch_confirmed}")
    
    # Test 2: Session-Aware Trading Rules
    print("\n2. Testing Session-Aware Trading Rules:")
    
    test_zones = [
        ("prior_day_high", 105.0, 104.8),  # Resistance zone
        ("prior_day_low", 95.0, 95.2),     # Support zone
    ]
    
    for zone_type, zone_level, current_price in test_zones:
        decision = trading_rules.analyze_trade_opportunity(
            bars, len(bars) - 1, zone_type, zone_level, current_price
        )
        print(f"   Zone: {zone_type} at {zone_level}")
        print(f"   Decision: {trading_rules.get_trade_summary(decision)}")
        print(f"   Should Take: {trading_rules.should_take_trade(decision)}")
        print()
    
    # Test 3: CHoCH Confirmation System
    print("3. Testing CHoCH Confirmation System:")
    choch_signal = choch_system.detect_choch(bars, len(bars) - 1)
    if choch_signal:
        print(f"   CHoCH Detected: {choch_signal.choch_type.value}")
        print(f"   Confidence: {choch_signal.confidence:.2f}")
    else:
        print("   No CHoCH detected")
    
    # Test 4: Integration Test
    print("\n4. Testing Integration:")
    
    # Simulate entry point
    entry = type('Entry', (), {
        'entry_id': 'TEST_1',
        'symbol': 'SPY',
        'zone_type': 'prior_day_high',
        'zone_level': 105.0,
        'price': 104.8,
        'hard_stop': 105.0,
        't1_price': 106.0,
        't2_price': 107.0,
        't3_price': 108.0,
        'timestamp': datetime.now()
    })()
    
    # Process through framework
    trade_decision = trading_rules.analyze_trade_opportunity(
        bars, len(bars) - 1, entry.zone_type, entry.zone_level, entry.price
    )
    
    if choch_signal:
        trade_decision = choch_system.enhance_trade_decision(trade_decision, choch_signal)
    
    print(f"   Entry: {entry.zone_type} at {entry.price}")
    print(f"   Final Decision: {trading_rules.get_trade_summary(trade_decision)}")
    print(f"   Should Take Trade: {trading_rules.should_take_trade(trade_decision)}")
    
    print("\n✅ Framework components test completed!")


def test_corrected_risk_calculation():
    """Test the corrected risk calculation."""
    
    print("\n🔧 Testing Corrected Risk Calculation:")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("LONG", 100.0, 99.0, 1.0),   # Long trade: entry 100, stop 99, risk = 1.0
        ("SHORT", 100.0, 101.0, 1.0), # Short trade: entry 100, stop 101, risk = 1.0
        ("LONG", 100.0, 100.0, 0.01), # Same price: fallback to 1%
        ("SHORT", 100.0, 100.0, 0.01), # Same price: fallback to 1%
    ]
    
    for direction, entry_price, hard_stop, expected_risk in test_cases:
        if direction == "LONG":
            risk_amount = entry_price - hard_stop
        else:  # SHORT
            risk_amount = hard_stop - entry_price
        
        if risk_amount <= 0:
            risk_amount = entry_price * 0.01
        
        print(f"   {direction}: Entry {entry_price}, Stop {hard_stop}")
        print(f"   Calculated Risk: {risk_amount:.2f}, Expected: {expected_risk:.2f}")
        print(f"   ✅ Correct" if abs(risk_amount - expected_risk) < 0.01 else "   ❌ Incorrect")
        print()


if __name__ == "__main__":
    test_framework_components()
    test_corrected_risk_calculation()