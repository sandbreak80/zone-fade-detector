# Strategy Development Guide

## Overview

This guide explains how to create new trading strategies for the framework. The framework is designed to be strategy-agnostic, meaning you can implement any trading logic without modifying the core framework code.

## Quick Start

```python
from zone_fade_detector.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signal(self, bars, params):
        # Your trading logic here
        return signals
    
    def get_parameter_space(self):
        # Define parameter ranges
        return parameter_space
    
    def get_name(self):
        return "My Strategy"
```

## BaseStrategy Interface

All strategies must implement the `BaseStrategy` interface with three required methods:

### 1. `generate_signal(bars, params) -> List[int]`
**Purpose**: Generate position signals for each bar

**Parameters**:
- `bars`: List of OHLCVBar objects
- `params`: Dictionary of strategy parameters

**Returns**: List of integers representing position:
- `1`: Long position
- `0`: No position (flat)
- `-1`: Short position

**Example**:
```python
def generate_signal(self, bars, params):
    signals = []
    for i, bar in enumerate(bars):
        if i < params['lookback']:
            signals.append(0)  # No signal for first bars
            continue
            
        # Your strategy logic
        if self._should_buy(bars, i, params):
            signals.append(1)
        elif self._should_sell(bars, i, params):
            signals.append(-1)
        else:
            signals.append(0)
    
    return signals
```

### 2. `get_parameter_space() -> Dict[str, List]`
**Purpose**: Define parameter optimization space

**Returns**: Dictionary mapping parameter names to lists of possible values

**Example**:
```python
def get_parameter_space(self):
    return {
        'fast_period': [10, 12, 14, 16, 18, 20],
        'slow_period': [20, 25, 30, 35, 40],
        'signal_period': [5, 7, 9, 11, 13],
        'threshold': [0.001, 0.002, 0.005, 0.01]
    }
```

### 3. `get_name() -> str`
**Purpose**: Return strategy name for reporting

**Returns**: String with strategy name

**Example**:
```python
def get_name(self):
    return "MACD Crossover Strategy"
```

## Strategy Implementation Examples

### MACD Crossover Strategy

```python
import numpy as np
from zone_fade_detector.strategies.base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    def __init__(self):
        self.name = "MACD Crossover"
    
    def generate_signal(self, bars, params):
        if len(bars) < params['slow_period']:
            return [0] * len(bars)
        
        # Calculate MACD
        closes = [bar.close for bar in bars]
        macd_line, signal_line = self._calculate_macd(closes, params)
        
        signals = []
        for i in range(len(bars)):
            if i < params['slow_period']:
                signals.append(0)
                continue
            
            # Crossover logic
            if i > 0:
                if (macd_line[i-1] <= signal_line[i-1] and 
                    macd_line[i] > signal_line[i]):
                    signals.append(1)  # Buy signal
                elif (macd_line[i-1] >= signal_line[i-1] and 
                      macd_line[i] < signal_line[i]):
                    signals.append(-1)  # Sell signal
                else:
                    signals.append(signals[-1] if signals else 0)
            else:
                signals.append(0)
        
        return signals
    
    def _calculate_macd(self, closes, params):
        # Calculate MACD line
        fast_ema = self._ema(closes, params['fast_period'])
        slow_ema = self._ema(closes, params['slow_period'])
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self._ema(macd_line, params['signal_period'])
        
        return macd_line, signal_line
    
    def _ema(self, data, period):
        # Exponential moving average calculation
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def get_parameter_space(self):
        return {
            'fast_period': [10, 12, 14, 16, 18, 20],
            'slow_period': [20, 25, 30, 35, 40],
            'signal_period': [5, 7, 9, 11, 13]
        }
    
    def get_name(self):
        return "MACD Crossover Strategy"
```

### RSI Mean Reversion Strategy

```python
class RSIMeanReversionStrategy(BaseStrategy):
    def generate_signal(self, bars, params):
        if len(bars) < params['period'] + 1:
            return [0] * len(bars)
        
        closes = [bar.close for bar in bars]
        rsi = self._calculate_rsi(closes, params['period'])
        
        signals = []
        for i, rsi_value in enumerate(rsi):
            if i < params['period']:
                signals.append(0)
                continue
            
            # Mean reversion logic
            if rsi_value < params['oversold']:
                signals.append(1)  # Buy signal
            elif rsi_value > params['overbought']:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)
        
        return signals
    
    def _calculate_rsi(self, closes, period):
        # Calculate RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(closes))
        avg_losses = np.zeros(len(closes))
        
        for i in range(period, len(closes)):
            avg_gains[i] = np.mean(gains[i-period:i])
            avg_losses[i] = np.mean(losses[i-period:i])
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_parameter_space(self):
        return {
            'period': [10, 14, 20, 25, 30],
            'oversold': [20, 25, 30],
            'overbought': [70, 75, 80]
        }
    
    def get_name(self):
        return "RSI Mean Reversion Strategy"
```

## Best Practices

### 1. Look-Ahead Prevention
**Critical**: Never use future information in signal generation.

**❌ Wrong**:
```python
def generate_signal(self, bars, params):
    signals = []
    for i, bar in enumerate(bars):
        # Using future data - WRONG!
        if bar.close > bars[i+1].close:  # Future data
            signals.append(1)
```

**✅ Correct**:
```python
def generate_signal(self, bars, params):
    signals = []
    for i, bar in enumerate(bars):
        # Only use current and past data
        if i > 0 and bar.close > bars[i-1].close:  # Past data
            signals.append(1)
```

### 2. Parameter Space Design
- **Start Simple**: Begin with 2-3 parameters
- **Economic Logic**: Parameters should have economic meaning
- **Reasonable Ranges**: Don't test extreme parameter values
- **Avoid Overfitting**: Don't create too many parameter combinations

### 3. Signal Generation
- **Clear Logic**: Strategy logic should be understandable
- **Consistent Signals**: Avoid rapid signal changes
- **Position Sizing**: Consider position sizing in signal generation
- **Risk Management**: Include stop-loss and take-profit logic

### 4. Performance Optimization
- **Vectorization**: Use NumPy for numerical operations
- **Caching**: Cache expensive calculations
- **Memory Management**: Avoid storing large intermediate arrays
- **Profiling**: Profile strategy performance

## Testing Your Strategy

### 1. Unit Testing
```python
import unittest
from zone_fade_detector.strategies.macd_strategy import MACDStrategy

class TestMACDStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MACDStrategy()
        self.bars = self._create_test_bars()
    
    def test_signal_generation(self):
        params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        signals = self.strategy.generate_signal(self.bars, params)
        
        self.assertEqual(len(signals), len(self.bars))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
    
    def _create_test_bars(self):
        # Create test data
        pass
```

### 2. Integration Testing
```python
def test_strategy_integration():
    strategy = MACDStrategy()
    
    # Test with real data
    bars = load_test_data()
    params = strategy.get_parameter_space()
    
    # Test signal generation
    signals = strategy.generate_signal(bars, params)
    
    # Validate signals
    assert len(signals) == len(bars)
    assert all(s in [-1, 0, 1] for s in signals)
```

### 3. Performance Testing
```python
def test_strategy_performance():
    strategy = MACDStrategy()
    bars = load_large_dataset()
    
    import time
    start_time = time.time()
    
    signals = strategy.generate_signal(bars, params)
    
    execution_time = time.time() - start_time
    assert execution_time < 1.0  # Should complete in under 1 second
```

## Common Pitfalls

### 1. Future Leakage
**Problem**: Using future information in signal generation
**Solution**: Only use current and past data, never future data

### 2. Overfitting
**Problem**: Too many parameters or complex logic
**Solution**: Start simple, use economic logic, avoid excessive parameters

### 3. Inconsistent Signals
**Problem**: Rapid signal changes or contradictory signals
**Solution**: Implement signal smoothing or confirmation logic

### 4. Poor Performance
**Problem**: Strategy is too slow for large datasets
**Solution**: Optimize calculations, use vectorization, profile performance

### 5. Parameter Space Issues
**Problem**: Parameter space too large or poorly defined
**Solution**: Use economic reasoning, test reasonable ranges, avoid extremes

## Strategy Registration

### 1. Create Strategy File
```python
# src/zone_fade_detector/strategies/my_strategy.py
from zone_fade_detector.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    # Implementation here
    pass
```

### 2. Register Strategy
```python
# src/zone_fade_detector/strategies/__init__.py
from .my_strategy import MyStrategy

STRATEGIES = {
    'my_strategy': MyStrategy,
    'macd': MACDStrategy,
    'rsi': RSIStrategy,
}
```

### 3. Use in Framework
```python
from zone_fade_detector.strategies import STRATEGIES

strategy_class = STRATEGIES['my_strategy']
strategy = strategy_class()
```

## Advanced Features

### 1. Custom Indicators
```python
class CustomIndicatorStrategy(BaseStrategy):
    def _calculate_custom_indicator(self, bars, params):
        # Your custom indicator calculation
        pass
    
    def generate_signal(self, bars, params):
        indicator = self._calculate_custom_indicator(bars, params)
        # Use indicator in signal generation
        pass
```

### 2. Multi-Timeframe Analysis
```python
class MultiTimeframeStrategy(BaseStrategy):
    def generate_signal(self, bars, params):
        # Analyze multiple timeframes
        daily_trend = self._analyze_daily_trend(bars)
        hourly_signal = self._analyze_hourly_signal(bars)
        
        # Combine timeframes
        if daily_trend == 'bullish' and hourly_signal == 'buy':
            return 1
        elif daily_trend == 'bearish' and hourly_signal == 'sell':
            return -1
        else:
            return 0
```

### 3. Risk Management
```python
class RiskManagedStrategy(BaseStrategy):
    def generate_signal(self, bars, params):
        base_signal = self._generate_base_signal(bars, params)
        
        # Apply risk management
        if self._is_high_risk_period(bars):
            return 0  # No position during high risk
        
        if self._exceeds_position_limit():
            return 0  # No position if limit exceeded
        
        return base_signal
```

## Conclusion

Creating strategies for the framework is straightforward once you understand the `BaseStrategy` interface. The key is to:

1. **Implement the interface correctly** - All three methods are required
2. **Prevent look-ahead bias** - Never use future information
3. **Design reasonable parameter spaces** - Use economic logic
4. **Test thoroughly** - Unit tests, integration tests, performance tests
5. **Follow best practices** - Clear logic, consistent signals, good performance

The framework will handle all the validation, optimization, and reporting automatically. Your job is to implement the trading logic correctly and let the framework do the rest.
