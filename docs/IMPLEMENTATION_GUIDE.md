# Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the Trading Strategy Testing Framework, using the [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) repository as a reference implementation.

## Reference Implementation

The [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) repository provides a complete reference implementation of the methodology described in this framework. Key files include:

- **`bar_permute.py`** - Bar permutation algorithm for destroying temporal structure
- **`donchian.py`** - Donchian strategy implementation (similar to our MACD test)
- **`insample_donchian_mcpt.py`** - In-sample Monte Carlo permutation test example
- **`walkforward_donchian_mcpt.py`** - Walk-forward permutation test example

## Implementation Phases

### Phase 1: Core Framework Foundation

#### 1.1 Base Strategy Interface
```python
# src/zone_fade_detector/strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from zone_fade_detector.core.models import OHLCVBar

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    @abstractmethod
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        """
        Generate position signals for each bar.
        
        Args:
            bars: List of OHLCVBar objects
            params: Dictionary of strategy parameters
            
        Returns:
            List of integers: 1 (long), 0 (flat), -1 (short)
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List]:
        """
        Define parameter optimization space.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return strategy name for reporting.
        
        Returns:
            String with strategy name
        """
        pass
```

#### 1.2 MACD Strategy Implementation
```python
# src/zone_fade_detector/strategies/macd_strategy.py
import numpy as np
from zone_fade_detector.strategies.base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    """MACD crossover strategy for framework validation."""
    
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        """Generate MACD crossover signals."""
        if len(bars) < params['slow_period']:
            return [0] * len(bars)
        
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
    
    def _calculate_macd(self, closes: List[float], params: Dict[str, Any]) -> tuple:
        """Calculate MACD line and signal line."""
        fast_ema = self._ema(closes, params['fast_period'])
        slow_ema = self._ema(closes, params['slow_period'])
        macd_line = fast_ema - slow_ema
        signal_line = self._ema(macd_line, params['signal_period'])
        return macd_line, signal_line
    
    def _ema(self, data: List[float], period: int) -> List[float]:
        """Calculate exponential moving average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def get_parameter_space(self) -> Dict[str, List]:
        """Define MACD parameter space."""
        return {
            'fast_period': [10, 12, 14, 16, 18, 20],
            'slow_period': [20, 25, 30, 35, 40],
            'signal_period': [5, 7, 9, 11, 13]
        }
    
    def get_name(self) -> str:
        """Return strategy name."""
        return "MACD Crossover Strategy"
```

#### 1.3 Bar Returns Engine
```python
# src/zone_fade_detector/utils/returns_engine.py
from typing import List
from zone_fade_detector.core.models import OHLCVBar

class ReturnsEngine:
    """Bar-level return calculation with look-ahead prevention."""
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        self.commission = commission
        self.slippage = slippage
    
    def calculate_strategy_returns(self, signals: List[int], 
                                  bars: List[OHLCVBar]) -> List[float]:
        """
        Calculate strategy returns with proper look-ahead prevention.
        
        Args:
            signals: List of position signals
            bars: List of OHLCVBar objects
            
        Returns:
            List of strategy returns
        """
        if len(signals) != len(bars):
            raise ValueError("Signals and bars must have same length")
        
        # Calculate bar returns (close-to-close)
        bar_returns = []
        for i in range(len(bars)):
            if i == 0:
                bar_returns.append(0.0)
            else:
                bar_returns.append((bars[i].close - bars[i-1].close) / bars[i-1].close)
        
        # Shift returns forward 1 bar (look-ahead prevention)
        shifted_returns = [0.0] + bar_returns[:-1]
        
        # Calculate strategy returns
        strategy_returns = []
        for i, (signal, ret) in enumerate(zip(signals, shifted_returns)):
            if signal == 0:
                strategy_returns.append(0.0)
            else:
                # Apply transaction costs
                cost = self.commission + self.slippage
                strategy_returns.append(signal * ret - cost)
        
        return strategy_returns
```

### Phase 2: Validation Components

#### 2.1 In-Sample Excellence
```python
# src/zone_fade_detector/validation/in_sample_excellence.py
from typing import List, Dict, Any, Tuple
from zone_fade_detector.strategies.base_strategy import BaseStrategy
from zone_fade_detector.core.models import OHLCVBar
from zone_fade_detector.utils.returns_engine import ReturnsEngine

class InSampleExcellence:
    """In-sample optimization and analysis."""
    
    def __init__(self, strategy: BaseStrategy, objective_func: str = 'profit_factor'):
        self.strategy = strategy
        self.objective_func = objective_func
        self.returns_engine = ReturnsEngine()
    
    def optimize_parameters(self, bars: List[OHLCVBar], 
                          train_start: int, train_end: int) -> Dict[str, Any]:
        """
        Optimize strategy parameters on training data.
        
        Args:
            bars: List of OHLCVBar objects
            train_start: Training start index
            train_end: Training end index
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        train_bars = bars[train_start:train_end]
        parameter_space = self.strategy.get_parameter_space()
        
        best_params = None
        best_score = float('-inf')
        optimization_trace = []
        
        # Grid search optimization
        for params in self._generate_parameter_combinations(parameter_space):
            try:
                signals = self.strategy.generate_signal(train_bars, params)
                returns = self.returns_engine.calculate_strategy_returns(signals, train_bars)
                score = self._calculate_objective(returns)
                
                optimization_trace.append({
                    'parameters': params,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return {
            'optimized_parameters': best_params,
            'best_score': best_score,
            'optimization_trace': optimization_trace
        }
    
    def _generate_parameter_combinations(self, parameter_space: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations."""
        import itertools
        
        keys = list(parameter_space.keys())
        values = list(parameter_space.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _calculate_objective(self, returns: List[float]) -> float:
        """Calculate objective function."""
        if self.objective_func == 'profit_factor':
            return self._calculate_profit_factor(returns)
        elif self.objective_func == 'sharpe':
            return self._calculate_sharpe_ratio(returns)
        else:
            raise ValueError(f"Unknown objective function: {self.objective_func}")
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor."""
        gains = sum(r for r in returns if r > 0)
        losses = sum(-r for r in returns if r < 0)
        return gains / max(losses, 1e-12)
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
```

#### 2.2 Permutation Testing (Based on mcpt)
```python
# src/zone_fade_detector/validation/permutation_tests.py
import numpy as np
from typing import List, Dict, Any
from zone_fade_detector.strategies.base_strategy import BaseStrategy
from zone_fade_detector.core.models import OHLCVBar
from zone_fade_detector.utils.returns_engine import ReturnsEngine

class PermutationTester:
    """Monte Carlo permutation testing based on mcpt repository."""
    
    def __init__(self, strategy: BaseStrategy, n_permutations: int = 1000):
        self.strategy = strategy
        self.n_permutations = n_permutations
        self.returns_engine = ReturnsEngine()
    
    def in_sample_permutation_test(self, bars: List[OHLCVBar], 
                                 train_start: int, train_end: int) -> Dict[str, Any]:
        """
        Run in-sample Monte Carlo permutation test.
        
        Based on mcpt/insample_donchian_mcpt.py
        """
        train_bars = bars[train_start:train_end]
        
        # Get real optimized score
        from zone_fade_detector.validation.in_sample_excellence import InSampleExcellence
        is_tester = InSampleExcellence(self.strategy)
        real_results = is_tester.optimize_parameters(bars, train_start, train_end)
        real_score = real_results['best_score']
        
        # Generate permutations and re-optimize
        permutation_scores = []
        
        for i in range(self.n_permutations):
            # Permute bars while preserving distributional stats
            permuted_bars = self._permute_bars(train_bars)
            
            # Re-optimize on permuted data
            try:
                perm_results = is_tester.optimize_parameters(permuted_bars, 0, len(permuted_bars))
                permutation_scores.append(perm_results['best_score'])
            except Exception as e:
                print(f"Error in permutation {i}: {e}")
                continue
        
        # Calculate p-value
        p_value = sum(1 for score in permutation_scores if score >= real_score) / len(permutation_scores)
        
        return {
            'real_score': real_score,
            'permutation_scores': permutation_scores,
            'p_value': p_value,
            'n_permutations': len(permutation_scores)
        }
    
    def _permute_bars(self, bars: List[OHLCVBar]) -> List[OHLCVBar]:
        """
        Permute bars while preserving distributional statistics.
        
        Based on mcpt/bar_permute.py
        """
        # Extract price data
        opens = [bar.open for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        closes = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]
        timestamps = [bar.timestamp for bar in bars]
        
        # Preserve first and last prices
        first_open, first_high, first_low, first_close = opens[0], highs[0], lows[0], closes[0]
        last_open, last_high, last_low, last_close = opens[-1], highs[-1], lows[-1], closes[-1]
        
        # Calculate intra-bar relatives
        open_relatives = [o / c for o, c in zip(opens, closes)]
        high_relatives = [h / c for h, c in zip(highs, closes)]
        low_relatives = [l / c for l, c in zip(lows, closes)]
        
        # Calculate gaps (close-to-open)
        gaps = []
        for i in range(1, len(bars)):
            gap = opens[i] / closes[i-1]
            gaps.append(gap)
        
        # Shuffle relatives and gaps
        np.random.shuffle(open_relatives)
        np.random.shuffle(high_relatives)
        np.random.shuffle(low_relatives)
        np.random.shuffle(gaps)
        
        # Reconstruct OHLC path
        permuted_bars = []
        current_close = first_close
        
        for i in range(len(bars)):
            if i == 0:
                # First bar - use original values
                permuted_bars.append(bars[0])
                continue
            
            # Calculate new close from gap
            gap = gaps[i-1] if i-1 < len(gaps) else 1.0
            new_close = current_close * gap
            
            # Calculate OHLC from relatives
            new_open = new_close * open_relatives[i]
            new_high = new_close * high_relatives[i]
            new_low = new_close * low_relatives[i]
            
            # Create new bar
            from zone_fade_detector.core.models import OHLCVBar
            new_bar = OHLCVBar(
                timestamp=timestamps[i],
                open=new_open,
                high=new_high,
                low=new_low,
                close=new_close,
                volume=volumes[i]
            )
            
            permuted_bars.append(new_bar)
            current_close = new_close
        
        return permuted_bars
```

### Phase 3: Reporting and Publishing

#### 3.1 Metrics Calculator
```python
# src/zone_fade_detector/reporting/metrics_calculator.py
from typing import List, Dict, Any
import numpy as np

class MetricsCalculator:
    """Standardized performance metrics calculation."""
    
    def __init__(self, initial_capital: float = 10000, 
                 commission: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def calculate_comprehensive_metrics(self, returns: List[float], 
                                       signals: List[int]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Returns metrics
        metrics.update(self._calculate_returns_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Trading metrics
        metrics.update(self._calculate_trading_metrics(signals, returns))
        
        return metrics
    
    def _calculate_returns_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate returns-based metrics."""
        if not returns:
            return {}
        
        total_return = sum(returns)
        annualized_return = total_return * 252 / len(returns)  # Assuming daily data
        
        # Volatility
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5 * (252 ** 0.5)  # Annualized
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if not returns:
            return {}
        
        # Calculate cumulative returns
        cumulative_returns = []
        cumsum = 0
        for ret in returns:
            cumsum += ret
            cumulative_returns.append(cumsum)
        
        # Maximum drawdown
        peak = cumulative_returns[0]
        max_drawdown = 0
        for cumret in cumulative_returns:
            if cumret > peak:
                peak = cumret
            drawdown = peak - cumret
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calmar ratio
        annualized_return = sum(returns) * 252 / len(returns)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_trading_metrics(self, signals: List[int], 
                                 returns: List[float]) -> Dict[str, float]:
        """Calculate trading-based metrics."""
        if not signals or not returns:
            return {}
        
        # Win rate
        winning_trades = sum(1 for ret in returns if ret > 0)
        total_trades = len([ret for ret in returns if ret != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gains = sum(ret for ret in returns if ret > 0)
        losses = sum(-ret for ret in returns if ret < 0)
        profit_factor = gains / max(losses, 1e-12)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades
        }
```

## Implementation Checklist

### Phase 1: Foundation
- [ ] Create BaseStrategy interface
- [ ] Implement MACD strategy
- [ ] Create bar returns engine
- [ ] Add Fortune 100 data source
- [ ] Test basic strategy execution

### Phase 2: Validation
- [ ] Implement in-sample excellence
- [ ] Create permutation testing engine
- [ ] Add walk-forward testing
- [ ] Integrate validation pipeline
- [ ] Test complete validation process

### Phase 3: Reporting
- [ ] Create metrics calculator
- [ ] Add visualization generator
- [ ] Implement report builder
- [ ] Add GitHub publishing
- [ ] Test complete reporting pipeline

### Phase 4: Testing
- [ ] Run MACD framework shakedown
- [ ] Validate all components
- [ ] Test reproducibility
- [ ] Document results
- [ ] Publish to GitHub

## Next Steps

1. **Start with Phase 1** - Implement the foundation components
2. **Use mcpt as Reference** - Follow the mcpt repository examples
3. **Test Incrementally** - Validate each component as you build it
4. **Document Everything** - Keep detailed documentation of your implementation
5. **Iterate and Improve** - Refine based on testing results

The [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) repository provides excellent reference code for implementing the permutation testing components. Use it as a guide for the statistical validation methodology.
