# API Reference

## Overview

This document provides detailed API reference for the Trading Strategy Testing Framework. All classes, methods, and functions are documented with their signatures, parameters, return values, and usage examples.

## Core Modules

### BaseStrategy

The abstract base class for all trading strategies.

```python
from zone_fade_detector.strategies.base_strategy import BaseStrategy

class BaseStrategy:
    """Abstract base class for all trading strategies."""
    
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        """
        Generate position signals for each bar.
        
        Args:
            bars: List of OHLCVBar objects
            params: Dictionary of strategy parameters
            
        Returns:
            List of integers: 1 (long), 0 (flat), -1 (short)
        """
        raise NotImplementedError
    
    def get_parameter_space(self) -> Dict[str, List]:
        """
        Define parameter optimization space.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        raise NotImplementedError
    
    def get_name(self) -> str:
        """
        Return strategy name for reporting.
        
        Returns:
            String with strategy name
        """
        raise NotImplementedError
```

### OHLCVBar

Data structure for price bars.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OHLCVBar:
    """OHLCV bar data structure."""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    def __post_init__(self):
        """Validate bar data."""
        if self.high < max(self.open, self.close):
            raise ValueError("High must be >= max(open, close)")
        if self.low > min(self.open, close):
            raise ValueError("Low must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Volume must be >= 0")
```

## Validation Module

### InSampleExcellence

Handles in-sample optimization and analysis.

```python
from zone_fade_detector.validation.in_sample_excellence import InSampleExcellence

class InSampleExcellence:
    """In-sample optimization and analysis."""
    
    def __init__(self, strategy: BaseStrategy, objective_func: str = 'profit_factor'):
        """
        Initialize in-sample excellence tester.
        
        Args:
            strategy: Strategy instance to test
            objective_func: Objective function name ('profit_factor', 'sharpe', 'return')
        """
        self.strategy = strategy
        self.objective_func = objective_func
    
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
        pass
    
    def analyze_stability(self, bars: List[OHLCVBar], 
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze parameter stability.
        
        Args:
            bars: List of OHLCVBar objects
            params: Parameter dictionary
            
        Returns:
            Dictionary with stability analysis results
        """
        pass
```

### PermutationTester

Handles Monte Carlo permutation testing.

```python
from zone_fade_detector.validation.permutation_tests import PermutationTester

class PermutationTester:
    """Monte Carlo permutation testing."""
    
    def __init__(self, strategy: BaseStrategy, n_permutations: int = 1000):
        """
        Initialize permutation tester.
        
        Args:
            strategy: Strategy instance to test
            n_permutations: Number of permutations to generate
        """
        self.strategy = strategy
        self.n_permutations = n_permutations
    
    def in_sample_permutation_test(self, bars: List[OHLCVBar], 
                                 train_start: int, train_end: int) -> Dict[str, Any]:
        """
        Run in-sample Monte Carlo permutation test.
        
        Args:
            bars: List of OHLCVBar objects
            train_start: Training start index
            train_end: Training end index
            
        Returns:
            Dictionary with p-value and permutation results
        """
        pass
    
    def walk_forward_permutation_test(self, bars: List[OHLCVBar], 
                                    train_years: int = 4, 
                                    retrain_days: int = 30) -> Dict[str, Any]:
        """
        Run walk-forward permutation test.
        
        Args:
            bars: List of OHLCVBar objects
            train_years: Training window length in years
            retrain_days: Retraining frequency in days
            
        Returns:
            Dictionary with p-value and permutation results
        """
        pass
```

### WalkForwardTester

Handles walk-forward validation.

```python
from zone_fade_detector.validation.walk_forward import WalkForwardTester

class WalkForwardTester:
    """Walk-forward validation testing."""
    
    def __init__(self, strategy: BaseStrategy, train_years: int = 4, 
                 retrain_days: int = 30):
        """
        Initialize walk-forward tester.
        
        Args:
            strategy: Strategy instance to test
            train_years: Training window length in years
            retrain_days: Retraining frequency in days
        """
        self.strategy = strategy
        self.train_years = train_years
        self.retrain_days = retrain_days
    
    def run_walk_forward(self, bars: List[OHLCVBar]) -> Dict[str, Any]:
        """
        Run walk-forward validation.
        
        Args:
            bars: List of OHLCVBar objects
            
        Returns:
            Dictionary with walk-forward results
        """
        pass
```

## Reporting Module

### MetricsCalculator

Calculates standardized performance metrics.

```python
from zone_fade_detector.reporting.metrics_calculator import MetricsCalculator

class MetricsCalculator:
    """Standardized performance metrics calculation."""
    
    def __init__(self, initial_capital: float = 10000, 
                 commission: float = 0.001, slippage: float = 0.0005):
        """
        Initialize metrics calculator.
        
        Args:
            initial_capital: Initial capital amount
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def calculate_returns_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate returns-based metrics.
        
        Args:
            returns: List of strategy returns
            
        Returns:
            Dictionary with returns metrics
        """
        pass
    
    def calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate risk-based metrics.
        
        Args:
            returns: List of strategy returns
            
        Returns:
            Dictionary with risk metrics
        """
        pass
    
    def calculate_trading_metrics(self, signals: List[int], 
                                returns: List[float]) -> Dict[str, float]:
        """
        Calculate trading-based metrics.
        
        Args:
            signals: List of position signals
            returns: List of strategy returns
            
        Returns:
            Dictionary with trading metrics
        """
        pass
```

### VisualizationGenerator

Generates charts and visualizations.

```python
from zone_fade_detector.reporting.visualization_generator import VisualizationGenerator

class VisualizationGenerator:
    """Generate charts and visualizations."""
    
    def __init__(self, output_dir: str = 'plots'):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = output_dir
    
    def generate_equity_curve(self, returns: List[float], 
                            title: str = 'Equity Curve') -> str:
        """
        Generate equity curve plot.
        
        Args:
            returns: List of strategy returns
            title: Plot title
            
        Returns:
            Path to generated plot file
        """
        pass
    
    def generate_drawdown_chart(self, returns: List[float], 
                             title: str = 'Drawdown Chart') -> str:
        """
        Generate drawdown chart.
        
        Args:
            returns: List of strategy returns
            title: Plot title
            
        Returns:
            Path to generated plot file
        """
        pass
    
    def generate_monthly_heatmap(self, returns: List[float], 
                               title: str = 'Monthly Returns Heatmap') -> str:
        """
        Generate monthly returns heatmap.
        
        Args:
            returns: List of strategy returns
            title: Plot title
            
        Returns:
            Path to generated plot file
        """
        pass
```

## Data Module

### DataManager

Unified interface for data sources.

```python
from zone_fade_detector.data.data_manager import DataManager

class DataManager:
    """Unified data management interface."""
    
    def __init__(self, config: DataManagerConfig):
        """
        Initialize data manager.
        
        Args:
            config: Data manager configuration
        """
        self.config = config
    
    async def get_bars(self, symbol: str, start: datetime, 
                     end: datetime) -> List[OHLCVBar]:
        """
        Get historical bars for symbol.
        
        Args:
            symbol: Symbol to fetch
            start: Start datetime
            end: End datetime
            
        Returns:
            List of OHLCVBar objects
        """
        pass
    
    def get_cached_bars(self, symbol: str, start: datetime, 
                       end: datetime) -> Optional[List[OHLCVBar]]:
        """
        Get cached bars if available.
        
        Args:
            symbol: Symbol to fetch
            start: Start datetime
            end: End datetime
            
        Returns:
            Cached bars or None if not available
        """
        pass
```

### Fortune100Client

Fortune 100 ticker management.

```python
from zone_fade_detector.data.fortune_100_client import Fortune100Client

class Fortune100Client:
    """Fortune 100 ticker management."""
    
    def __init__(self, year: int = 2024):
        """
        Initialize Fortune 100 client.
        
        Args:
            year: Year for Fortune 100 list
        """
        self.year = year
    
    def get_fortune_100_tickers(self) -> List[str]:
        """
        Get Fortune 100 ticker symbols.
        
        Returns:
            List of ticker symbols
        """
        pass
    
    def select_random_tickers(self, n: int = 5, seed: int = None) -> List[str]:
        """
        Select random tickers from Fortune 100.
        
        Args:
            n: Number of tickers to select
            seed: Random seed for reproducibility
            
        Returns:
            List of selected ticker symbols
        """
        pass
```

## Publishing Module

### GitHubPublisher

Automated GitHub result publishing.

```python
from zone_fade_detector.publishing.github_publisher import GitHubPublisher

class GitHubPublisher:
    """Automated GitHub result publishing."""
    
    def __init__(self, repo_path: str, token: str = None):
        """
        Initialize GitHub publisher.
        
        Args:
            repo_path: Path to Git repository
            token: GitHub personal access token
        """
        self.repo_path = repo_path
        self.token = token
    
    def publish_results(self, strategy_name: str, results: Dict[str, Any]) -> str:
        """
        Publish strategy results to GitHub.
        
        Args:
            strategy_name: Name of strategy
            results: Results dictionary
            
        Returns:
            Commit hash of published results
        """
        pass
    
    def create_result_directory(self, strategy_name: str, 
                               timestamp: str) -> str:
        """
        Create result directory structure.
        
        Args:
            strategy_name: Name of strategy
            timestamp: Timestamp for directory name
            
        Returns:
            Path to created directory
        """
        pass
```

## Utility Functions

### Returns Engine

Bar-level return calculation with look-ahead prevention.

```python
from zone_fade_detector.utils.returns_engine import ReturnsEngine

class ReturnsEngine:
    """Bar-level return calculation."""
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        """
        Initialize returns engine.
        
        Args:
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
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
        pass
    
    def calculate_costs(self, signals: List[int], 
                       bars: List[OHLCVBar]) -> List[float]:
        """
        Calculate transaction costs.
        
        Args:
            signals: List of position signals
            bars: List of OHLCVBar objects
            
        Returns:
            List of transaction costs
        """
        pass
```

### Statistical Functions

Statistical analysis utilities.

```python
from zone_fade_detector.utils.statistical_functions import StatisticalFunctions

class StatisticalFunctions:
    """Statistical analysis utilities."""
    
    @staticmethod
    def calculate_p_value(real_score: float, 
                          permutation_scores: List[float]) -> float:
        """
        Calculate p-value for permutation test.
        
        Args:
            real_score: Real strategy score
            permutation_scores: List of permutation scores
            
        Returns:
            P-value
        """
        pass
    
    @staticmethod
    def calculate_confidence_interval(data: List[float], 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval.
        
        Args:
            data: List of values
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        pass
    
    @staticmethod
    def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, 
                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: List of values
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        pass
```

## Configuration

### Strategy Testing Configuration

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class StrategyTestingConfig:
    """Configuration for strategy testing."""
    
    # Data configuration
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    timeframe: str = '1h'
    
    # Validation configuration
    train_years: int = 4
    retrain_days: int = 30
    n_permutations_imcpt: int = 1000
    n_permutations_wfpt: int = 200
    
    # Performance configuration
    initial_capital: float = 10000
    commission: float = 0.001
    slippage: float = 0.0005
    max_position_size: float = 0.2
    
    # Reporting configuration
    output_dir: str = 'results'
    generate_plots: bool = True
    publish_to_github: bool = True
    
    # Random seeds
    random_seed: int = 42
    permutation_seed: int = 123
    ticker_selection_seed: int = 456
```

## Error Handling

### Custom Exceptions

```python
class StrategyTestingError(Exception):
    """Base exception for strategy testing."""
    pass

class DataError(StrategyTestingError):
    """Data-related errors."""
    pass

class ValidationError(StrategyTestingError):
    """Validation-related errors."""
    pass

class ReportingError(StrategyTestingError):
    """Reporting-related errors."""
    pass

class PublishingError(StrategyTestingError):
    """Publishing-related errors."""
    pass
```

## Usage Examples

### Basic Strategy Testing

```python
from zone_fade_detector.strategies.macd_strategy import MACDStrategy
from zone_fade_detector.validation import InSampleExcellence, PermutationTester
from zone_fade_detector.reporting import MetricsCalculator

# Create strategy
strategy = MACDStrategy()

# Load data
bars = load_data('QQQ', start_date, end_date)

# In-sample optimization
is_tester = InSampleExcellence(strategy)
optimized_params = is_tester.optimize_parameters(bars, 0, len(bars)//2)

# Permutation testing
perm_tester = PermutationTester(strategy)
imcpt_results = perm_tester.in_sample_permutation_test(bars, 0, len(bars)//2)

# Calculate metrics
metrics_calc = MetricsCalculator()
signals = strategy.generate_signal(bars, optimized_params)
returns = calculate_returns(signals, bars)
metrics = metrics_calc.calculate_returns_metrics(returns)
```

### Advanced Strategy Testing

```python
from zone_fade_detector.validation import WalkForwardTester
from zone_fade_detector.reporting import VisualizationGenerator
from zone_fade_detector.publishing import GitHubPublisher

# Walk-forward testing
wf_tester = WalkForwardTester(strategy)
wf_results = wf_tester.run_walk_forward(bars)

# Generate visualizations
viz_gen = VisualizationGenerator()
equity_plot = viz_gen.generate_equity_curve(returns)
drawdown_plot = viz_gen.generate_drawdown_chart(returns)

# Publish results
github_pub = GitHubPublisher()
commit_hash = github_pub.publish_results('macd_strategy', {
    'parameters': optimized_params,
    'metrics': metrics,
    'plots': [equity_plot, drawdown_plot]
})
```

## Performance Considerations

### Memory Management
- Use streaming processing for large datasets
- Implement lazy loading for data
- Cache intermediate results appropriately
- Clean up large objects explicitly

### Computational Efficiency
- Use parallel processing for permutation tests
- Vectorize numerical operations with NumPy
- Profile and optimize bottlenecks
- Use appropriate data structures

### Scalability
- Design for horizontal scaling
- Use cloud resources for large tests
- Implement batch processing
- Monitor resource usage

## Reference Implementation

The [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) repository provides a complete reference implementation of the methodology described in this framework. Key files include:

- **`bar_permute.py`** - Bar permutation algorithm for destroying temporal structure
- **`donchian.py`** - Donchian strategy implementation (similar to our MACD test)
- **`insample_donchian_mcpt.py`** - In-sample Monte Carlo permutation test example
- **`walkforward_donchian_mcpt.py`** - Walk-forward permutation test example

This repository demonstrates the exact 4-step validation process we implement in our framework.

This API reference provides comprehensive documentation for all framework components. For more detailed examples and usage patterns, see the [Strategy Development Guide](STRATEGY_DEVELOPMENT.md) and [Architecture Documentation](ARCHITECTURE.md).
