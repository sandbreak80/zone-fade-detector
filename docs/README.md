# Trading Strategy Testing Framework

A scientific, reproducible system for validating trading strategies using rigorous statistical methods and automated GitHub publishing.

## 🎯 Overview

The **Trading Strategy Testing Framework** transforms the Zone Fade Detector into a standardized, reproducible system for validating trading strategies. Built on scientific principles, this framework uses a 4-step validation battery to prevent overfitting, ensure reproducibility, and automatically publish results to GitHub for historical tracking.

### Key Features
- **Scientific Validation**: 4-step statistical testing battery prevents overfitting
- **Reproducibility**: Docker environment + version control + comprehensive metadata
- **Zero-Friction Extension**: New strategies require zero framework changes
- **Professional Reporting**: Standardized metrics and GitHub publishing
- **Bar-Level Returns**: Proper look-ahead prevention and robust statistical analysis

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/zone-fade-detector.git
cd zone-fade-detector

# Build Docker environment
docker-compose build

# Run MACD framework shakedown test
docker-compose run zone-fade-detector python -m strategy_testing.macd_shakedown

# View results
ls results/strategy_001_macd_crossover_*/
```

## 📚 Documentation

### Core Documentation
- **[Framework Overview](STRATEGY_TESTING_FRAMEWORK.md)** - Complete framework introduction
- **[Architecture Guide](ARCHITECTURE.md)** - System design and components
- **[Validation Methodology](VALIDATION_METHODOLOGY.md)** - 4-step validation process
- **[Strategy Development](STRATEGY_DEVELOPMENT.md)** - How to create new strategies
- **[API Reference](API_REFERENCE.md)** - Technical API documentation

### Standards & Best Practices
- **[Reporting Standards](REPORTING_STANDARDS.md)** - Metrics and visualization standards
- **[Reproducibility Guide](REPRODUCIBILITY.md)** - Ensuring reproducible results
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategy Testing Framework               │
├─────────────────────────────────────────────────────────────┤
│ Publishing Layer    │ GitHub integration, result commits   │
├─────────────────────────────────────────────────────────────┤
│ Reporting Layer     │ Standardized metrics, plots, reports │
├─────────────────────────────────────────────────────────────┤
│ Validation Layer    │ 4-step testing battery (IS/WF/PT)    │
├─────────────────────────────────────────────────────────────┤
│ Strategy Layer      │ BaseStrategy interface, implementations│
├─────────────────────────────────────────────────────────────┤
│ Data Layer          │ Alpaca/Polygon APIs, Fortune 100     │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 4-Step Validation Process

### 1. In-Sample Excellence
- Parameter optimization on training data
- Equity curve analysis and stability checks
- Parameter surface mapping

### 2. In-Sample Monte Carlo Permutation Test (IMCPT)
- 1,000 permutations of training data
- Re-optimize on each permutation
- **Target: p < 1%** (genuine patterns vs. selection bias)

### 3. Walk-Forward Test (WFT)
- Rolling 4-year training windows, 30-day retrain
- True out-of-sample performance
- Performance degradation analysis

### 4. Walk-Forward Permutation Test (WFPT)
- 200 permutations of OOS segments only
- Re-run walk-forward pipeline
- **Target: p ≤ 5%** (1 year OOS) or **p ≤ 1%** (2+ years OOS)

## 📊 Standard Metrics

### Performance Metrics
- **Returns**: Total return, annualized return, volatility, Sharpe ratio
- **Risk**: Maximum drawdown, Calmar ratio, Sortino ratio, tail ratio
- **Trading**: Win rate, profit factor, average win/loss, trade frequency
- **Risk-Adjusted**: MAR, Omega ratio, recovery time

### Statistical Validation
- **P-values**: IMCPT and WFPT significance levels
- **Confidence Intervals**: Bootstrap 95% CIs for key metrics
- **Regime Analysis**: Performance by market conditions
- **Cost Sensitivity**: Transaction cost impact analysis

## 🛠️ Creating New Strategies

### 1. Implement BaseStrategy Interface
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

### 2. Register Strategy
```python
# Add to strategies/__init__.py
STRATEGIES = {
    'my_strategy': MyStrategy,
    # ... other strategies
}
```

### 3. Run Validation
```bash
# Test your strategy
docker-compose run zone-fade-detector python -m strategy_testing.run_strategy my_strategy
```

## 📈 Example: MACD Strategy

The framework includes a MACD crossover strategy as a framework shakedown test:

```python
class MACDStrategy(BaseStrategy):
    def generate_signal(self, bars, params):
        # Calculate MACD indicators
        macd_line, signal_line = self._calculate_macd(bars, params)
        
        # Generate crossover signals
        signals = []
        for i in range(len(bars)):
            if i > 0 and macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                signals.append(1)  # Buy signal
            elif i > 0 and macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)
        
        return signals
```

## 🎯 Expected Results

### MACD Shakedown Test
- **Expected Outcome**: Framework correctly identifies unprofitable strategy
- **IMCPT P-value**: Likely >5% (no genuine edge)
- **WFPT P-value**: Likely >10% (OOS success due to luck)
- **Validation**: Proves framework works correctly

### Successful Strategy
- **IS Excellence**: Clear performance with stable parameters
- **IMCPT**: p < 1% (genuine patterns, not selection bias)
- **WFT**: Positive OOS performance with reasonable degradation
- **WFPT**: p ≤ 5% (OOS success not due to luck)

## 🔧 Configuration

### Environment Variables
```bash
# Data sources
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
POLYGON_API_KEY=your_polygon_key

# Discord notifications (optional)
DISCORD_WEBHOOK_URL=your_webhook_url

# Random seeds for reproducibility
RANDOM_SEED=42
PERMUTATION_SEED=123
TICKER_SELECTION_SEED=456
```

### Strategy Configuration
```yaml
# config/strategy_testing.yaml
strategy:
  name: "MACD Crossover"
  symbols: ["QQQ", "SPY"]
  start_date: "2010-01-01"
  end_date: "2025-01-01"
  timeframe: "1h"

validation:
  train_years: 4
  retrain_days: 30
  n_permutations_imcpt: 1000
  n_permutations_wfpt: 200

performance:
  initial_capital: 10000
  commission: 0.001
  slippage: 0.0005
  max_position_size: 0.2
```

## 📊 Results Structure

```
results/
├── strategy_001_macd_crossover_2024-01-15/
│   ├── SUMMARY.md                    # Executive summary
│   ├── metrics.csv                   # Machine-readable metrics
│   ├── metadata.json                 # Run metadata and configuration
│   ├── plots/
│   │   ├── equity_curve.png          # Equity curve visualization
│   │   ├── drawdown_chart.png        # Drawdown analysis
│   │   ├── monthly_heatmap.png       # Monthly returns heatmap
│   │   ├── parameter_surface.png     # 3D parameter optimization
│   │   ├── permutation_histogram.png # Statistical significance
│   │   └── regime_analysis.png       # Performance by market regime
│   └── data/
│       ├── signals.csv              # Position signals
│       ├── returns.csv              # Strategy returns
│       └── trades.csv               # Individual trade analysis
```

## 🚀 Getting Started

### 1. Read the Documentation
- Start with [Framework Overview](STRATEGY_TESTING_FRAMEWORK.md)
- Review [Architecture Guide](ARCHITECTURE.md)
- Understand [Validation Methodology](VALIDATION_METHODOLOGY.md)

### 2. Set Up Environment
```bash
# Clone repository
git clone https://github.com/your-org/zone-fade-detector.git
cd zone-fade-detector

# Build Docker environment
docker-compose build

# Set up configuration
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Framework Shakedown
```bash
# Run MACD test to validate framework
docker-compose run zone-fade-detector python -m strategy_testing.macd_shakedown

# View results
ls results/strategy_001_macd_crossover_*/
```

### 4. Create Your Strategy
- Follow [Strategy Development Guide](STRATEGY_DEVELOPMENT.md)
- Implement `BaseStrategy` interface
- Test your strategy
- Publish results

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 src/
python -m black src/
```

### Adding New Strategies
1. Implement `BaseStrategy` interface
2. Add to `strategies/__init__.py`
3. Create unit tests
4. Test with framework
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

- **Timothy Masters** - "Permutation and Randomization Tests for Trading System Development"
- **mcpt Repository** - [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) - Monte Carlo permutation tests implementation
- **Alpaca API** - Historical data and paper trading
- **Polygon API** - Alternative data source for redundancy

## 📞 Support

- **Documentation**: Check this documentation first
- **Issues**: Report issues on GitHub
- **Community**: Join the community discussions
- **Professional Support**: Contact for commercial support

---

*This framework provides a professional-grade strategy testing system that can validate any trading idea with scientific rigor while maintaining complete reproducibility and historical tracking through GitHub integration.*