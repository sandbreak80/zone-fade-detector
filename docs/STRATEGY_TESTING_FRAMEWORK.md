# Trading Strategy Testing Framework

## Overview

The **Trading Strategy Testing Framework** is a scientific, reproducible system for validating trading strategies using rigorous statistical methods. Built on the foundation of the Zone Fade Detector project, this framework standardizes strategy testing through a 4-step validation battery that prevents overfitting, ensures reproducibility, and automatically publishes results to GitHub for historical tracking.

## Key Features

- **Scientific Validation**: 4-step statistical testing battery prevents overfitting
- **Reproducibility**: Docker environment + version control + comprehensive metadata
- **Zero-Friction Extension**: New strategies require zero framework changes
- **Professional Reporting**: Standardized metrics and GitHub publishing
- **Bar-Level Returns**: Proper look-ahead prevention and robust statistical analysis

## Quick Start

```bash
# Run MACD framework shakedown test
docker-compose run zone-fade-detector python -m strategy_testing.macd_shakedown

# View results
ls results/strategy_001_macd_crossover_*/
```

## Architecture

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

## 4-Step Validation Process

### 1. In-Sample Excellence
- Parameter optimization on training data
- Equity curve analysis and stability checks
- Parameter surface mapping

### 2. In-Sample Monte Carlo Permutation Test (IMCPT)
- 1,000 permutations of training data
- Re-optimize on each permutation
- Target: p < 1% (genuine patterns vs. selection bias)

### 3. Walk-Forward Test (WFT)
- Rolling 4-year training windows, 30-day retrain
- True out-of-sample performance
- Performance degradation analysis

### 4. Walk-Forward Permutation Test (WFPT)
- 200 permutations of OOS segments only
- Re-run walk-forward pipeline
- Target: p ≤ 5% (1 year OOS) or p ≤ 1% (2+ years OOS)

## Standard Metrics

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

## Getting Started

1. **Read the [Architecture Guide](ARCHITECTURE.md)** - Understand the system design
2. **Review [Validation Methodology](VALIDATION_METHODOLOGY.md)** - Learn the 4-step process
3. **Follow [Strategy Development Guide](STRATEGY_DEVELOPMENT.md)** - Create your first strategy
4. **Check [API Reference](API_REFERENCE.md)** - Detailed technical documentation

## Documentation Structure

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[VALIDATION_METHODOLOGY.md](VALIDATION_METHODOLOGY.md)** - 4-step validation process
- **[STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md)** - How to create new strategies
- **[API_REFERENCE.md](API_REFERENCE.md)** - Technical API documentation
- **[REPORTING_STANDARDS.md](REPORTING_STANDARDS.md)** - Metrics and visualization standards
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** - Ensuring reproducible results

## Reference Implementation

The [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) repository provides a complete reference implementation of the methodology described in this framework. Key files include:

- **`bar_permute.py`** - Bar permutation algorithm for destroying temporal structure
- **`donchian.py`** - Donchian strategy implementation (similar to our MACD test)
- **`insample_donchian_mcpt.py`** - In-sample Monte Carlo permutation test example
- **`walkforward_donchian_mcpt.py`** - Walk-forward permutation test example

This repository demonstrates the exact 4-step validation process we implement in our framework.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and contribution process.

## License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.
