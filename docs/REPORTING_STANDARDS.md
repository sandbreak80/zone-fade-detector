# Reporting Standards

## Overview

The Trading Strategy Testing Framework uses standardized reporting to ensure consistent, comparable results across all strategies. This document defines the reporting standards, metrics, visualizations, and output formats used throughout the framework.

## Report Structure

### Directory Layout

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

## Standard Metrics

### Performance Metrics

#### Returns Metrics
- **Total Return**: Cumulative return over entire period
- **Annualized Return**: Annualized return rate
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **MAR (Return/Max DD)**: Return divided by maximum drawdown

#### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Maximum Drawdown Duration**: Longest time in drawdown
- **VaR (95%)**: 95% Value at Risk
- **CVaR (95%)**: 95% Conditional Value at Risk
- **Tail Ratio**: 95th percentile / 5th percentile returns
- **Omega Ratio**: Probability-weighted return measure

#### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Average Win**: Mean profit per winning trade
- **Average Loss**: Mean loss per losing trade
- **Risk/Reward**: Average win / average loss
- **Trade Count**: Total number of trades
- **Trade Frequency**: Trades per month/quarter

#### Risk-Adjusted Metrics
- **Recovery Time**: Time to recover from maximum drawdown
- **Win Streak**: Longest consecutive winning trades
- **Loss Streak**: Longest consecutive losing trades
- **Exposure**: Percentage of time in market
- **Turnover**: Portfolio turnover rate

### Statistical Validation Metrics

#### Permutation Test Results
- **IMCPT P-value**: In-sample permutation test significance
- **WFPT P-value**: Walk-forward permutation test significance
- **Real Score Percentile**: Where real score ranks among permutations
- **Permutation Count**: Number of permutations tested

#### Confidence Intervals
- **Profit Factor CI**: 95% confidence interval for profit factor
- **Sharpe Ratio CI**: 95% confidence interval for Sharpe ratio
- **Max Drawdown CI**: 95% confidence interval for maximum drawdown
- **Win Rate CI**: 95% confidence interval for win rate

#### Regime Analysis
- **Bull Market Performance**: Performance during bull markets
- **Bear Market Performance**: Performance during bear markets
- **High Volatility Performance**: Performance during high volatility
- **Low Volatility Performance**: Performance during low volatility

## Visualization Standards

### Required Plots

#### 1. Equity Curve
**Purpose**: Show cumulative performance over time
**Elements**:
- Strategy equity curve
- Buy-and-hold benchmark
- In-sample vs. out-of-sample periods
- Key performance milestones

**Format**: Line chart with dual y-axis for equity and drawdown

#### 2. Drawdown Chart
**Purpose**: Visualize risk and recovery periods
**Elements**:
- Drawdown percentage over time
- Maximum drawdown highlight
- Recovery periods
- Risk thresholds

**Format**: Area chart with negative values

#### 3. Monthly Returns Heatmap
**Purpose**: Show performance patterns by calendar month
**Elements**:
- Monthly returns matrix
- Color coding for positive/negative returns
- Year-over-year comparison
- Seasonal patterns

**Format**: Heatmap with months on x-axis, years on y-axis

#### 4. Parameter Surface
**Purpose**: Visualize optimization landscape
**Elements**:
- 3D surface plot of parameter combinations
- Performance contours
- Optimal parameter region
- Stability analysis

**Format**: 3D surface plot or contour plot

#### 5. Permutation Histogram
**Purpose**: Show statistical significance of results
**Elements**:
- Distribution of permutation scores
- Real score marker
- P-value annotation
- Confidence intervals

**Format**: Histogram with vertical line for real score

### Optional Plots

#### 6. Regime Analysis
**Purpose**: Performance by market conditions
**Elements**:
- Performance by market regime
- Regime identification
- Performance comparison
- Risk-adjusted returns

**Format**: Bar chart or box plot

#### 7. Trade Analysis
**Purpose**: Individual trade performance
**Elements**:
- Trade P&L distribution
- Win/loss analysis
- Trade duration
- Entry/exit analysis

**Format**: Histogram or scatter plot

#### 8. Rolling Metrics
**Purpose**: Time-varying performance metrics
**Elements**:
- Rolling Sharpe ratio
- Rolling maximum drawdown
- Rolling win rate
- Performance stability

**Format**: Line chart with time series

## Report Formats

### SUMMARY.md

Executive summary report in Markdown format:

```markdown
# Strategy Test Results: MACD Crossover

## Test Configuration
- **Strategy**: MACD Crossover
- **Symbols**: QQQ, SPY, AAPL, MSFT, GOOGL, AMZN, TSLA
- **Period**: 2010-01-01 to 2025-01-01
- **Timeframe**: 1-hour bars
- **Parameters**: Fast=12, Slow=26, Signal=9

## Key Results
- **Total Return**: 15.2%
- **Sharpe Ratio**: 0.85
- **Max Drawdown**: -12.3%
- **Win Rate**: 45.2%
- **Profit Factor**: 1.34

## Validation Results
- **IMCPT P-value**: 0.023 (PASS)
- **WFPT P-value**: 0.087 (PASS)
- **Acceptance**: ✅ ACCEPTED

## Conclusion
Strategy shows genuine edge with statistical significance.
```

### metrics.csv

Machine-readable metrics in CSV format:

```csv
metric,value,confidence_interval_lower,confidence_interval_upper
total_return,0.152,0.134,0.170
sharpe_ratio,0.85,0.72,0.98
max_drawdown,-0.123,-0.145,-0.101
win_rate,0.452,0.421,0.483
profit_factor,1.34,1.21,1.47
```

### metadata.json

Run metadata and configuration:

```json
{
  "strategy_name": "MACD Crossover",
  "test_timestamp": "2024-01-15T10:30:00Z",
  "framework_version": "1.0.0",
  "python_version": "3.11.0",
  "git_commit": "abc123def456",
  "random_seeds": {
    "permutation": 42,
    "ticker_selection": 123
  },
  "data_sources": {
    "alpaca_version": "2.0.0",
    "polygon_version": "1.0.0"
  },
  "hardware": {
    "cpu": "Intel i7-12700K",
    "memory": "32GB",
    "os": "Linux 6.14.0-33-generic"
  },
  "configuration": {
    "symbols": ["QQQ", "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "start_date": "2010-01-01",
    "end_date": "2025-01-01",
    "timeframe": "1h",
    "initial_capital": 10000,
    "commission": 0.001,
    "slippage": 0.0005
  }
}
```

## Quality Standards

### Data Quality
- **Completeness**: All required data points present
- **Accuracy**: Data validated against source
- **Consistency**: Uniform data format across sources
- **Timeliness**: Data up-to-date and relevant

### Calculation Accuracy
- **Precision**: Calculations to appropriate decimal places
- **Validation**: Results cross-checked against benchmarks
- **Reproducibility**: Identical results across runs
- **Documentation**: All calculations documented

### Visualization Quality
- **Clarity**: Charts clearly labeled and readable
- **Consistency**: Uniform styling across all plots
- **Accessibility**: Color-blind friendly palettes
- **Resolution**: High-resolution images for publication

### Report Completeness
- **Coverage**: All required sections present
- **Accuracy**: Information factually correct
- **Clarity**: Language clear and concise
- **Consistency**: Format consistent across reports

## Acceptance Criteria

### Statistical Significance
- **IMCPT P-value**: < 1% (strong evidence of genuine patterns)
- **WFPT P-value**: < 5% (1 year OOS) or < 1% (2+ years OOS)
- **Confidence Intervals**: All key metrics have reasonable CIs
- **Bootstrap Validation**: Results stable across bootstrap samples

### Performance Thresholds
- **Profit Factor**: > 1.2 (net of costs)
- **Sharpe Ratio**: > 0.5 (risk-adjusted returns)
- **Maximum Drawdown**: < -20% (risk management)
- **Win Rate**: > 40% (trading success rate)

### Economic Viability
- **Transaction Costs**: Performance net of realistic costs
- **Capacity**: Strategy capacity sufficient for intended use
- **Liquidity**: Sufficient liquidity for strategy execution
- **Market Impact**: Minimal market impact assumptions

## Report Generation

### Automated Generation
- **Template-based**: Use standardized report templates
- **Data-driven**: Generate reports from test results
- **Version-controlled**: Track report versions and changes
- **Quality-assured**: Automated validation of report completeness

### Manual Review
- **Expert Review**: Human review of critical results
- **Peer Review**: Independent validation of methodology
- **Quality Assurance**: Systematic quality checks
- **Documentation**: Review process documented

## Customization

### Strategy-Specific Metrics
- **Custom Indicators**: Strategy-specific performance measures
- **Regime Analysis**: Market condition-specific performance
- **Risk Metrics**: Strategy-specific risk measures
- **Benchmark Comparison**: Relevant benchmark comparisons

### Visualization Customization
- **Chart Types**: Strategy-appropriate chart types
- **Color Schemes**: Brand-consistent color palettes
- **Layout**: Custom report layouts
- **Annotations**: Strategy-specific annotations

### Report Templates
- **Executive Summary**: High-level strategy overview
- **Technical Report**: Detailed technical analysis
- **Research Report**: Academic-style research report
- **Presentation**: Slide-ready presentation format

## Best Practices

### Report Design
- **Consistency**: Uniform formatting across all reports
- **Clarity**: Clear, concise language
- **Completeness**: All required information included
- **Accuracy**: Factually correct and validated

### Data Presentation
- **Appropriate Precision**: Right number of decimal places
- **Significant Figures**: Consistent significant figures
- **Units**: Clear units and measurements
- **Context**: Sufficient context for interpretation

### Visualization
- **Appropriate Charts**: Right chart type for data
- **Clear Labels**: All axes and data clearly labeled
- **Consistent Styling**: Uniform appearance across charts
- **Accessibility**: Accessible to all users

### Quality Assurance
- **Validation**: All results validated
- **Review**: Human review of critical results
- **Documentation**: Process and results documented
- **Reproducibility**: Results can be reproduced

## Conclusion

Standardized reporting ensures consistent, comparable results across all strategies tested in the framework. By following these standards, users can confidently compare strategies, track performance over time, and make informed decisions about strategy deployment.

The reporting standards are designed to be comprehensive yet practical, providing both human-readable summaries and machine-readable data for further analysis. All reports include sufficient metadata to ensure reproducibility and enable historical tracking of strategy performance.
