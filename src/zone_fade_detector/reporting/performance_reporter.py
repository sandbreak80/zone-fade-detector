"""
Performance Reporter for Trading Strategy Testing Framework.

Generates QuantStats-style performance reports with comprehensive metrics,
visualizations, and buy-and-hold comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceReporter:
    """
    Generates comprehensive performance reports similar to QuantStats.
    
    Creates detailed performance analysis with:
    - Strategy vs Benchmark comparison
    - Key performance metrics
    - Visual charts (equity curves, drawdowns, monthly heatmaps)
    - Risk analysis
    - Trade statistics
    """
    
    def __init__(self, initial_capital: float = 10000, benchmark_ticker: str = "SPY"):
        """
        Initialize the performance reporter.
        
        Args:
            initial_capital: Starting portfolio value
            benchmark_ticker: Benchmark for comparison (SPY, QQQ, etc.)
        """
        self.initial_capital = initial_capital
        self.benchmark_ticker = benchmark_ticker
    
    def generate_performance_report(self, 
                                  strategy_returns: List[float],
                                  benchmark_returns: List[float],
                                  timestamps: List[datetime],
                                  ticker: str,
                                  strategy_name: str,
                                  output_dir: str) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            strategy_returns: List of strategy returns
            benchmark_returns: List of benchmark returns
            timestamps: List of timestamps for returns
            ticker: Instrument ticker
            strategy_name: Name of the strategy
            output_dir: Directory to save reports
        
        Returns:
            Dictionary with all performance metrics
        """
        # Convert to pandas Series for easier analysis
        strategy_series = pd.Series(strategy_returns, index=timestamps)
        benchmark_series = pd.Series(benchmark_returns, index=timestamps)
        
        # Calculate cumulative returns
        strategy_cumulative = (1 + strategy_series).cumprod()
        benchmark_cumulative = (1 + benchmark_series).cumprod()
        
        # Calculate portfolio values
        strategy_portfolio = self.initial_capital * strategy_cumulative
        benchmark_portfolio = self.initial_capital * benchmark_cumulative
        
        # Generate metrics
        metrics = self._calculate_metrics(strategy_series, benchmark_series, 
                                        strategy_portfolio, benchmark_portfolio)
        
        # Generate visualizations
        self._create_equity_curve_plot(strategy_portfolio, benchmark_portfolio, 
                                     ticker, strategy_name, output_dir)
        self._create_drawdown_plot(strategy_series, benchmark_series, 
                                  ticker, strategy_name, output_dir)
        self._create_monthly_heatmap(strategy_series, ticker, strategy_name, output_dir)
        self._create_rolling_metrics_plot(strategy_series, benchmark_series, 
                                         ticker, strategy_name, output_dir)
        
        # Save metrics to JSON
        self._save_metrics_json(metrics, ticker, strategy_name, output_dir)
        
        # Generate HTML report
        self._generate_html_report(metrics, ticker, strategy_name, output_dir)
        
        return metrics
    
    def _calculate_metrics(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,
                          strategy_portfolio: pd.Series, benchmark_portfolio: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Basic returns
        total_return_strategy = (strategy_portfolio.iloc[-1] / self.initial_capital - 1) * 100
        total_return_benchmark = (benchmark_portfolio.iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized metrics
        years = len(strategy_returns) / 252  # Assuming 252 trading days per year
        cagr_strategy = (strategy_portfolio.iloc[-1] / self.initial_capital) ** (1/years) - 1
        cagr_benchmark = (benchmark_portfolio.iloc[-1] / self.initial_capital) ** (1/years) - 1
        
        # Risk metrics
        sharpe_strategy = self._calculate_sharpe_ratio(strategy_returns)
        sharpe_benchmark = self._calculate_sharpe_ratio(benchmark_returns)
        
        max_dd_strategy = self._calculate_max_drawdown(strategy_portfolio)
        max_dd_benchmark = self._calculate_max_drawdown(benchmark_portfolio)
        
        volatility_strategy = strategy_returns.std() * np.sqrt(252) * 100
        volatility_benchmark = benchmark_returns.std() * np.sqrt(252) * 100
        
        # Win rate and other metrics
        win_rate_strategy = (strategy_returns > 0).mean() * 100
        win_rate_benchmark = (benchmark_returns > 0).mean() * 100
        
        # Sortino ratio
        sortino_strategy = self._calculate_sortino_ratio(strategy_returns)
        sortino_benchmark = self._calculate_sortino_ratio(benchmark_returns)
        
        # Calmar ratio
        calmar_strategy = cagr_strategy / abs(max_dd_strategy) if max_dd_strategy != 0 else 0
        calmar_benchmark = cagr_benchmark / abs(max_dd_benchmark) if max_dd_benchmark != 0 else 0
        
        # Beta and Alpha
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        alpha = cagr_strategy - (0.02 + beta * (cagr_benchmark - 0.02))  # Assuming 2% risk-free rate
        
        # Correlation
        correlation = strategy_returns.corr(benchmark_returns) * 100
        
        # Best and worst days
        best_day_strategy = strategy_returns.max() * 100
        worst_day_strategy = strategy_returns.min() * 100
        best_day_benchmark = benchmark_returns.max() * 100
        worst_day_benchmark = benchmark_returns.min() * 100
        
        # Monthly returns analysis
        monthly_strategy = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_benchmark = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        best_month_strategy = monthly_strategy.max() * 100
        worst_month_strategy = monthly_strategy.min() * 100
        best_month_benchmark = monthly_benchmark.max() * 100
        worst_month_benchmark = monthly_benchmark.min() * 100
        
        return {
            'strategy_name': 'MACD Crossover Strategy',
            'ticker': strategy_returns.index[0].strftime('%Y-%m-%d') if len(strategy_returns) > 0 else 'N/A',
            'date_range': {
                'start': strategy_returns.index[0].strftime('%Y-%m-%d') if len(strategy_returns) > 0 else 'N/A',
                'end': strategy_returns.index[-1].strftime('%Y-%m-%d') if len(strategy_returns) > 0 else 'N/A',
                'days': len(strategy_returns)
            },
            'portfolio_values': {
                'strategy_start': self.initial_capital,
                'strategy_end': strategy_portfolio.iloc[-1],
                'benchmark_start': self.initial_capital,
                'benchmark_end': benchmark_portfolio.iloc[-1]
            },
            'returns': {
                'strategy_total_return': total_return_strategy,
                'benchmark_total_return': total_return_benchmark,
                'strategy_cagr': cagr_strategy * 100,
                'benchmark_cagr': cagr_benchmark * 100,
                'outperformance': total_return_strategy - total_return_benchmark
            },
            'risk_metrics': {
                'strategy_sharpe': sharpe_strategy,
                'benchmark_sharpe': sharpe_benchmark,
                'strategy_sortino': sortino_strategy,
                'benchmark_sortino': sortino_benchmark,
                'strategy_calmar': calmar_strategy,
                'benchmark_calmar': calmar_benchmark,
                'strategy_volatility': volatility_strategy,
                'benchmark_volatility': volatility_benchmark
            },
            'drawdowns': {
                'strategy_max_dd': max_dd_strategy * 100,
                'benchmark_max_dd': max_dd_benchmark * 100,
                'strategy_max_dd_date': self._get_max_dd_date(strategy_portfolio),
                'benchmark_max_dd_date': self._get_max_dd_date(benchmark_portfolio)
            },
            'win_rates': {
                'strategy_win_rate': win_rate_strategy,
                'benchmark_win_rate': win_rate_benchmark
            },
            'best_worst': {
                'strategy_best_day': best_day_strategy,
                'strategy_worst_day': worst_day_strategy,
                'benchmark_best_day': best_day_benchmark,
                'benchmark_worst_day': worst_day_benchmark,
                'strategy_best_month': best_month_strategy,
                'strategy_worst_month': worst_month_strategy,
                'benchmark_best_month': best_month_benchmark,
                'benchmark_worst_month': worst_month_benchmark
            },
            'correlation': {
                'beta': beta,
                'alpha': alpha * 100,
                'correlation': correlation
            }
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_volatility if downside_volatility != 0 else 0
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def _calculate_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta."""
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    def _get_max_dd_date(self, portfolio_values: pd.Series) -> str:
        """Get the date of maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_dd_idx = drawdown.idxmin()
        return max_dd_idx.strftime('%Y-%m-%d') if max_dd_idx else 'N/A'
    
    def _create_equity_curve_plot(self, strategy_portfolio: pd.Series, benchmark_portfolio: pd.Series,
                                 ticker: str, strategy_name: str, output_dir: str):
        """Create equity curve comparison plot."""
        plt.figure(figsize=(12, 8))
        
        plt.plot(strategy_portfolio.index, strategy_portfolio, label=f'{strategy_name}', linewidth=2)
        plt.plot(benchmark_portfolio.index, benchmark_portfolio, label=f'{ticker} (Benchmark)', linewidth=2)
        
        plt.title(f'Portfolio Performance: {strategy_name} vs {ticker}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curve_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_plot(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,
                             ticker: str, strategy_name: str, output_dir: str):
        """Create drawdown comparison plot."""
        # Calculate drawdowns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        strategy_peak = strategy_cumulative.expanding().max()
        benchmark_peak = benchmark_cumulative.expanding().max()
        
        strategy_dd = (strategy_cumulative - strategy_peak) / strategy_peak * 100
        benchmark_dd = (benchmark_cumulative - benchmark_peak) / benchmark_peak * 100
        
        plt.figure(figsize=(12, 8))
        
        plt.fill_between(strategy_dd.index, strategy_dd, 0, alpha=0.7, label=f'{strategy_name}')
        plt.fill_between(benchmark_dd.index, benchmark_dd, 0, alpha=0.7, label=f'{ticker} (Benchmark)')
        
        plt.title(f'Drawdown Analysis: {strategy_name} vs {ticker}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drawdown_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_monthly_heatmap(self, strategy_returns: pd.Series, ticker: str, 
                               strategy_name: str, output_dir: str):
        """Create monthly returns heatmap."""
        # Resample to monthly returns
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        
        pivot_table = monthly_returns.pivot_table(values=0, index='Year', columns='Month', fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'})
        
        plt.title(f'Monthly Returns Heatmap: {strategy_name} ({ticker})', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(range(1, 13), month_labels)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monthly_heatmap_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rolling_metrics_plot(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,
                                   ticker: str, strategy_name: str, output_dir: str):
        """Create rolling Sharpe ratio plot."""
        # Calculate rolling Sharpe ratios (252-day window)
        window = 252
        strategy_rolling_sharpe = strategy_returns.rolling(window).apply(
            lambda x: self._calculate_sharpe_ratio(x), raw=False
        )
        benchmark_rolling_sharpe = benchmark_returns.rolling(window).apply(
            lambda x: self._calculate_sharpe_ratio(x), raw=False
        )
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe, 
                label=f'{strategy_name}', linewidth=2)
        plt.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe, 
                label=f'{ticker} (Benchmark)', linewidth=2)
        
        plt.title(f'Rolling Sharpe Ratio (252-day): {strategy_name} vs {ticker}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rolling Sharpe Ratio', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rolling_sharpe_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics_json(self, metrics: Dict[str, Any], ticker: str, 
                          strategy_name: str, output_dir: str):
        """Save metrics to JSON file."""
        output_path = f'{output_dir}/performance_metrics_{ticker}.json'
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def _generate_html_report(self, metrics: Dict[str, Any], ticker: str, 
                            strategy_name: str, output_dir: str):
        """Generate HTML performance report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report: {strategy_name} ({ticker})</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; font-weight: bold; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Report: {strategy_name}</h1>
                <h2>Instrument: {ticker}</h2>
                <p>Date Range: {metrics['date_range']['start']} to {metrics['date_range']['end']}</p>
            </div>
            
            <div class="summary">
                <h3>Quick Summary</h3>
                <p><strong>Strategy Return:</strong> {metrics['returns']['strategy_total_return']:.2f}%</p>
                <p><strong>Benchmark Return:</strong> {metrics['returns']['benchmark_total_return']:.2f}%</p>
                <p><strong>Outperformance:</strong> {metrics['returns']['outperformance']:.2f}%</p>
                <p><strong>Final Portfolio Value:</strong> ${metrics['portfolio_values']['strategy_end']:,.2f}</p>
            </div>
            
            <h3>Key Performance Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Strategy</th>
                    <th>Benchmark</th>
                </tr>
                <tr>
                    <td>Cumulative Return</td>
                    <td class="{'positive' if metrics['returns']['strategy_total_return'] > 0 else 'negative'}">
                        {metrics['returns']['strategy_total_return']:.2f}%
                    </td>
                    <td class="{'positive' if metrics['returns']['benchmark_total_return'] > 0 else 'negative'}">
                        {metrics['returns']['benchmark_total_return']:.2f}%
                    </td>
                </tr>
                <tr>
                    <td>CAGR</td>
                    <td>{metrics['returns']['strategy_cagr']:.2f}%</td>
                    <td>{metrics['returns']['benchmark_cagr']:.2f}%</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{metrics['risk_metrics']['strategy_sharpe']:.2f}</td>
                    <td>{metrics['risk_metrics']['benchmark_sharpe']:.2f}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td class="negative">{metrics['drawdowns']['strategy_max_dd']:.2f}%</td>
                    <td class="negative">{metrics['drawdowns']['benchmark_max_dd']:.2f}%</td>
                </tr>
                <tr>
                    <td>Volatility</td>
                    <td>{metrics['risk_metrics']['strategy_volatility']:.2f}%</td>
                    <td>{metrics['risk_metrics']['benchmark_volatility']:.2f}%</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{metrics['win_rates']['strategy_win_rate']:.1f}%</td>
                    <td>{metrics['win_rates']['benchmark_win_rate']:.1f}%</td>
                </tr>
            </table>
            
            <h3>Charts</h3>
            <p><img src="equity_curve_{ticker}.png" alt="Equity Curve" style="max-width: 100%;"></p>
            <p><img src="drawdown_{ticker}.png" alt="Drawdown Analysis" style="max-width: 100%;"></p>
            <p><img src="monthly_heatmap_{ticker}.png" alt="Monthly Heatmap" style="max-width: 100%;"></p>
            <p><img src="rolling_sharpe_{ticker}.png" alt="Rolling Sharpe" style="max-width: 100%;"></p>
        </body>
        </html>
        """
        
        with open(f'{output_dir}/performance_report_{ticker}.html', 'w') as f:
            f.write(html_content)
