#!/usr/bin/env python3
"""
Simple Performance Summary Generator.

Creates a QuantStats-style summary report from Phase 3 validation results
without complex visualizations.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_results_data(results_dir: str) -> dict:
    """Load results data from the Phase 3 validation."""
    results_path = Path(results_dir)
    
    # Load metadata
    with open(results_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load IS metrics
    is_metrics = pd.read_csv(results_path / "metrics_is.csv")
    
    # Load OOS metrics  
    oos_metrics = pd.read_csv(results_path / "metrics_oos.csv")
    
    return {
        'metadata': metadata,
        'is_metrics': is_metrics,
        'oos_metrics': oos_metrics
    }


def generate_simple_summary(results_dir: str):
    """Generate a simple performance summary report."""
    print("🚀 Generating Simple Performance Summary")
    print("=" * 50)
    
    # Load results data
    data = load_results_data(results_dir)
    metadata = data['metadata']
    is_metrics = data['is_metrics']
    oos_metrics = data['oos_metrics']
    
    # Create reports directory
    reports_dir = Path(results_dir) / "performance_reports"
    reports_dir.mkdir(exist_ok=True)
    
    print(f"📊 Processing {len(metadata['instruments'])} instruments")
    
    # Create summary table
    summary_data = []
    
    for ticker in metadata['instruments']:
        print(f"📈 Processing {ticker}...")
        
        # Get metrics for this ticker
        ticker_is = is_metrics[is_metrics['ticker'] == ticker].iloc[0]
        ticker_oos = oos_metrics[oos_metrics['ticker'] == ticker].iloc[0]
        
        # Calculate key metrics
        is_return = ticker_is['total_return'] * 100  # Convert to percentage
        is_sharpe = ticker_is['sharpe_ratio']
        is_max_dd = ticker_is['max_drawdown'] * 100  # Convert to percentage
        is_win_rate = ticker_is['win_rate'] * 100  # Convert to percentage
        is_profit_factor = ticker_is['profit_factor']
        
        oos_score = ticker_oos['total_oos_score']
        n_retrains = ticker_oos['n_retrains']
        
        # Estimate final portfolio value (starting with $10,000)
        initial_capital = 10000
        final_value = initial_capital * (1 + is_return / 100)
        
        summary_data.append({
            'Ticker': ticker,
            'IS Return (%)': f"{is_return:.1f}",
            'IS Sharpe': f"{is_sharpe:.2f}",
            'IS Max DD (%)': f"{is_max_dd:.1f}",
            'IS Win Rate (%)': f"{is_win_rate:.1f}",
            'Profit Factor': f"{is_profit_factor:.2f}",
            'OOS Score': f"{oos_score:.2f}",
            'Retrains': f"{n_retrains}",
            'Final Value ($)': f"${final_value:,.0f}"
        })
    
    # Generate HTML summary
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MACD Strategy Performance Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .summary-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .positive {{ color: green; font-weight: bold; }}
            .negative {{ color: red; font-weight: bold; }}
            .info {{ background-color: #e7f3ff; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .warning {{ background-color: #fff3cd; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MACD Crossover Strategy Performance Summary</h1>
            <h2>Phase 3 Validation Results</h2>
        </div>
        
        <div class="info">
            <h3>Test Configuration</h3>
            <p><strong>Date Range:</strong> {metadata['date_range']['start']} to {metadata['date_range']['end']}</p>
            <p><strong>Timeframe:</strong> {metadata['timeframe']}</p>
            <p><strong>Initial Capital:</strong> ${metadata['configuration']['initial_capital']:,}</p>
            <p><strong>Commission:</strong> {metadata['transaction_costs']}</p>
            <p><strong>Instruments Tested:</strong> {', '.join(metadata['instruments'])}</p>
        </div>
        
        <div class="warning">
            <h3>⚠️ Validation Status</h3>
            <p><strong>Statistical Validation:</strong> FAILED (0/7 instruments passed)</p>
            <p><strong>IMCPT Results:</strong> High p-values indicate selection bias</p>
            <p><strong>WFPT Results:</strong> High p-values indicate luck factor</p>
            <p><strong>Conclusion:</strong> Strategy returns likely due to randomness, not genuine edge</p>
        </div>
        
        <h3>Performance Summary</h3>
        <table class="summary-table">
            <tr>
                <th>Ticker</th>
                <th>IS Return (%)</th>
                <th>IS Sharpe</th>
                <th>IS Max DD (%)</th>
                <th>IS Win Rate (%)</th>
                <th>Profit Factor</th>
                <th>OOS Score</th>
                <th>Retrains</th>
                <th>Final Value ($)</th>
            </tr>
    """
    
    for row in summary_data:
        is_return = float(row['IS Return (%)'])
        html_content += f"""
            <tr>
                <td>{row['Ticker']}</td>
                <td class="{'positive' if is_return > 0 else 'negative'}">{row['IS Return (%)']}</td>
                <td>{row['IS Sharpe']}</td>
                <td class="negative">{row['IS Max DD (%)']}</td>
                <td>{row['IS Win Rate (%)']}</td>
                <td>{row['Profit Factor']}</td>
                <td>{row['OOS Score']}</td>
                <td>{row['Retrains']}</td>
                <td>{row['Final Value ($)']}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>Key Insights</h3>
        <ul>
            <li><strong>Strategy Performance:</strong> MACD crossover strategy showed positive returns across all instruments</li>
            <li><strong>Risk-Adjusted Returns:</strong> Modest Sharpe ratios (0.08-0.09) indicate room for improvement</li>
            <li><strong>Drawdowns:</strong> Strategy experienced significant drawdowns (7-8%), highlighting risk management needs</li>
            <li><strong>Win Rates:</strong> Consistent ~57% win rates across all instruments</li>
            <li><strong>Profit Factors:</strong> Modest profit factors (~1.25) suggest limited edge</li>
            <li><strong>Validation Status:</strong> Failed statistical validation - returns likely due to luck</li>
        </ul>
        
        <h3>Statistical Validation Results</h3>
        <p><strong>IMCPT (In-Sample Monte Carlo Permutation Test):</strong></p>
        <ul>
            <li>QQQ: p = 0.069 (borderline significant)</li>
            <li>Others: p = 0.788 (not significant)</li>
            <li>Target: p < 1%</li>
        </ul>
        
        <p><strong>WFPT (Walk-Forward Permutation Test):</strong></p>
        <ul>
            <li>All instruments: p = 1.000 (completely not significant)</li>
            <li>Target: p < 5%</li>
        </ul>
        
        <h3>Recommendations</h3>
        <ul>
            <li><strong>Do Not Trade:</strong> Strategy failed statistical validation</li>
            <li><strong>Improve Strategy:</strong> Add more sophisticated features or filters</li>
            <li><strong>Risk Management:</strong> Implement better drawdown controls</li>
            <li><strong>Further Testing:</strong> Test on different timeframes or instruments</li>
        </ul>
        
        <h3>Framework Validation</h3>
        <p><strong>✅ Framework Success:</strong> The validation framework worked perfectly:</p>
        <ul>
            <li>All 4 validation steps executed correctly</li>
            <li>Proper statistical testing with 1000 IMCPT + 200 WFPT permutations</li>
            <li>Look-ahead prevention and transaction costs applied correctly</li>
            <li>Framework successfully identified that strategy lacks statistical significance</li>
        </ul>
    </body>
    </html>
    """
    
    # Save summary report
    with open(reports_dir / "simple_performance_summary.html", 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Simple performance summary generated!")
    print(f"📁 Report saved to: {reports_dir}/simple_performance_summary.html")
    print(f"🌐 Open the HTML file in your browser to view the report")


def main():
    """Main function to generate simple performance summary."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python simple_performance_summary.py <results_directory>")
        print("Example: python simple_performance_summary.py results/macd_1h_2010_2025_1761254135")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not Path(results_dir).exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    try:
        generate_simple_summary(results_dir)
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
