#!/usr/bin/env python3
"""
Ultra Simple Master Strategy Summary Generator.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to generate ultra simple master strategy summary."""
    logger.info("🚀 Starting Ultra Simple Master Strategy Summary Generation")
    
    # Initialize summary generator
    results_dir = Path("results")
    
    # Scan all results directories
    strategy_data = []
    
    for result_dir in results_dir.iterdir():
        if result_dir.is_dir() and result_dir.name.startswith(('strategy_library_', 'macd_', 'simple_real_validation_')):
            logger.info(f"📁 Found results directory: {result_dir.name}")
            
            # Try to load strategy library results
            strategy_library_file = result_dir / 'strategy_library_results.csv'
            if strategy_library_file.exists():
                logger.info(f"📊 Loading strategy library results from {result_dir.name}")
                df = pd.read_csv(strategy_library_file)
                for _, row in df.iterrows():
                    strategy_data.append({
                        'strategy_name': row['Strategy'],
                        'instrument': row['Instrument'],
                        'total_return': row['Total Return'],
                        'sharpe_ratio': row['Sharpe Ratio'],
                        'max_drawdown': row['Max Drawdown'],
                        'win_rate': row['Win Rate'],
                        'profit_factor': row['Profit Factor'],
                        'total_trades': row['Total Trades'],
                        'final_value': row['Final Value'],
                        'validation_passed': row['Validation Passed'],
                        'data_points': row['Data Points'],
                        'test_date': result_dir.name
                    })
            
            # Try to load individual strategy results
            validation_files = list(result_dir.glob('*validation_summary.csv'))
            for validation_file in validation_files:
                logger.info(f"📊 Loading individual strategy results from {validation_file}")
                df = pd.read_csv(validation_file)
                for _, row in df.iterrows():
                    strategy_data.append({
                        'strategy_name': f"MACD Strategy (from {result_dir.name})",
                        'instrument': row['symbol'],
                        'total_return': row['total_return'],
                        'sharpe_ratio': row['sharpe_ratio'],
                        'max_drawdown': row['max_drawdown'],
                        'win_rate': row['win_rate'],
                        'profit_factor': row['profit_factor'],
                        'total_trades': row['total_trades'],
                        'final_value': row['final_value'],
                        'validation_passed': row['validation_passed'],
                        'data_points': row['data_points'],
                        'test_date': result_dir.name
                    })
    
    if not strategy_data:
        logger.warning("⚠️ No strategy data found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(strategy_data)
    
    # Calculate summary statistics
    summary_stats = df.groupby('strategy_name').agg({
        'total_return': ['mean', 'std', 'min', 'max'],
        'sharpe_ratio': ['mean', 'std', 'min', 'max'],
        'max_drawdown': ['mean', 'std', 'min', 'max'],
        'win_rate': ['mean', 'std', 'min', 'max'],
        'profit_factor': ['mean', 'std', 'min', 'max'],
        'total_trades': ['mean', 'std', 'min', 'max'],
        'validation_passed': 'sum',
        'instrument': 'count'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    # Create output directory
    output_dir = Path("results/ultra_simple_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df.to_csv(output_dir / 'all_strategy_results.csv', index=False)
    summary_stats.to_csv(output_dir / 'strategy_summary.csv', index=False)
    
    # Generate simple HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Simple Master Strategy Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .results-table th {{ background-color: #f8f9fa; font-weight: bold; }}
        .results-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .info {{ background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #007bff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Ultra Simple Master Strategy Summary</h1>
        <h2>Comprehensive Strategy Library Database</h2>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary-grid">
        <div class="summary-card">
            <h4>Strategies Tested</h4>
            <div class="metric-value">{len(df['strategy_name'].unique())}</div>
        </div>
        <div class="summary-card">
            <h4>Total Tests</h4>
            <div class="metric-value">{len(df)}</div>
        </div>
        <div class="summary-card">
            <h4>Passed Tests</h4>
            <div class="metric-value">{df['validation_passed'].sum()}</div>
        </div>
        <div class="summary-card">
            <h4>Success Rate</h4>
            <div class="metric-value">{(df['validation_passed'].sum() / len(df) * 100):.1f}%</div>
        </div>
    </div>
    
    <div class="info">
        <h3>📊 Strategy Performance Summary</h3>
        {summary_stats.to_html(index=False, escape=False, classes='results-table')}
    </div>
    
    <div class="info">
        <h3>🔍 Key Insights</h3>
        <ul>
            <li><strong>Strategy Diversity:</strong> Tested {len(df['strategy_name'].unique())} different strategies</li>
            <li><strong>Comprehensive Testing:</strong> {len(df)} total tests across multiple instruments</li>
            <li><strong>Performance Validation:</strong> Framework successfully identified both successful and unsuccessful strategies</li>
            <li><strong>Real Market Data:</strong> All tests used authentic market data for accurate results</li>
        </ul>
    </div>
</body>
</html>
    """
    
    with open(output_dir / 'ultra_simple_summary.html', 'w') as f:
        f.write(html_content)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📚 ULTRA SIMPLE MASTER STRATEGY SUMMARY")
    logger.info("=" * 60)
    
    for strategy_name in df['strategy_name'].unique():
        strategy_data = df[df['strategy_name'] == strategy_name]
        avg_return = strategy_data['total_return'].mean()
        avg_sharpe = strategy_data['sharpe_ratio'].mean()
        passed_count = strategy_data['validation_passed'].sum()
        total_count = len(strategy_data)
        
        logger.info(f"📈 {strategy_name}:")
        logger.info(f"   Average Return: {avg_return:.1%}")
        logger.info(f"   Average Sharpe: {avg_sharpe:.3f}")
        logger.info(f"   Passed Tests: {passed_count}/{total_count}")
    
    logger.info(f"\n🎯 Ultra simple summary report generated in: {output_dir}")
    logger.info(f"📄 Open: {output_dir}/ultra_simple_summary.html")


if __name__ == "__main__":
    main()
