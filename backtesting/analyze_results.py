#!/usr/bin/env python3
"""
Bitcoin Zone Fade Backtest Results Analyzer.

This script analyzes backtest results and generates detailed performance reports.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestAnalyzer:
    """Analyzes backtest results and generates reports."""
    
    def __init__(self, results_file: str = "backtest_results.json"):
        """Initialize analyzer with results file."""
        self.results_file = results_file
        self.results = self.load_results()
    
    def load_results(self) -> dict:
        """Load results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file {self.results_file} not found.")
            print("Run a backtest first: python backtest_bitcoin.py")
            return {}
    
    def print_summary(self):
        """Print a summary of the backtest results."""
        if not self.results:
            return
        
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 40)
        print(f"Total Trades: {self.results.get('total_trades', 0)}")
        print(f"Winning Trades: {self.results.get('winning_trades', 0)}")
        print(f"Losing Trades: {self.results.get('losing_trades', 0)}")
        print(f"Win Rate: {self.results.get('win_rate', 0):.2%}")
        print(f"Total Return: {self.results.get('total_return', 0):.2%}")
        print(f"Profit Factor: {self.results.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {self.results.get('max_drawdown', 0):.2%}")
        print(f"Average Win: ${self.results.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${self.results.get('avg_loss', 0):.2f}")
        print(f"Total P&L: ${self.results.get('total_pnl', 0):.2f}")
        print()
    
    def analyze_performance(self):
        """Analyze performance metrics."""
        if not self.results:
            return
        
        print("📈 PERFORMANCE ANALYSIS")
        print("=" * 30)
        
        # Risk-adjusted returns
        sharpe = self.results.get('sharpe_ratio', 0)
        if sharpe > 1.0:
            print(f"✅ Excellent Sharpe Ratio: {sharpe:.2f}")
        elif sharpe > 0.5:
            print(f"⚠️  Moderate Sharpe Ratio: {sharpe:.2f}")
        else:
            print(f"❌ Poor Sharpe Ratio: {sharpe:.2f}")
        
        # Win rate analysis
        win_rate = self.results.get('win_rate', 0)
        if win_rate > 0.6:
            print(f"✅ High Win Rate: {win_rate:.2%}")
        elif win_rate > 0.5:
            print(f"⚠️  Moderate Win Rate: {win_rate:.2%}")
        else:
            print(f"❌ Low Win Rate: {win_rate:.2%}")
        
        # Profit factor analysis
        profit_factor = self.results.get('profit_factor', 0)
        if profit_factor > 1.5:
            print(f"✅ Strong Profit Factor: {profit_factor:.2f}")
        elif profit_factor > 1.0:
            print(f"⚠️  Moderate Profit Factor: {profit_factor:.2f}")
        else:
            print(f"❌ Poor Profit Factor: {profit_factor:.2f}")
        
        # Drawdown analysis
        max_dd = self.results.get('max_drawdown', 0)
        if max_dd < -0.1:
            print(f"❌ High Drawdown: {max_dd:.2%}")
        elif max_dd < -0.05:
            print(f"⚠️  Moderate Drawdown: {max_dd:.2%}")
        else:
            print(f"✅ Low Drawdown: {max_dd:.2%}")
        
        print()
    
    def generate_recommendations(self):
        """Generate trading recommendations based on results."""
        if not self.results:
            return
        
        print("💡 TRADING RECOMMENDATIONS")
        print("=" * 35)
        
        win_rate = self.results.get('win_rate', 0)
        profit_factor = self.results.get('profit_factor', 0)
        max_dd = self.results.get('max_drawdown', 0)
        
        # Position sizing recommendation
        if max_dd < -0.15:
            print("🔴 REDUCE POSITION SIZE - High drawdown detected")
        elif max_dd < -0.1:
            print("🟡 CONSIDER REDUCING POSITION SIZE - Moderate drawdown")
        else:
            print("🟢 POSITION SIZE OK - Drawdown within acceptable limits")
        
        # Strategy adjustment recommendations
        if win_rate < 0.5:
            print("🔴 INCREASE QRS THRESHOLD - Low win rate suggests need for higher quality setups")
        elif win_rate > 0.7:
            print("🟢 CONSIDER LOWERING QRS THRESHOLD - High win rate suggests room for more trades")
        else:
            print("🟡 QRS THRESHOLD OK - Win rate is reasonable")
        
        if profit_factor < 1.0:
            print("🔴 REVIEW STOP LOSSES - Poor profit factor suggests losses are too large")
        elif profit_factor > 2.0:
            print("🟢 EXCELLENT PROFIT FACTOR - Strategy is very profitable")
        else:
            print("🟡 PROFIT FACTOR OK - Reasonable profitability")
        
        print()
    
    def save_detailed_report(self, output_file: str = "backtest_analysis_report.txt"):
        """Save detailed analysis to file."""
        if not self.results:
            return
        
        with open(output_file, 'w') as f:
            f.write("BITCOIN ZONE FADE DETECTOR - BACKTEST ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            for key, value in self.results.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\nANALYSIS COMPLETE\n")
        
        print(f"📄 Detailed report saved to {output_file}")

def main():
    """Main analysis function."""
    print("🔍 Bitcoin Zone Fade Backtest Results Analyzer")
    print("=" * 50)
    print()
    
    # Initialize analyzer
    analyzer = BacktestAnalyzer()
    
    # Run analysis
    analyzer.print_summary()
    analyzer.analyze_performance()
    analyzer.generate_recommendations()
    analyzer.save_detailed_report()
    
    print("✅ Analysis complete!")
    print("\n📚 Next Steps:")
    print("1. Review the recommendations above")
    print("2. Adjust strategy parameters if needed")
    print("3. Run additional backtests")
    print("4. Consider paper trading before going live")

if __name__ == "__main__":
    main()