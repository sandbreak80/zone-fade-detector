#!/usr/bin/env python3
"""
Run Zone Fade Strategy Performance Analysis

This script runs comprehensive P&L and key trading metrics analysis
on the Zone Fade strategy backtesting results.
"""

import subprocess
import sys
from pathlib import Path

def run_performance_analysis():
    """Run the performance analysis on Zone Fade results."""
    print("🚀 Running Zone Fade Strategy Performance Analysis")
    print("=" * 60)
    print("📊 Calculating comprehensive P&L and key trading metrics")
    print("💰 Evaluating profitability, risk management, and overall performance")
    
    # Check if we're in Docker
    try:
        # Try to run the performance analysis script
        result = subprocess.run([
            "python", "backtesting/performance_analyzer.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Performance analysis completed successfully!")
        print("\n📊 Results:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Warnings/Info:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Performance analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Python not found. Make sure you're running in Docker.")
        print("Try: docker compose run --rm zone-fade-detector-test python backtesting/run_performance_analysis.py")
        return False
    
    return True

def show_usage():
    """Show usage instructions."""
    print("\n📋 USAGE INSTRUCTIONS:")
    print("=" * 40)
    print("\n🐳 Docker (Recommended):")
    print("   docker compose run --rm zone-fade-detector-test python backtesting/run_performance_analysis.py")
    
    print("\n🔧 Direct Python (if in container):")
    print("   python backtesting/run_performance_analysis.py")
    
    print("\n📁 Input Files:")
    print("   • results/2024/corrected/zone_fade_entry_points_2024_corrected.csv")
    
    print("\n📊 Performance Metrics Calculated:")
    print("   💰 PROFITABILITY:")
    print("     • Net Profit/Return")
    print("     • Win Rate (Win Percentage)")
    print("     • Average Win vs Average Loss")
    print("     • Profit Factor")
    print("     • Expectancy")
    
    print("\n   ⚠️  RISK MANAGEMENT:")
    print("     • Maximum Drawdown (MDD)")
    print("     • Risk-Reward Ratio")
    print("     • Value at Risk (VaR)")
    print("     • Conditional Value at Risk (CVaR)")
    
    print("\n   📈 RISK-ADJUSTED PERFORMANCE:")
    print("     • Sharpe Ratio")
    print("     • Sortino Ratio")
    print("     • Alpha (vs market benchmark)")
    print("     • Beta (volatility vs market)")
    
    print("\n   ⚙️  OPERATIONAL METRICS:")
    print("     • Slippage and Transaction Costs")
    print("     • Trade Frequency")
    print("     • Commission Analysis")
    
    print("\n   🎯 QRS GRADE ANALYSIS:")
    print("     • Performance by QRS Grade (A, B, C)")
    print("     • Win Rate by Grade")
    print("     • P&L by Grade")
    
    print("\n🔍 Key Features:")
    print("   • Comprehensive P&L calculation")
    print("   • Multiple position sizing methods")
    print("   • Risk-adjusted performance metrics")
    print("   • QRS grade performance breakdown")
    print("   • Professional trading metrics")
    print("   • Detailed performance reporting")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_usage()
    else:
        success = run_performance_analysis()
        if success:
            print("\n🎉 Performance analysis completed successfully!")
            print("📊 Comprehensive P&L and key trading metrics calculated!")
            print("💰 Profitability, risk management, and performance evaluated!")
        else:
            print("\n❌ Performance analysis failed. Check the error messages above.")
            show_usage()