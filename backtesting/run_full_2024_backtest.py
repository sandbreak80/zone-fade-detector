#!/usr/bin/env python3
"""
Run Full 2024 Backtest with Corrected Logic

This script runs the complete 2024 backtest with:
1. Corrected window duration tracking
2. Risk/reward filtering (1:2 or better)
3. All 2024 data (not just sample)
"""

import subprocess
import sys
from pathlib import Path

def run_full_2024_backtest():
    """Run the full 2024 backtest with corrected logic."""
    print("🚀 Running FULL 2024 Zone Fade Backtest")
    print("=" * 60)
    print("⏱️  CORRECTED: Window duration tracks how long entry conditions remain valid")
    print("💰 RISK/REWARD: Only accepts entry points with 1:2 or better risk/reward ratio")
    print("📊 FULL DATA: Using complete 2024 dataset (not sample)")
    
    # Check if we're in Docker
    try:
        # Try to run the corrected validation script with full data
        result = subprocess.run([
            "python", "backtesting/backtest_2024_corrected_window_tracking.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Full 2024 backtest completed successfully!")
        print("\n📊 Results:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Warnings/Info:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Full 2024 backtest failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Python not found. Make sure you're running in Docker.")
        print("Try: docker-compose run --rm zone-fade-detector-test python backtesting/run_full_2024_backtest.py")
        return False
    
    return True

def show_usage():
    """Show usage instructions."""
    print("\n📋 USAGE INSTRUCTIONS:")
    print("=" * 40)
    print("\n🐳 Docker (Recommended):")
    print("   docker-compose run --rm zone-fade-detector-test python backtesting/run_full_2024_backtest.py")
    
    print("\n🔧 Direct Python (if in container):")
    print("   python backtesting/run_full_2024_backtest.py")
    
    print("\n📁 Output Files:")
    print("   • results/2024/corrected/zone_fade_entry_points_2024_corrected.csv")
    print("   • results/2024/corrected/corrected_backtesting_summary.txt")
    
    print("\n⏱️  CORRECTED Window Duration Logic:")
    print("   • Tracks how long entry conditions remain valid")
    print("   • Checks ALL conditions every minute:")
    print("     - Zone still being touched?")
    print("     - Still rejection candle pattern?")
    print("     - Still volume spike?")
    print("     - QRS score still ≥ 7?")
    print("   • Stops when ANY condition is no longer met")
    print("   • Reports actual trading opportunity window")
    
    print("\n💰 RISK/REWARD FILTERING:")
    print("   • Only accepts entry points with 1:2 or better risk/reward ratio")
    print("   • Calculates stop loss and take profit levels")
    print("   • Ensures profitable trading opportunities")
    print("   • Filters out low-quality setups")
    
    print("\n📊 FULL 2024 DATA:")
    print("   • Uses complete 2024 dataset (not sample)")
    print("   • Processes all available bars for each symbol")
    print("   • Provides comprehensive backtesting results")
    print("   • More accurate statistics and metrics")
    
    print("\n🔍 Key Differences from Original:")
    print("   • Original: Stops when price moves away from zone")
    print("   • Corrected: Stops when entry conditions are invalid")
    print("   • Original: May overestimate execution time")
    print("   • Corrected: Accurate execution time assessment")
    print("   • Original: No risk/reward filtering")
    print("   • Corrected: Only 1:2+ risk/reward setups")
    
    print("\n📊 Expected Results:")
    print("   • Shorter average window durations (1-5 minutes vs 28.9)")
    print("   • More accurate trading opportunity assessment")
    print("   • Better risk management data")
    print("   • Realistic execution time expectations")
    print("   • Only high-quality setups with 1:2+ risk/reward")
    print("   • Fewer but better entry points")
    print("   • Improved trading performance potential")
    print("   • Complete 2024 dataset coverage")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_usage()
    else:
        success = run_full_2024_backtest()
        if success:
            print("\n🎉 FULL 2024 backtest completed successfully!")
            print("📁 Check results/2024/corrected/ for detailed output files")
            print("⏱️  Window durations now accurately reflect trading opportunity!")
            print("💰 All entry points have 1:2+ risk/reward ratio!")
            print("📊 Complete 2024 dataset processed!")
        else:
            print("\n❌ FULL 2024 backtest failed. Check the error messages above.")
            show_usage()