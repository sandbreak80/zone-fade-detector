#!/usr/bin/env python3
"""
Run Enhanced 2024 Zone Fade Validation

This script runs the enhanced validation that includes all the metrics
shown in the visualization table.
"""

import subprocess
import sys
from pathlib import Path

def run_enhanced_validation():
    """Run the enhanced validation script."""
    print("🚀 Running Enhanced 2024 Zone Fade Validation")
    print("=" * 60)
    
    # Check if we're in Docker
    try:
        # Try to run the enhanced validation script
        result = subprocess.run([
            "python", "backtesting/backtest_2024_enhanced_validation.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Enhanced validation completed successfully!")
        print("\n📊 Results:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Warnings/Info:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Enhanced validation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Python not found. Make sure you're running in Docker.")
        print("Try: docker-compose run --rm zone-fade-detector-test python backtesting/run_enhanced_validation.py")
        return False
    
    return True

def show_usage():
    """Show usage instructions."""
    print("\n📋 USAGE INSTRUCTIONS:")
    print("=" * 40)
    print("\n🐳 Docker (Recommended):")
    print("   docker-compose run --rm zone-fade-detector-test python backtesting/run_enhanced_validation.py")
    
    print("\n🔧 Direct Python (if in container):")
    print("   python backtesting/run_enhanced_validation.py")
    
    print("\n📁 Output Files:")
    print("   • results/2024/enhanced/zone_fade_entry_points_2024_enhanced.csv")
    print("   • results/2024/enhanced/enhanced_backtesting_summary.txt")
    
    print("\n📊 Enhanced Metrics Include:")
    print("   • QRS Score Breakdown (Quality, Risk, Setup)")
    print("   • Risk Management (Stop Loss, Take Profit, Risk/Reward)")
    print("   • Price Analysis (Range, Zone Position, High/Low)")
    print("   • Volume Analysis (Ratio, VWAP Distance)")
    print("   • Entry Window Details (Start/End Times, Duration)")
    
    print("\n🔍 Comparison:")
    print("   • Original: 16 columns, basic metrics")
    print("   • Enhanced: 40+ columns, comprehensive analysis")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_usage()
    else:
        success = run_enhanced_validation()
        if success:
            print("\n🎉 Enhanced validation completed successfully!")
            print("📁 Check results/2024/enhanced/ for detailed output files")
        else:
            print("\n❌ Enhanced validation failed. Check the error messages above.")
            show_usage()