#!/usr/bin/env python3
"""
Test script for the Entry Visualizer

This script tests the visualization functionality with a small subset of data
to ensure everything works correctly before running on the full dataset.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from entry_visualizer import EntryVisualizer, VisualizationConfig
import pandas as pd


def test_visualizer():
    """Test the visualizer with a small subset of data."""
    print("🧪 Testing Entry Visualizer")
    print("=" * 40)
    
    # Create a custom config for testing
    config = VisualizationConfig(
        hours_before=1,  # Shorter window for testing
        hours_after=1,
        figure_size=(12, 8),
        dpi=150  # Lower DPI for faster testing
    )
    
    # Initialize visualizer
    visualizer = EntryVisualizer(config)
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals_test")
    
    try:
        # Load data
        print("📊 Loading test data...")
        symbols_data = visualizer.load_backtest_data(data_dir)
        entry_points_df = visualizer.load_entry_points(entry_points_file)
        
        if not symbols_data:
            print("❌ No backtesting data found")
            return False
        
        # Test with first 3 entry points only
        test_entries = entry_points_df.head(3)
        print(f"🎯 Testing with {len(test_entries)} entry points...")
        
        # Generate visualizations
        results = visualizer.generate_entry_visuals(test_entries, symbols_data, output_dir)
        
        # Create HTML report
        html_file = visualizer.create_html_report(test_entries, output_dir)
        
        # Print results
        print(f"\n📊 Test Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        print(f"   📄 HTML Report: {html_file}")
        
        if results['successful'] > 0:
            print(f"\n🎉 Test successful! Check {output_dir.absolute()}")
            return True
        else:
            print(f"\n❌ Test failed - no successful visualizations")
            return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_visualizer()
    sys.exit(0 if success else 1)