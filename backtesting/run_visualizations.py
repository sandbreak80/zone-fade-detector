#!/usr/bin/env python3
"""
Zone Fade Entry Point Visualization Runner

This script provides an easy way to run the entry point visualizations
with different configurations and options.

Usage:
    python run_visualizations.py --help
    python run_visualizations.py --test
    python run_visualizations.py --full
    python run_visualizations.py --custom --hours-before 3 --hours-after 3
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from entry_visualizer import EntryVisualizer, VisualizationConfig


def create_custom_config(args) -> VisualizationConfig:
    """Create a custom configuration based on command line arguments."""
    return VisualizationConfig(
        hours_before=args.hours_before,
        hours_after=args.hours_after,
        figure_size=(args.width, args.height),
        dpi=args.dpi,
        style=args.style,
        show_grid=not args.no_grid,
        volume_alpha=args.volume_alpha
    )


def run_test_mode():
    """Run in test mode with a small subset of data."""
    print("🧪 Running in TEST mode (first 5 entry points)")
    
    config = VisualizationConfig(
        hours_before=1,
        hours_after=1,
        figure_size=(12, 8),
        dpi=150
    )
    
    visualizer = EntryVisualizer(config)
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals_test")
    
    try:
        # Load data
        symbols_data = visualizer.load_backtest_data(data_dir)
        entry_points_df = visualizer.load_entry_points(entry_points_file)
        
        if not symbols_data:
            print("❌ No backtesting data found")
            return False
        
        # Test with first 5 entry points only
        test_entries = entry_points_df.head(5)
        print(f"🎯 Testing with {len(test_entries)} entry points...")
        
        # Generate visualizations
        results = visualizer.generate_entry_visuals(test_entries, symbols_data, output_dir)
        
        # Create HTML report
        html_file = visualizer.create_html_report(test_entries, output_dir)
        
        print(f"\n📊 Test Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        
        return results['successful'] > 0
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def run_full_mode(args):
    """Run in full mode with all entry points."""
    print("🚀 Running in FULL mode (all entry points)")
    
    config = create_custom_config(args)
    visualizer = EntryVisualizer(config)
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals")
    
    try:
        # Load data
        symbols_data = visualizer.load_backtest_data(data_dir)
        entry_points_df = visualizer.load_entry_points(entry_points_file)
        
        if not symbols_data:
            print("❌ No backtesting data found")
            return False
        
        print(f"🎯 Processing {len(entry_points_df)} entry points...")
        
        # Generate visualizations
        results = visualizer.generate_entry_visuals(entry_points_df, symbols_data, output_dir)
        
        # Create HTML report
        html_file = visualizer.create_html_report(entry_points_df, output_dir)
        
        print(f"\n📊 Full Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        print(f"   📄 HTML Report: {html_file}")
        
        return results['successful'] > 0
        
    except Exception as e:
        print(f"❌ Full mode error: {e}")
        return False


def run_custom_mode(args):
    """Run in custom mode with user-specified parameters."""
    print("⚙️ Running in CUSTOM mode")
    
    config = create_custom_config(args)
    visualizer = EntryVisualizer(config)
    
    # Set up paths
    data_dir = Path("data/2024")
    entry_points_file = Path("results/manual_validation/entry_points/zone_fade_entry_points_2024_efficient.csv")
    output_dir = Path("outputs/visuals_custom")
    
    try:
        # Load data
        symbols_data = visualizer.load_backtest_data(data_dir)
        entry_points_df = visualizer.load_entry_points(entry_points_file)
        
        if not symbols_data:
            print("❌ No backtesting data found")
            return False
        
        # Apply filters if specified
        if args.min_qrs:
            entry_points_df = entry_points_df[entry_points_df['qrs_score'] >= args.min_qrs]
            print(f"🔍 Filtered to QRS >= {args.min_qrs}: {len(entry_points_df)} entries")
        
        if args.symbols:
            entry_points_df = entry_points_df[entry_points_df['symbol'].isin(args.symbols)]
            print(f"🔍 Filtered to symbols {args.symbols}: {len(entry_points_df)} entries")
        
        if args.limit:
            entry_points_df = entry_points_df.head(args.limit)
            print(f"🔍 Limited to {args.limit} entries: {len(entry_points_df)} entries")
        
        print(f"🎯 Processing {len(entry_points_df)} entry points...")
        
        # Generate visualizations
        results = visualizer.generate_entry_visuals(entry_points_df, symbols_data, output_dir)
        
        # Create HTML report
        html_file = visualizer.create_html_report(entry_points_df, output_dir)
        
        print(f"\n📊 Custom Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ❌ No Data: {results['no_data']}")
        print(f"   📁 Output Directory: {output_dir.absolute()}")
        print(f"   📄 HTML Report: {html_file}")
        
        return results['successful'] > 0
        
    except Exception as e:
        print(f"❌ Custom mode error: {e}")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Zone Fade Entry Point Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_visualizations.py --test
  python run_visualizations.py --full
  python run_visualizations.py --custom --hours-before 3 --hours-after 3
  python run_visualizations.py --custom --min-qrs 8 --symbols SPY QQQ
  python run_visualizations.py --custom --limit 10 --no-grid
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--test', action='store_true', help='Run in test mode (first 5 entries)')
    mode_group.add_argument('--full', action='store_true', help='Run in full mode (all entries)')
    mode_group.add_argument('--custom', action='store_true', help='Run in custom mode with parameters')
    
    # Time window settings
    parser.add_argument('--hours-before', type=int, default=2, help='Hours before entry point (default: 2)')
    parser.add_argument('--hours-after', type=int, default=2, help='Hours after entry point (default: 2)')
    
    # Chart settings
    parser.add_argument('--width', type=int, default=16, help='Chart width in inches (default: 16)')
    parser.add_argument('--height', type=int, default=10, help='Chart height in inches (default: 10)')
    parser.add_argument('--dpi', type=int, default=300, help='Chart DPI (default: 300)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid', 
                       help='Matplotlib style (default: seaborn-v0_8-whitegrid)')
    parser.add_argument('--no-grid', action='store_true', help='Disable grid lines')
    parser.add_argument('--volume-alpha', type=float, default=0.6, help='Volume bar transparency (default: 0.6)')
    
    # Filtering options
    parser.add_argument('--min-qrs', type=float, help='Minimum QRS score filter')
    parser.add_argument('--symbols', nargs='+', choices=['SPY', 'QQQ', 'IWM'], help='Symbols to include')
    parser.add_argument('--limit', type=int, help='Limit number of entries to process')
    
    args = parser.parse_args()
    
    print("🎨 Zone Fade Entry Point Visualizer")
    print("=" * 50)
    
    success = False
    
    if args.test:
        success = run_test_mode()
    elif args.full:
        success = run_full_mode(args)
    elif args.custom:
        success = run_custom_mode(args)
    
    if success:
        print(f"\n🎉 Visualization complete!")
        sys.exit(0)
    else:
        print(f"\n❌ Visualization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()