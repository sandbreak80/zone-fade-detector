#!/usr/bin/env python3
"""
Modern Visualization Runner

Runs the new Plotly + Seaborn visualization system for Zone Fade entry points.
"""

import sys
from pathlib import Path
import argparse
from modern_plotly_visualizer import ModernPlotlyVisualizer, ModernVisualizationConfig


def main():
    """Main function for running modern visualizations."""
    parser = argparse.ArgumentParser(description='Modern Plotly + Seaborn Zone Fade Visualizer')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--test', action='store_true', help='Run in test mode (first 5 entries)')
    mode_group.add_argument('--full', action='store_true', help='Run in full mode (all entries)')
    mode_group.add_argument('--custom', action='store_true', help='Run in custom mode with parameters')
    
    # Time window settings
    parser.add_argument('--hours-before', type=int, default=2, 
                       help='Hours before entry point (default: 2)')
    parser.add_argument('--hours-after', type=int, default=2,
                       help='Hours after entry point (default: 2)')
    
    # Chart settings
    parser.add_argument('--width', type=int, default=1200, help='Chart width in pixels (default: 1200)')
    parser.add_argument('--height', type=int, default=800, help='Chart height in pixels (default: 800)')
    parser.add_argument('--dpi', type=int, default=300, help='Chart DPI (default: 300)')
    
    # Filtering options
    parser.add_argument('--min-qrs', type=float, help='Minimum QRS score filter')
    parser.add_argument('--symbols', nargs='+', choices=['SPY', 'QQQ', 'IWM'], 
                       help='Symbols to include')
    parser.add_argument('--limit', type=int, help='Limit number of entries to process')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='outputs/modern_visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Determine mode and set parameters
    if args.test:
        limit = 5
        print("🧪 Running in TEST mode (first 5 entry points)")
    elif args.full:
        limit = None
        print("🚀 Running in FULL mode (all entry points)")
    else:  # custom
        limit = args.limit
        print("⚙️ Running in CUSTOM mode")
    
    # Create configuration
    config = ModernVisualizationConfig(
        hours_before=args.hours_before,
        hours_after=args.hours_after,
        figure_width=args.width,
        figure_height=args.height,
        dpi=args.dpi
    )
    
    # Create visualizer
    visualizer = ModernPlotlyVisualizer(config)
    
    # Set up paths
    data_dir = Path('data')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data and entry points
    data = visualizer.load_backtest_data(data_dir)
    if not data:
        print("❌ No data loaded. Exiting.")
        return 1
    
    entry_points = visualizer.load_entry_points(data_dir)
    if not entry_points:
        print("❌ No entry points loaded. Exiting.")
        return 1
    
    # Apply filters
    if args.min_qrs:
        entry_points = [ep for ep in entry_points if ep.get('qrs_score', 0) >= args.min_qrs]
        print(f"🔍 Filtered to {len(entry_points)} entries with QRS >= {args.min_qrs}")
    
    if args.symbols:
        entry_points = [ep for ep in entry_points if ep.get('symbol') in args.symbols]
        print(f"🔍 Filtered to {len(entry_points)} entries for symbols: {args.symbols}")
    
    # Apply limit
    if limit:
        entry_points = entry_points[:limit]
        print(f"🎯 Processing {len(entry_points)} entry points (limited)")
    
    if not entry_points:
        print("❌ No entry points to process after filtering.")
        return 1
    
    # Create visualizations
    print(f"\n🎨 Creating modern visualizations...")
    success = visualizer.create_visualization(data, entry_points, output_dir)
    
    # Create Seaborn analysis
    print(f"\n📊 Creating Seaborn analysis...")
    visualizer.create_seaborn_analysis(entry_points, output_dir)
    
    if success:
        print(f"\n🎉 Modern visualization complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🌐 Open HTML files in your browser for interactive charts!")
        print(f"📊 Check seaborn_analysis/ for statistical charts!")
        return 0
    else:
        print(f"\n❌ Modern visualization failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())