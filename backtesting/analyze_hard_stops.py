#!/usr/bin/env python3
"""
Analyze Hard Stop Patterns from 2024 Backtest Results

This script analyzes why 85% of trades hit hard stops to identify root causes
and patterns that can inform zone quality and entry criteria improvements.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


def load_backtest_results():
    """Load the 2024 backtest results."""
    results_file = Path("results/2024/1year_backtest/backtest_results_2024.json")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data


def analyze_hard_stop_patterns(trades):
    """Analyze patterns in hard stop trades vs winning trades."""
    
    print("=" * 80)
    print("🔍 HARD STOP PATTERN ANALYSIS")
    print("=" * 80)
    
    # Separate trades by exit type
    hard_stop_trades = [t for t in trades if t['exit_type'] == 'HARD_STOP']
    winning_trades = [t for t in trades if t['pnl'] > 0]
    t3_exits = [t for t in trades if t['exit_type'] == 'T3']
    eod_exits = [t for t in trades if t['exit_type'] == 'EOD']
    
    print(f"\n📊 Trade Distribution:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Hard Stops: {len(hard_stop_trades)} ({len(hard_stop_trades)/len(trades)*100:.1f}%)")
    print(f"   T3 Exits: {len(t3_exits)} ({len(t3_exits)/len(trades)*100:.1f}%)")
    print(f"   EOD Exits: {len(eod_exits)} ({len(eod_exits)/len(trades)*100:.1f}%)")
    print(f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    
    # Analyze QRS scores
    print(f"\n📈 QRS Score Analysis:")
    hard_stop_qrs = [t['qrs_score'] for t in hard_stop_trades]
    winning_qrs = [t['qrs_score'] for t in winning_trades]
    all_qrs = [t['qrs_score'] for t in trades]
    
    print(f"   Hard Stop Avg QRS: {statistics.mean(hard_stop_qrs):.2f}")
    print(f"   Winning Trade Avg QRS: {statistics.mean(winning_qrs):.2f}")
    print(f"   All Trade Avg QRS: {statistics.mean(all_qrs):.2f}")
    print(f"   QRS Difference: {statistics.mean(winning_qrs) - statistics.mean(hard_stop_qrs):.2f} (winning - hard stop)")
    
    # Analyze bars in trade (how quickly they hit stops)
    print(f"\n⏱️ Time to Exit Analysis:")
    hard_stop_bars = [t['bars_in_trade'] for t in hard_stop_trades]
    winning_bars = [t['bars_in_trade'] for t in winning_trades]
    
    print(f"   Hard Stop Avg Bars: {statistics.mean(hard_stop_bars):.1f} bars")
    print(f"   Winning Trade Avg Bars: {statistics.mean(winning_bars):.1f} bars")
    print(f"   Hard Stop Median Bars: {statistics.median(hard_stop_bars):.1f} bars")
    
    # Quick exits (< 10 bars = immediate reversal)
    quick_stops = [t for t in hard_stop_trades if t['bars_in_trade'] < 10]
    print(f"   Quick Stops (<10 bars): {len(quick_stops)} ({len(quick_stops)/len(hard_stop_trades)*100:.1f}%)")
    
    # Analyze by direction
    print(f"\n📊 Direction Analysis:")
    long_hard_stops = [t for t in hard_stop_trades if t['direction'] == 'LONG']
    short_hard_stops = [t for t in hard_stop_trades if t['direction'] == 'SHORT']
    long_winners = [t for t in winning_trades if t['direction'] == 'LONG']
    short_winners = [t for t in winning_trades if t['direction'] == 'SHORT']
    
    print(f"   LONG Hard Stops: {len(long_hard_stops)} ({len(long_hard_stops)/(len(long_hard_stops)+len(long_winners))*100:.1f}% of LONG trades)")
    print(f"   SHORT Hard Stops: {len(short_hard_stops)} ({len(short_hard_stops)/(len(short_hard_stops)+len(short_winners))*100:.1f}% of SHORT trades)")
    print(f"   LONG Winners: {len(long_winners)}")
    print(f"   SHORT Winners: {len(short_winners)}")
    
    # Analyze by symbol
    print(f"\n📊 Symbol Analysis:")
    for symbol in ['SPY', 'QQQ', 'IWM']:
        symbol_hard_stops = [t for t in hard_stop_trades if t['symbol'] == symbol]
        symbol_winners = [t for t in winning_trades if t['symbol'] == symbol]
        symbol_trades = [t for t in trades if t['symbol'] == symbol]
        
        if symbol_trades:
            print(f"   {symbol}:")
            print(f"      Total: {len(symbol_trades)} trades")
            print(f"      Hard Stops: {len(symbol_hard_stops)} ({len(symbol_hard_stops)/len(symbol_trades)*100:.1f}%)")
            print(f"      Winners: {len(symbol_winners)} ({len(symbol_winners)/len(symbol_trades)*100:.1f}%)")
            
            if symbol_hard_stops:
                symbol_hard_qrs = [t['qrs_score'] for t in symbol_hard_stops]
                print(f"      Hard Stop Avg QRS: {statistics.mean(symbol_hard_qrs):.2f}")
            
            if symbol_winners:
                symbol_win_qrs = [t['qrs_score'] for t in symbol_winners]
                print(f"      Winner Avg QRS: {statistics.mean(symbol_win_qrs):.2f}")
    
    # Analyze P&L distribution
    print(f"\n💰 P&L Analysis:")
    hard_stop_pnl = [t['pnl'] for t in hard_stop_trades]
    winning_pnl = [t['pnl'] for t in winning_trades]
    
    print(f"   Hard Stop Avg Loss: ${statistics.mean(hard_stop_pnl):.2f}")
    print(f"   Hard Stop Median Loss: ${statistics.median(hard_stop_pnl):.2f}")
    print(f"   Winning Avg Profit: ${statistics.mean(winning_pnl):.2f}")
    print(f"   Winning Median Profit: ${statistics.median(winning_pnl):.2f}")
    
    # Analyze stop distance
    print(f"\n🎯 Stop Distance Analysis:")
    hard_stop_distances = []
    for t in hard_stop_trades:
        entry = t['entry_price']
        stop = t['hard_stop']
        distance_pct = abs((stop - entry) / entry) * 100
        hard_stop_distances.append(distance_pct)
    
    print(f"   Avg Stop Distance: {statistics.mean(hard_stop_distances):.3f}%")
    print(f"   Median Stop Distance: {statistics.median(hard_stop_distances):.3f}%")
    print(f"   Min Stop Distance: {min(hard_stop_distances):.3f}%")
    print(f"   Max Stop Distance: {max(hard_stop_distances):.3f}%")
    
    return {
        'hard_stop_trades': hard_stop_trades,
        'winning_trades': winning_trades,
        'quick_stops': quick_stops,
        'hard_stop_qrs': hard_stop_qrs,
        'winning_qrs': winning_qrs,
        'hard_stop_bars': hard_stop_bars,
        'stop_distances': hard_stop_distances
    }


def identify_root_causes(analysis_results):
    """Identify root causes based on analysis."""
    
    print("\n" + "=" * 80)
    print("🎯 ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    hard_stop_qrs = analysis_results['hard_stop_qrs']
    winning_qrs = analysis_results['winning_qrs']
    quick_stops = analysis_results['quick_stops']
    hard_stop_bars = analysis_results['hard_stop_bars']
    stop_distances = analysis_results['stop_distances']
    
    root_causes = []
    
    # Root Cause 1: QRS Score not discriminating enough
    qrs_diff = statistics.mean(winning_qrs) - statistics.mean(hard_stop_qrs)
    if abs(qrs_diff) < 1.0:
        root_causes.append({
            'cause': 'QRS Score Not Discriminating',
            'severity': 'CRITICAL',
            'evidence': f'QRS difference between winners and losers: {qrs_diff:.2f} (too small)',
            'recommendation': 'Enhance QRS scoring to better differentiate quality setups'
        })
    
    # Root Cause 2: Quick reversals (immediate invalidation)
    quick_stop_pct = len(quick_stops) / len(analysis_results['hard_stop_trades']) * 100
    if quick_stop_pct > 30:
        root_causes.append({
            'cause': 'Immediate Zone Invalidation',
            'severity': 'CRITICAL',
            'evidence': f'{quick_stop_pct:.1f}% of hard stops hit in <10 bars (immediate reversal)',
            'recommendation': 'Require stronger zone confirmation (balance detection, ATR compression, multiple touches)'
        })
    
    # Root Cause 3: Stops too tight
    avg_stop_distance = statistics.mean(stop_distances)
    if avg_stop_distance < 0.25:
        root_causes.append({
            'cause': 'Stops Too Tight',
            'severity': 'HIGH',
            'evidence': f'Average stop distance: {avg_stop_distance:.3f}% (too tight for normal volatility)',
            'recommendation': 'Use ATR-based stops or wider zone buffers (0.5-1% minimum)'
        })
    
    # Root Cause 4: Poor entry timing
    median_bars = statistics.median(hard_stop_bars) if hard_stop_bars else 0
    if median_bars < 20:
        root_causes.append({
            'cause': 'Poor Entry Timing',
            'severity': 'HIGH',
            'evidence': f'Median hard stop at {median_bars:.0f} bars (entering too early/late)',
            'recommendation': 'Wait for zone approach with balance, require CHoCH confirmation'
        })
    
    # Root Cause 5: Low base QRS
    avg_qrs = statistics.mean(hard_stop_qrs + winning_qrs)
    if avg_qrs < 7.0:
        root_causes.append({
            'cause': 'Overall QRS Too Low',
            'severity': 'MEDIUM',
            'evidence': f'Average QRS across all trades: {avg_qrs:.2f} (below 7.0 threshold)',
            'recommendation': 'Increase minimum QRS threshold to 7.0+ and enhance scoring factors'
        })
    
    print(f"\n🔍 Identified {len(root_causes)} Root Causes:\n")
    
    for i, cause in enumerate(root_causes, 1):
        print(f"{i}. [{cause['severity']}] {cause['cause']}")
        print(f"   Evidence: {cause['evidence']}")
        print(f"   Recommendation: {cause['recommendation']}")
        print()
    
    return root_causes


def generate_recommendations(root_causes):
    """Generate specific implementation recommendations."""
    
    print("=" * 80)
    print("💡 IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    print("\n🎯 Immediate Actions (Week 1):\n")
    
    # Recommendation 1: Enhanced Zone Validation
    print("1. ENHANCE ZONE VALIDATION")
    print("   - Require ATR compression before zone approach")
    print("   - Check for balance (low volatility) in 10 bars before touch")
    print("   - Minimum zone age: 4 hours (zones need to mature)")
    print("   - Code: Implement in ZoneApproachAnalyzer")
    recommendations.append("enhance_zone_validation")
    
    # Recommendation 2: Stricter Entry Criteria
    print("\n2. STRICTER ENTRY CRITERIA")
    print("   - Increase minimum QRS threshold: 5.0 → 7.0")
    print("   - Require CHoCH confirmation (not optional)")
    print("   - Volume spike: 1.8x → 2.0x minimum")
    print("   - Wick ratio: 30% → 40% minimum")
    print("   - Code: Update SignalProcessor filters")
    recommendations.append("stricter_entry_criteria")
    
    # Recommendation 3: Better Stop Placement
    print("\n3. BETTER STOP PLACEMENT")
    print("   - Use ATR-based stops: 1.5 * ATR(14)")
    print("   - Minimum stop distance: 0.5% of entry price")
    print("   - Place stops beyond recent swing high/low, not just zone")
    print("   - Code: Update stop calculation in entry logic")
    recommendations.append("better_stop_placement")
    
    # Recommendation 4: Zone Touch Tracking
    print("\n4. ZONE TOUCH TRACKING")
    print("   - Only trade 1st and 2nd touches per session")
    print("   - Reset touch count at 9:30 AM ET daily")
    print("   - Track zone ID persistence across session")
    print("   - Code: Implement ZoneTouchTracker")
    recommendations.append("zone_touch_tracking")
    
    # Recommendation 5: Enhanced QRS Scoring
    print("\n5. ENHANCED QRS SCORING")
    print("   - Add ATR compression factor (0-2 points)")
    print("   - Add balance detection factor (0-2 points)")
    print("   - Increase CHoCH weight (0-2 → 0-3 points)")
    print("   - Add zone age factor (0-2 points)")
    print("   - New max score: 15 points, threshold: 10+")
    print("   - Code: Update EnhancedQRSScorer")
    recommendations.append("enhanced_qrs_scoring")
    
    print("\n🔄 Testing Approach:\n")
    print("1. Implement changes incrementally")
    print("2. Run backtest after each change")
    print("3. Measure impact on hard stop rate and win rate")
    print("4. Target: Hard stop rate <50%, Win rate >40%")
    
    return recommendations


def main():
    """Main analysis function."""
    
    print("\n" + "=" * 80)
    print("🔍 HARD STOP ANALYSIS - ZONE FADE DETECTOR")
    print("=" * 80)
    print("Analyzing 2024 backtest results to identify root causes...")
    print("=" * 80)
    
    # Load results
    results = load_backtest_results()
    if not results:
        return
    
    trades = results['trades']
    print(f"\n✅ Loaded {len(trades)} trades from backtest")
    
    # Analyze patterns
    analysis_results = analyze_hard_stop_patterns(trades)
    
    # Identify root causes
    root_causes = identify_root_causes(analysis_results)
    
    # Generate recommendations
    recommendations = generate_recommendations(root_causes)
    
    # Save analysis report
    report = {
        'analysis_date': datetime.now().isoformat(),
        'total_trades': len(trades),
        'hard_stop_count': len(analysis_results['hard_stop_trades']),
        'hard_stop_rate': len(analysis_results['hard_stop_trades']) / len(trades) * 100,
        'winning_count': len(analysis_results['winning_trades']),
        'win_rate': len(analysis_results['winning_trades']) / len(trades) * 100,
        'root_causes': root_causes,
        'recommendations': recommendations,
        'key_metrics': {
            'avg_hard_stop_qrs': float(statistics.mean(analysis_results['hard_stop_qrs'])),
            'avg_winning_qrs': float(statistics.mean(analysis_results['winning_qrs'])),
            'avg_bars_to_hard_stop': float(statistics.mean(analysis_results['hard_stop_bars'])),
            'quick_stops_pct': len(analysis_results['quick_stops']) / len(analysis_results['hard_stop_trades']) * 100,
            'avg_stop_distance_pct': float(statistics.mean(analysis_results['stop_distances']))
        }
    }
    
    report_file = Path("results/2024/1year_backtest/hard_stop_analysis_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Analysis report saved to: {report_file}")
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review root causes and recommendations above")
    print("2. Implement changes incrementally")
    print("3. Run backtests to measure impact")
    print("4. Target: Hard stop rate <50%, Win rate >40%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
