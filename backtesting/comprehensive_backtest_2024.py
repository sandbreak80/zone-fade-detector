#!/usr/bin/env python3
"""
Comprehensive 1-Year Backtest with Full P&L Analysis

Compares Phase 2.1 strategy vs Buy & Hold for QQQ
- Starting capital: $10,000
- Position sizing: 90% equity per trade
- Full metrics: Sharpe ratio, max drawdown, win/loss analysis, etc.
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import statistics
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zone_fade_detector.core.models import OHLCVBar


@dataclass
class Trade:
    """Complete trade with all metrics."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    direction: str
    symbol: str
    shares: int
    
    # Entry details
    qrs: float
    volume_spike: float
    wick_ratio: float
    
    # Targets and stops
    hard_stop: float
    t1_price: float
    t2_price: float
    t3_price: float
    
    # Exit information
    exit_type: Optional[str] = None
    bars_held: int = 0
    
    # P&L
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    pnl_percent: float = 0.0
    
    # Trade state
    risk_units: float = 0.0
    t1_hit: bool = False
    trailing_stop: Optional[float] = None
    
    def is_winner(self) -> bool:
        return self.net_pnl > 0
    
    def hit_hard_stop(self) -> bool:
        return self.exit_type == 'HARD_STOP'


@dataclass
class Portfolio:
    """Track portfolio state over time."""
    starting_capital: float = 10000.0
    current_capital: float = 10000.0
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    open_trade: Optional[Trade] = None
    
    # Performance metrics
    peak_capital: float = 10000.0
    max_drawdown: float = 0.0
    total_commission: float = 0.0
    
    def update_equity(self, timestamp: datetime):
        """Update equity curve."""
        self.equity_curve.append((timestamp, self.current_capital))
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def open_position(self, trade: Trade):
        """Open a new position."""
        self.open_trade = trade
    
    def close_position(self, exit_time: datetime, exit_price: float, exit_type: str):
        """Close the current position."""
        if not self.open_trade:
            return
        
        trade = self.open_trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_type = exit_type
        
        # Calculate P&L
        if trade.direction == 'LONG':
            trade.gross_pnl = (exit_price - trade.entry_price) * trade.shares
        else:  # SHORT
            trade.gross_pnl = (trade.entry_price - exit_price) * trade.shares
        
        # Commission: $0.005 per share, $1 minimum
        trade.commission = max(1.0, 0.005 * trade.shares * 2)  # Entry + exit
        trade.net_pnl = trade.gross_pnl - trade.commission
        trade.pnl_percent = (trade.net_pnl / (trade.entry_price * trade.shares)) * 100
        
        # Update portfolio
        self.current_capital += trade.net_pnl
        self.total_commission += trade.commission
        self.trades.append(trade)
        self.open_trade = None
        
        return trade


@dataclass
class BuyAndHold:
    """Track buy and hold strategy."""
    starting_capital: float = 10000.0
    shares: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    commission: float = 0.0
    
    def buy(self, price: float, timestamp: datetime):
        """Buy and hold."""
        self.shares = int(self.starting_capital / price)
        self.entry_price = price
        self.current_price = price
        self.commission = max(1.0, 0.005 * self.shares)
        self.equity_curve.append((timestamp, self.starting_capital - self.commission))
    
    def update(self, price: float, timestamp: datetime):
        """Update current value."""
        self.current_price = price
        current_value = self.shares * price
        self.equity_curve.append((timestamp, current_value))
    
    def get_final_value(self) -> float:
        """Get final portfolio value."""
        return self.shares * self.current_price - self.commission


def load_2024_data() -> Dict[str, List[OHLCVBar]]:
    """Load QQQ 2024 1-minute data."""
    print("📊 Loading 2024 data...")
    
    data_dir = Path(__file__).parent.parent / "data" / "2024"
    file_path = data_dir / "QQQ_2024.pkl"
    
    if not file_path.exists():
        print(f"   ❌ QQQ data not found at {file_path}")
        return {}
    
    with open(file_path, 'rb') as f:
        bars = pickle.load(f)
    
    print(f"   ✅ QQQ: {len(bars):,} bars")
    print(f"   ✅ Date range: {bars[0].timestamp.date()} to {bars[-1].timestamp.date()}")
    print()
    return {'QQQ': bars}


def load_improved_entries() -> List[Dict]:
    """Load improved entry points (QQQ only)."""
    print("📋 Loading improved entry points...")
    
    file_path = Path(__file__).parent.parent / "results" / "2024" / "improved_backtest" / "improved_entry_points.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'entry_points' in data:
        entries = data['entry_points']
    else:
        entries = data
    
    # Filter for QQQ only
    qqq_entries = [e for e in entries if e['symbol'] == 'QQQ']
    
    print(f"✅ Loaded {len(qqq_entries)} QQQ entries\n")
    return qqq_entries


def calculate_phase21_targets(entry_price: float, direction: str) -> Dict[str, float]:
    """Calculate Phase 2.1 targets."""
    stop_distance = entry_price * 0.005  # 0.5% stop
    
    if direction == 'LONG':
        hard_stop = entry_price - stop_distance
        t1 = entry_price + (stop_distance * 0.5)  # 0.5R
        t2 = entry_price + (stop_distance * 1.0)  # 1R
        t3 = entry_price + (stop_distance * 1.5)  # 1.5R
    else:  # SHORT
        hard_stop = entry_price + stop_distance
        t1 = entry_price - (stop_distance * 0.5)
        t2 = entry_price - (stop_distance * 1.0)
        t3 = entry_price - (stop_distance * 1.5)
    
    return {
        'hard_stop': hard_stop,
        't1': t1,
        't2': t2,
        't3': t3,
        'risk_units': stop_distance
    }


def should_exit_time_based(entry_time: datetime, current_time: datetime, 
                           direction: str, entry_price: float, current_price: float,
                           t1_hit: bool) -> Tuple[bool, str]:
    """Phase 2.1 time-based exit rules."""
    hours_held = (current_time - entry_time).total_seconds() / 3600
    
    # EOD cutoff at 3:30 PM
    if current_time.hour >= 15 and current_time.minute >= 30:
        return True, 'EOD_EARLY'
    
    # Close after 4 hours if < 0.5R move and T1 not hit
    if hours_held >= 4.0 and not t1_hit:
        if direction == 'LONG':
            move_r = (current_price - entry_price) / (entry_price * 0.005)
        else:
            move_r = (entry_price - current_price) / (entry_price * 0.005)
        
        if move_r < 0.5:
            return True, 'TIME_EXIT_NO_MOMENTUM'
    
    return False, ''


def simulate_comprehensive_backtest(entries: List[Dict], bars: List[OHLCVBar]) -> Tuple[Portfolio, BuyAndHold]:
    """
    Run comprehensive backtest with Phase 2.1 strategy.
    
    Returns trading portfolio and buy & hold portfolio.
    """
    print("🎮 Running comprehensive backtest...")
    print(f"   Starting capital: $10,000")
    print(f"   Position sizing: 90% equity")
    print(f"   Strategy: Phase 2.1 (QQQ)")
    print()
    
    # Initialize portfolios
    trading_portfolio = Portfolio(starting_capital=10000.0)
    buy_hold = BuyAndHold(starting_capital=10000.0)
    
    # Buy and hold at first bar
    buy_hold.buy(bars[0].close, bars[0].timestamp)
    
    # Create entry lookup
    entry_map = {}
    for entry in entries:
        entry_time = datetime.fromisoformat(entry['timestamp'])
        entry_map[entry_time] = entry
    
    # Track next entry to process
    sorted_entries = sorted(entries, key=lambda e: datetime.fromisoformat(e['timestamp']))
    entry_idx = 0
    
    # Simulate bar by bar
    for i, bar in enumerate(bars):
        # Update buy and hold (sample every 30 mins to reduce data)
        if i % 30 == 0:
            buy_hold.update(bar.close, bar.timestamp)
        
        # Check if we should enter a new trade
        if entry_idx < len(sorted_entries) and trading_portfolio.open_trade is None:
            next_entry = sorted_entries[entry_idx]
            next_entry_time = datetime.fromisoformat(next_entry['timestamp'])
            
            if bar.timestamp >= next_entry_time:
                # Enter trade
                entry_price = next_entry['entry_price']
                direction = next_entry['direction']
                
                # Calculate position size (90% of equity)
                position_value = trading_portfolio.current_capital * 0.90
                shares = int(position_value / entry_price)
                
                if shares > 0:
                    # Calculate targets
                    targets = calculate_phase21_targets(entry_price, direction)
                    
                    # Create trade
                    trade = Trade(
                        entry_time=bar.timestamp,
                        entry_price=entry_price,
                        exit_time=None,
                        exit_price=None,
                        direction=direction,
                        symbol='QQQ',
                        shares=shares,
                        qrs=next_entry.get('qrs_score', 0),
                        volume_spike=next_entry.get('volume_spike', 0),
                        wick_ratio=next_entry.get('wick_ratio', 0),
                        hard_stop=targets['hard_stop'],
                        t1_price=targets['t1'],
                        t2_price=targets['t2'],
                        t3_price=targets['t3'],
                        risk_units=targets['risk_units']
                    )
                    
                    trading_portfolio.open_position(trade)
                
                entry_idx += 1
        
        # Manage open trade
        if trading_portfolio.open_trade:
            trade = trading_portfolio.open_trade
            trade.bars_held += 1
            
            # Check time-based exits
            should_exit, exit_reason = should_exit_time_based(
                trade.entry_time, bar.timestamp, trade.direction, 
                trade.entry_price, bar.close, trade.t1_hit
            )
            
            if should_exit:
                trading_portfolio.close_position(bar.timestamp, bar.close, exit_reason)
                trading_portfolio.update_equity(bar.timestamp)
                continue
            
            # Check price-based exits
            if trade.direction == 'LONG':
                # Check hard stop
                if bar.low <= trade.hard_stop:
                    trading_portfolio.close_position(bar.timestamp, trade.hard_stop, 'HARD_STOP')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check breakeven/trailing stop
                if trade.t1_hit and trade.trailing_stop and bar.low <= trade.trailing_stop:
                    trading_portfolio.close_position(bar.timestamp, trade.trailing_stop, 'BREAKEVEN_STOP')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check T3
                if bar.high >= trade.t3_price:
                    trading_portfolio.close_position(bar.timestamp, trade.t3_price, 'T3')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check T2
                if bar.high >= trade.t2_price:
                    trading_portfolio.close_position(bar.timestamp, trade.t2_price, 'T2')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check T1 - activate trailing stop
                if bar.high >= trade.t1_price and not trade.t1_hit:
                    trade.t1_hit = True
                    trade.trailing_stop = trade.entry_price
            
            else:  # SHORT
                # Check hard stop
                if bar.high >= trade.hard_stop:
                    trading_portfolio.close_position(bar.timestamp, trade.hard_stop, 'HARD_STOP')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check breakeven/trailing stop
                if trade.t1_hit and trade.trailing_stop and bar.high >= trade.trailing_stop:
                    trading_portfolio.close_position(bar.timestamp, trade.trailing_stop, 'BREAKEVEN_STOP')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check T3
                if bar.low <= trade.t3_price:
                    trading_portfolio.close_position(bar.timestamp, trade.t3_price, 'T3')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check T2
                if bar.low <= trade.t2_price:
                    trading_portfolio.close_position(bar.timestamp, trade.t2_price, 'T2')
                    trading_portfolio.update_equity(bar.timestamp)
                    continue
                
                # Check T1 - activate trailing stop
                if bar.low <= trade.t1_price and not trade.t1_hit:
                    trade.t1_hit = True
                    trade.trailing_stop = trade.entry_price
        
        # Update portfolio equity periodically (every 30 mins)
        if i % 30 == 0:
            trading_portfolio.update_equity(bar.timestamp)
    
    # Close any remaining open position
    if trading_portfolio.open_trade:
        trading_portfolio.close_position(bars[-1].timestamp, bars[-1].close, 'EOD_FINAL')
        trading_portfolio.update_equity(bars[-1].timestamp)
    
    # Final updates
    trading_portfolio.update_equity(bars[-1].timestamp)
    buy_hold.update(bars[-1].close, bars[-1].timestamp)
    
    print(f"✅ Backtest complete")
    print(f"   Trading: {len(trading_portfolio.trades)} trades executed")
    print(f"   Buy & Hold: {buy_hold.shares} shares held")
    print()
    
    return trading_portfolio, buy_hold


def calculate_sharpe_ratio(equity_curve: List[Tuple[datetime, float]], risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(equity_curve)):
        prev_value = equity_curve[i-1][1]
        curr_value = equity_curve[i][1]
        if prev_value > 0:
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
    
    if not returns:
        return 0.0
    
    # Annualize
    mean_return = statistics.mean(returns)
    std_return = statistics.stdev(returns) if len(returns) > 1 else 0
    
    if std_return == 0:
        return 0.0
    
    # Assume ~252 trading days per year, but we're using intraday
    # For intraday, annualize differently
    periods_per_year = 252 * 6.5 * 2  # Trading days * hours * 30-min periods
    
    sharpe = (mean_return * periods_per_year - risk_free_rate) / (std_return * math.sqrt(periods_per_year))
    return sharpe


def analyze_results(trading: Portfolio, buy_hold: BuyAndHold):
    """Comprehensive performance analysis."""
    
    print("=" * 80)
    print("📊 COMPREHENSIVE BACKTEST RESULTS - 2024")
    print("=" * 80)
    print()
    
    # ===== PORTFOLIO SUMMARY =====
    print("💼 **PORTFOLIO SUMMARY**")
    print()
    
    trading_return = ((trading.current_capital - trading.starting_capital) / trading.starting_capital) * 100
    buy_hold_final = buy_hold.get_final_value()
    buy_hold_return = ((buy_hold_final - buy_hold.starting_capital) / buy_hold.starting_capital) * 100
    
    print(f"{'Strategy':<20} {'Starting':<15} {'Ending':<15} {'Return':<15} {'Max DD':<15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'Trading (Phase 2.1)':<20} {'$10,000':<15} {f'${trading.current_capital:,.2f}':<15} {f'{trading_return:+.2f}%':<15} {f'{trading.max_drawdown:.2f}%':<15}")
    print(f"{'Buy & Hold QQQ':<20} {'$10,000':<15} {f'${buy_hold_final:,.2f}':<15} {f'{buy_hold_return:+.2f}%':<15} {'N/A':<15}")
    print()
    
    outperformance = trading_return - buy_hold_return
    print(f"🎯 **Outperformance**: {outperformance:+.2f}%")
    print()
    
    # ===== TRADING STATISTICS =====
    print("=" * 80)
    print("📈 **TRADING STATISTICS**")
    print("=" * 80)
    print()
    
    trades = trading.trades
    winners = [t for t in trades if t.is_winner()]
    losers = [t for t in trades if not t.is_winner()]
    hard_stops = [t for t in trades if t.hit_hard_stop()]
    
    total_trades = len(trades)
    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    hard_stop_rate = (len(hard_stops) / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"Losers: {len(losers)} ({100-win_rate:.1f}%)")
    print(f"Hard Stops: {len(hard_stops)} ({hard_stop_rate:.1f}%)")
    print()
    
    # ===== WIN/LOSS ANALYSIS =====
    print("💰 **WIN/LOSS ANALYSIS**")
    print()
    
    if winners:
        avg_win = statistics.mean(t.net_pnl for t in winners)
        avg_win_pct = statistics.mean(t.pnl_percent for t in winners)
        max_win = max(t.net_pnl for t in winners)
        total_win_pnl = sum(t.net_pnl for t in winners)
        print(f"Average Win: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
        print(f"Largest Win: ${max_win:.2f}")
        print(f"Total Winning P&L: ${total_win_pnl:.2f}")
    else:
        avg_win = 0
        total_win_pnl = 0
        print(f"Average Win: $0.00")
    print()
    
    if losers:
        avg_loss = statistics.mean(t.net_pnl for t in losers)
        avg_loss_pct = statistics.mean(t.pnl_percent for t in losers)
        max_loss = min(t.net_pnl for t in losers)
        total_loss_pnl = sum(t.net_pnl for t in losers)
        print(f"Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
        print(f"Largest Loss: ${max_loss:.2f}")
        print(f"Total Losing P&L: ${total_loss_pnl:.2f}")
    else:
        avg_loss = 0
        total_loss_pnl = 0
        print(f"Average Loss: $0.00")
    print()
    
    # Win/Loss Ratio
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
        print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    else:
        print(f"Win/Loss Ratio: N/A")
    
    # Profit Factor
    if total_loss_pnl != 0:
        profit_factor = abs(total_win_pnl / total_loss_pnl)
        print(f"Profit Factor: {profit_factor:.2f}")
    else:
        print(f"Profit Factor: N/A")
    print()
    
    # ===== TIME IN POSITION =====
    print("⏱️  **TIME IN POSITION**")
    print()
    
    if trades:
        avg_bars = statistics.mean(t.bars_held for t in trades)
        avg_minutes = avg_bars
        avg_hours = avg_minutes / 60
        
        median_bars = statistics.median(t.bars_held for t in trades)
        median_minutes = median_bars
        
        max_bars = max(t.bars_held for t in trades)
        max_minutes = max_bars
        max_hours = max_minutes / 60
        
        print(f"Average Time: {avg_bars:.0f} bars ({avg_hours:.1f} hours)")
        print(f"Median Time: {median_bars:.0f} bars ({median_minutes/60:.1f} hours)")
        print(f"Max Time: {max_bars:.0f} bars ({max_hours:.1f} hours)")
    print()
    
    # ===== EXIT BREAKDOWN =====
    print("🚪 **EXIT BREAKDOWN**")
    print()
    
    exit_counts = {}
    for trade in trades:
        exit_type = trade.exit_type or 'UNKNOWN'
        exit_counts[exit_type] = exit_counts.get(exit_type, 0) + 1
    
    for exit_type, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_trades * 100) if total_trades > 0 else 0
        print(f"   {exit_type}: {count} ({pct:.1f}%)")
    print()
    
    # ===== RISK METRICS =====
    print("=" * 80)
    print("⚠️  **RISK METRICS**")
    print("=" * 80)
    print()
    
    print(f"Max Drawdown: {trading.max_drawdown:.2f}%")
    print(f"Peak Portfolio Value: ${trading.peak_capital:,.2f}")
    print(f"Total Commission Paid: ${trading.total_commission:.2f}")
    print()
    
    # Sharpe Ratio
    trading_sharpe = calculate_sharpe_ratio(trading.equity_curve)
    buy_hold_sharpe = calculate_sharpe_ratio(buy_hold.equity_curve)
    
    print(f"Sharpe Ratio (Trading): {trading_sharpe:.2f}")
    print(f"Sharpe Ratio (Buy & Hold): {buy_hold_sharpe:.2f}")
    print()
    
    # ===== TRADE DISTRIBUTION =====
    print("=" * 80)
    print("📊 **TRADE DISTRIBUTION**")
    print("=" * 80)
    print()
    
    long_trades = [t for t in trades if t.direction == 'LONG']
    short_trades = [t for t in trades if t.direction == 'SHORT']
    
    if long_trades:
        long_winners = [t for t in long_trades if t.is_winner()]
        long_win_rate = (len(long_winners) / len(long_trades) * 100)
        long_pnl = sum(t.net_pnl for t in long_trades)
        print(f"LONG Trades: {len(long_trades)}")
        print(f"   Win Rate: {long_win_rate:.1f}%")
        print(f"   Total P&L: ${long_pnl:.2f}")
        print()
    
    if short_trades:
        short_winners = [t for t in short_trades if t.is_winner()]
        short_win_rate = (len(short_winners) / len(short_trades) * 100)
        short_pnl = sum(t.net_pnl for t in short_trades)
        print(f"SHORT Trades: {len(short_trades)}")
        print(f"   Win Rate: {short_win_rate:.1f}%")
        print(f"   Total P&L: ${short_pnl:.2f}")
        print()
    
    # ===== MONTHLY BREAKDOWN =====
    print("=" * 80)
    print("📅 **MONTHLY PERFORMANCE**")
    print("=" * 80)
    print()
    
    monthly_pnl = {}
    for trade in trades:
        if trade.exit_time:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = []
            monthly_pnl[month_key].append(trade.net_pnl)
    
    print(f"{'Month':<15} {'Trades':<10} {'P&L':<15} {'Avg P&L':<15}")
    print(f"{'-'*15} {'-'*10} {'-'*15} {'-'*15}")
    
    for month in sorted(monthly_pnl.keys()):
        trades_in_month = len(monthly_pnl[month])
        total_pnl = sum(monthly_pnl[month])
        avg_pnl = total_pnl / trades_in_month if trades_in_month > 0 else 0
        print(f"{month:<15} {trades_in_month:<10} ${total_pnl:>12.2f} ${avg_pnl:>12.2f}")
    
    print()
    
    # ===== COMPARISON SUMMARY =====
    print("=" * 80)
    print("🏆 **STRATEGY COMPARISON**")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<30} {'Trading':<20} {'Buy & Hold':<20} {'Winner':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    print(f"{'Final Value':<30} {f'${trading.current_capital:,.2f}':<20} {f'${buy_hold_final:,.2f}':<20} {'Trading' if trading.current_capital > buy_hold_final else 'Buy & Hold':<15}")
    print(f"{'Total Return':<30} {f'{trading_return:+.2f}%':<20} {f'{buy_hold_return:+.2f}%':<20} {'Trading' if trading_return > buy_hold_return else 'Buy & Hold':<15}")
    print(f"{'Sharpe Ratio':<30} {f'{trading_sharpe:.2f}':<20} {f'{buy_hold_sharpe:.2f}':<20} {'Trading' if trading_sharpe > buy_hold_sharpe else 'Buy & Hold':<15}")
    print(f"{'Max Drawdown':<30} {f'{trading.max_drawdown:.2f}%':<20} {'N/A':<20} {'N/A':<15}")
    print()
    
    # ===== FINAL VERDICT =====
    print("=" * 80)
    print("🎯 **FINAL VERDICT**")
    print("=" * 80)
    print()
    
    if trading.current_capital > buy_hold_final:
        print(f"✅ **TRADING STRATEGY WINS!**")
        print(f"   Outperformance: ${trading.current_capital - buy_hold_final:.2f} ({outperformance:+.2f}%)")
    else:
        print(f"⚠️  **BUY & HOLD WINS**")
        print(f"   Underperformance: ${trading.current_capital - buy_hold_final:.2f} ({outperformance:+.2f}%)")
    
    print()
    print(f"Trading Strategy Highlights:")
    print(f"   - Win Rate: {win_rate:.1f}% (Target: >40%)")
    print(f"   - Hard Stop Rate: {hard_stop_rate:.1f}% (Target: <50%)")
    print(f"   - Total Trades: {total_trades}")
    print(f"   - Max Drawdown: {trading.max_drawdown:.2f}%")
    
    print()
    print("=" * 80)
    
    # Return summary dict
    return {
        'trading': {
            'starting_capital': trading.starting_capital,
            'ending_capital': trading.current_capital,
            'total_return_pct': trading_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'hard_stop_rate': hard_stop_rate,
            'profit_factor': profit_factor if total_loss_pnl != 0 else 0,
            'sharpe_ratio': trading_sharpe,
            'max_drawdown': trading.max_drawdown,
            'total_commission': trading.total_commission,
            'avg_win': avg_win if winners else 0,
            'avg_loss': avg_loss if losers else 0,
        },
        'buy_hold': {
            'starting_capital': buy_hold.starting_capital,
            'ending_capital': buy_hold_final,
            'total_return_pct': buy_hold_return,
            'sharpe_ratio': buy_hold_sharpe,
            'shares': buy_hold.shares,
            'entry_price': buy_hold.entry_price,
            'exit_price': buy_hold.current_price,
        },
        'comparison': {
            'outperformance_pct': outperformance,
            'outperformance_dollars': trading.current_capital - buy_hold_final,
            'winner': 'Trading' if trading.current_capital > buy_hold_final else 'Buy & Hold'
        }
    }


def run_comprehensive_backtest():
    """Main execution."""
    
    print("=" * 80)
    print("🚀 COMPREHENSIVE 1-YEAR BACKTEST - QQQ")
    print("=" * 80)
    print()
    print("Configuration:")
    print("   Strategy: Phase 2.1 (Validated)")
    print("   Symbol: QQQ only")
    print("   Starting Capital: $10,000")
    print("   Position Sizing: 90% equity per trade")
    print("   Comparison: Buy & Hold QQQ")
    print()
    
    # Load data
    data = load_2024_data()
    if 'QQQ' not in data:
        print("❌ No QQQ data available")
        return
    
    bars = data['QQQ']
    entries = load_improved_entries()
    
    # Run backtest
    trading, buy_hold = simulate_comprehensive_backtest(entries, bars)
    
    # Analyze results
    results = analyze_results(trading, buy_hold)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "2024" / "comprehensive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    output_file = output_dir / "comprehensive_backtest_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Save equity curves (CSV for plotting)
    equity_file = output_dir / "equity_curves.csv"
    with open(equity_file, 'w') as f:
        f.write("timestamp,trading_equity,buy_hold_equity\n")
        
        # Align timestamps
        trading_dict = {t: v for t, v in trading.equity_curve}
        buy_hold_dict = {t: v for t, v in buy_hold.equity_curve}
        
        all_timestamps = sorted(set(trading_dict.keys()) | set(buy_hold_dict.keys()))
        
        for ts in all_timestamps:
            trading_val = trading_dict.get(ts, trading.current_capital)
            buy_hold_val = buy_hold_dict.get(ts, buy_hold.get_final_value())
            f.write(f"{ts.isoformat()},{trading_val:.2f},{buy_hold_val:.2f}\n")
    
    print(f"📈 Equity curves saved to: {equity_file}")
    print()
    
    return results


if __name__ == "__main__":
    run_comprehensive_backtest()
