#!/usr/bin/env python3
"""
Multi-Day Swing Trading Strategies

This module implements 5 swing trading strategies designed to potentially
outperform buy-and-hold with realistic expectations and proper risk management.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yfinance as yf
import talib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Container for individual trade information."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    strategy_name: str
    symbol: str
    entry_reason: str
    exit_reason: Optional[str]
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    days_held: Optional[int] = None

@dataclass
class StrategyResult:
    """Container for strategy backtest results."""
    strategy_name: str
    symbol: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    buy_hold_return: float
    excess_return: float
    trades: List[Trade]


class SwingTradingFramework:
    """Framework for implementing and backtesting swing trading strategies."""
    
    def __init__(self, initial_capital: float = 10000, position_size_pct: float = 0.03):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        
        logger.info(f"🚀 Initialized Swing Trading Framework")
        logger.info(f"💰 Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"📊 Position Size: {position_size_pct:.1%}")
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Add technical indicators
            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            df['SMA_20'] = talib.SMA(df['Close'].values, timeperiod=20)
            df['SMA_50'] = talib.SMA(df['Close'].values, timeperiod=50)
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'].values, timeperiod=20)
            df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            
            # Add volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Add volatility indicators
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
            
            logger.info(f"📊 Loaded {len(df)} bars for {symbol}")
            logger.info(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management."""
        risk_amount = self.current_capital * 0.02  # 2% risk per trade
        price_risk = abs(price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        max_position = self.current_capital * self.position_size_pct / price
        
        return min(position_size, max_position)
    
    def execute_trade(self, trade: Trade, current_price: float) -> Trade:
        """Execute a trade and update capital."""
        if trade.exit_price is None:
            trade.exit_price = current_price
        
        trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
        trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        trade.days_held = (trade.exit_date - trade.entry_date).days
        
        self.current_capital += trade.pnl
        self.trades.append(trade)
        
        return trade


class OversoldBounceStrategy:
    """Strategy 1: Oversold Bounce (2-3 day holds)"""
    
    def __init__(self, name: str = "Oversold Bounce"):
        self.name = name
        self.rsi_oversold = 30
        self.rsi_exit = 50
        self.consecutive_down_days = 3
        self.profit_target = 0.03
        self.stop_loss = 0.02
        self.max_hold_days = 3
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for oversold bounce strategy."""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        signals['exit_reason'] = ''
        
        # Find consecutive down days
        df['down_day'] = df['Close'] < df['Close'].shift(1)
        df['consecutive_down'] = df['down_day'].groupby((df['down_day'] != df['down_day'].shift()).cumsum()).cumsum()
        
        # Entry conditions
        entry_condition = (
            (df['consecutive_down'] >= self.consecutive_down_days) &
            (df['RSI'] < self.rsi_oversold) &
            (df['Volume_ratio'] > 1.2)  # Volume spike
        )
        
        signals.loc[entry_condition, 'position'] = 1
        signals.loc[entry_condition, 'entry_reason'] = 'Oversold bounce'
        
        # Exit conditions
        exit_condition = (
            (signals['position'].shift(1) == 1) &
            (
                (df['RSI'] > self.rsi_exit) |
                (df['Close'] / df['Close'].shift(1) > 1 + self.profit_target) |
                (df['Close'] / df['Close'].shift(1) < 1 - self.stop_loss)
            )
        )
        
        signals.loc[exit_condition, 'position'] = 0
        signals.loc[exit_condition, 'exit_reason'] = 'Exit conditions met'
        
        return signals


class BreakoutContinuationStrategy:
    """Strategy 2: Breakout Continuation (3-5 day holds)"""
    
    def __init__(self, name: str = "Breakout Continuation"):
        self.name = name
        self.lookback_period = 20
        self.volume_threshold = 1.5
        self.rsi_min = 50
        self.profit_target = 0.05
        self.stop_loss = 0.03
        self.max_hold_days = 5
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for breakout continuation strategy."""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        signals['exit_reason'] = ''
        
        # Calculate 20-day high
        df['high_20'] = df['High'].rolling(self.lookback_period).max()
        
        # Entry conditions
        entry_condition = (
            (df['Close'] > df['high_20'].shift(1)) &  # Break above 20-day high
            (df['RSI'] > self.rsi_min) &  # Momentum
            (df['Volume_ratio'] > self.volume_threshold) &  # Volume confirmation
            (df['Close'] > df['SMA_20'])  # Above moving average
        )
        
        signals.loc[entry_condition, 'position'] = 1
        signals.loc[entry_condition, 'entry_reason'] = 'Breakout continuation'
        
        # Exit conditions
        exit_condition = (
            (signals['position'].shift(1) == 1) &
            (
                (df['Close'] / df['Close'].shift(1) > 1 + self.profit_target) |
                (df['Close'] / df['Close'].shift(1) < 1 - self.stop_loss) |
                (df['RSI'] < 40)  # Momentum loss
            )
        )
        
        signals.loc[exit_condition, 'position'] = 0
        signals.loc[exit_condition, 'exit_reason'] = 'Exit conditions met'
        
        return signals


class VolatilityExpansionStrategy:
    """Strategy 3: Volatility Expansion (4-7 day holds)"""
    
    def __init__(self, name: str = "Volatility Expansion"):
        self.name = name
        self.low_vol_threshold = 0.15  # 15% annualized volatility
        self.high_vol_threshold = 0.25  # 25% annualized volatility
        self.profit_target = 0.04
        self.stop_loss = 0.025
        self.max_hold_days = 7
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for volatility expansion strategy."""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        signals['exit_reason'] = ''
        
        # Calculate volatility regime
        df['vol_regime'] = 'normal'
        df.loc[df['Volatility'] < self.low_vol_threshold, 'vol_regime'] = 'low'
        df.loc[df['Volatility'] > self.high_vol_threshold, 'vol_regime'] = 'high'
        
        # Entry conditions (buy the dip after volatility expansion)
        entry_condition = (
            (df['vol_regime'].shift(1) == 'low') &  # Was in low vol
            (df['vol_regime'] == 'high') &  # Now in high vol
            (df['Close'] < df['Close'].shift(1)) &  # Price down
            (df['RSI'] < 40)  # Oversold
        )
        
        signals.loc[entry_condition, 'position'] = 1
        signals.loc[entry_condition, 'entry_reason'] = 'Volatility expansion'
        
        # Exit conditions
        exit_condition = (
            (signals['position'].shift(1) == 1) &
            (
                (df['Close'] / df['Close'].shift(1) > 1 + self.profit_target) |
                (df['Close'] / df['Close'].shift(1) < 1 - self.stop_loss) |
                (df['vol_regime'] == 'low')  # Volatility normalized
            )
        )
        
        signals.loc[exit_condition, 'position'] = 0
        signals.loc[exit_condition, 'exit_reason'] = 'Exit conditions met'
        
        return signals


class SectorRotationStrategy:
    """Strategy 4: Sector Rotation (5-7 day holds)"""
    
    def __init__(self, name: str = "Sector Rotation"):
        self.name = name
        self.lookback_period = 10
        self.relative_strength_threshold = 1.02  # 2% outperformance
        self.profit_target = 0.06
        self.stop_loss = 0.03
        self.max_hold_days = 7
    
    def generate_signals(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for sector rotation strategy."""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        signals['exit_reason'] = ''
        
        # Calculate relative strength
        df['relative_strength'] = df['Close'] / benchmark_df['Close']
        df['rs_ma'] = df['relative_strength'].rolling(self.lookback_period).mean()
        
        # Entry conditions
        entry_condition = (
            (df['relative_strength'] > df['rs_ma'] * self.relative_strength_threshold) &
            (df['Close'] > df['SMA_20']) &
            (df['Volume_ratio'] > 1.1) &
            (df['RSI'] > 45)
        )
        
        signals.loc[entry_condition, 'position'] = 1
        signals.loc[entry_condition, 'entry_reason'] = 'Sector rotation'
        
        # Exit conditions
        exit_condition = (
            (signals['position'].shift(1) == 1) &
            (
                (df['Close'] / df['Close'].shift(1) > 1 + self.profit_target) |
                (df['Close'] / df['Close'].shift(1) < 1 - self.stop_loss) |
                (df['relative_strength'] < df['rs_ma'])  # Relative strength lost
            )
        )
        
        signals.loc[exit_condition, 'position'] = 0
        signals.loc[exit_condition, 'exit_reason'] = 'Exit conditions met'
        
        return signals


class EarningsMomentumStrategy:
    """Strategy 5: Earnings Momentum (3-5 day holds)"""
    
    def __init__(self, name: str = "Earnings Momentum"):
        self.name = name
        self.gap_threshold = 0.02  # 2% gap up
        self.volume_threshold = 2.0  # 2x average volume
        self.profit_target = 0.08
        self.stop_loss = 0.04
        self.max_hold_days = 5
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for earnings momentum strategy."""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        signals['exit_reason'] = ''
        
        # Calculate gap up
        df['gap_up'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Entry conditions (simulate earnings beat)
        entry_condition = (
            (df['gap_up'] > self.gap_threshold) &  # Gap up
            (df['Volume_ratio'] > self.volume_threshold) &  # High volume
            (df['Close'] > df['Open']) &  # Positive day
            (df['RSI'] > 50)  # Momentum
        )
        
        signals.loc[entry_condition, 'position'] = 1
        signals.loc[entry_condition, 'entry_reason'] = 'Earnings momentum'
        
        # Exit conditions
        exit_condition = (
            (signals['position'].shift(1) == 1) &
            (
                (df['Close'] / df['Close'].shift(1) > 1 + self.profit_target) |
                (df['Close'] / df['Close'].shift(1) < 1 - self.stop_loss) |
                (df['RSI'] < 45)  # Momentum loss
            )
        )
        
        signals.loc[exit_condition, 'position'] = 0
        signals.loc[exit_condition, 'exit_reason'] = 'Exit conditions met'
        
        return signals


class SwingTradingBacktester:
    """Backtester for swing trading strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.framework = SwingTradingFramework(initial_capital)
        
    def backtest_strategy(self, strategy, symbol: str, start_date: str, end_date: str, 
                         benchmark_symbol: str = 'SPY') -> StrategyResult:
        """Backtest a single strategy."""
        logger.info(f"🧪 Backtesting {strategy.name} on {symbol}")
        
        # Load data
        df = self.framework.load_data(symbol, start_date, end_date)
        if df.empty:
            return None
        
        # Load benchmark data
        benchmark_df = self.framework.load_data(benchmark_symbol, start_date, end_date)
        
        # Generate signals
        if strategy.name == "Sector Rotation":
            signals = strategy.generate_signals(df, benchmark_df)
        else:
            signals = strategy.generate_signals(df)
        
        # Execute trades
        trades = self._execute_trades(df, signals, strategy.name, symbol)
        
        # Calculate performance metrics
        result = self._calculate_performance(trades, df, benchmark_df, strategy.name, symbol)
        
        return result
    
    def _execute_trades(self, df: pd.DataFrame, signals: pd.DataFrame, 
                       strategy_name: str, symbol: str) -> List[Trade]:
        """Execute trades based on signals."""
        trades = []
        current_position = None
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            signal = signals.loc[timestamp, 'position']
            entry_reason = signals.loc[timestamp, 'entry_reason']
            exit_reason = signals.loc[timestamp, 'exit_reason']
            
            # Entry signal
            if signal == 1 and current_position is None:
                current_position = Trade(
                    entry_date=timestamp,
                    exit_date=None,
                    entry_price=row['Close'],
                    exit_price=None,
                    position_size=self.framework.calculate_position_size(row['Close'], row['Close'] * 0.98),
                    strategy_name=strategy_name,
                    symbol=symbol,
                    entry_reason=entry_reason,
                    exit_reason=None
                )
            
            # Exit signal
            elif signal == 0 and current_position is not None:
                current_position.exit_date = timestamp
                current_position.exit_price = row['Close']
                current_position.exit_reason = exit_reason
                
                # Calculate P&L
                current_position.pnl = (current_position.exit_price - current_position.entry_price) * current_position.position_size
                current_position.pnl_pct = (current_position.exit_price - current_position.entry_price) / current_position.entry_price
                current_position.days_held = (current_position.exit_date - current_position.entry_date).days
                
                trades.append(current_position)
                current_position = None
        
        return trades
    
    def _calculate_performance(self, trades: List[Trade], df: pd.DataFrame, 
                             benchmark_df: pd.DataFrame, strategy_name: str, symbol: str) -> StrategyResult:
        """Calculate performance metrics."""
        if not trades:
            return StrategyResult(
                strategy_name=strategy_name,
                symbol=symbol,
                total_return=0,
                annualized_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                avg_trade_duration=0,
                buy_hold_return=0,
                excess_return=0,
                trades=[]
            )
        
        # Calculate returns
        total_pnl = sum(trade.pnl for trade in trades)
        total_return = total_pnl / self.initial_capital
        
        # Calculate buy-and-hold return
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        # Calculate other metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) / self.initial_capital if len(drawdown) > 0 else 0
        
        # Calculate average trade duration
        avg_trade_duration = np.mean([t.days_held for t in trades if t.days_held is not None])
        
        return StrategyResult(
            strategy_name=strategy_name,
            symbol=symbol,
            total_return=total_return,
            annualized_return=total_return * (252 / len(df)),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            buy_hold_return=buy_hold_return,
            excess_return=total_return - buy_hold_return,
            trades=trades
        )


def run_swing_trading_backtest():
    """Run backtest for all swing trading strategies."""
    logger.info("🚀 Starting Swing Trading Strategy Backtest")
    
    # Initialize backtester
    backtester = SwingTradingBacktester(initial_capital=10000)
    
    # Define test parameters
    symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # Initialize strategies
    strategies = [
        OversoldBounceStrategy(),
        BreakoutContinuationStrategy(),
        VolatilityExpansionStrategy(),
        SectorRotationStrategy(),
        EarningsMomentumStrategy()
    ]
    
    results = []
    
    # Run backtests
    for strategy in strategies:
        for symbol in symbols:
            try:
                result = backtester.backtest_strategy(strategy, symbol, start_date, end_date)
                if result:
                    results.append(result)
                    logger.info(f"✅ {strategy.name} on {symbol}: "
                              f"Return={result.total_return:.1%}, "
                              f"Sharpe={result.sharpe_ratio:.2f}, "
                              f"Trades={result.total_trades}, "
                              f"Win Rate={result.win_rate:.1%}")
            except Exception as e:
                logger.error(f"❌ Error backtesting {strategy.name} on {symbol}: {e}")
    
    # Generate summary report
    generate_summary_report(results)
    
    return results


def generate_summary_report(results: List[StrategyResult]):
    """Generate a summary report of backtest results."""
    if not results:
        logger.warning("⚠️ No results to summarize")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 SWING TRADING STRATEGY BACKTEST SUMMARY")
    logger.info("=" * 80)
    
    # Group by strategy
    strategy_results = {}
    for result in results:
        if result.strategy_name not in strategy_results:
            strategy_results[result.strategy_name] = []
        strategy_results[result.strategy_name].append(result)
    
    # Calculate strategy averages
    for strategy_name, strategy_list in strategy_results.items():
        avg_return = np.mean([r.total_return for r in strategy_list])
        avg_sharpe = np.mean([r.sharpe_ratio for r in strategy_list])
        avg_win_rate = np.mean([r.win_rate for r in strategy_list])
        total_trades = sum([r.total_trades for r in strategy_list])
        avg_excess = np.mean([r.excess_return for r in strategy_list])
        
        logger.info(f"\n📈 {strategy_name}:")
        logger.info(f"   Average Return: {avg_return:.1%}")
        logger.info(f"   Average Sharpe: {avg_sharpe:.2f}")
        logger.info(f"   Average Win Rate: {avg_win_rate:.1%}")
        logger.info(f"   Total Trades: {total_trades}")
        logger.info(f"   Average Excess Return: {avg_excess:.1%}")
    
    # Overall statistics
    all_returns = [r.total_return for r in results]
    all_excess = [r.excess_return for r in results]
    all_sharpe = [r.sharpe_ratio for r in results]
    
    logger.info(f"\n🎯 OVERALL STATISTICS:")
    logger.info(f"   Average Strategy Return: {np.mean(all_returns):.1%}")
    logger.info(f"   Average Excess Return: {np.mean(all_excess):.1%}")
    logger.info(f"   Average Sharpe Ratio: {np.mean(all_sharpe):.2f}")
    logger.info(f"   Strategies Tested: {len(strategy_results)}")
    logger.info(f"   Total Tests: {len(results)}")
    
    # Best performing strategies
    best_strategies = sorted(strategy_results.items(), 
                           key=lambda x: np.mean([r.total_return for r in x[1]]), 
                           reverse=True)
    
    logger.info(f"\n🏆 TOP PERFORMING STRATEGIES:")
    for i, (strategy_name, strategy_list) in enumerate(best_strategies[:3]):
        avg_return = np.mean([r.total_return for r in strategy_list])
        logger.info(f"   {i+1}. {strategy_name}: {avg_return:.1%}")


if __name__ == "__main__":
    # Run the backtest
    results = run_swing_trading_backtest()
    
    logger.info("\n🎉 Swing Trading Strategy Backtest Complete!")
    logger.info("📊 Check the results above for strategy performance")
