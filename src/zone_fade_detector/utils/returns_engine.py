"""
Bar-Level Returns Engine for Trading Strategy Testing Framework.

This module provides the core functionality for calculating bar-level returns
with proper look-ahead prevention and transaction cost modeling.
"""

from typing import List, Tuple
from zone_fade_detector.core.models import OHLCVBar


class ReturnsEngine:
    """
    Bar-level return calculation with look-ahead prevention.
    
    This engine ensures that strategy returns are calculated correctly without
    any look-ahead bias, which is critical for valid backtesting results.
    """
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        """
        Initialize returns engine.
        
        Args:
            commission: Commission rate per trade (default: 0.1%)
            slippage: Slippage rate per trade (default: 0.05%)
        """
        self.commission = commission
        self.slippage = slippage
    
    def calculate_strategy_returns(self, signals: List[int], 
                                  bars: List[OHLCVBar]) -> List[float]:
        """
        Calculate strategy returns with proper look-ahead prevention.
        
        Args:
            signals: List of position signals (-1, 0, 1)
            bars: List of OHLCVBar objects
            
        Returns:
            List of strategy returns
            
        Raises:
            ValueError: If signals and bars have different lengths
        """
        if len(signals) != len(bars):
            raise ValueError("Signals and bars must have same length")
        
        # Calculate bar returns (close-to-close)
        bar_returns = self._calculate_bar_returns(bars)
        
        # Shift returns forward 1 bar (look-ahead prevention)
        shifted_returns = [0.0] + bar_returns[:-1]
        
        # Calculate strategy returns
        strategy_returns = []
        for i, (signal, ret) in enumerate(zip(signals, shifted_returns)):
            if signal == 0:
                strategy_returns.append(0.0)
            else:
                # Apply transaction costs
                cost = self._calculate_transaction_cost(signal, bars[i])
                strategy_returns.append(signal * ret - cost)
        
        return strategy_returns
    
    def _calculate_bar_returns(self, bars: List[OHLCVBar]) -> List[float]:
        """
        Calculate close-to-close returns for each bar.
        
        Args:
            bars: List of OHLCVBar objects
            
        Returns:
            List of bar returns
        """
        bar_returns = []
        
        for i in range(len(bars)):
            if i == 0:
                bar_returns.append(0.0)  # First bar has no return
            else:
                # Close-to-close return
                return_val = (bars[i].close - bars[i-1].close) / bars[i-1].close
                bar_returns.append(return_val)
        
        return bar_returns
    
    def _calculate_transaction_cost(self, signal: int, bar: OHLCVBar) -> float:
        """
        Calculate transaction cost for a trade.
        
        Args:
            signal: Position signal (-1, 0, 1)
            bar: OHLCVBar object for price reference
            
        Returns:
            Transaction cost as a fraction of price
        """
        if signal == 0:
            return 0.0
        
        # Commission + slippage
        total_cost = self.commission + self.slippage
        
        return total_cost
    
    def calculate_cumulative_returns(self, returns: List[float]) -> List[float]:
        """
        Calculate cumulative returns from period returns.
        
        Args:
            returns: List of period returns
            
        Returns:
            List of cumulative returns
        """
        cumulative = []
        cumsum = 0.0
        
        for ret in returns:
            cumsum += ret
            cumulative.append(cumsum)
        
        return cumulative
    
    def calculate_equity_curve(self, returns: List[float], 
                              initial_capital: float = 10000) -> List[float]:
        """
        Calculate equity curve from returns.
        
        Args:
            returns: List of period returns
            initial_capital: Initial capital amount
            
        Returns:
            List of equity values
        """
        equity = []
        current_equity = initial_capital
        
        for ret in returns:
            current_equity *= (1 + ret)
            equity.append(current_equity)
        
        return equity
    
    def calculate_drawdown(self, equity_curve: List[float]) -> List[float]:
        """
        Calculate drawdown from equity curve.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            List of drawdown values (negative)
        """
        drawdown = []
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            dd = (equity - peak) / peak
            drawdown.append(dd)
        
        return drawdown
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Maximum drawdown (negative value)
        """
        drawdown = self.calculate_drawdown(equity_curve)
        return min(drawdown)
    
    def calculate_sharpe_ratio(self, returns: List[float], 
                              risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0
        
        # Calculate excess returns
        excess_returns = [r - risk_free_rate for r in returns]
        
        # Calculate mean and standard deviation
        mean_return = sum(excess_returns) / len(excess_returns)
        variance = sum((r - mean_return) ** 2 for r in excess_returns) / len(excess_returns)
        std_return = variance ** 0.5
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def calculate_profit_factor(self, returns: List[float]) -> float:
        """
        Calculate profit factor from returns.
        
        Args:
            returns: List of period returns
            
        Returns:
            Profit factor (gross profit / gross loss)
        """
        gains = sum(r for r in returns if r > 0)
        losses = sum(-r for r in returns if r < 0)
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses
    
    def calculate_win_rate(self, returns: List[float]) -> float:
        """
        Calculate win rate from returns.
        
        Args:
            returns: List of period returns
            
        Returns:
            Win rate (fraction of positive returns)
        """
        if not returns:
            return 0.0
        
        positive_returns = sum(1 for r in returns if r > 0)
        total_returns = len(returns)
        
        return positive_returns / total_returns
    
    def calculate_metrics(self, returns: List[float], 
                         initial_capital: float = 10000) -> dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: List of period returns
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary of performance metrics
        """
        if not returns:
            return {}
        
        # Basic metrics
        total_return = sum(returns)
        mean_return = sum(returns) / len(returns)
        
        # Risk metrics
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        # Equity curve and drawdown
        equity_curve = self.calculate_equity_curve(returns, initial_capital)
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        
        # Performance metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        profit_factor = self.calculate_profit_factor(returns)
        win_rate = self.calculate_win_rate(returns)
        
        return {
            'total_return': total_return,
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_trades': len([r for r in returns if r != 0])
        }
