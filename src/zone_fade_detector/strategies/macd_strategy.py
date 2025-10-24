"""
MACD Crossover Strategy Implementation.

This module implements a MACD (Moving Average Convergence Divergence) crossover strategy
for the Trading Strategy Testing Framework. This strategy serves as the framework
shakedown test to validate the testing methodology.
"""

import numpy as np
from typing import List, Dict, Any
from zone_fade_detector.strategies.base_strategy import BaseStrategy
from zone_fade_detector.core.models import OHLCVBar


class MACDStrategy(BaseStrategy):
    """
    MACD Crossover Strategy.
    
    This strategy generates buy/sell signals based on MACD line and signal line crossovers.
    It serves as the framework shakedown test to validate the testing methodology.
    
    Strategy Logic:
    - Buy signal: MACD line crosses above signal line
    - Sell signal: MACD line crosses below signal line
    - No signal: MACD line and signal line are on same side
    """
    
    def __init__(self):
        """Initialize MACD strategy."""
        self.name = "MACD Crossover Strategy"
    
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        """
        Generate MACD crossover signals.
        
        Args:
            bars: List of OHLCVBar objects
            params: Dictionary containing:
                - fast_period: Fast EMA period (default: 12)
                - slow_period: Slow EMA period (default: 26)
                - signal_period: Signal line EMA period (default: 9)
        
        Returns:
            List of position signals: 1 (long), 0 (flat), -1 (short)
        """
        if len(bars) < params['slow_period']:
            return [0] * len(bars)
        
        # Extract closing prices
        closes = [bar.close for bar in bars]
        
        # Calculate MACD line and signal line
        macd_line, signal_line = self._calculate_macd(closes, params)
        
        # Generate signals based on crossovers
        signals = []
        for i in range(len(bars)):
            if i < params['slow_period']:
                signals.append(0)  # No signal for first bars
                continue
            
            # Crossover logic
            if i > 0:
                # Buy signal: MACD crosses above signal line
                if (macd_line[i-1] <= signal_line[i-1] and 
                    macd_line[i] > signal_line[i]):
                    signals.append(1)
                # Sell signal: MACD crosses below signal line
                elif (macd_line[i-1] >= signal_line[i-1] and 
                      macd_line[i] < signal_line[i]):
                    signals.append(-1)
                # No signal: maintain previous position
                else:
                    signals.append(signals[-1] if signals else 0)
            else:
                signals.append(0)
        
        return signals
    
    def _calculate_macd(self, closes: List[float], params: Dict[str, Any]) -> tuple:
        """
        Calculate MACD line and signal line.
        
        Args:
            closes: List of closing prices
            params: Strategy parameters
            
        Returns:
            Tuple of (macd_line, signal_line) as numpy arrays
        """
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        signal_period = params['signal_period']
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(closes, fast_period)
        slow_ema = self._calculate_ema(closes, slow_period)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Signal line = EMA of MACD line
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        return macd_line, signal_line
    
    def _calculate_ema(self, data: List[float], period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: List of values
            period: EMA period
            
        Returns:
            NumPy array of EMA values
        """
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def get_parameter_space(self) -> Dict[str, List]:
        """
        Define MACD parameter optimization space.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            'fast_period': [10, 12, 14, 16, 18, 20],
            'slow_period': [20, 25, 30, 35, 40],
            'signal_period': [5, 7, 9, 11, 13]
        }
    
    def get_name(self) -> str:
        """
        Return strategy name.
        
        Returns:
            Strategy name string
        """
        return self.name
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default MACD parameters.
        
        Returns:
            Dictionary with default parameter values
        """
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate MACD parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Check required parameters
        required_params = ['fast_period', 'slow_period', 'signal_period']
        for param in required_params:
            if param not in params:
                return False
        
        # Validate parameter values
        if params['fast_period'] >= params['slow_period']:
            return False
        
        if params['fast_period'] <= 0 or params['slow_period'] <= 0 or params['signal_period'] <= 0:
            return False
        
        return super().validate_parameters(params)
    
    def get_strategy_description(self) -> str:
        """
        Get detailed strategy description.
        
        Returns:
            String describing the strategy logic
        """
        return """
        MACD Crossover Strategy:
        
        The MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator.
        It consists of three components:
        1. MACD Line: Fast EMA - Slow EMA
        2. Signal Line: EMA of MACD Line
        3. Histogram: MACD Line - Signal Line
        
        Trading Signals:
        - Buy: MACD line crosses above signal line
        - Sell: MACD line crosses below signal line
        
        This strategy is commonly used in technical analysis and serves as a good
        framework shakedown test because it's well-known and likely overfitted.
        """
