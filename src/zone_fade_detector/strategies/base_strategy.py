"""
Base Strategy Interface for Trading Strategy Testing Framework.

This module provides the abstract base class that all trading strategies must implement.
The interface ensures consistent strategy implementation across the framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from zone_fade_detector.core.models import OHLCVBar


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the three required methods:
    - generate_signal: Generate position signals for each bar
    - get_parameter_space: Define parameter optimization space
    - get_name: Return strategy name for reporting
    """
    
    @abstractmethod
    def generate_signal(self, bars: List[OHLCVBar], params: Dict[str, Any]) -> List[int]:
        """
        Generate position signals for each bar.
        
        Args:
            bars: List of OHLCVBar objects containing price data
            params: Dictionary of strategy parameters
            
        Returns:
            List of integers representing position signals:
            - 1: Long position
            - 0: No position (flat)
            - -1: Short position
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List]:
        """
        Define parameter optimization space.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values.
            Used by the optimization engine to find optimal parameters.
            
        Example:
            {
                'fast_period': [10, 12, 14, 16, 18, 20],
                'slow_period': [20, 25, 30, 35, 40],
                'signal_period': [5, 7, 9, 11, 13]
            }
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return strategy name for reporting and identification.
        
        Returns:
            String with strategy name
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        parameter_space = self.get_parameter_space()
        
        for param_name, param_value in params.items():
            if param_name not in parameter_space:
                return False
            
            if param_value not in parameter_space[param_name]:
                return False
        
        return True
    
    def get_parameter_count(self) -> int:
        """
        Get the number of parameters in the parameter space.
        
        Returns:
            Number of parameters
        """
        return len(self.get_parameter_space())
    
    def get_total_combinations(self) -> int:
        """
        Get the total number of parameter combinations.
        
        Returns:
            Total number of parameter combinations
        """
        import math
        parameter_space = self.get_parameter_space()
        return math.prod(len(values) for values in parameter_space.values())
