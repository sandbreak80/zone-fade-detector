"""
Strategy Registry for Trading Strategy Testing Framework.

This module provides a registry of all available strategies and their implementations.
"""

from typing import List
from zone_fade_detector.strategies.base_strategy import BaseStrategy
from zone_fade_detector.strategies.macd_strategy import MACDStrategy

# Strategy registry
STRATEGIES = {
    'macd': MACDStrategy,
    'macd_crossover': MACDStrategy,  # Alias
}

def get_strategy_class(strategy_name: str) -> type:
    """
    Get strategy class by name.
    
    Args:
        strategy_name: Name of strategy to get
        
    Returns:
        Strategy class
        
    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {available}")
    
    return STRATEGIES[strategy_name]

def get_strategy(strategy_name: str) -> BaseStrategy:
    """
    Get strategy instance by name.
    
    Args:
        strategy_name: Name of strategy to get
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {available}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class()

def list_strategies() -> List[str]:
    """
    List all available strategies.
    
    Returns:
        List of strategy names
    """
    return list(STRATEGIES.keys())

def register_strategy(name: str, strategy_class: type) -> None:
    """
    Register a new strategy.
    
    Args:
        name: Strategy name
        strategy_class: Strategy class
        
    Raises:
        ValueError: If strategy class doesn't inherit from BaseStrategy
    """
    if not issubclass(strategy_class, BaseStrategy):
        raise ValueError("Strategy class must inherit from BaseStrategy")
    
    STRATEGIES[name] = strategy_class

def unregister_strategy(name: str) -> None:
    """
    Unregister a strategy.
    
    Args:
        name: Strategy name to unregister
        
    Raises:
        ValueError: If strategy name is not found
    """
    if name not in STRATEGIES:
        raise ValueError(f"Strategy '{name}' not found")
    
    del STRATEGIES[name]

def get_strategy_info(strategy_name: str) -> dict:
    """
    Get information about a strategy.
    
    Args:
        strategy_name: Name of strategy
        
    Returns:
        Dictionary with strategy information
        
    Raises:
        ValueError: If strategy name is not found
    """
    strategy = get_strategy(strategy_name)
    
    return {
        'name': strategy.get_name(),
        'parameter_count': strategy.get_parameter_count(),
        'total_combinations': strategy.get_total_combinations(),
        'parameter_space': strategy.get_parameter_space(),
        'description': getattr(strategy, 'get_strategy_description', lambda: 'No description available')()
    }