"""
Parameter Optimization Engine for Trading Strategy Testing Framework.

This module provides optimization capabilities for strategy parameters,
supporting both grid search and more sophisticated optimization methods.
"""

import itertools
import random
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    optimization_time: float
    total_evaluations: int


class OptimizationEngine:
    """
    Parameter optimization engine for trading strategies.
    
    Supports grid search, random search, and custom optimization methods.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the optimization engine.
        
        Args:
            random_seed: Optional random seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    def grid_search(self, 
                   param_space: Dict[str, List[Any]], 
                   objective_function: Callable[[Dict[str, Any]], float],
                   max_evaluations: Optional[int] = None) -> OptimizationResult:
        """
        Perform grid search optimization over parameter space.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            objective_function: Function that takes parameters and returns a score
            max_evaluations: Maximum number of evaluations (None for exhaustive)
        
        Returns:
            OptimizationResult with best parameters and scores
        """
        import time
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if max_evaluations is specified
        if max_evaluations and len(all_combinations) > max_evaluations:
            logger.info(f"Limiting grid search to {max_evaluations} evaluations from {len(all_combinations)} total combinations")
            all_combinations = random.sample(all_combinations, max_evaluations)
        
        logger.info(f"Starting grid search with {len(all_combinations)} parameter combinations")
        
        # Evaluate all combinations
        all_results = []
        best_score = float('-inf')
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            
            try:
                score = objective_function(params)
                all_results.append((params, score))
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{len(all_combinations)} evaluations")
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {e}")
                all_results.append((params, float('-inf')))
        
        optimization_time = time.time() - start_time
        
        logger.info(f"Grid search completed in {optimization_time:.2f}s")
        logger.info(f"Best score: {best_score} with parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            total_evaluations=len(all_combinations)
        )
    
    def random_search(self, 
                    param_space: Dict[str, List[Any]], 
                    objective_function: Callable[[Dict[str, Any]], float],
                    n_evaluations: int) -> OptimizationResult:
        """
        Perform random search optimization over parameter space.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            objective_function: Function that takes parameters and returns a score
            n_evaluations: Number of random evaluations to perform
        
        Returns:
            OptimizationResult with best parameters and scores
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting random search with {n_evaluations} evaluations")
        
        all_results = []
        best_score = float('-inf')
        best_params = None
        
        for i in range(n_evaluations):
            # Randomly sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = random.choice(param_values)
            
            try:
                score = objective_function(params)
                all_results.append((params, score))
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{n_evaluations} evaluations")
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {e}")
                all_results.append((params, float('-inf')))
        
        optimization_time = time.time() - start_time
        
        logger.info(f"Random search completed in {optimization_time:.2f}s")
        logger.info(f"Best score: {best_score} with parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            total_evaluations=n_evaluations
        )
    
    def optimize(self, 
                param_space: Dict[str, List[Any]], 
                objective_function: Callable[[Dict[str, Any]], float],
                method: str = "grid",
                max_evaluations: Optional[int] = None) -> OptimizationResult:
        """
        Optimize parameters using specified method.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            objective_function: Function that takes parameters and returns a score
            method: Optimization method ("grid" or "random")
            max_evaluations: Maximum number of evaluations (for grid search)
        
        Returns:
            OptimizationResult with best parameters and scores
        """
        if method == "grid":
            return self.grid_search(param_space, objective_function, max_evaluations)
        elif method == "random":
            if max_evaluations is None:
                max_evaluations = 1000  # Default for random search
            return self.random_search(param_space, objective_function, max_evaluations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def calculate_parameter_stability(self, results: List[Tuple[Dict[str, Any], float]], 
                                    top_n: int = 10) -> Dict[str, Any]:
        """
        Calculate parameter stability metrics from optimization results.
        
        Args:
            results: List of (parameters, score) tuples
            top_n: Number of top results to analyze
        
        Returns:
            Dictionary with stability metrics
        """
        # Sort by score (descending)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:top_n]
        
        if not top_results:
            return {"stability_score": 0.0, "parameter_variance": {}}
        
        # Calculate parameter variance for top results
        param_variance = {}
        for param_name in top_results[0][0].keys():
            values = [result[0][param_name] for result in top_results]
            
            if isinstance(values[0], (int, float)):
                # Numeric parameters
                variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
                param_variance[param_name] = {
                    "variance": variance,
                    "std": variance**0.5,
                    "values": values
                }
            else:
                # Categorical parameters
                unique_values = list(set(values))
                param_variance[param_name] = {
                    "variance": len(unique_values) / len(values),
                    "std": (len(unique_values) / len(values))**0.5,
                    "values": values,
                    "unique_count": len(unique_values)
                }
        
        # Calculate overall stability score (lower variance = higher stability)
        numeric_variances = [v["variance"] for v in param_variance.values() 
                           if isinstance(v["values"][0], (int, float))]
        
        if numeric_variances:
            avg_variance = sum(numeric_variances) / len(numeric_variances)
            stability_score = max(0, 1 - avg_variance)  # Higher score = more stable
        else:
            stability_score = 1.0
        
        return {
            "stability_score": stability_score,
            "parameter_variance": param_variance,
            "top_n": top_n,
            "total_evaluations": len(results)
        }
