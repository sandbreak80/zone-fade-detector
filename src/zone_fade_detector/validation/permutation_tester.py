"""
Monte Carlo Permutation Testing for Trading Strategy Validation.

This module implements the permutation testing components for the 4-step validation battery:
- In-Sample Monte-Carlo Permutation Test (IMCPT)
- Walk-Forward Permutation Test (WFPT)
"""

import random
import time
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PermutationResult:
    """Results from permutation testing."""
    real_score: float
    permutation_scores: List[float]
    p_value: float
    n_permutations: int
    test_time: float
    significant: bool
    significance_threshold: float


class PermutationTester:
    """
    Monte Carlo Permutation Tester for trading strategy validation.
    
    Implements both IMCPT and WFPT permutation testing methods.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the permutation tester.
        
        Args:
            random_seed: Optional random seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    def permute_bars(self, bars: List[Any], preserve_first_order: bool = True) -> List[Any]:
        """
        Permute bars while optionally preserving first-order statistics.
        
        Args:
            bars: List of bar objects to permute
            preserve_first_order: If True, preserves mean and variance of returns
        
        Returns:
            List of permuted bars
        """
        if not bars:
            return bars
        
        # Create a copy to avoid modifying original
        permuted_bars = bars.copy()
        
        if preserve_first_order:
            # Calculate original returns for first-order preservation
            if hasattr(bars[0], 'close') and len(bars) > 1:
                original_returns = []
                for i in range(1, len(bars)):
                    ret = (bars[i].close - bars[i-1].close) / bars[i-1].close
                    original_returns.append(ret)
                
                # Shuffle returns
                random.shuffle(original_returns)
                
                # Reconstruct bars with shuffled returns
                permuted_bars[0] = bars[0]  # Keep first bar unchanged
                for i in range(1, len(bars)):
                    new_close = permuted_bars[i-1].close * (1 + original_returns[i-1])
                    # Create new bar with permuted close price
                    new_bar = type(bars[i])(
                        timestamp=bars[i].timestamp,
                        open=bars[i].open,
                        high=bars[i].high,
                        low=bars[i].low,
                        close=new_close,
                        volume=bars[i].volume
                    )
                    permuted_bars[i] = new_bar
            else:
                # Simple random shuffle if no close prices available
                random.shuffle(permuted_bars)
        else:
            # Simple random shuffle
            random.shuffle(permuted_bars)
        
        return permuted_bars
    
    def in_sample_permutation_test(self, 
                                 bars: List[Any],
                                 strategy_class: type,
                                 param_space: Dict[str, List[Any]],
                                 optimization_function: Callable[[List[Any], Dict[str, Any]], float],
                                 n_permutations: int = 1000,
                                 significance_threshold: float = 0.01) -> PermutationResult:
        """
        Perform In-Sample Monte-Carlo Permutation Test (IMCPT).
        
        This test permutes the training data to destroy temporal structure,
        then re-optimizes the strategy on the permuted data to assess selection bias.
        
        Args:
            bars: List of bar objects for training data
            strategy_class: Strategy class to test
            param_space: Parameter space for optimization
            optimization_function: Function that optimizes strategy and returns best score
            n_permutations: Number of permutations to perform
            significance_threshold: P-value threshold for significance (default 1%)
        
        Returns:
            PermutationResult with test results
        """
        logger.info(f"Starting IMCPT with {n_permutations} permutations")
        start_time = time.time()
        
        # Get real score on original data
        real_score = optimization_function(bars, param_space)
        logger.info(f"Real optimization score: {real_score}")
        
        # Perform permutations
        permutation_scores = []
        
        for i in range(n_permutations):
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{n_permutations} permutations")
            
            # Permute the bars
            permuted_bars = self.permute_bars(bars, preserve_first_order=True)
            
            # Optimize on permuted data
            try:
                permuted_score = optimization_function(permuted_bars, param_space)
                permutation_scores.append(permuted_score)
            except Exception as e:
                logger.warning(f"Permutation {i + 1} failed: {e}")
                permutation_scores.append(float('-inf'))
        
        # Calculate p-value
        better_permutations = sum(1 for score in permutation_scores if score >= real_score)
        p_value = better_permutations / len(permutation_scores)
        
        test_time = time.time() - start_time
        significant = p_value < significance_threshold
        
        logger.info(f"IMCPT completed in {test_time:.2f}s")
        logger.info(f"P-value: {p_value:.4f} (threshold: {significance_threshold})")
        logger.info(f"Significant: {significant}")
        
        return PermutationResult(
            real_score=real_score,
            permutation_scores=permutation_scores,
            p_value=p_value,
            n_permutations=n_permutations,
            test_time=test_time,
            significant=significant,
            significance_threshold=significance_threshold
        )
    
    def walk_forward_permutation_test(self, 
                                    bars: List[Any],
                                    strategy_class: type,
                                    param_space: Dict[str, List[Any]],
                                    walk_forward_function: Callable[[List[Any], Dict[str, Any]], float],
                                    train_window_size: int,
                                    retrain_frequency: int,
                                    n_permutations: int = 200,
                                    significance_threshold: float = 0.05) -> PermutationResult:
        """
        Perform Walk-Forward Permutation Test (WFPT).
        
        This test permutes only the out-of-sample segments while keeping
        the training window intact, then re-runs the walk-forward process.
        
        Args:
            bars: List of bar objects for full dataset
            strategy_class: Strategy class to test
            param_space: Parameter space for optimization
            walk_forward_function: Function that performs walk-forward analysis
            train_window_size: Size of training window
            retrain_frequency: Frequency of retraining (in bars)
            n_permutations: Number of permutations to perform
            significance_threshold: P-value threshold for significance
        
        Returns:
            PermutationResult with test results
        """
        logger.info(f"Starting WFPT with {n_permutations} permutations")
        start_time = time.time()
        
        # Get real walk-forward score
        real_score = walk_forward_function(bars, param_space)
        logger.info(f"Real walk-forward score: {real_score}")
        
        # Perform permutations
        permutation_scores = []
        
        for i in range(n_permutations):
            if (i + 1) % 50 == 0:
                logger.info(f"Completed {i + 1}/{n_permutations} permutations")
            
            # Permute only OOS segments, keep training window intact
            permuted_bars = self._permute_oos_segments(bars, train_window_size, retrain_frequency)
            
            # Run walk-forward on permuted data
            try:
                permuted_score = walk_forward_function(permuted_bars, param_space)
                permutation_scores.append(permuted_score)
            except Exception as e:
                logger.warning(f"Permutation {i + 1} failed: {e}")
                permutation_scores.append(float('-inf'))
        
        # Calculate p-value
        better_permutations = sum(1 for score in permutation_scores if score >= real_score)
        p_value = better_permutations / len(permutation_scores)
        
        test_time = time.time() - start_time
        significant = p_value < significance_threshold
        
        logger.info(f"WFPT completed in {test_time:.2f}s")
        logger.info(f"P-value: {p_value:.4f} (threshold: {significance_threshold})")
        logger.info(f"Significant: {significant}")
        
        return PermutationResult(
            real_score=real_score,
            permutation_scores=permutation_scores,
            p_value=p_value,
            n_permutations=n_permutations,
            test_time=test_time,
            significant=significant,
            significance_threshold=significance_threshold
        )
    
    def _permute_oos_segments(self, bars: List[Any], train_window_size: int, 
                            retrain_frequency: int) -> List[Any]:
        """
        Permute only the out-of-sample segments while keeping training windows intact.
        
        Args:
            bars: List of bar objects
            train_window_size: Size of training window
            retrain_frequency: Frequency of retraining
        
        Returns:
            List of bars with permuted OOS segments
        """
        if len(bars) <= train_window_size:
            return bars.copy()
        
        permuted_bars = bars.copy()
        
        # Identify OOS segments
        oos_segments = []
        current_pos = train_window_size
        
        while current_pos < len(bars):
            # Find the end of this OOS segment
            segment_end = min(current_pos + retrain_frequency, len(bars))
            oos_segments.append((current_pos, segment_end))
            current_pos = segment_end
        
        # Extract and shuffle OOS segments
        oos_data = []
        for start, end in oos_segments:
            oos_data.extend(bars[start:end])
        
        random.shuffle(oos_data)
        
        # Reconstruct bars with shuffled OOS segments
        data_idx = 0
        for start, end in oos_segments:
            for i in range(start, end):
                permuted_bars[i] = oos_data[data_idx]
                data_idx += 1
        
        return permuted_bars
    
    def calculate_permutation_statistics(self, result: PermutationResult) -> Dict[str, Any]:
        """
        Calculate additional statistics from permutation test results.
        
        Args:
            result: PermutationResult from a permutation test
        
        Returns:
            Dictionary with additional statistics
        """
        if not result.permutation_scores:
            return {}
        
        scores = result.permutation_scores
        
        # Basic statistics
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score)**2 for s in scores) / len(scores)
        std_score = variance**0.5
        
        # Percentiles
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        percentiles = {
            "p5": sorted_scores[int(0.05 * n)] if n > 0 else 0,
            "p25": sorted_scores[int(0.25 * n)] if n > 0 else 0,
            "p50": sorted_scores[int(0.50 * n)] if n > 0 else 0,
            "p75": sorted_scores[int(0.75 * n)] if n > 0 else 0,
            "p95": sorted_scores[int(0.95 * n)] if n > 0 else 0
        }
        
        # Effect size (how much better is real score than mean permutation score)
        effect_size = (result.real_score - mean_score) / std_score if std_score > 0 else 0
        
        return {
            "mean_permutation_score": mean_score,
            "std_permutation_score": std_score,
            "variance_permutation_score": variance,
            "percentiles": percentiles,
            "effect_size": effect_size,
            "real_score_percentile": sum(1 for s in scores if s <= result.real_score) / len(scores),
            "n_permutations": result.n_permutations,
            "test_time": result.test_time
        }
