"""
Walk-Forward Analysis for Trading Strategy Validation.

This module implements the Walk-Forward Test (WFT) component of the 4-step validation battery.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    total_score: float
    oos_scores: List[float]
    is_scores: List[float]
    retrain_dates: List[Any]
    best_params_history: List[Dict[str, Any]]
    analysis_time: float
    n_retrains: int
    oos_periods: List[Tuple[int, int]]  # (start_idx, end_idx) for each OOS period


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for trading strategy validation.
    
    Implements rolling retraining with configurable windows and frequencies.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the walk-forward analyzer.
        
        Args:
            random_seed: Optional random seed for reproducible results
        """
        self.random_seed = random_seed
    
    def analyze(self, 
               bars: List[Any],
               strategy_class: type,
               param_space: Dict[str, List[Any]],
               optimization_function: Callable[[List[Any], Dict[str, Any]], Tuple[Dict[str, Any], float]],
               train_window_size: int,
               retrain_frequency: int,
               min_train_bars: int = 100) -> WalkForwardResult:
        """
        Perform walk-forward analysis with rolling retraining.
        
        Args:
            bars: List of bar objects for full dataset
            strategy_class: Strategy class to test
            param_space: Parameter space for optimization
            optimization_function: Function that optimizes and returns (best_params, best_score)
            train_window_size: Size of training window (in bars)
            retrain_frequency: Frequency of retraining (in bars)
            min_train_bars: Minimum number of bars required for training
        
        Returns:
            WalkForwardResult with analysis results
        """
        logger.info(f"Starting walk-forward analysis")
        logger.info(f"Train window: {train_window_size} bars, Retrain frequency: {retrain_frequency} bars")
        start_time = time.time()
        
        if len(bars) < train_window_size + retrain_frequency:
            raise ValueError(f"Insufficient data: need at least {train_window_size + retrain_frequency} bars")
        
        oos_scores = []
        is_scores = []
        retrain_dates = []
        best_params_history = []
        oos_periods = []
        
        current_pos = train_window_size
        
        while current_pos < len(bars):
            # Define training window
            train_start = max(0, current_pos - train_window_size)
            train_end = current_pos
            train_bars = bars[train_start:train_end]
            
            if len(train_bars) < min_train_bars:
                logger.warning(f"Insufficient training data at position {current_pos}, skipping")
                current_pos += retrain_frequency
                continue
            
            # Define OOS window
            oos_start = current_pos
            oos_end = min(current_pos + retrain_frequency, len(bars))
            oos_bars = bars[oos_start:oos_end]
            
            if len(oos_bars) == 0:
                break
            
            logger.info(f"Training on bars {train_start}-{train_end}, Testing on bars {oos_start}-{oos_end}")
            
            # Optimize on training data
            try:
                best_params, is_score = optimization_function(train_bars, param_space)
                is_scores.append(is_score)
                best_params_history.append(best_params)
                
                # Test on OOS data
                oos_score = self._evaluate_oos_performance(bars, best_params, oos_start, oos_end, strategy_class)
                oos_scores.append(oos_score)
                oos_periods.append((oos_start, oos_end))
                
                # Record retrain date (using timestamp from first OOS bar)
                if hasattr(oos_bars[0], 'timestamp'):
                    retrain_dates.append(oos_bars[0].timestamp)
                else:
                    retrain_dates.append(oos_start)
                
                logger.info(f"IS score: {is_score:.4f}, OOS score: {oos_score:.4f}")
                
            except Exception as e:
                logger.error(f"Walk-forward step failed at position {current_pos}: {e}")
                oos_scores.append(0.0)
                is_scores.append(0.0)
                best_params_history.append({})
                oos_periods.append((oos_start, oos_end))
            
            current_pos += retrain_frequency
        
        total_score = sum(oos_scores)
        analysis_time = time.time() - start_time
        
        logger.info(f"Walk-forward analysis completed in {analysis_time:.2f}s")
        logger.info(f"Total OOS score: {total_score:.4f}")
        logger.info(f"Average OOS score: {total_score/len(oos_scores):.4f}")
        logger.info(f"Number of retrains: {len(retrain_dates)}")
        
        return WalkForwardResult(
            total_score=total_score,
            oos_scores=oos_scores,
            is_scores=is_scores,
            retrain_dates=retrain_dates,
            best_params_history=best_params_history,
            analysis_time=analysis_time,
            n_retrains=len(retrain_dates),
            oos_periods=oos_periods
        )
    
    def _evaluate_oos_performance(self, 
                                bars: List[Any], 
                                params: Dict[str, Any], 
                                oos_start: int, 
                                oos_end: int,
                                strategy_class: type) -> float:
        """
        Evaluate strategy performance on out-of-sample data.
        
        Args:
            bars: Full dataset
            params: Strategy parameters
            oos_start: Start index of OOS period
            oos_end: End index of OOS period
            strategy_class: Strategy class to test
        
        Returns:
            Performance score for OOS period
        """
        oos_bars = bars[oos_start:oos_end]
        
        if len(oos_bars) < 2:
            return 0.0
        
        # Create strategy instance and generate signals
        strategy = strategy_class()
        signals = strategy.generate_signal(oos_bars, params)
        
        if not signals or len(signals) != len(oos_bars):
            return 0.0
        
        # Calculate returns
        from zone_fade_detector.utils.returns_engine import ReturnsEngine
        returns_engine = ReturnsEngine()
        strategy_returns = returns_engine.calculate_strategy_returns(signals, oos_bars)
        
        # Calculate performance metrics
        metrics = returns_engine.calculate_metrics(strategy_returns)
        
        # Return total return as the score
        return metrics.get('total_return', 0.0)
    
    def calculate_walk_forward_statistics(self, result: WalkForwardResult) -> Dict[str, Any]:
        """
        Calculate additional statistics from walk-forward results.
        
        Args:
            result: WalkForwardResult from walk-forward analysis
        
        Returns:
            Dictionary with additional statistics
        """
        if not result.oos_scores:
            return {}
        
        oos_scores = result.oos_scores
        is_scores = result.is_scores
        
        # Basic statistics
        oos_mean = sum(oos_scores) / len(oos_scores)
        oos_std = (sum((s - oos_mean)**2 for s in oos_scores) / len(oos_scores))**0.5
        
        is_mean = sum(is_scores) / len(is_scores) if is_scores else 0
        is_std = (sum((s - is_mean)**2 for s in is_scores) / len(is_scores))**0.5 if is_scores else 0
        
        # Performance degradation
        performance_degradation = is_mean - oos_mean if is_scores else 0
        
        # Consistency metrics
        positive_oos_periods = sum(1 for s in oos_scores if s > 0)
        oos_consistency = positive_oos_periods / len(oos_scores)
        
        # Risk metrics
        max_drawdown = 0
        current_peak = 0
        cumulative_return = 0
        
        for score in oos_scores:
            cumulative_return += score
            if cumulative_return > current_peak:
                current_peak = cumulative_return
            drawdown = current_peak - cumulative_return
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "oos_mean": oos_mean,
            "oos_std": oos_std,
            "is_mean": is_mean,
            "is_std": is_std,
            "performance_degradation": performance_degradation,
            "oos_consistency": oos_consistency,
            "max_drawdown": max_drawdown,
            "n_retrains": result.n_retrains,
            "analysis_time": result.analysis_time,
            "total_oos_score": result.total_score
        }
