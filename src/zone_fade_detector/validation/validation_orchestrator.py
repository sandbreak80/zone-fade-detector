"""
Validation Orchestrator for 4-Step Trading Strategy Validation Battery.

This module orchestrates the complete 4-step validation process:
1. In-Sample (IS) Excellence
2. In-Sample Monte-Carlo Permutation Test (IMCPT)
3. Walk-Forward Test (WFT)
4. Walk-Forward Permutation Test (WFPT)
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .optimization_engine import OptimizationEngine, OptimizationResult
from .permutation_tester import PermutationTester, PermutationResult
from .walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Complete results from 4-step validation battery."""
    # IS Excellence results
    is_optimization: OptimizationResult
    
    # IMCPT results
    imcpt_result: PermutationResult
    
    # WFT results
    wft_result: WalkForwardResult
    
    # WFPT results
    wfpt_result: PermutationResult
    
    # Overall validation
    validation_passed: bool
    total_time: float
    strategy_name: str
    instrument: str
    date_range: Tuple[str, str]


class ValidationOrchestrator:
    """
    Orchestrates the complete 4-step validation battery for trading strategies.
    
    Manages the entire validation process from optimization through permutation testing.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the validation orchestrator.
        
        Args:
            random_seed: Optional random seed for reproducible results
        """
        self.random_seed = random_seed
        self.optimization_engine = OptimizationEngine(random_seed)
        self.permutation_tester = PermutationTester(random_seed)
        self.walk_forward_analyzer = WalkForwardAnalyzer(random_seed)
    
    def validate_strategy(self, 
                         bars: List[Any],
                         strategy_class: type,
                         param_space: Dict[str, List[Any]],
                         instrument: str,
                         date_range: Tuple[str, str],
                         train_window_size: int = 1000,
                         retrain_frequency: int = 30,
                         imcpt_permutations: int = 1000,
                         wfpt_permutations: int = 200,
                         optimization_method: str = "grid",
                         max_optimization_evaluations: Optional[int] = None) -> ValidationResult:
        """
        Perform complete 4-step validation battery.
        
        Args:
            bars: List of bar objects for full dataset
            strategy_class: Strategy class to validate
            param_space: Parameter space for optimization
            instrument: Instrument symbol (e.g., "QQQ")
            date_range: Tuple of (start_date, end_date) strings
            train_window_size: Size of training window for walk-forward
            retrain_frequency: Frequency of retraining (in bars)
            imcpt_permutations: Number of permutations for IMCPT
            wfpt_permutations: Number of permutations for WFPT
            optimization_method: Optimization method ("grid" or "random")
            max_optimization_evaluations: Max evaluations for optimization
        
        Returns:
            ValidationResult with complete validation results
        """
        logger.info(f"Starting 4-step validation battery for {strategy_class.__name__}")
        logger.info(f"Instrument: {instrument}, Date range: {date_range}")
        start_time = time.time()
        
        strategy_name = strategy_class().get_name()
        
        # Step 1: In-Sample Excellence
        logger.info("=" * 60)
        logger.info("STEP 1: In-Sample Excellence")
        logger.info("=" * 60)
        
        is_optimization = self._perform_is_optimization(
            bars, strategy_class, param_space, optimization_method, max_optimization_evaluations
        )
        
        # Step 2: In-Sample Monte-Carlo Permutation Test (IMCPT)
        logger.info("=" * 60)
        logger.info("STEP 2: In-Sample Monte-Carlo Permutation Test (IMCPT)")
        logger.info("=" * 60)
        
        imcpt_result = self._perform_imcpt(
            bars, strategy_class, param_space, imcpt_permutations
        )
        
        # Step 3: Walk-Forward Test (WFT)
        logger.info("=" * 60)
        logger.info("STEP 3: Walk-Forward Test (WFT)")
        logger.info("=" * 60)
        
        wft_result = self._perform_wft(
            bars, strategy_class, param_space, train_window_size, retrain_frequency
        )
        
        # Step 4: Walk-Forward Permutation Test (WFPT)
        logger.info("=" * 60)
        logger.info("STEP 4: Walk-Forward Permutation Test (WFPT)")
        logger.info("=" * 60)
        
        wfpt_result = self._perform_wfpt(
            bars, strategy_class, param_space, train_window_size, retrain_frequency, wfpt_permutations
        )
        
        # Determine overall validation result
        validation_passed = self._evaluate_validation_results(
            is_optimization, imcpt_result, wft_result, wfpt_result
        )
        
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Validation passed: {validation_passed}")
        
        return ValidationResult(
            is_optimization=is_optimization,
            imcpt_result=imcpt_result,
            wft_result=wft_result,
            wfpt_result=wfpt_result,
            validation_passed=validation_passed,
            total_time=total_time,
            strategy_name=strategy_name,
            instrument=instrument,
            date_range=date_range
        )
    
    def _perform_is_optimization(self, 
                               bars: List[Any],
                               strategy_class: type,
                               param_space: Dict[str, List[Any]],
                               method: str,
                               max_evaluations: Optional[int]) -> OptimizationResult:
        """Perform in-sample optimization."""
        logger.info("Optimizing strategy parameters on full dataset...")
        
        def objective_function(params: Dict[str, Any]) -> float:
            return self._evaluate_strategy_performance(bars, strategy_class, params)
        
        return self.optimization_engine.optimize(
            param_space=param_space,
            objective_function=objective_function,
            method=method,
            max_evaluations=max_evaluations
        )
    
    def _perform_imcpt(self, 
                      bars: List[Any],
                      strategy_class: type,
                      param_space: Dict[str, List[Any]],
                      n_permutations: int) -> PermutationResult:
        """Perform In-Sample Monte-Carlo Permutation Test."""
        logger.info(f"Performing IMCPT with {n_permutations} permutations...")
        
        def optimization_function(data: List[Any], params: Dict[str, Any]) -> float:
            return self._evaluate_strategy_performance(data, strategy_class, params)
        
        return self.permutation_tester.in_sample_permutation_test(
            bars=bars,
            strategy_class=strategy_class,
            param_space=param_space,
            optimization_function=optimization_function,
            n_permutations=n_permutations,
            significance_threshold=0.01
        )
    
    def _perform_wft(self, 
                    bars: List[Any],
                    strategy_class: type,
                    param_space: Dict[str, List[Any]],
                    train_window_size: int,
                    retrain_frequency: int) -> WalkForwardResult:
        """Perform Walk-Forward Test."""
        logger.info("Performing Walk-Forward Test...")
        
        def optimization_function(data: List[Any], params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
            # Optimize and return best parameters and score
            result = self.optimization_engine.optimize(
                param_space=params,
                objective_function=lambda p: self._evaluate_strategy_performance(data, strategy_class, p),
                method="grid"
            )
            return result.best_params, result.best_score
        
        return self.walk_forward_analyzer.analyze(
            bars=bars,
            strategy_class=strategy_class,
            param_space=param_space,
            optimization_function=optimization_function,
            train_window_size=train_window_size,
            retrain_frequency=retrain_frequency
        )
    
    def _perform_wfpt(self, 
                     bars: List[Any],
                     strategy_class: type,
                     param_space: Dict[str, List[Any]],
                     train_window_size: int,
                     retrain_frequency: int,
                     n_permutations: int) -> PermutationResult:
        """Perform Walk-Forward Permutation Test."""
        logger.info(f"Performing WFPT with {n_permutations} permutations...")
        
        def walk_forward_function(data: List[Any], params: Dict[str, Any]) -> float:
            # Perform walk-forward analysis and return total score
            result = self.walk_forward_analyzer.analyze(
                bars=data,
                strategy_class=strategy_class,
                param_space=params,
                optimization_function=lambda d, p: self._optimize_and_return_params(d, strategy_class, p),
                train_window_size=train_window_size,
                retrain_frequency=retrain_frequency
            )
            return result.total_score
        
        return self.permutation_tester.walk_forward_permutation_test(
            bars=bars,
            strategy_class=strategy_class,
            param_space=param_space,
            walk_forward_function=walk_forward_function,
            train_window_size=train_window_size,
            retrain_frequency=retrain_frequency,
            n_permutations=n_permutations,
            significance_threshold=0.05
        )
    
    def _evaluate_strategy_performance(self, 
                                    bars: List[Any], 
                                    strategy_class: type, 
                                    params: Dict[str, Any]) -> float:
        """Evaluate strategy performance on given data."""
        if len(bars) < 2:
            return 0.0
        
        try:
            # Create strategy instance and generate signals
            strategy = strategy_class()
            signals = strategy.generate_signal(bars, params)
            
            if not signals or len(signals) != len(bars):
                return 0.0
            
            # Calculate returns
            from zone_fade_detector.utils.returns_engine import ReturnsEngine
            returns_engine = ReturnsEngine()
            strategy_returns = returns_engine.calculate_strategy_returns(signals, bars)
            
            # Calculate performance metrics
            metrics = returns_engine.calculate_metrics(strategy_returns)
            
            # Return total return as the score
            return metrics.get('total_return', 0.0)
            
        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")
            return 0.0
    
    def _optimize_and_return_params(self, 
                                  data: List[Any], 
                                  strategy_class: type, 
                                  param_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Optimize strategy and return parameters and score."""
        result = self.optimization_engine.optimize(
            param_space=param_space,
            objective_function=lambda p: self._evaluate_strategy_performance(data, strategy_class, p),
            method="grid"
        )
        return result.best_params, result.best_score
    
    def _evaluate_validation_results(self, 
                                   is_optimization: OptimizationResult,
                                   imcpt_result: PermutationResult,
                                   wft_result: WalkForwardResult,
                                   wfpt_result: PermutationResult) -> bool:
        """Evaluate if validation results pass acceptance criteria."""
        
        # Check IMCPT significance (p < 1%)
        imcpt_passed = imcpt_result.p_value < 0.01
        
        # Check WFPT significance (p ≤ 5% for 1 OOS year, p ≤ 1% for 2+ OOS years)
        # For now, use 5% threshold
        wfpt_passed = wfpt_result.p_value < 0.05
        
        # Check that optimization found reasonable parameters
        optimization_passed = is_optimization.best_score > float('-inf')
        
        # Check that walk-forward analysis completed successfully
        wft_passed = wft_result.total_score > float('-inf') and wft_result.n_retrains > 0
        
        validation_passed = imcpt_passed and wfpt_passed and optimization_passed and wft_passed
        
        logger.info(f"Validation criteria:")
        logger.info(f"  IMCPT passed (p < 1%): {imcpt_passed} (p = {imcpt_result.p_value:.4f})")
        logger.info(f"  WFPT passed (p < 5%): {wfpt_passed} (p = {wfpt_result.p_value:.4f})")
        logger.info(f"  Optimization passed: {optimization_passed}")
        logger.info(f"  Walk-forward passed: {wft_passed}")
        logger.info(f"  Overall validation: {validation_passed}")
        
        return validation_passed
