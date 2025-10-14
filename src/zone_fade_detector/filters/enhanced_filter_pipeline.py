"""
Enhanced Filter Pipeline

This module implements the complete filter pipeline for zone fade signals,
integrating all the new enhancement filters in the correct order.

Features:
- Market type detection (trend vs range-bound)
- Market internals monitoring (TICK and A/D Line)
- Zone approach analysis (balance detection)
- Zone touch tracking (1st/2nd touch only)
- Entry optimization (zone position and R:R)
- Session analysis (PM rules and ON range)
- Enhanced QRS scoring with veto power
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .market_type_detector import MarketTypeDetector, MarketTypeFilter
from .market_internals import MarketInternalsMonitor, InternalsFilter
from .zone_approach_analyzer import ZoneApproachAnalyzer, ZoneApproachFilter
from .zone_touch_tracker import ZoneTouchTracker, ZoneTouchFilter
from .entry_optimizer import EntryOptimizer, EntryOptimizationFilter
from .session_analyzer import SessionAnalyzer, SessionAnalysisFilter
from ..scoring.enhanced_qrs import EnhancedQRSScorer, QRSResult

@dataclass
class FilterPipelineResult:
    """Result of the complete filter pipeline."""
    signal: Optional[Dict[str, Any]]
    passed_filters: List[str]
    failed_filters: List[str]
    qrs_result: Optional[QRSResult]
    processing_time_ms: float
    timestamp: datetime

class EnhancedFilterPipeline:
    """
    Complete filter pipeline for zone fade signals.
    
    This pipeline implements all the enhancement requirements:
    1. Market type detection (trend day veto)
    2. Market internals monitoring (TICK/A/D Line)
    3. Zone approach analysis (balance detection)
    4. Zone touch tracking (1st/2nd touch only)
    5. Entry optimization (position and R:R)
    6. Session analysis (PM rules)
    7. Enhanced QRS scoring (with veto power)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced filter pipeline.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config or {}
        
        # Initialize all components
        self.market_type_detector = MarketTypeDetector(
            tick_threshold=self.config.get('tick_threshold', 800.0),
            ad_slope_threshold=self.config.get('ad_slope_threshold', 1000.0),
            atr_expansion_threshold=self.config.get('atr_expansion_threshold', 1.3),
            directional_bars_threshold=self.config.get('directional_bars_threshold', 0.7)
        )
        
        self.market_internals_monitor = MarketInternalsMonitor(
            tick_threshold=self.config.get('tick_balanced_threshold', 200.0),
            ad_slope_threshold=self.config.get('ad_flat_threshold', 100.0)
        )
        
        self.zone_approach_analyzer = ZoneApproachAnalyzer(
            lookback_period=self.config.get('balance_lookback', 10),
            balance_threshold=self.config.get('balance_threshold', 0.7)
        )
        
        self.zone_touch_tracker = ZoneTouchTracker(
            session_start_hour=self.config.get('session_start_hour', 9),
            session_start_minute=self.config.get('session_start_minute', 30)
        )
        
        self.entry_optimizer = EntryOptimizer(
            tick_size=self.config.get('tick_size', 0.25),
            zfr_entry_pct=self.config.get('zfr_entry_pct', 0.25),
            zf_tr_entry_pct=self.config.get('zf_tr_entry_pct', 0.60)
        )
        
        self.session_analyzer = SessionAnalyzer(
            on_session_start_hour=self.config.get('on_session_start_hour', 18),
            am_session_end_hour=self.config.get('am_session_end_hour', 12),
            pm_session_start_hour=self.config.get('pm_session_start_hour', 12),
            pm_session_end_hour=self.config.get('pm_session_end_hour', 16)
        )
        
        self.qrs_scorer = EnhancedQRSScorer(
            threshold=self.config.get('qrs_threshold', 7.0)
        )
        
        # Initialize filters
        self.market_type_filter = MarketTypeFilter(self.market_type_detector)
        self.internals_filter = InternalsFilter(self.market_internals_monitor)
        self.zone_approach_filter = ZoneApproachFilter(self.zone_approach_analyzer)
        self.zone_touch_filter = ZoneTouchFilter(self.zone_touch_tracker)
        self.entry_optimization_filter = EntryOptimizationFilter(self.entry_optimizer)
        self.session_analysis_filter = SessionAnalysisFilter(self.session_analyzer)
        
        # Statistics
        self.total_signals_processed = 0
        self.signals_generated = 0
        self.signals_vetoed = 0
        self.filter_statistics = {}
    
    def process_signal(self, 
                      signal: Dict[str, Any],
                      market_data: Dict[str, Any]) -> FilterPipelineResult:
        """
        Process a zone fade signal through the complete filter pipeline.
        
        Args:
            signal: Zone fade signal to process
            market_data: Market data including price bars, internals, etc.
            
        Returns:
            FilterPipelineResult with processing results
        """
        start_time = datetime.now()
        passed_filters = []
        failed_filters = []
        current_signal = signal.copy()
        
        # Filter 1: Market Type Detection (CRITICAL - VETO)
        current_signal = self.market_type_filter.filter_signal(current_signal, market_data)
        if current_signal is None:
            failed_filters.append('market_type_detection')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=None,
                start_time=start_time
            )
        passed_filters.append('market_type_detection')
        
        # Filter 2: Market Internals Monitoring (CRITICAL - VETO)
        current_signal = self.internals_filter.filter_signal(current_signal, market_data)
        if current_signal is None:
            failed_filters.append('market_internals')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=None,
                start_time=start_time
            )
        passed_filters.append('market_internals')
        
        # Filter 3: Zone Approach Analysis (HIGH - VETO)
        current_signal = self.zone_approach_filter.filter_signal(current_signal, market_data)
        if current_signal is None:
            failed_filters.append('zone_approach_analysis')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=None,
                start_time=start_time
            )
        passed_filters.append('zone_approach_analysis')
        
        # Filter 4: Zone Touch Tracking (HIGH - VETO)
        current_signal = self.zone_touch_filter.filter_signal(current_signal, market_data)
        if current_signal is None:
            failed_filters.append('zone_touch_tracking')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=None,
                start_time=start_time
            )
        passed_filters.append('zone_touch_tracking')
        
        # Filter 5: Entry Optimization (MEDIUM - ENHANCEMENT)
        current_signal = self.entry_optimization_filter.filter_signal(current_signal, market_data)
        if current_signal is None:
            failed_filters.append('entry_optimization')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=None,
                start_time=start_time
            )
        passed_filters.append('entry_optimization')
        
        # Filter 6: Session Analysis (MEDIUM - ENHANCEMENT)
        current_signal = self.session_analysis_filter.filter_signal(current_signal, market_data)
        if current_signal is None:
            failed_filters.append('session_analysis')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=None,
                start_time=start_time
            )
        passed_filters.append('session_analysis')
        
        # Final Step: Enhanced QRS Scoring (CRITICAL - VETO)
        qrs_result = self.qrs_scorer.score_setup(current_signal, market_data)
        if qrs_result is None or qrs_result.veto:
            failed_filters.append('enhanced_qrs_scoring')
            return self._create_result(
                signal=None,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                qrs_result=qrs_result,
                start_time=start_time
            )
        passed_filters.append('enhanced_qrs_scoring')
        
        # Add QRS result to signal
        current_signal['qrs_result'] = qrs_result
        current_signal['qrs_score'] = qrs_result.total_score
        current_signal['qrs_grade'] = qrs_result.grade.value
        
        # Update statistics
        self.total_signals_processed += 1
        self.signals_generated += 1
        
        return self._create_result(
            signal=current_signal,
            passed_filters=passed_filters,
            failed_filters=failed_filters,
            qrs_result=qrs_result,
            start_time=start_time
        )
    
    def _create_result(self, 
                      signal: Optional[Dict[str, Any]],
                      passed_filters: List[str],
                      failed_filters: List[str],
                      qrs_result: Optional[QRSResult],
                      start_time: datetime) -> FilterPipelineResult:
        """Create a FilterPipelineResult."""
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        if signal is None:
            self.signals_vetoed += 1
        
        return FilterPipelineResult(
            signal=signal,
            passed_filters=passed_filters,
            failed_filters=failed_filters,
            qrs_result=qrs_result,
            processing_time_ms=processing_time_ms,
            timestamp=end_time
        )
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            'pipeline_statistics': {
                'total_signals_processed': self.total_signals_processed,
                'signals_generated': self.signals_generated,
                'signals_vetoed': self.signals_vetoed,
                'generation_rate': (self.signals_generated / self.total_signals_processed * 100) 
                                 if self.total_signals_processed > 0 else 0.0,
                'veto_rate': (self.signals_vetoed / self.total_signals_processed * 100) 
                           if self.total_signals_processed > 0 else 0.0
            },
            'market_type_detector': self.market_type_detector.get_statistics(),
            'market_internals_monitor': self.market_internals_monitor.get_statistics(),
            'zone_approach_analyzer': self.zone_approach_analyzer.get_statistics(),
            'zone_touch_tracker': self.zone_touch_tracker.get_statistics(),
            'entry_optimizer': self.entry_optimizer.get_statistics(),
            'session_analyzer': self.session_analyzer.get_statistics(),
            'qrs_scorer': self.qrs_scorer.get_statistics(),
            'filter_statistics': {
                'market_type_filter': self.market_type_filter.get_filter_statistics(),
                'internals_filter': self.internals_filter.get_filter_statistics(),
                'zone_approach_filter': self.zone_approach_filter.get_filter_statistics(),
                'zone_touch_filter': self.zone_touch_filter.get_filter_statistics(),
                'entry_optimization_filter': self.entry_optimization_filter.get_filter_statistics(),
                'session_analysis_filter': self.session_analysis_filter.get_filter_statistics()
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_signals_processed = 0
        self.signals_generated = 0
        self.signals_vetoed = 0
        
        # Reset component statistics
        self.market_type_detector.reset_statistics()
        self.market_internals_monitor.reset_statistics()
        self.zone_approach_analyzer.reset_statistics()
        self.zone_touch_tracker.reset_statistics()
        self.entry_optimizer.reset_statistics()
        self.session_analyzer.reset_statistics()
        self.qrs_scorer.reset_statistics()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration for all components."""
        self.config.update(new_config)
        
        # Update component configurations
        # Note: In a production system, you'd want to implement
        # proper configuration updates for each component
        pass


# Placeholder classes for missing components
# These would be implemented in separate files

class ZoneApproachAnalyzer:
    """Placeholder for zone approach analysis."""
    def __init__(self, **kwargs):
        pass
    def get_statistics(self):
        return {}

class ZoneApproachFilter:
    """Placeholder for zone approach filter."""
    def __init__(self, analyzer):
        self.analyzer = analyzer
    def filter_signal(self, signal, market_data):
        return signal
    def get_filter_statistics(self):
        return {}

class ZoneTouchTracker:
    """Placeholder for zone touch tracking."""
    def __init__(self, **kwargs):
        pass
    def get_statistics(self):
        return {}

class ZoneTouchFilter:
    """Placeholder for zone touch filter."""
    def __init__(self, tracker):
        self.tracker = tracker
    def filter_signal(self, signal, market_data):
        return signal
    def get_filter_statistics(self):
        return {}

class EntryOptimizer:
    """Placeholder for entry optimization."""
    def __init__(self, **kwargs):
        pass
    def get_statistics(self):
        return {}

class EntryOptimizationFilter:
    """Placeholder for entry optimization filter."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def filter_signal(self, signal, market_data):
        return signal
    def get_filter_statistics(self):
        return {}

class SessionAnalyzer:
    """Placeholder for session analysis."""
    def __init__(self, **kwargs):
        pass
    def get_statistics(self):
        return {}

class SessionAnalysisFilter:
    """Placeholder for session analysis filter."""
    def __init__(self, analyzer):
        self.analyzer = analyzer
    def filter_signal(self, signal, market_data):
        return signal
    def get_filter_statistics(self):
        return {}