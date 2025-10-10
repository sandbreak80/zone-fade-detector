"""
Parallel Cross-Symbol Processor for Zone Fade Strategy.

This module provides real-time intermarket analysis by processing multiple symbols
in parallel, enabling detection of divergences, correlations, and market-wide patterns
that are critical for Zone Fade setups.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from collections import defaultdict, deque
import statistics

from zone_fade_detector.core.models import OHLCVBar, MarketContext
from zone_fade_detector.core.rolling_window_manager import RollingWindowManager, WindowType
from zone_fade_detector.core.session_state_manager import SessionStateManager
from zone_fade_detector.core.micro_window_analyzer import MicroWindowAnalyzer


class IntermarketSignal(Enum):
    """Types of intermarket signals."""
    BULLISH_DIVERGENCE = "bullish_divergence"      # SPY up, QQQ down
    BEARISH_DIVERGENCE = "bearish_divergence"      # SPY down, QQQ up
    CORRELATION_BREAK = "correlation_break"        # Normal correlation broken
    SECTOR_ROTATION = "sector_rotation"            # Money moving between sectors
    RISK_OFF = "risk_off"                          # Flight to safety
    RISK_ON = "risk_on"                            # Risk appetite increasing
    VOLATILITY_SPIKE = "volatility_spike"          # VIX spike detected
    VOLATILITY_SUPPRESSION = "volatility_suppression"  # VIX suppression
    MOMENTUM_SHIFT = "momentum_shift"              # Momentum changing direction
    CONSOLIDATION = "consolidation"                # Range-bound movement


class SymbolType(Enum):
    """Types of symbols for intermarket analysis."""
    BROAD_MARKET = "broad_market"      # SPY, QQQ, IWM
    SECTOR = "sector"                  # Sector ETFs
    VOLATILITY = "volatility"          # VIX, VXX
    BOND = "bond"                      # TLT, TBT
    COMMODITY = "commodity"            # GLD, SLV, OIL
    CURRENCY = "currency"              # DXY, EUR/USD
    CRYPTO = "crypto"                  # BTC, ETH


@dataclass
class SymbolConfig:
    """Configuration for a symbol in intermarket analysis."""
    symbol: str
    symbol_type: SymbolType
    weight: float = 1.0  # Weight in correlation calculations
    enabled: bool = True
    min_bars: int = 10  # Minimum bars required for analysis
    lookback_minutes: int = 60  # Lookback period for analysis


@dataclass
class IntermarketMetrics:
    """Metrics for intermarket analysis."""
    timestamp: datetime
    symbol: str
    price_change: float  # Price change percentage
    volume_ratio: float  # Current vs average volume
    momentum: float  # Price momentum
    volatility: float  # Volatility measure
    relative_strength: float  # Relative strength vs other symbols
    trend_direction: str  # "bullish", "bearish", "neutral"
    is_outlier: bool  # Is this an outlier move
    correlation_score: float  # Correlation with other symbols


@dataclass
class IntermarketAnalysis:
    """Complete intermarket analysis result."""
    timestamp: datetime
    primary_symbol: str
    signals: List[IntermarketSignal]
    correlations: Dict[str, float]  # Symbol -> correlation
    divergences: List[Tuple[str, str, str]]  # (symbol1, symbol2, type)
    sector_rotation: Dict[str, float]  # Sector -> strength
    risk_sentiment: str  # "risk_on", "risk_off", "neutral"
    volatility_regime: str  # "low", "normal", "high"
    momentum_alignment: float  # 0.0 to 1.0 alignment score
    confidence_score: float  # 0.0 to 1.0 confidence
    market_context: MarketContext


@dataclass
class CrossSymbolWindow:
    """Window for cross-symbol analysis."""
    symbol: str
    symbol_type: SymbolType
    bars: List[OHLCVBar] = field(default_factory=list)
    last_update: Optional[datetime] = None
    is_ready: bool = False
    metrics: Optional[IntermarketMetrics] = None


class ParallelCrossSymbolProcessor:
    """
    Processes multiple symbols in parallel for real-time intermarket analysis.
    
    Provides comprehensive intermarket analysis including divergences, correlations,
    sector rotation, and risk sentiment that are critical for Zone Fade setups.
    """
    
    def __init__(
        self,
        window_manager: RollingWindowManager,
        session_manager: SessionStateManager,
        micro_analyzer: MicroWindowAnalyzer,
        max_workers: int = 4,
        analysis_interval_seconds: int = 30,
        correlation_threshold: float = 0.7,
        divergence_threshold: float = 0.3
    ):
        """
        Initialize parallel cross-symbol processor.
        
        Args:
            window_manager: Rolling window manager instance
            session_manager: Session state manager instance
            micro_analyzer: Micro window analyzer instance
            max_workers: Maximum number of parallel workers
            analysis_interval_seconds: Interval between analyses
            correlation_threshold: Threshold for correlation significance
            divergence_threshold: Threshold for divergence detection
        """
        self.window_manager = window_manager
        self.session_manager = session_manager
        self.micro_analyzer = micro_analyzer
        self.max_workers = max_workers
        self.analysis_interval = analysis_interval_seconds
        self.correlation_threshold = correlation_threshold
        self.divergence_threshold = divergence_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Symbol configurations
        self.symbol_configs: Dict[str, SymbolConfig] = {}
        self.cross_symbol_windows: Dict[str, CrossSymbolWindow] = {}
        
        # Analysis results
        self.recent_analyses: deque = deque(maxlen=100)
        self.signal_history: Dict[IntermarketSignal, List[datetime]] = defaultdict(list)
        
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Analysis state
        self.is_running = False
        self.last_analysis_time: Optional[datetime] = None
        
        self.logger.info("ParallelCrossSymbolProcessor initialized")
    
    def add_symbol(self, symbol: str, symbol_type: SymbolType, weight: float = 1.0) -> None:
        """Add a symbol for intermarket analysis."""
        config = SymbolConfig(
            symbol=symbol,
            symbol_type=symbol_type,
            weight=weight,
            enabled=True
        )
        self.symbol_configs[symbol] = config
        
        # Initialize cross-symbol window
        self.cross_symbol_windows[symbol] = CrossSymbolWindow(
            symbol=symbol,
            symbol_type=symbol_type
        )
        
        self.logger.info(f"Added symbol {symbol} ({symbol_type.value}) for intermarket analysis")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from intermarket analysis."""
        if symbol in self.symbol_configs:
            del self.symbol_configs[symbol]
            del self.cross_symbol_windows[symbol]
            self.logger.info(f"Removed symbol {symbol} from intermarket analysis")
    
    def update_symbol_data(self, symbol: str, bar: OHLCVBar) -> bool:
        """Update symbol data and return if analysis should be triggered."""
        if symbol not in self.cross_symbol_windows:
            return False
        
        window = self.cross_symbol_windows[symbol]
        window.bars.append(bar)
        window.last_update = bar.timestamp
        
        # Keep only recent bars (last 2 hours)
        cutoff_time = bar.timestamp - timedelta(hours=2)
        window.bars = [b for b in window.bars if b.timestamp >= cutoff_time]
        
        # Check if window is ready for analysis
        config = self.symbol_configs.get(symbol)
        if config and len(window.bars) >= config.min_bars:
            window.is_ready = True
            return True
        
        return False
    
    async def analyze_intermarket(self, primary_symbol: str = "SPY") -> Optional[IntermarketAnalysis]:
        """Perform comprehensive intermarket analysis."""
        try:
            # Check if we have enough data
            ready_symbols = [s for s, w in self.cross_symbol_windows.items() if w.is_ready]
            if len(ready_symbols) < 2:
                self.logger.warning("Insufficient symbols ready for intermarket analysis")
                return None
            
            # Calculate metrics for all symbols in parallel
            metrics_tasks = []
            for symbol in ready_symbols:
                task = asyncio.create_task(self._calculate_symbol_metrics(symbol))
                metrics_tasks.append(task)
            
            # Wait for all metrics to complete
            metrics_results = await asyncio.gather(*metrics_tasks, return_exceptions=True)
            
            # Filter out exceptions and create metrics dict
            symbol_metrics = {}
            for i, result in enumerate(metrics_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error calculating metrics for {ready_symbols[i]}: {result}")
                else:
                    symbol_metrics[ready_symbols[i]] = result
            
            if len(symbol_metrics) < 2:
                self.logger.warning("Insufficient valid metrics for intermarket analysis")
                return None
            
            # Perform intermarket analysis
            analysis = await self._perform_intermarket_analysis(primary_symbol, symbol_metrics)
            
            if analysis:
                # Store analysis
                self.recent_analyses.append(analysis)
                self.last_analysis_time = analysis.timestamp
                
                # Update signal history
                for signal in analysis.signals:
                    self.signal_history[signal].append(analysis.timestamp)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in intermarket analysis: {e}")
            return None
    
    async def _calculate_symbol_metrics(self, symbol: str) -> IntermarketMetrics:
        """Calculate metrics for a single symbol."""
        window = self.cross_symbol_windows[symbol]
        bars = window.bars
        
        if not bars:
            raise ValueError(f"No bars available for {symbol}")
        
        # Calculate basic metrics
        price_change = self._calculate_price_change(bars)
        volume_ratio = self._calculate_volume_ratio(bars)
        momentum = self._calculate_momentum(bars)
        volatility = self._calculate_volatility(bars)
        
        # Calculate relative strength
        relative_strength = self._calculate_relative_strength(symbol, bars)
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(price_change, momentum)
        
        # Check for outlier moves
        is_outlier = self._is_outlier_move(price_change, volatility)
        
        # Calculate correlation score (simplified)
        correlation_score = self._calculate_correlation_score(symbol, bars)
        
        return IntermarketMetrics(
            timestamp=bars[-1].timestamp,
            symbol=symbol,
            price_change=price_change,
            volume_ratio=volume_ratio,
            momentum=momentum,
            volatility=volatility,
            relative_strength=relative_strength,
            trend_direction=trend_direction,
            is_outlier=is_outlier,
            correlation_score=correlation_score
        )
    
    def _calculate_price_change(self, bars: List[OHLCVBar]) -> float:
        """Calculate price change percentage."""
        if len(bars) < 2:
            return 0.0
        
        first_price = bars[0].close
        last_price = bars[-1].close
        
        return (last_price - first_price) / first_price if first_price > 0 else 0.0
    
    def _calculate_volume_ratio(self, bars: List[OHLCVBar]) -> float:
        """Calculate current volume vs average volume ratio."""
        if not bars:
            return 1.0
        
        current_volume = bars[-1].volume
        avg_volume = sum(b.volume for b in bars) / len(bars)
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_momentum(self, bars: List[OHLCVBar]) -> float:
        """Calculate price momentum."""
        if len(bars) < 5:
            return 0.0
        
        # Use 5-bar momentum
        recent_bars = bars[-5:]
        first_price = recent_bars[0].close
        last_price = recent_bars[-1].close
        
        return (last_price - first_price) / first_price if first_price > 0 else 0.0
    
    def _calculate_volatility(self, bars: List[OHLCVBar]) -> float:
        """Calculate volatility measure."""
        if len(bars) < 3:
            return 0.0
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(bars)):
            change = (bars[i].close - bars[i-1].close) / bars[i-1].close
            price_changes.append(abs(change))
        
        return statistics.stdev(price_changes) if len(price_changes) > 1 else 0.0
    
    def _calculate_relative_strength(self, symbol: str, bars: List[OHLCVBar]) -> float:
        """Calculate relative strength vs other symbols."""
        if not bars:
            return 0.0
        
        # Simplified relative strength calculation
        # In practice, this would compare against a benchmark
        price_change = self._calculate_price_change(bars)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, (price_change + 0.1) / 0.2))
    
    def _determine_trend_direction(self, price_change: float, momentum: float) -> str:
        """Determine trend direction based on price change and momentum."""
        if price_change > 0.01 and momentum > 0.005:
            return "bullish"
        elif price_change < -0.01 and momentum < -0.005:
            return "bearish"
        else:
            return "neutral"
    
    def _is_outlier_move(self, price_change: float, volatility: float) -> bool:
        """Check if this is an outlier move."""
        # Outlier if price change is more than 2 standard deviations from normal
        return abs(price_change) > volatility * 2
    
    def _calculate_correlation_score(self, symbol: str, bars: List[OHLCVBar]) -> float:
        """Calculate correlation score with other symbols."""
        # Simplified correlation calculation
        # In practice, this would calculate actual correlation
        return 0.5  # Placeholder
    
    async def _perform_intermarket_analysis(
        self, 
        primary_symbol: str, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> IntermarketAnalysis:
        """Perform comprehensive intermarket analysis."""
        # Detect signals
        signals = await self._detect_intermarket_signals(symbol_metrics)
        
        # Calculate correlations
        correlations = self._calculate_correlations(symbol_metrics)
        
        # Detect divergences
        divergences = self._detect_divergences(symbol_metrics)
        
        # Analyze sector rotation
        sector_rotation = self._analyze_sector_rotation(symbol_metrics)
        
        # Determine risk sentiment
        risk_sentiment = self._determine_risk_sentiment(symbol_metrics)
        
        # Determine volatility regime
        volatility_regime = self._determine_volatility_regime(symbol_metrics)
        
        # Calculate momentum alignment
        momentum_alignment = self._calculate_momentum_alignment(symbol_metrics)
        
        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(symbol_metrics, signals)
        
        # Create market context
        market_context = self._create_market_context(
            symbol_metrics, risk_sentiment, volatility_regime
        )
        
        return IntermarketAnalysis(
            timestamp=datetime.now(),
            primary_symbol=primary_symbol,
            signals=signals,
            correlations=correlations,
            divergences=divergences,
            sector_rotation=sector_rotation,
            risk_sentiment=risk_sentiment,
            volatility_regime=volatility_regime,
            momentum_alignment=momentum_alignment,
            confidence_score=confidence_score,
            market_context=market_context
        )
    
    async def _detect_intermarket_signals(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> List[IntermarketSignal]:
        """Detect intermarket signals."""
        signals = []
        
        # Group symbols by type
        broad_market = {s: m for s, m in symbol_metrics.items() 
                       if self.cross_symbol_windows[s].symbol_type == SymbolType.BROAD_MARKET}
        volatility = {s: m for s, m in symbol_metrics.items() 
                     if self.cross_symbol_windows[s].symbol_type == SymbolType.VOLATILITY}
        bonds = {s: m for s, m in symbol_metrics.items() 
                if self.cross_symbol_windows[s].symbol_type == SymbolType.BOND}
        
        # Detect divergences
        if len(broad_market) >= 2:
            divergence_signals = self._detect_broad_market_divergences(broad_market)
            signals.extend(divergence_signals)
        
        # Detect volatility signals
        if volatility:
            volatility_signals = self._detect_volatility_signals(volatility)
            signals.extend(volatility_signals)
        
        # Detect risk sentiment
        if bonds:
            risk_signals = self._detect_risk_sentiment_signals(bonds, broad_market)
            signals.extend(risk_signals)
        
        # Detect momentum shifts
        momentum_signals = self._detect_momentum_shift_signals(symbol_metrics)
        signals.extend(momentum_signals)
        
        return signals
    
    def _detect_broad_market_divergences(
        self, 
        broad_market: Dict[str, IntermarketMetrics]
    ) -> List[IntermarketSignal]:
        """Detect divergences in broad market symbols."""
        signals = []
        
        if len(broad_market) < 2:
            return signals
        
        # Get SPY and QQQ metrics
        spy_metrics = broad_market.get("SPY")
        qqq_metrics = broad_market.get("QQQ")
        
        if spy_metrics and qqq_metrics:
            # Check for divergence
            spy_change = spy_metrics.price_change
            qqq_change = qqq_metrics.price_change
            
            if abs(spy_change - qqq_change) > self.divergence_threshold:
                if spy_change > qqq_change:
                    signals.append(IntermarketSignal.BULLISH_DIVERGENCE)
                else:
                    signals.append(IntermarketSignal.BEARISH_DIVERGENCE)
        
        return signals
    
    def _detect_volatility_signals(
        self, 
        volatility: Dict[str, IntermarketMetrics]
    ) -> List[IntermarketSignal]:
        """Detect volatility-related signals."""
        signals = []
        
        for symbol, metrics in volatility.items():
            # Check for volatility spike
            if metrics.volatility > 0.05:  # 5% volatility threshold
                signals.append(IntermarketSignal.VOLATILITY_SPIKE)
            
            # Check for volatility suppression
            if metrics.volatility < 0.01:  # 1% volatility threshold
                signals.append(IntermarketSignal.VOLATILITY_SUPPRESSION)
        
        return signals
    
    def _detect_risk_sentiment_signals(
        self, 
        bonds: Dict[str, IntermarketMetrics],
        broad_market: Dict[str, IntermarketMetrics]
    ) -> List[IntermarketSignal]:
        """Detect risk sentiment signals."""
        signals = []
        
        if not bonds or not broad_market:
            return signals
        
        # Get TLT (bonds) and SPY metrics
        tlt_metrics = bonds.get("TLT")
        spy_metrics = broad_market.get("SPY")
        
        if tlt_metrics and spy_metrics:
            # Risk off: bonds up, stocks down
            if tlt_metrics.price_change > 0.01 and spy_metrics.price_change < -0.01:
                signals.append(IntermarketSignal.RISK_OFF)
            
            # Risk on: bonds down, stocks up
            elif tlt_metrics.price_change < -0.01 and spy_metrics.price_change > 0.01:
                signals.append(IntermarketSignal.RISK_ON)
        
        return signals
    
    def _detect_momentum_shift_signals(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> List[IntermarketSignal]:
        """Detect momentum shift signals."""
        signals = []
        
        # Check if momentum is shifting across symbols
        bullish_count = sum(1 for m in symbol_metrics.values() if m.trend_direction == "bullish")
        bearish_count = sum(1 for m in symbol_metrics.values() if m.trend_direction == "bearish")
        total_count = len(symbol_metrics)
        
        # If momentum is shifting from one direction to another
        if total_count > 0:
            bullish_ratio = bullish_count / total_count
            bearish_ratio = bearish_count / total_count
            
            if abs(bullish_ratio - bearish_ratio) < 0.3:  # Close to 50/50
                signals.append(IntermarketSignal.MOMENTUM_SHIFT)
        
        return signals
    
    def _calculate_correlations(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> Dict[str, float]:
        """Calculate correlations between symbols."""
        correlations = {}
        
        symbols = list(symbol_metrics.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Simplified correlation calculation
                # In practice, this would use actual price data
                correlation = 0.7  # Placeholder
                correlations[f"{symbol1}-{symbol2}"] = correlation
        
        return correlations
    
    def _detect_divergences(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> List[Tuple[str, str, str]]:
        """Detect divergences between symbols."""
        divergences = []
        
        symbols = list(symbol_metrics.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                metrics1 = symbol_metrics[symbol1]
                metrics2 = symbol_metrics[symbol2]
                
                # Check for divergence
                if abs(metrics1.price_change - metrics2.price_change) > self.divergence_threshold:
                    if metrics1.price_change > metrics2.price_change:
                        divergences.append((symbol1, symbol2, "bullish"))
                    else:
                        divergences.append((symbol1, symbol2, "bearish"))
        
        return divergences
    
    def _analyze_sector_rotation(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> Dict[str, float]:
        """Analyze sector rotation."""
        sector_rotation = {}
        
        # Group by sector type
        sectors = defaultdict(list)
        for symbol, metrics in symbol_metrics.items():
            symbol_type = self.cross_symbol_windows[symbol].symbol_type
            if symbol_type == SymbolType.SECTOR:
                sectors[symbol].append(metrics.relative_strength)
        
        # Calculate sector strength
        for sector, strengths in sectors.items():
            if strengths:
                sector_rotation[sector] = statistics.mean(strengths)
        
        return sector_rotation
    
    def _determine_risk_sentiment(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> str:
        """Determine overall risk sentiment."""
        # Analyze bonds vs stocks
        bond_metrics = [m for s, m in symbol_metrics.items() 
                       if self.cross_symbol_windows[s].symbol_type == SymbolType.BOND]
        stock_metrics = [m for s, m in symbol_metrics.items() 
                        if self.cross_symbol_windows[s].symbol_type == SymbolType.BROAD_MARKET]
        
        if bond_metrics and stock_metrics:
            avg_bond_change = statistics.mean(m.price_change for m in bond_metrics)
            avg_stock_change = statistics.mean(m.price_change for m in stock_metrics)
            
            if avg_bond_change > 0.01 and avg_stock_change < -0.01:
                return "risk_off"
            elif avg_bond_change < -0.01 and avg_stock_change > 0.01:
                return "risk_on"
        
        return "neutral"
    
    def _determine_volatility_regime(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> str:
        """Determine volatility regime."""
        volatilities = [m.volatility for m in symbol_metrics.values()]
        
        if not volatilities:
            return "normal"
        
        avg_volatility = statistics.mean(volatilities)
        
        if avg_volatility > 0.03:  # 3% threshold
            return "high"
        elif avg_volatility < 0.01:  # 1% threshold
            return "low"
        else:
            return "normal"
    
    def _calculate_momentum_alignment(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics]
    ) -> float:
        """Calculate momentum alignment across symbols."""
        if not symbol_metrics:
            return 0.0
        
        # Count trend directions
        bullish_count = sum(1 for m in symbol_metrics.values() if m.trend_direction == "bullish")
        bearish_count = sum(1 for m in symbol_metrics.values() if m.trend_direction == "bearish")
        total_count = len(symbol_metrics)
        
        if total_count == 0:
            return 0.0
        
        # Calculate alignment (closer to 1.0 means more aligned)
        max_count = max(bullish_count, bearish_count)
        return max_count / total_count
    
    def _calculate_analysis_confidence(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics],
        signals: List[IntermarketSignal]
    ) -> float:
        """Calculate confidence score for the analysis."""
        if not symbol_metrics:
            return 0.0
        
        # Base confidence on number of symbols and signals
        symbol_confidence = min(len(symbol_metrics) / 5.0, 1.0)  # Max at 5 symbols
        signal_confidence = min(len(signals) / 3.0, 1.0)  # Max at 3 signals
        
        # Average the confidences
        return (symbol_confidence + signal_confidence) / 2.0
    
    def _create_market_context(
        self, 
        symbol_metrics: Dict[str, IntermarketMetrics],
        risk_sentiment: str,
        volatility_regime: str
    ) -> MarketContext:
        """Create market context from intermarket analysis."""
        # Calculate overall trend
        trend_directions = [m.trend_direction for m in symbol_metrics.values()]
        bullish_count = trend_directions.count("bullish")
        bearish_count = trend_directions.count("bearish")
        
        is_trend_day = abs(bullish_count - bearish_count) > len(trend_directions) * 0.6
        
        # Calculate VWAP slope (simplified)
        vwap_slope = 0.0
        if symbol_metrics:
            avg_momentum = statistics.mean(m.momentum for m in symbol_metrics.values())
            vwap_slope = avg_momentum * 100  # Scale for VWAP slope
        
        # Determine market balance
        market_balance = 0.5
        if risk_sentiment == "risk_on":
            market_balance = 0.8
        elif risk_sentiment == "risk_off":
            market_balance = 0.2
        
        return MarketContext(
            is_trend_day=is_trend_day,
            vwap_slope=vwap_slope,
            value_area_overlap=False,  # Would need more data
            market_balance=market_balance,
            volatility_regime=volatility_regime,
            session_type="regular"
        )
    
    async def start_analysis_loop(self) -> None:
        """Start the continuous analysis loop."""
        self.is_running = True
        self.logger.info("Started intermarket analysis loop")
        
        while self.is_running:
            try:
                # Perform analysis
                analysis = await self.analyze_intermarket()
                
                if analysis:
                    self.logger.info(
                        f"Intermarket analysis completed: {len(analysis.signals)} signals, "
                        f"confidence: {analysis.confidence_score:.2f}"
                    )
                
                # Wait for next analysis
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(self.analysis_interval)
    
    def stop_analysis_loop(self) -> None:
        """Stop the continuous analysis loop."""
        self.is_running = False
        self.logger.info("Stopped intermarket analysis loop")
    
    def get_recent_analyses(self, limit: int = 10) -> List[IntermarketAnalysis]:
        """Get recent intermarket analyses."""
        return list(self.recent_analyses)[-limit:]
    
    def get_signal_frequency(self, signal: IntermarketSignal, hours: int = 24) -> int:
        """Get frequency of a specific signal in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return sum(1 for t in self.signal_history[signal] if t >= cutoff_time)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of intermarket analyses."""
        if not self.recent_analyses:
            return {"status": "no_analyses"}
        
        recent = list(self.recent_analyses)
        
        # Count signals
        signal_counts = defaultdict(int)
        for analysis in recent:
            for signal in analysis.signals:
                signal_counts[signal] += 1
        
        # Calculate averages
        avg_confidence = statistics.mean(a.confidence_score for a in recent)
        avg_momentum_alignment = statistics.mean(a.momentum_alignment for a in recent)
        
        # Risk sentiment distribution
        risk_sentiments = [a.risk_sentiment for a in recent]
        risk_sentiment_counts = {sentiment: risk_sentiments.count(sentiment) 
                               for sentiment in set(risk_sentiments)}
        
        return {
            "total_analyses": len(recent),
            "avg_confidence": avg_confidence,
            "avg_momentum_alignment": avg_momentum_alignment,
            "signal_counts": dict(signal_counts),
            "risk_sentiment_distribution": risk_sentiment_counts,
            "last_analysis_time": recent[-1].timestamp.isoformat() if recent else None
        }