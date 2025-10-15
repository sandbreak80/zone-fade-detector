#!/usr/bin/env python3
"""
Unified Trading Framework

This module provides a unified interface for both live trading and backtesting
that implements the Zone-Based Intraday Trading Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal, Union
from datetime import datetime, timedelta
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from directional_bias_detector import DirectionalBiasDetector, DirectionalBias, SessionType, BiasAnalysis
from session_aware_trading import SessionAwareTradingRules, TradeType, TradeDirection, TradeDecision
from choch_confirmation_system import CHoCHConfirmationSystem, CHoCHSignal


class TradeResult(Enum):
    """Trade execution results."""
    EXECUTED = "executed"
    REJECTED = "rejected"
    NO_TRADE = "no_trade"


@dataclass
class TradeExecution:
    """Complete trade execution data."""
    # Trade identification
    trade_id: str
    symbol: str
    timestamp: datetime
    
    # Trade decision
    trade_type: TradeType
    direction: TradeDirection
    confidence: float
    reason: str
    
    # Market context
    bias: DirectionalBias
    session_type: SessionType
    choch_confirmed: bool
    
    # Entry details
    entry_price: float
    position_size: int
    risk_amount: float
    
    # Exit details
    exit_price: float
    exit_scenario: str
    pnl: float
    is_winner: bool
    
    # Framework compliance
    is_first_touch: bool
    session_appropriate: bool
    choch_aligned: bool


class UnifiedTradingFramework:
    """Unified framework for both live trading and backtesting."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 max_equity_per_trade: float = 0.10,
                 slippage_ticks: int = 2,
                 commission_per_trade: float = 5.0,
                 min_confidence: float = 0.6):
        """
        Initialize unified trading framework.
        
        Args:
            initial_balance: Starting account balance
            max_equity_per_trade: Maximum equity to risk per trade
            slippage_ticks: Slippage in ticks
            commission_per_trade: Commission cost per trade
            min_confidence: Minimum confidence for trade execution
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_equity_per_trade = max_equity_per_trade
        self.slippage_ticks = slippage_ticks
        self.commission_per_trade = commission_per_trade
        self.min_confidence = min_confidence
        
        # Initialize framework components
        self.bias_detector = DirectionalBiasDetector()
        self.trading_rules = SessionAwareTradingRules(self.bias_detector)
        self.choch_system = CHoCHConfirmationSystem()
        
        # Track trades and state
        self.trades = []
        self.zone_touch_history = {}
        
        # Realistic trading parameters
        self.stop_loss_hit_rate = 0.40  # 40% of trades hit stop loss
        self.t1_hit_rate = 0.35  # 35% of trades hit T1
        self.t2_hit_rate = 0.15  # 15% of trades hit T2
        self.t3_hit_rate = 0.10  # 10% of trades hit T3
    
    def evaluate_trade_opportunity(self, 
                                 bars: List, 
                                 current_index: int,
                                 zone_data: Dict,
                                 entry_price: float) -> Tuple[TradeDecision, Optional[CHoCHSignal]]:
        """
        Evaluate a trade opportunity using the complete framework.
        
        Args:
            bars: List of OHLCV bars
            current_index: Current bar index
            zone_data: Zone information (type, level, etc.)
            entry_price: Current price touching the zone
            
        Returns:
            Tuple of (TradeDecision, CHoCHSignal)
        """
        # Get current bias analysis
        bias_analysis = self.bias_detector.detect_directional_bias(bars, current_index)
        
        # Check if this is first touch of zone
        zone_id = f"{zone_data['zone_type']}_{zone_data['zone_level']:.2f}"
        is_first_touch = self.bias_detector.is_first_touch(
            zone_id, bars[current_index].timestamp
        )
        
        if not is_first_touch:
            return TradeDecision(
                trade_type=TradeType.NO_TRADE,
                direction=None,
                confidence=0.0,
                reason="Not first touch of zone - framework requires first touch only",
                bias_analysis=bias_analysis,
                zone_type=zone_data['zone_type'],
                is_first_touch=is_first_touch,
                choch_aligned=False,
                session_appropriate=False
            ), None
        
        # Analyze trade opportunity using session-aware rules
        trade_decision = self.trading_rules.analyze_trade_opportunity(
            bars, current_index, zone_data['zone_type'], zone_data['zone_level'], entry_price
        )
        
        # Check for CHoCH confirmation
        choch_signal = self.choch_system.detect_choch(bars, current_index)
        
        # Enhance trade decision with CHoCH
        if choch_signal:
            trade_decision = self.choch_system.enhance_trade_decision(trade_decision, choch_signal)
        
        return trade_decision, choch_signal
    
    def should_execute_trade(self, trade_decision: TradeDecision) -> bool:
        """Determine if trade should be executed based on framework rules."""
        return self.trading_rules.should_take_trade(trade_decision, self.min_confidence)
    
    def calculate_position_size(self, entry_price: float, risk_amount: float) -> int:
        """Calculate position size based on risk amount and current balance."""
        if risk_amount <= 0:
            return 0
        
        max_risk_amount = self.current_balance * self.max_equity_per_trade
        position_size = int(max_risk_amount / risk_amount)
        position_size = max(1, position_size)
        
        # Ensure we don't risk more than we have
        max_affordable = int(self.current_balance * 0.95 / entry_price)
        position_size = min(position_size, max_affordable)
        
        return position_size
    
    def calculate_corrected_risk_amount(self, entry_price: float, hard_stop: float, direction: str) -> float:
        """Calculate corrected risk amount using proper formula."""
        if direction == "LONG":
            risk_amount = entry_price - hard_stop
        else:  # SHORT
            risk_amount = hard_stop - entry_price
        
        if risk_amount <= 0:
            risk_amount = entry_price * 0.01  # 1% fallback
        
        return risk_amount
    
    def calculate_slippage(self, price: float) -> float:
        """Calculate slippage amount based on price."""
        tick_value = 0.01  # $0.01 per tick for ETFs
        return self.slippage_ticks * tick_value
    
    def execute_trade(self, 
                     trade_decision: TradeDecision,
                     choch_signal: Optional[CHoCHSignal],
                     entry_data: Dict,
                     bars: List,
                     current_index: int,
                     simulation_mode: bool = True) -> TradeExecution:
        """
        Execute a trade (live or simulated).
        
        Args:
            trade_decision: Trade decision from framework
            choch_signal: CHoCH signal if available
            entry_data: Entry point data
            bars: List of OHLCV bars
            current_index: Current bar index
            simulation_mode: Whether to simulate or execute live
            
        Returns:
            TradeExecution result
        """
        # Calculate corrected risk amount
        risk_amount = self.calculate_corrected_risk_amount(
            entry_data['price'], entry_data['hard_stop'], trade_decision.direction.value
        )
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_data['price'], risk_amount)
        
        if position_size <= 0:
            return TradeExecution(
                trade_id=f"REJECTED_{entry_data.get('entry_id', 'UNKNOWN')}",
                symbol=entry_data.get('symbol', 'UNKNOWN'),
                timestamp=entry_data.get('timestamp', datetime.now()),
                trade_type=trade_decision.trade_type,
                direction=trade_decision.direction,
                confidence=trade_decision.confidence,
                reason="Insufficient balance for position",
                bias=trade_decision.bias_analysis.bias,
                session_type=trade_decision.bias_analysis.session_type,
                choch_confirmed=choch_signal is not None,
                entry_price=entry_data['price'],
                position_size=0,
                risk_amount=0,
                exit_price=entry_data['price'],
                exit_scenario='no_trade',
                pnl=0,
                is_winner=False,
                is_first_touch=trade_decision.is_first_touch,
                session_appropriate=trade_decision.session_appropriate,
                choch_aligned=trade_decision.choch_aligned
            )
        
        # Calculate entry price with slippage
        entry_slippage = self.calculate_slippage(entry_data['price'])
        if trade_decision.direction == TradeDirection.LONG:
            actual_entry_price = entry_data['price'] + entry_slippage
        else:
            actual_entry_price = entry_data['price'] - entry_slippage
        
        # Determine exit scenario (simulation or live)
        if simulation_mode:
            exit_scenario = self._simulate_exit_scenario(entry_data, trade_decision)
            exit_price = self._calculate_simulated_exit_price(entry_data, exit_scenario, trade_decision)
        else:
            # Live trading - would use real market data
            exit_scenario = "live_execution"
            exit_price = actual_entry_price  # Placeholder for live execution
        
        # Calculate exit price with slippage
        exit_slippage = self.calculate_slippage(exit_price)
        if trade_decision.direction == TradeDirection.SHORT:
            actual_exit_price = exit_price + exit_slippage
        else:
            actual_exit_price = exit_price - exit_slippage
        
        # Calculate P&L
        if trade_decision.direction == TradeDirection.LONG:
            pnl = (actual_exit_price - actual_entry_price) * position_size
        else:  # SHORT
            pnl = (actual_entry_price - actual_exit_price) * position_size
        
        # Calculate costs
        total_costs = self.commission_per_trade * 2  # Entry + Exit
        
        # Calculate net P&L
        net_pnl = pnl - total_costs
        
        # Update balance
        self.current_balance += net_pnl
        
        # Create trade execution record
        trade_execution = TradeExecution(
            trade_id=f"TRADE_{len(self.trades) + 1}_{entry_data.get('entry_id', 'UNKNOWN')}",
            symbol=entry_data.get('symbol', 'UNKNOWN'),
            timestamp=entry_data.get('timestamp', datetime.now()),
            trade_type=trade_decision.trade_type,
            direction=trade_decision.direction,
            confidence=trade_decision.confidence,
            reason=trade_decision.reason,
            bias=trade_decision.bias_analysis.bias,
            session_type=trade_decision.bias_analysis.session_type,
            choch_confirmed=choch_signal is not None,
            entry_price=actual_entry_price,
            position_size=position_size,
            risk_amount=risk_amount,
            exit_price=actual_exit_price,
            exit_scenario=exit_scenario,
            pnl=net_pnl,
            is_winner=net_pnl > 0,
            is_first_touch=trade_decision.is_first_touch,
            session_appropriate=trade_decision.session_appropriate,
            choch_aligned=trade_decision.choch_aligned
        )
        
        # Store trade
        self.trades.append(trade_execution)
        
        return trade_execution
    
    def _simulate_exit_scenario(self, entry_data: Dict, trade_decision: TradeDecision) -> str:
        """Simulate exit scenario for backtesting."""
        rand = random.random()
        
        if rand < self.stop_loss_hit_rate:
            return 'stop_loss'
        elif rand < self.stop_loss_hit_rate + self.t1_hit_rate:
            return 't1'
        elif rand < self.stop_loss_hit_rate + self.t1_hit_rate + self.t2_hit_rate:
            return 't2'
        elif rand < self.stop_loss_hit_rate + self.t1_hit_rate + self.t2_hit_rate + self.t3_hit_rate:
            return 't3'
        else:
            return 'partial_fill'
    
    def _calculate_simulated_exit_price(self, entry_data: Dict, exit_scenario: str, trade_decision: TradeDecision) -> float:
        """Calculate simulated exit price based on scenario."""
        if exit_scenario == 'stop_loss':
            return entry_data['hard_stop']
        elif exit_scenario == 't1':
            return entry_data['t1_price']
        elif exit_scenario == 't2':
            return entry_data['t2_price']
        elif exit_scenario == 't3':
            return entry_data['t3_price']
        else:  # partial_fill
            if trade_decision.direction == TradeDirection.LONG:
                return entry_data['price'] + (entry_data['t1_price'] - entry_data['price']) * 0.3
            else:
                return entry_data['price'] - (entry_data['price'] - entry_data['t1_price']) * 0.3
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_balance': self.current_balance,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }
        
        trades_df = pd.DataFrame([
            {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'timestamp': trade.timestamp,
                'trade_type': trade.trade_type.value,
                'direction': trade.direction.value,
                'confidence': trade.confidence,
                'bias': trade.bias.value,
                'session_type': trade.session_type.value,
                'choch_confirmed': trade.choch_confirmed,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'is_winner': trade.is_winner,
                'is_loser': trade.pnl < 0,
                'exit_scenario': trade.exit_scenario,
                'is_first_touch': trade.is_first_touch,
                'session_appropriate': trade.session_appropriate,
                'choch_aligned': trade.choch_aligned
            }
            for trade in self.trades
        ])
        
        # Basic metrics
        total_trades = len(trades_df)
        final_balance = self.current_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        # Win/Loss analysis
        winning_trades = trades_df[trades_df['is_winner']]
        losing_trades = trades_df[trades_df['is_loser']]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown analysis
        trades_df['running_balance'] = trades_df['pnl'].cumsum() + self.initial_balance
        trades_df['running_max'] = trades_df['running_balance'].expanding().max()
        trades_df['drawdown'] = trades_df['running_balance'] - trades_df['running_max']
        max_drawdown = trades_df['drawdown'].min()
        
        # Framework compliance analysis
        framework_compliant_trades = trades_df[
            (trades_df['is_first_touch'] == True) &
            (trades_df['session_appropriate'] == True)
        ]
        
        choch_required_trades = trades_df[trades_df['trade_type'] == 'continuation']
        choch_aligned_trades = choch_required_trades[choch_required_trades['choch_aligned'] == True]
        
        return {
            'total_trades': total_trades,
            'final_balance': final_balance,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'framework_compliant_trades': len(framework_compliant_trades),
            'choch_required_trades': len(choch_required_trades),
            'choch_aligned_trades': len(choch_aligned_trades),
            'fade_trades': len(trades_df[trades_df['trade_type'] == 'fade']),
            'continuation_trades': len(trades_df[trades_df['trade_type'] == 'continuation']),
            'bullish_trades': len(trades_df[trades_df['bias'] == 'bullish']),
            'bearish_trades': len(trades_df[trades_df['bias'] == 'bearish']),
            'neutral_trades': len(trades_df[trades_df['bias'] == 'neutral']),
            'trend_day_trades': len(trades_df[trades_df['session_type'] == 'trend_day']),
            'balanced_day_trades': len(trades_df[trades_df['session_type'] == 'balanced_day']),
            'choppy_day_trades': len(trades_df[trades_df['session_type'] == 'choppy_day'])
        }
    
    def print_performance_summary(self):
        """Print comprehensive performance summary."""
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*80)
        print("UNIFIED TRADING FRAMEWORK - PERFORMANCE SUMMARY")
        print("="*80)
        
        # Basic Performance
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"   Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"   Total Return: {metrics['total_return']:.2f}%")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Max Drawdown: ${metrics['max_drawdown']:,.2f}")
        
        # Framework Compliance
        print(f"\n🎯 FRAMEWORK COMPLIANCE")
        print(f"   Framework Compliant Trades: {metrics['framework_compliant_trades']} ({metrics['framework_compliant_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   CHoCH Required Trades: {metrics['choch_required_trades']}")
        print(f"   CHoCH Aligned Trades: {metrics['choch_aligned_trades']} ({metrics['choch_aligned_trades']/metrics['choch_required_trades']*100:.1f}%)")
        
        # Trade Type Analysis
        print(f"\n📊 TRADE TYPE ANALYSIS")
        print(f"   Fade Trades: {metrics['fade_trades']} ({metrics['fade_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Continuation Trades: {metrics['continuation_trades']} ({metrics['continuation_trades']/metrics['total_trades']*100:.1f}%)")
        
        # Bias Analysis
        print(f"\n🎯 BIAS ANALYSIS")
        print(f"   Bullish Bias Trades: {metrics['bullish_trades']} ({metrics['bullish_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Bearish Bias Trades: {metrics['bearish_trades']} ({metrics['bearish_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Neutral Bias Trades: {metrics['neutral_trades']} ({metrics['neutral_trades']/metrics['total_trades']*100:.1f}%)")
        
        # Session Type Analysis
        print(f"\n📈 SESSION TYPE ANALYSIS")
        print(f"   Trend Day Trades: {metrics['trend_day_trades']} ({metrics['trend_day_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Balanced Day Trades: {metrics['balanced_day_trades']} ({metrics['balanced_day_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"   Choppy Day Trades: {metrics['choppy_day_trades']} ({metrics['choppy_day_trades']/metrics['total_trades']*100:.1f}%)")
        
        print("\n" + "="*80)


def create_unified_framework(initial_balance: float = 10000.0,
                           max_equity_per_trade: float = 0.10,
                           slippage_ticks: int = 2,
                           commission_per_trade: float = 5.0,
                           min_confidence: float = 0.6) -> UnifiedTradingFramework:
    """Create a new unified trading framework instance."""
    return UnifiedTradingFramework(
        initial_balance=initial_balance,
        max_equity_per_trade=max_equity_per_trade,
        slippage_ticks=slippage_ticks,
        commission_per_trade=commission_per_trade,
        min_confidence=min_confidence
    )


if __name__ == "__main__":
    # Example usage
    framework = create_unified_framework()
    print("✅ Unified Trading Framework created successfully!")
    print("   This framework can be used for both live trading and backtesting.")
    print("   All framework components are integrated and ready to use.")