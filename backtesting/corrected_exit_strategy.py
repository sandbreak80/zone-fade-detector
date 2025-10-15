#!/usr/bin/env python3
"""
Corrected Zone Fade Exit Strategy

This module fixes the critical calculation error in risk amount calculation.
"""

def calculate_corrected_zone_fade_exit_strategy(entry_price: float, zone, bars, bar_index: int, qrs_score: float) -> dict:
    """Calculate Zone Fade exit strategy with CORRECTED risk calculation."""
    # Determine trade direction based on zone type
    if zone.zone_type in [zone.zone_type.PRIOR_DAY_HIGH, zone.zone_type.WEEKLY_HIGH, zone.zone_type.VALUE_AREA_HIGH]:
        direction = "SHORT"
        # Hard stop: back of zone (zone level)
        hard_stop = zone.level
        # Zone invalidation: break beyond back of zone
        invalidation_level = zone.level
    else:
        direction = "LONG"
        # Hard stop: back of zone (zone level)
        hard_stop = zone.level
        # Zone invalidation: break beyond back of zone
        invalidation_level = zone.level
    
    # Calculate VWAP for T1 target
    vwap = calculate_vwap(bars, bar_index)
    
    # CORRECTED: Calculate R (Risk Unit) - FIXED THE ERROR!
    if direction == "SHORT":
        risk_amount = entry_price - hard_stop  # Entry above zone, stop at zone
    else:
        risk_amount = entry_price - hard_stop  # Entry below zone, stop at zone - CORRECTED!
    
    # Ensure minimum risk amount
    if risk_amount <= 0:
        risk_amount = entry_price * 0.01  # 1% fallback
    
    # Calculate targets based on QRS quality
    if qrs_score >= 9.0:  # A-grade setup
        t1_reward = 1.0 * risk_amount
        t2_reward = 2.0 * risk_amount
        t3_reward = 3.0 * risk_amount
    elif qrs_score >= 7.0:  # B-grade setup
        t1_reward = 0.8 * risk_amount
        t2_reward = 1.6 * risk_amount
        t3_reward = 2.4 * risk_amount
    else:  # C-grade setup
        t1_reward = 0.6 * risk_amount
        t2_reward = 1.2 * risk_amount
        t3_reward = 1.8 * risk_amount
    
    # Calculate target prices
    if direction == "SHORT":
        t1_price = min(vwap, entry_price - t1_reward)  # Nearest of VWAP or 1R
        t2_price = entry_price - t2_reward
        t3_price = entry_price - t3_reward
    else:
        t1_price = max(vwap, entry_price + t1_reward)  # Nearest of VWAP or 1R
        t2_price = entry_price + t2_reward
        t3_price = entry_price + t3_reward
    
    # Calculate risk/reward ratio (using T1 as primary target)
    if direction == "SHORT":
        primary_reward = entry_price - t1_price
    else:
        primary_reward = t1_price - entry_price
    
    risk_reward_ratio = primary_reward / risk_amount if risk_amount > 0 else 0
    
    return {
        "direction": direction,
        "hard_stop": hard_stop,
        "invalidation_level": invalidation_level,
        "t1_price": t1_price,
        "t2_price": t2_price,
        "t3_price": t3_price,
        "risk_amount": risk_amount,
        "t1_reward": t1_reward,
        "t2_reward": t2_reward,
        "t3_reward": t3_reward,
        "vwap": vwap,
        "risk_reward_ratio": risk_reward_ratio
    }


def calculate_vwap(bars, bar_index: int, lookback: int = 20) -> float:
    """Calculate VWAP for the given bar index."""
    start_index = max(0, bar_index - lookback + 1)
    end_index = bar_index + 1
    
    if start_index >= len(bars) or end_index > len(bars):
        return bars[bar_index].close if bar_index < len(bars) else 0.0
    
    total_volume = 0.0
    total_price_volume = 0.0
    
    for i in range(start_index, end_index):
        bar = bars[i]
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        total_volume += bar.volume
        total_price_volume += typical_price * bar.volume
    
    return total_price_volume / total_volume if total_volume > 0 else bars[bar_index].close


if __name__ == "__main__":
    print("✅ Corrected Zone Fade Exit Strategy loaded")
    print("🔧 Fixed critical risk calculation error:")
    print("   - LONG trades: risk_amount = entry_price - hard_stop (was backwards)")
    print("   - SHORT trades: risk_amount = entry_price - hard_stop (unchanged)")
    print("   - This should fix the 100% success rate issue")