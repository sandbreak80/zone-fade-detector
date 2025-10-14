"""
Zone Touch Tracking

This module implements zone touch tracking to ensure only 1st and 2nd touches
are allowed per trading session, as 3rd+ touches have higher breakout probability.

Features:
- Session-based touch counting
- Zone ID generation and persistence
- 1st/2nd touch filtering only
- Session reset at 9:30 AM ET
- Persistent storage across restarts
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, time, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

class TouchStatus(Enum):
    """Touch status classifications."""
    FIRST_TOUCH = "FIRST_TOUCH"
    SECOND_TOUCH = "SECOND_TOUCH"
    THIRD_PLUS_TOUCH = "THIRD_PLUS_TOUCH"
    UNKNOWN = "UNKNOWN"

@dataclass
class ZoneTouch:
    """Individual zone touch record."""
    zone_id: str
    touch_number: int
    timestamp: datetime
    price_level: float
    trade_direction: str
    session_date: str

@dataclass
class ZoneTouchResult:
    """Zone touch tracking result."""
    touch_status: TouchStatus
    touch_number: int
    zone_id: str
    session_date: str
    previous_touches: List[ZoneTouch]
    is_valid: bool
    recommendation: str

class ZoneTouchTracker:
    """
    Tracks zone touches per session to filter out 3rd+ touches.
    
    This tracker implements the requirement that only 1st and 2nd touches
    should be allowed, as 3rd+ touches have higher breakout probability.
    """
    
    def __init__(self,
                 session_start_hour: int = 9,
                 session_start_minute: int = 30,
                 data_file: str = "zone_touches.json"):
        """
        Initialize zone touch tracker.
        
        Args:
            session_start_hour: Hour when trading session starts (ET)
            session_start_minute: Minute when trading session starts (ET)
            data_file: File to persist touch data
        """
        self.session_start_hour = session_start_hour
        self.session_start_minute = session_start_minute
        self.data_file = data_file
        
        # Touch data storage: {zone_id: {session_date: touch_count}}
        self.touch_data: Dict[str, Dict[str, int]] = {}
        
        # Touch history: {zone_id: [ZoneTouch]}
        self.touch_history: Dict[str, List[ZoneTouch]] = {}
        
        # Current session date
        self.current_session_date = self._get_current_session_date()
        
        # Statistics
        self.total_touches_tracked = 0
        self.first_touches = 0
        self.second_touches = 0
        self.third_plus_touches = 0
        
        # Load existing data
        self._load_touch_data()
    
    def track_zone_touch(self, 
                        symbol: str,
                        zone_type: str,
                        price_level: float,
                        trade_direction: str,
                        timestamp: Optional[datetime] = None) -> ZoneTouchResult:
        """
        Track a zone touch and determine if it's valid.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            zone_type: Type of zone (e.g., 'SUPPORT', 'RESISTANCE')
            price_level: Price level of the zone
            trade_direction: 'LONG' or 'SHORT'
            timestamp: Touch timestamp (default: now)
            
        Returns:
            ZoneTouchResult with touch status and validity
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate zone ID
        zone_id = self._generate_zone_id(symbol, zone_type, price_level)
        
        # Get current session date
        session_date = self._get_session_date(timestamp)
        
        # Check if we need to reset for new session
        if session_date != self.current_session_date:
            self._reset_for_new_session(session_date)
        
        # Get current touch count for this zone in this session
        current_touch_count = self.touch_data.get(zone_id, {}).get(session_date, 0)
        new_touch_number = current_touch_count + 1
        
        # Determine touch status
        if new_touch_number == 1:
            touch_status = TouchStatus.FIRST_TOUCH
            is_valid = True
            recommendation = "1st touch - proceed with setup"
        elif new_touch_number == 2:
            touch_status = TouchStatus.SECOND_TOUCH
            is_valid = True
            recommendation = "2nd touch - proceed with setup"
        else:
            touch_status = TouchStatus.THIRD_PLUS_TOUCH
            is_valid = False
            recommendation = f"{new_touch_number}rd+ touch - SKIP setup (zone likely to break)"
        
        # Create touch record
        touch_record = ZoneTouch(
            zone_id=zone_id,
            touch_number=new_touch_number,
            timestamp=timestamp,
            price_level=price_level,
            trade_direction=trade_direction,
            session_date=session_date
        )
        
        # Update touch data
        if zone_id not in self.touch_data:
            self.touch_data[zone_id] = {}
        self.touch_data[zone_id][session_date] = new_touch_number
        
        # Update touch history
        if zone_id not in self.touch_history:
            self.touch_history[zone_id] = []
        self.touch_history[zone_id].append(touch_record)
        
        # Update statistics
        self.total_touches_tracked += 1
        if touch_status == TouchStatus.FIRST_TOUCH:
            self.first_touches += 1
        elif touch_status == TouchStatus.SECOND_TOUCH:
            self.second_touches += 1
        else:
            self.third_plus_touches += 1
        
        # Get previous touches for this zone in this session
        previous_touches = [
            touch for touch in self.touch_history.get(zone_id, [])
            if touch.session_date == session_date and touch.touch_number < new_touch_number
        ]
        
        # Save data
        self._save_touch_data()
        
        return ZoneTouchResult(
            touch_status=touch_status,
            touch_number=new_touch_number,
            zone_id=zone_id,
            session_date=session_date,
            previous_touches=previous_touches,
            is_valid=is_valid,
            recommendation=recommendation
        )
    
    def _generate_zone_id(self, symbol: str, zone_type: str, price_level: float) -> str:
        """Generate unique zone ID."""
        # Round price level to 0.25 ticks for grouping nearby levels
        rounded_price = round(price_level * 4) / 4
        
        return f"{symbol}_{zone_type}_{rounded_price:.2f}"
    
    def _get_current_session_date(self) -> str:
        """Get current session date string."""
        return self._get_session_date(datetime.now())
    
    def _get_session_date(self, timestamp: datetime) -> str:
        """Get session date for a given timestamp."""
        # Convert to ET (assuming UTC input)
        # In production, you'd use proper timezone handling
        et_hour = timestamp.hour - 5  # Simple UTC to ET conversion
        
        # If before session start, it belongs to previous day's session
        if et_hour < self.session_start_hour or (et_hour == self.session_start_hour and timestamp.minute < self.session_start_minute):
            session_date = (timestamp - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            session_date = timestamp.strftime("%Y-%m-%d")
        
        return session_date
    
    def _reset_for_new_session(self, new_session_date: str):
        """Reset touch counts for new session."""
        self.current_session_date = new_session_date
        # Note: We don't clear historical data, just start fresh counts
        # The touch_data structure maintains separate counts per session
    
    def _load_touch_data(self):
        """Load touch data from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.touch_data = data.get('touch_data', {})
                    # Convert touch history back to objects
                    history_data = data.get('touch_history', {})
                    self.touch_history = {}
                    for zone_id, touches in history_data.items():
                        self.touch_history[zone_id] = [
                            ZoneTouch(**touch) for touch in touches
                        ]
            except Exception as e:
                print(f"Warning: Could not load touch data: {e}")
                self.touch_data = {}
                self.touch_history = {}
    
    def _save_touch_data(self):
        """Save touch data to file."""
        try:
            data = {
                'touch_data': self.touch_data,
                'touch_history': {
                    zone_id: [asdict(touch) for touch in touches]
                    for zone_id, touches in self.touch_history.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save touch data: {e}")
    
    def get_zone_touch_count(self, zone_id: str, session_date: Optional[str] = None) -> int:
        """Get touch count for a specific zone in a session."""
        if session_date is None:
            session_date = self.current_session_date
        
        return self.touch_data.get(zone_id, {}).get(session_date, 0)
    
    def get_zone_touch_history(self, zone_id: str, session_date: Optional[str] = None) -> List[ZoneTouch]:
        """Get touch history for a specific zone in a session."""
        if session_date is None:
            session_date = self.current_session_date
        
        return [
            touch for touch in self.touch_history.get(zone_id, [])
            if touch.session_date == session_date
        ]
    
    def is_valid_touch(self, zone_id: str, session_date: Optional[str] = None) -> bool:
        """Check if next touch for a zone would be valid (1st or 2nd)."""
        current_count = self.get_zone_touch_count(zone_id, session_date)
        return current_count < 2
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        if self.total_touches_tracked == 0:
            return {
                'total_touches_tracked': 0,
                'first_touches': 0,
                'second_touches': 0,
                'third_plus_touches': 0,
                'first_touch_rate': 0.0,
                'second_touch_rate': 0.0,
                'third_plus_touch_rate': 0.0,
                'active_zones': 0,
                'current_session': self.current_session_date
            }
        
        return {
            'total_touches_tracked': self.total_touches_tracked,
            'first_touches': self.first_touches,
            'second_touches': self.second_touches,
            'third_plus_touches': self.third_plus_touches,
            'first_touch_rate': (self.first_touches / self.total_touches_tracked) * 100,
            'second_touch_rate': (self.second_touches / self.total_touches_tracked) * 100,
            'third_plus_touch_rate': (self.third_plus_touches / self.total_touches_tracked) * 100,
            'active_zones': len(self.touch_data),
            'current_session': self.current_session_date
        }
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_touches_tracked = 0
        self.first_touches = 0
        self.second_touches = 0
        self.third_plus_touches = 0
    
    def clear_session_data(self, session_date: str):
        """Clear touch data for a specific session."""
        for zone_id in self.touch_data:
            if session_date in self.touch_data[zone_id]:
                del self.touch_data[zone_id][session_date]
        
        # Remove touch history for the session
        for zone_id in self.touch_history:
            self.touch_history[zone_id] = [
                touch for touch in self.touch_history[zone_id]
                if touch.session_date != session_date
            ]
        
        self._save_touch_data()


class ZoneTouchFilter:
    """
    Filter that applies zone touch tracking to zone fade signals.
    
    This filter implements the requirement that only 1st and 2nd touches
    should be allowed, as 3rd+ touches have higher breakout probability.
    """
    
    def __init__(self, tracker: ZoneTouchTracker):
        """
        Initialize zone touch filter.
        
        Args:
            tracker: ZoneTouchTracker instance
        """
        self.tracker = tracker
        self.signals_vetoed = 0
        self.signals_passed = 0
    
    def filter_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter signal based on zone touch tracking.
        
        Args:
            signal: Zone fade signal to filter
            market_data: Market data (not used but kept for consistency)
            
        Returns:
            Filtered signal if touch is valid, None if vetoed
        """
        # Extract required data
        symbol = signal.get('symbol', 'UNKNOWN')
        zone_type = signal.get('zone_type', 'UNKNOWN')
        price_level = signal.get('zone_level', 0.0)
        trade_direction = signal.get('trade_direction', 'LONG')
        timestamp = signal.get('timestamp', datetime.now())
        
        # Track the zone touch
        result = self.tracker.track_zone_touch(
            symbol, zone_type, price_level, trade_direction, timestamp
        )
        
        # Apply filter
        if not result.is_valid:
            self.signals_vetoed += 1
            return None  # VETO: 3rd+ touch detected
        
        # Add touch tracking info to signal
        signal['touch_tracking'] = {
            'touch_status': result.touch_status.value,
            'touch_number': result.touch_number,
            'zone_id': result.zone_id,
            'session_date': result.session_date,
            'is_valid': result.is_valid,
            'recommendation': result.recommendation
        }
        
        self.signals_passed += 1
        return signal
    
    def get_filter_statistics(self) -> Dict:
        """Get filter statistics."""
        total_signals = self.signals_vetoed + self.signals_passed
        
        return {
            'total_signals_processed': total_signals,
            'signals_vetoed': self.signals_vetoed,
            'signals_passed': self.signals_passed,
            'veto_percentage': (self.signals_vetoed / total_signals * 100) if total_signals > 0 else 0.0,
            'pass_percentage': (self.signals_passed / total_signals * 100) if total_signals > 0 else 0.0,
            'tracker_statistics': self.tracker.get_statistics()
        }