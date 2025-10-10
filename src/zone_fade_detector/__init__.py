"""
Zone Fade Detector

A Python-based trading system that identifies high-probability Zone Fade reversal setups
using 15-minute delayed market data from Alpaca and Polygon APIs.

The system monitors SPY, QQQ, and IWM for trading opportunities based on:
- Higher-timeframe zone approaches
- Market exhaustion signals
- Change of Character (CHoCH) patterns
- Quality Rating System (QRS) scoring

Author: Zone Fade Detector Team
Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Zone Fade Detector Team"
__email__ = "team@zonefadedetector.com"
__license__ = "MIT"

# Core modules
from zone_fade_detector.core.models import (
    OHLCVBar,
    Zone,
    ZoneType,
    SetupDirection,
    CandleType,
    VWAPData,
    OpeningRange,
    SwingPoint,
    SwingStructure,
    VolumeAnalysis,
    QRSFactors,
    ZoneFadeSetup,
    MarketContext,
    Alert,
)

# Utility modules
from zone_fade_detector.utils.config import load_config
from zone_fade_detector.utils.logging import setup_logging

__all__ = [
    # Core Models
    "OHLCVBar",
    "Zone",
    "ZoneType",
    "SetupDirection",
    "CandleType",
    "VWAPData",
    "OpeningRange",
    "SwingPoint",
    "SwingStructure",
    "VolumeAnalysis",
    "QRSFactors",
    "ZoneFadeSetup",
    "MarketContext",
    "Alert",
    
    # Utils
    "load_config",
    "setup_logging",
]