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
from zone_fade_detector.core.detector import ZoneFadeDetector
from zone_fade_detector.core.setup import ZoneFadeSetup

# Data modules
from zone_fade_detector.data.alpaca_client import AlpacaClient
from zone_fade_detector.data.polygon_client import PolygonClient

# Indicator modules
from zone_fade_detector.indicators.vwap import VWAPCalculator
from zone_fade_detector.indicators.swing_structure import SwingStructureDetector
from zone_fade_detector.indicators.opening_range import OpeningRangeCalculator

# Strategy modules
from zone_fade_detector.strategies.zone_fade_strategy import ZoneFadeStrategy
from zone_fade_detector.strategies.qrs_scorer import QRSScorer

# Utility modules
from zone_fade_detector.utils.config import load_config
from zone_fade_detector.utils.logging import setup_logging

__all__ = [
    # Core
    "ZoneFadeDetector",
    "ZoneFadeSetup",
    
    # Data
    "AlpacaClient",
    "PolygonClient",
    
    # Indicators
    "VWAPCalculator",
    "SwingStructureDetector",
    "OpeningRangeCalculator",
    
    # Strategies
    "ZoneFadeStrategy",
    "QRSScorer",
    
    # Utils
    "load_config",
    "setup_logging",
]