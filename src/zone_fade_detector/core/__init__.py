"""
Core module for Zone Fade Detector.

This module contains the core data models, main detector, and alert system
used throughout the trading system.
"""

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
from zone_fade_detector.core.detector import ZoneFadeDetector
from zone_fade_detector.core.alert_system import (
    AlertSystem,
    AlertChannelConfig,
    ConsoleAlertChannel,
    FileAlertChannel,
    EmailAlertChannel,
    WebhookAlertChannel,
)

__all__ = [
    # Models
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
    
    # Core Components
    "ZoneFadeDetector",
    "AlertSystem",
    "AlertChannelConfig",
    "ConsoleAlertChannel",
    "FileAlertChannel",
    "EmailAlertChannel",
    "WebhookAlertChannel",
]