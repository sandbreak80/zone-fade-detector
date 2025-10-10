# Zone Fade Detector - Quick Start Guide
**Session Restored - Ready to Continue Backtesting**

## 🎯 **CURRENT TASK: 2024 Backtesting & Validation**

You were downloading 2024 historical data and running Zone Fade detection against it to validate trading logic.

## 🚀 Immediate Next Steps

### 1. Download 2024 Historical Data
```bash
cd /home/brad/zone-fade-detector
python download_2024_data.py
```
*This downloads full year 2024 data for SPY, QQQ, IWM and saves to `/tmp/zone_fade_data_2024/`*

### 2. Run Zone Fade Detection on 2024 Data
```bash
python test_2024_detection.py
```
*This runs the detection logic against the cached 2024 data and sends alerts to Discord*

### 3. Test API Connectivity (if needed)
```bash
python test_alpaca_final.py
```

### 4. Test Discord Alerts (if needed)
```bash
python test_discord_simple.py
```

## 📊 Current Configuration
- **Symbols**: SPY, QQQ, IWM
- **APIs**: Alpaca (Paper) + Polygon configured
- **Alerts**: Discord webhook ready
- **Trading Hours**: 9:30 AM - 4:00 PM ET
- **QRS Threshold**: 7+ points for A-setups

## 🔧 Quick Commands
```bash
# View logs
docker-compose logs -f zone-fade-detector

# Test alerts only
docker-compose run --rm zone-fade-detector --test-alerts

# Run in background
docker-compose up -d zone-fade-detector

# Stop system
docker-compose down
```

## 📋 Status Check
- ✅ Environment variables configured
- ✅ Docker setup ready
- ✅ API credentials loaded
- ✅ Discord webhook configured
- ✅ Test files ready

**Ready to resume trading system monitoring!**