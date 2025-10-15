# Zone Fade Detector - 2024 Backtesting Plan
**Session Restored - Ready to Continue Validation**

## 🎯 **Current Objective**
Download 2024 historical data and run Zone Fade Detector against it to validate trading logic and identify A-grade setups.

## 📊 **Backtesting Strategy**

### Phase 1: Data Download
- **Target**: Full year 2024 data (Jan 1 - Dec 31)
- **Symbols**: SPY, QQQ, IWM (proxies for /ES, /NQ, /RTY futures)
- **Data Type**: 1-minute OHLCV bars
- **Storage**: `/tmp/zone_fade_data_2024/` (host-persistent)
- **Format**: Pickled for fast loading

### Phase 2: Detection Testing
- **QRS Threshold**: Lowered to 5 (from 7) for more detections during testing
- **Max Setups**: 10 per symbol (increased for testing)
- **Cooldown**: 5 minutes (reduced for testing)
- **Filters**: Intermarket and volume filtering disabled for initial testing

### Phase 3: Analysis & Validation
- **Alert Distribution**: By symbol, QRS score, date
- **Setup Quality**: Analyze A-grade vs B-grade setups
- **Performance**: Success rate of detected setups
- **Discord Alerts**: Real-time notifications for validation

## 🚀 **Execution Steps**

### Step 1: Download 2024 Data
```bash
cd /path/to/zone-fade-detector
python download_2024_data.py
```

**Expected Output:**
- Downloads ~250 trading days of data
- ~390 bars per day × 250 days = ~97,500 bars per symbol
- Total: ~292,500 bars across 3 symbols
- Estimated size: 50-100 MB

### Step 2: Run Detection
```bash
python test_2024_detection.py
```

**Expected Output:**
- Loads cached 2024 data
- Processes through Zone Fade detection logic
- Generates alerts for setups meeting QRS criteria
- Sends alerts to Discord for validation

### Step 3: Analyze Results
- Review Discord alerts for setup quality
- Check console output for detection summary
- Analyze QRS score distribution
- Identify patterns in successful setups

## 📈 **Expected Results**

### Conservative Estimate
- **Total Setups**: 50-100 across all symbols for 2024
- **A-Grade (7+ QRS)**: 10-20 setups
- **B-Grade (5-6 QRS)**: 30-50 setups
- **C-Grade (3-4 QRS)**: 10-30 setups

### Success Metrics
- **Detection Rate**: Setups per trading day
- **Quality Distribution**: QRS score spread
- **Symbol Performance**: SPY vs QQQ vs IWM
- **Seasonal Patterns**: Monthly setup frequency

## 🔧 **Configuration for Testing**

### Data Download Settings
- **Cache TTL**: 30 days (86400 * 30)
- **Primary Source**: Alpaca API
- **Fallback**: Polygon API
- **Chunking**: Automatic API rate limit handling

### Detection Settings
- **QRS Threshold**: 5 (lowered for testing)
- **Max Setups**: 10 per symbol
- **Cooldown**: 5 minutes
- **Deduplication**: 2 minutes
- **Filters**: Disabled for initial testing

### Alert Settings
- **Console**: Enabled with details
- **File**: `/tmp/alerts_2024.log`
- **Discord**: Real-time webhook notifications
- **Rate Limiting**: 1 second between alerts

## 📋 **Validation Checklist**

### Data Quality
- [ ] All 3 symbols downloaded successfully
- [ ] Data covers full 2024 trading year
- [ ] No missing trading days
- [ ] Price ranges look reasonable

### Detection Logic
- [ ] Zone Fade setups detected
- [ ] QRS scoring working correctly
- [ ] Alerts generated and sent
- [ ] Discord notifications received

### Performance
- [ ] Processing time reasonable (<5 minutes)
- [ ] Memory usage acceptable
- [ ] No API rate limit issues
- [ ] Error handling working

## 🎯 **Success Criteria**

### Minimum Viable Results
- At least 10 Zone Fade setups detected
- QRS scores range from 3-10
- Alerts successfully sent to Discord
- No critical errors during processing

### Optimal Results
- 50+ Zone Fade setups detected
- Good distribution of QRS scores
- Clear A-grade setups identified
- Successful validation of trading logic

## 🚨 **Troubleshooting**

### If No Data Downloaded
- Check API credentials in `.env`
- Verify Alpaca paper trading access
- Check API rate limits
- Try smaller date ranges first

### If No Setups Detected
- Lower QRS threshold to 3
- Disable all filters
- Check detection logic
- Verify data quality

### If Discord Alerts Fail
- Test webhook URL
- Check Discord permissions
- Verify alert formatting
- Test with simple alert first

## 📊 **Next Steps After Backtesting**

1. **Analyze Results**: Review all detected setups
2. **Tune Parameters**: Adjust QRS thresholds and filters
3. **Validate Logic**: Confirm setups match expected patterns
4. **Optimize Detection**: Improve detection accuracy
5. **Prepare Live Trading**: Ready system for real-time monitoring

---
*Backtesting plan created for 2024 validation session*