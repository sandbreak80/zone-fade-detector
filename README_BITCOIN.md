# Bitcoin Zone Fade Detector - 24/7 Cryptocurrency Trading System

A specialized version of the Zone Fade Detector optimized for Bitcoin and other cryptocurrencies that trade 24/7. This system uses the same core Zone Fade strategy but adapted for crypto market characteristics.

## 🚀 Features

### 24/7 Trading
- **Continuous monitoring** of Bitcoin, Ethereum, and other cryptocurrencies
- **No market hours restrictions** - crypto markets never close
- **Real-time alerts** via Discord webhook
- **Optimized polling** every 60 seconds for crypto volatility

### Crypto-Optimized Strategy
- **Lower QRS threshold** (6/10) for crypto volatility
- **Higher volume thresholds** for crypto volume spikes
- **Extended timeframes** (12-hour VWAP, 50-bar swing detection)
- **Monthly zone weights** - monthly levels are very important in crypto

### Data Sources
- **CoinGecko API** - Free, reliable cryptocurrency data
- **Rate limiting** - Respects API limits with built-in delays
- **Caching** - 5-minute cache for frequent updates
- **Multiple coins** - Bitcoin, Ethereum, Binance Coin support

## 📁 New Files Created

### Core System
- `bitcoin_zone_fade_detector.py` - Main Bitcoin detector class
- `src/zone_fade_detector/data/crypto_client.py` - CoinGecko API client
- `src/zone_fade_detector/data/bitcoin_data_manager.py` - Bitcoin data manager

### Configuration
- `config/bitcoin_config.yaml` - Bitcoin-specific configuration
- `docker-compose.bitcoin.yml` - Docker setup for Bitcoin detector

### Scripts
- `test_bitcoin_detector.py` - Test Bitcoin detector functionality
- `run_bitcoin_detector.py` - Run Bitcoin detector in live mode

## 🛠️ Setup

### 1. Environment Variables
Add to your `.env` file:
```bash
# Discord webhook (same as stock detector)
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/1426359199591305237/w6HghgFEE8XVYnIZey4jpz8GYmcf7kT56hdfpba9UWD7v07UdedDXLhkcye5ze5vYtTC

# CoinGecko API (optional - free tier available)
COINGECKO_API_KEY=your_api_key_here
```

### 2. Install Dependencies
The Bitcoin detector uses the same dependencies as the main system, plus:
- `aiohttp` for async HTTP requests
- `diskcache` for caching

### 3. Test the System
```bash
# Test Bitcoin detector
python test_bitcoin_detector.py

# Test with Docker
docker-compose -f docker-compose.bitcoin.yml run --rm bitcoin-test
```

### 4. Run Live Mode
```bash
# Run Bitcoin detector
python run_bitcoin_detector.py

# Run with Docker
docker-compose -f docker-compose.bitcoin.yml up bitcoin-zone-fade-detector
```

## 📊 Configuration

### Supported Cryptocurrencies
- **Bitcoin (BTC)** - Primary focus
- **Ethereum (ETH)** - Secondary
- **Binance Coin (BNB)** - Optional

### Key Settings
- **Polling interval**: 60 seconds (crypto moves fast)
- **Cache TTL**: 5 minutes (more frequent updates)
- **Min QRS score**: 6/10 (lower threshold for crypto)
- **Volume threshold**: 2.0x (higher for crypto spikes)
- **Zone proximity**: 0.8% (crypto price levels)

### Timeframes
- **Micro**: 5 minutes
- **Short**: 15 minutes  
- **Medium**: 1 hour
- **Long**: 4 hours
- **VWAP**: 12 hours (longer than stocks)

## 🎯 How It Works

### 1. Data Fetching
- Fetches OHLC data from CoinGecko API
- Caches data for 5 minutes to respect rate limits
- Monitors Bitcoin, Ethereum, and other configured coins

### 2. Zone Detection
- **Prior day high/low** - Previous day's extremes
- **Weekly high/low** - Weekly extremes (higher weight)
- **Monthly high/low** - Monthly extremes (highest weight)

### 3. Signal Processing
- Applies Zone Fade strategy adapted for crypto
- Lower QRS threshold due to crypto volatility
- Higher volume thresholds for crypto volume spikes
- Extended lookback periods for crypto patterns

### 4. Alert Generation
- Sends rich Discord alerts with Bitcoin-specific formatting
- Includes price levels, QRS scores, and risk/reward ratios
- 24/7 monitoring means alerts can come anytime

## 📱 Discord Alerts

Bitcoin alerts include:
- **Symbol**: BTC, ETH, BNB
- **Direction**: LONG/SHORT
- **Zone Level**: Price level in USD
- **QRS Score**: Quality rating (6-10 for crypto)
- **Entry/Stop/Targets**: Calculated levels
- **Risk/Reward**: Risk-reward ratios
- **QRS Breakdown**: Detailed scoring factors

## 🔧 Docker Usage

### Run Bitcoin Detector
```bash
# Start Bitcoin detector
docker-compose -f docker-compose.bitcoin.yml up -d bitcoin-zone-fade-detector

# View logs
docker-compose -f docker-compose.bitcoin.yml logs -f bitcoin-zone-fade-detector

# Stop detector
docker-compose -f docker-compose.bitcoin.yml down
```

### Test Bitcoin Detector
```bash
# Run test
docker-compose -f docker-compose.bitcoin.yml run --rm bitcoin-test
```

## 📈 Expected Behavior

### During Market Hours
- **Continuous monitoring** every 60 seconds
- **Zone Fade detection** when price reaches key levels
- **Discord alerts** sent immediately when setups are found
- **Rich formatting** with Bitcoin-specific details

### Alert Examples
```
🚨 ZONE FADE ALERT - BTC_ZF_20241231_160000
🎯 Zone Fade Setup - BTC
Direction: LONG
Zone Level: $45,000.00
QRS Score: 7/10
Priority: HIGH

📊 Symbol: BTC
📈 Direction: LONG
🎯 Zone Level: $45,000.00
⭐ QRS Score: 7/10
💰 Entry Price: $44,800.00
🛑 Stop Loss: $44,500.00
🎯 Target 1: $46,000.00
🎯 Target 2: $47,500.00
📊 Risk/Reward 1: 1:4.0
📊 Risk/Reward 2: 1:9.0
💵 Risk Amount: $300.00
🕐 Timestamp: Friday, December 31, 2024 at 4:00 PM

📋 QRS Breakdown:
• Zone Quality: 2/2
• Rejection Clarity: 2/2
• Structure Flip: 1/2
• Context: 2/2
• Intermarket Divergence: 0/2
```

## 🚨 Important Notes

### Rate Limiting
- CoinGecko free tier: 10-50 calls/minute
- Built-in rate limiting with 1-second delays
- Caching reduces API calls

### Market Differences
- **24/7 trading** - no market hours
- **Higher volatility** - wider price ranges
- **Different patterns** - crypto has unique behavior
- **Volume spikes** - more dramatic volume changes

### Configuration
- Uses separate config file (`bitcoin_config.yaml`)
- Different thresholds optimized for crypto
- Extended timeframes for crypto patterns
- Higher weights for longer-term levels

## 🔍 Troubleshooting

### No Data Received
- Check CoinGecko API status
- Verify internet connection
- Check rate limiting (too many requests)

### Discord Alerts Not Working
- Verify `DISCORD_WEBHOOK_URL` in `.env`
- Test webhook with `test_bitcoin_detector.py`
- Check Discord channel permissions

### High Memory Usage
- Reduce cache size in config
- Lower polling frequency
- Monitor with `docker stats`

## 🎉 Success!

The Bitcoin Zone Fade Detector is now ready for 24/7 cryptocurrency trading! It will:

1. **Monitor Bitcoin and Ethereum** continuously
2. **Detect Zone Fade setups** using crypto-optimized parameters
3. **Send rich Discord alerts** with all the details you need
4. **Trade 24/7** - no market hours restrictions
5. **Adapt to crypto volatility** with optimized thresholds

The system uses the same proven Zone Fade strategy but adapted for the unique characteristics of cryptocurrency markets. Since crypto trades 24/7, you can get alerts at any time of day or night! 🚀