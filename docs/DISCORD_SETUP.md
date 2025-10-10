# Discord Webhook Setup Guide

This guide explains how to set up Discord webhooks for receiving Zone Fade Detector alerts.

## 🎯 Overview

The Zone Fade Detector can send real-time alerts directly to Discord channels using webhooks. This provides instant notifications for high-probability trading setups.

## 🔧 Setup Instructions

### 1. Create a Discord Webhook

1. **Open Discord** and navigate to your server
2. **Right-click** on the channel where you want alerts
3. Select **"Edit Channel"**
4. Go to **"Integrations"** tab
5. Click **"Create Webhook"**
6. **Copy the Webhook URL** (keep this secure!)

### 2. Configure the Application

#### Option A: Using Environment Variables (Recommended)

Add the webhook URL to your `.env` file:

```bash
# Discord Webhook Configuration
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

#### Option B: Direct Configuration

Update `config/config.yaml`:

```yaml
alerts:
  channels: ['console', 'file', 'webhook']
  
  webhook:
    enabled: true
    url: https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
    timeout: 5
```

### 3. Test the Webhook

```bash
# Test Discord webhook integration
docker-compose run --rm zone-fade-detector --test-alerts
```

## 📱 Alert Format

Discord alerts include rich formatting with:

- **Symbol** and **Direction** (Long/Short)
- **Zone Level** and **Zone Type**
- **QRS Score** (Quality Rating)
- **Rejection Candle** details
- **Target Levels** (T1, T2)
- **Timestamp** and **Confirmation Status**

### Example Alert

```
🚨 **ZONE FADE ALERT** 🚨

**Symbol:** SPY
**Direction:** LONG
**Zone Level:** 485.50
**Zone Type:** Prior Day High
**QRS Score:** 8/10 ⭐

**Rejection Candle:**
- Open: 485.20
- High: 485.80
- Low: 484.90
- Close: 485.10
- Volume: 1,500,000

**Targets:**
- T1: 485.25 (VWAP)
- T2: 486.00 (Range Edge)

**Status:** CHoCH Confirmed ✅
**Time:** 2024-01-15 14:30:00 EST
```

## ⚙️ Configuration Options

### Webhook Settings

```yaml
alerts:
  webhook:
    enabled: true                    # Enable/disable webhook alerts
    url: ${DISCORD_WEBHOOK_URL}      # Webhook URL (from environment)
    timeout: 5                       # Request timeout in seconds
    secret: ""                       # Optional webhook secret
```

### Alert Filtering

```yaml
alerts:
  min_qrs_score: 7                  # Minimum QRS score for alerts
  deduplication_minutes: 5          # Prevent duplicate alerts
  max_alerts_per_hour: 10           # Rate limiting
```

## 🔒 Security Best Practices

### 1. Protect Your Webhook URL

- **Never commit** webhook URLs to version control
- Store in environment variables or secure configuration
- Rotate webhook URLs periodically

### 2. Channel Permissions

- Use a **dedicated channel** for trading alerts
- Set appropriate **permissions** (read-only for most users)
- Consider **role-based access** for sensitive alerts

### 3. Rate Limiting

- Discord has **rate limits** (30 requests per minute per webhook)
- The system includes built-in rate limiting
- Monitor for rate limit errors in logs

## 🐛 Troubleshooting

### Common Issues

**Webhook not sending alerts:**
```bash
# Check webhook URL format
echo $DISCORD_WEBHOOK_URL

# Test webhook manually
curl -X POST "$DISCORD_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test message"}'
```

**Rate limit errors:**
```bash
# Check logs for rate limit messages
docker-compose logs zone-fade-detector | grep -i "rate limit"

# Reduce alert frequency in config
# max_alerts_per_hour: 5
```

**Permission errors:**
```bash
# Verify webhook permissions in Discord
# Check if webhook is still active
# Regenerate webhook if needed
```

### Debug Mode

```bash
# Run with debug logging
docker-compose run --rm zone-fade-detector \
  --log-level DEBUG \
  --test-alerts
```

## 📊 Monitoring

### Check Alert Status

```bash
# View recent alerts
docker-compose logs zone-fade-detector | grep -i "discord"

# Check webhook health
docker-compose exec zone-fade-detector python -c "
from zone_fade_detector.core.alert_system import AlertSystem
import asyncio
asyncio.run(AlertSystem.test_webhook())
"
```

### Log Analysis

```bash
# Monitor webhook success rate
docker-compose logs zone-fade-detector | grep -E "(webhook|discord)" | tail -20

# Check for errors
docker-compose logs zone-fade-detector | grep -i "error.*webhook"
```

## 🎨 Customization

### Custom Message Format

You can customize the Discord message format by modifying the alert system:

```python
# In src/zone_fade_detector/core/alert_system.py
def format_discord_message(self, alert_data):
    return {
        "content": f"🚨 **ZONE FADE ALERT** 🚨",
        "embeds": [{
            "title": f"{alert_data['symbol']} - {alert_data['direction'].upper()}",
            "color": 0x00ff00 if alert_data['direction'] == 'long' else 0xff0000,
            "fields": [
                {"name": "Zone Level", "value": f"${alert_data['zone_level']:.2f}", "inline": True},
                {"name": "QRS Score", "value": f"{alert_data['qrs_score']}/10", "inline": True},
                {"name": "Status", "value": "CHoCH Confirmed ✅" if alert_data['choch_confirmed'] else "Pending", "inline": True}
            ]
        }]
    }
```

### Multiple Webhooks

For multiple Discord channels:

```yaml
alerts:
  webhook:
    enabled: true
    urls:
      - https://discordapp.com/api/webhooks/WEBHOOK1
      - https://discordapp.com/api/webhooks/WEBHOOK2
    timeout: 5
```

## 📚 Additional Resources

- [Discord Webhook Documentation](https://discord.com/developers/docs/resources/webhook)
- [Discord Rate Limits](https://discord.com/developers/docs/topics/rate-limits)
- [Zone Fade Detector Configuration](README.md#configuration)

## 🆘 Support

If you encounter issues with Discord webhook setup:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the application logs
3. Verify Discord webhook permissions
4. Test with a simple curl command
5. Open an issue on GitHub with logs and configuration details