# Zone Fade Detector - Setup Guide

## 🚀 Quick Setup

### 1. Add Your API Credentials

Edit the `.env` file with your actual API credentials:

```bash
nano .env
```

**Required credentials:**
- `ALPACA_API_KEY` - Your Alpaca API key
- `ALPACA_SECRET_KEY` - Your Alpaca secret key  
- `POLYGON_API_KEY` - Your Polygon API key
- `DISCORD_WEBHOOK_URL` - Your Discord webhook URL

### 2. Configure Your Settings (Optional)

Edit the `config/config.yaml` file to customize your preferences:

```bash
nano config/config.yaml
```

**Key settings you might want to change:**
- `symbols` - Which ETFs to monitor
- `polling.interval_seconds` - How often to check for data
- `alerts.min_qrs_score` - Minimum quality score for alerts
- `alerts.channels` - Which alert channels to use

### 3. Build and Run

```bash
# Build Docker images
make build

# Test your setup
make test-alerts

# Run the detector
make run
```

## 📁 File Structure

```
zone-fade-detector/
├── .env                    # YOUR API credentials (persistent)
├── .env.example           # Template file (safe to overwrite)
├── config/
│   ├── config.yaml        # YOUR settings (persistent)
│   └── config.example.yaml # Template file (safe to overwrite)
└── ...
```

## 🔒 Important Notes

- **`.env`** and **`config/config.yaml`** are in `.gitignore` and will NOT be overwritten
- These files contain your personal settings and API credentials
- Template files (`.env.example`, `config.example.yaml`) are safe to overwrite
- Always use `make` commands instead of direct Docker commands

## 🆘 Troubleshooting

**If you lose your credentials:**
1. Check `.env.backup` for your previous settings
2. Re-enter your API credentials in `.env`
3. Your `config/config.yaml` should still be intact

**If configuration gets reset:**
1. Check `config/config.yaml` - it should persist
2. If needed, copy from `config/config.example.yaml` and customize

## 🎯 Next Steps

1. **Add your API credentials** to `.env`
2. **Customize settings** in `config/config.yaml` (optional)
3. **Test the setup**: `make test-alerts`
4. **Run the detector**: `make run` or `make live`

Happy trading! 📈