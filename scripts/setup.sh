#!/bin/bash
# Zone Fade Detector Setup Script

set -e

echo "🚀 Setting up Zone Fade Detector..."

# Check if Python 3.11+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ required, found $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Check for jq (optional but recommended for signal inspection)
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found. Install with: sudo apt install jq (for signal inspection)"
else
    echo "✅ jq found"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Generate configuration files
echo "⚙️  Setting up configuration..."
make gen-config
make setup-env

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Edit config/config.yaml with your preferences"
echo "3. Test the system: make test-alerts"
echo "4. Run smoke test: make test && make typecheck && make format"
echo "5. Run replay: make replay START=2025-01-06 END=2025-01-10 SYMBOLS=SPY,QQQ,IWM PROVIDER=alpaca"
echo "6. Run live: make live"
echo ""
echo "Happy trading! 📈"