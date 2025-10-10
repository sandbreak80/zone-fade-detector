#!/bin/bash
# Zone Fade Detector - Docker Entrypoint Script
# Provides flexible container startup with various modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Default values
MODE="standard"
CONFIG_FILE="config/config.yaml"
LOG_LEVEL="INFO"
VERBOSE=false
DRY_RUN=false
SYMBOLS=""
POLL_INTERVAL=""
START_DATE=""
END_DATE=""
PROVIDER="alpaca"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --test-alerts)
            MODE="test-alerts"
            shift
            ;;
        --replay)
            MODE="replay"
            shift
            ;;
        --live)
            MODE="live"
            shift
            ;;
        --help)
            echo "Zone Fade Detector - Docker Entrypoint"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE              Set operation mode (standard, live, replay, test-alerts)"
            echo "  --config FILE            Configuration file path (default: config/config.yaml)"
            echo "  --log-level LEVEL        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
            echo "  --verbose                Enable verbose output"
            echo "  --dry-run                Run in dry-run mode"
            echo "  --symbols SYMBOLS        Comma-separated list of symbols"
            echo "  --poll-interval SECONDS  Polling interval in seconds"
            echo "  --start-date DATE        Start date for replay mode (YYYY-MM-DD)"
            echo "  --end-date DATE          End date for replay mode (YYYY-MM-DD)"
            echo "  --provider PROVIDER      Data provider (alpaca, polygon, both)"
            echo "  --test-alerts            Test alert channels and exit"
            echo "  --replay                 Run in replay mode"
            echo "  --live                   Run in live mode during RTH"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --mode live --verbose"
            echo "  $0 --replay --start-date 2024-01-01 --end-date 2024-01-31"
            echo "  $0 --test-alerts"
            echo "  $0 --mode standard --symbols SPY,QQQ --poll-interval 60"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if required files exist
check_requirements() {
    log "Checking requirements..."
    
    # Check if configuration file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file not found: $CONFIG_FILE"
        error "Please ensure the configuration file is mounted or exists"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        warn ".env file not found, using environment variables only"
    fi
    
    # Check if logs directory exists and is writable
    if [[ ! -d "logs" ]]; then
        log "Creating logs directory..."
        mkdir -p logs
    fi
    
    # Ensure logs directory is writable
    chmod 755 logs 2>/dev/null || true
    chown -R appuser:appuser logs 2>/dev/null || true
    
    # Check if signals directory exists and is writable
    if [[ ! -d "signals" ]]; then
        log "Creating signals directory..."
        mkdir -p signals
    fi
    
    # Ensure signals directory is writable
    chmod 755 signals 2>/dev/null || true
    chown -R appuser:appuser signals 2>/dev/null || true
    
    # Check if data directory exists and is writable
    if [[ ! -d "data" ]]; then
        log "Creating data directory..."
        mkdir -p data
    fi
    
    # Ensure data directory is writable
    chmod 755 data 2>/dev/null || true
    chown -R appuser:appuser data 2>/dev/null || true
    
    success "Requirements check passed"
}

# Function to build command arguments
build_command() {
    local cmd_args=()
    
    # Add configuration file
    cmd_args+=("--config" "$CONFIG_FILE")
    
    # Add log level
    cmd_args+=("--log-level" "$LOG_LEVEL")
    
    # Add verbose flag
    if [[ "$VERBOSE" == "true" ]]; then
        cmd_args+=("--verbose")
    fi
    
    # Add dry-run flag
    if [[ "$DRY_RUN" == "true" ]]; then
        cmd_args+=("--dry-run")
    fi
    
    # Add symbols
    if [[ -n "$SYMBOLS" ]]; then
        IFS=',' read -ra SYMBOL_ARRAY <<< "$SYMBOLS"
        for symbol in "${SYMBOL_ARRAY[@]}"; do
            cmd_args+=("--symbols" "$symbol")
        done
    fi
    
    # Add poll interval
    if [[ -n "$POLL_INTERVAL" ]]; then
        cmd_args+=("--poll-interval" "$POLL_INTERVAL")
    fi
    
    # Add mode-specific arguments
    case "$MODE" in
        "replay")
            if [[ -z "$START_DATE" || -z "$END_DATE" ]]; then
                error "Replay mode requires --start-date and --end-date"
                exit 1
            fi
            cmd_args+=("--replay" "--start-date" "$START_DATE" "--end-date" "$END_DATE" "--provider" "$PROVIDER")
            ;;
        "live")
            cmd_args+=("--live")
            ;;
        "test-alerts")
            cmd_args+=("--test-alerts")
            ;;
        "standard")
            # No additional arguments needed
            ;;
        *)
            error "Unknown mode: $MODE"
            exit 1
            ;;
    esac
    
    echo "${cmd_args[@]}"
}

# Function to run the application
run_application() {
    local cmd_args=($(build_command))
    
    log "Starting Zone Fade Detector in $MODE mode..."
    log "Configuration: $CONFIG_FILE"
    log "Log Level: $LOG_LEVEL"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log "Verbose mode enabled"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry run mode enabled"
    fi
    
    # Display command being executed
    log "Executing: python -m zone_fade_detector.main ${cmd_args[*]}"
    echo ""
    
    # Execute the application
    exec python -m zone_fade_detector.main "${cmd_args[@]}"
}

# Main execution
main() {
    log "Zone Fade Detector - Docker Entrypoint"
    log "Mode: $MODE"
    log "Config: $CONFIG_FILE"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Run the application
    run_application
}

# Handle signals for graceful shutdown
trap 'log "Received shutdown signal, exiting gracefully..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"