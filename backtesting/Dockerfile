# Bitcoin Zone Fade Detector Backtesting Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backtesting files
COPY backtesting/ ./backtesting/
COPY src/ ./src/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app

# Create results directory
RUN mkdir -p /app/backtesting/results

# Set working directory to backtesting
WORKDIR /app/backtesting

# Default command
CMD ["python", "backtest_bitcoin.py"]