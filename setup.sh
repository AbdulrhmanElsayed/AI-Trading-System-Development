#!/bin/bash

# AI Trading System Setup Script

set -e

echo "🚀 Setting up AI Trading System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create environment file from template
if [ ! -f .env ]; then
    echo "🔧 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️ Please edit .env file with your actual API keys and passwords!"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p notebooks
mkdir -p sql

# Initialize database schemas
echo "🗄️ Creating database initialization scripts..."
cat > sql/init.sql << 'EOF'
-- Trading System Database Initialization

-- Create tables for configuration and metadata
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4),
    price DECIMAL(15,6),
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(15,6),
    status VARCHAR(20) NOT NULL,
    filled_quantity DECIMAL(15,6) DEFAULT 0,
    filled_price DECIMAL(15,6),
    commission DECIMAL(10,6) DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for positions
CREATE TABLE IF NOT EXISTS positions (
    symbol VARCHAR(20) PRIMARY KEY,
    quantity DECIMAL(15,6) NOT NULL,
    average_price DECIMAL(15,6) NOT NULL,
    market_price DECIMAL(15,6),
    unrealized_pnl DECIMAL(15,6),
    realized_pnl DECIMAL(15,6) DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON trading_signals(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_timestamp ON orders(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
EOF

cat > sql/timescale_init.sql << 'EOF'
-- TimescaleDB Time Series Database Initialization

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable for OHLCV data
CREATE TABLE IF NOT EXISTS ohlcv_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open_price DECIMAL(15,6),
    high_price DECIMAL(15,6),
    low_price DECIMAL(15,6),
    close_price DECIMAL(15,6),
    volume DECIMAL(20,6),
    timeframe VARCHAR(10)
);

-- Convert to hypertable
SELECT create_hypertable('ohlcv_data', 'timestamp', if_not_exists => TRUE);

-- Create table for technical indicators
CREATE TABLE IF NOT EXISTS technical_indicators (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    value DECIMAL(15,6),
    metadata JSONB
);

SELECT create_hypertable('technical_indicators', 'timestamp', if_not_exists => TRUE);

-- Create table for news sentiment
CREATE TABLE IF NOT EXISTS sentiment_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20),
    source VARCHAR(100),
    headline TEXT,
    sentiment_score DECIMAL(5,4),
    confidence DECIMAL(5,4),
    metadata JSONB
);

SELECT create_hypertable('sentiment_data', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_symbol_name_time ON technical_indicators(symbol, indicator_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time ON sentiment_data(symbol, timestamp DESC);
EOF

# Create monitoring configuration
echo "📊 Creating monitoring configuration..."
mkdir -p monitoring

cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "rules/*.yml"

scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['trading-app:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'
EOF

# Create Jupyter notebooks directory with sample notebook
echo "📓 Creating sample notebooks..."
cat > notebooks/01_data_exploration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Trading System Data Exploration\n",
    "\n",
    "This notebook provides examples of how to interact with the trading system data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set up plotting\n",
    "plt.style.use('default')\n",
    "sns.set_palette('husl')\n",
    "\n",
    "print(\"✅ Trading System Analysis Environment Ready\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create .gitignore
echo "🔒 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Environment and secrets
.env
*.log
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
data/models/*
!data/models/.gitkeep

# Jupyter
.ipynb_checkpoints/

# Database
*.db
*.sqlite3

# Docker
docker-compose.override.yml
EOF

# Create placeholder files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/models/.gitkeep

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Run 'docker-compose up -d' to start the infrastructure"
echo "3. Run 'python src/main.py' to start the trading system"
echo "4. Access Grafana at http://localhost:3000 (admin/grafana123)"
echo "5. Access Jupyter at http://localhost:8888 (token: trading123)"
echo ""
echo "For paper trading, ensure TRADING_MODE=paper in your .env file"
echo "⚠️ WARNING: Only use live trading after thorough testing!"