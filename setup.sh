#!/bin/bash

# AI Trading System Setup Script

set -e

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"
}

# Check for Python 3
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        log_info "Using python3: $(python3 --version)"
    elif command -v python &> /dev/null && [[ $(python --version 2>&1) == *"Python 3"* ]]; then
        PYTHON_CMD="python"
        log_info "Using python: $(python --version)"
    else
        log_error "Python 3 is required but not found!"
        log_error "Please install Python 3.8+ and try again"
        exit 1
    fi
}

# Check for pip
check_pip() {
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    else
        log_error "pip is required but not found!"
        log_error "Please install pip and try again"
        exit 1
    fi
    log_info "Using pip: $PIP_CMD"
}

echo "🚀 Setting up AI Trading System..."

# Check requirements
check_python
check_pip

# Create virtual environment
log "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    log_info "Activated Windows virtual environment"
else
    source venv/bin/activate
    log_info "Activated Linux/macOS virtual environment"
fi

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    log "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    log_warning "Virtual environment may not be properly activated"
fi

# Upgrade pip
log "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    log_error "requirements.txt not found!"
    log_error "Please ensure requirements.txt exists in the current directory"
    exit 1
fi

# Install requirements
log "📚 Installing Python dependencies..."
log_info "This may take several minutes for ML libraries..."

# Try to install requirements, handle externally-managed environment
if ! pip install -r requirements.txt 2>/dev/null; then
    log_warning "Standard pip install failed, trying alternative approaches..."
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        # We're in a virtual environment, try different approaches
        log_info "In virtual environment, trying alternative installation methods..."
        
        # Try upgrading pip first
        if pip install --upgrade pip 2>/dev/null; then
            log_info "Pip upgraded, retrying package installation..."
            if pip install -r requirements.txt; then
                log "✅ Requirements installed after pip upgrade"
            else
                log_error "Failed to install requirements even after pip upgrade"
                log_info "You may need to check individual package compatibility"
                exit 1
            fi
        else
            log_error "Failed to upgrade pip or install requirements in virtual environment"
            log_info "Try: pip install --force-reinstall -r requirements.txt"
            exit 1
        fi
    else
        # Not in virtual environment, try --user flag for externally-managed environments
        log_info "Not in virtual environment, trying --user installation..."
        if ! pip install --user -r requirements.txt; then
            log_error "Failed to install requirements with --user flag"
            log_info "This may be due to externally-managed Python environment"
            log_info "Recommended solutions:"
            log_info "1. Create and activate a virtual environment:"
            log_info "   python3 -m venv venv && source venv/bin/activate"
            log_info "2. Or use system package manager (apt, dnf, etc.)"
            exit 1
        fi
    fi
fi

# Create environment file from template
if [ ! -f .env ]; then
    log "🔧 Creating .env file from template..."
    if [ -f .env.template ]; then
        cp .env.template .env
        log_warning "Please edit .env file with your actual API keys and passwords!"
    else
        log_error ".env.template not found!"
        log_info "Creating basic .env file..."
        cat > .env << 'EOF'
# Environment Variables
# Please fill in your actual values

# Database Configuration
DATABASE_PASSWORD=changeme
TIMESERIES_PASSWORD=changeme
REDIS_PASSWORD=changeme

# API Keys - Data Providers
ALPHA_VANTAGE_API_KEY=your_key_here
TRADING_MODE=paper

# System Configuration
LOG_LEVEL=INFO
MAX_POSITIONS=10
RISK_PER_TRADE=0.02
EOF
        log_warning "Created basic .env file - please update with your configuration!"
    fi
else
    log_info ".env file already exists"
fi

# Create necessary directories
log "📁 Creating directories..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p notebooks
mkdir -p sql
mkdir -p monitoring

# Set appropriate permissions for logs directory
chmod 755 logs data data/raw data/processed data/models
log_info "Created directory structure with proper permissions"

# Initialize database schemas
log "🗄️ Creating database initialization scripts..."
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
log "📊 Creating monitoring configuration..."

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
log "📓 Creating sample notebooks..."
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
log "🔒 Creating .gitignore..."
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
log_info "Created placeholder files for git tracking"

# Verify Python packages installation
log "🔍 Verifying Python package installation..."
if python -c "import numpy, pandas, sklearn" 2>/dev/null; then
    log "✅ Core ML packages installed successfully"
else
    log_warning "Some core packages may not be properly installed"
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        log_info "Docker is available and running"
    else
        log_warning "Docker is installed but not running. Start Docker to use containerized services."
    fi
else
    log_warning "Docker not found. Install Docker to use containerized services."
    log_info "You can install Docker using the install-linux.sh script"
fi

log "✅ Setup complete!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Run 'docker-compose up -d' to start the infrastructure (if Docker is available)"
echo "3. Run 'python src/main.py' to start the trading system"
echo "4. Access Grafana at http://localhost:3000 (admin/grafana123)"
echo "5. Access Jupyter at http://localhost:8888 (token: trading123)"
echo ""
echo -e "${YELLOW}For paper trading, ensure TRADING_MODE=paper in your .env file${NC}"
echo -e "${RED}⚠️ WARNING: Only use live trading after thorough testing!${NC}"
echo ""
echo -e "${GREEN}To install system dependencies and Docker, run:${NC}"
echo "chmod +x install-linux.sh && ./install-linux.sh"