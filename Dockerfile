# AI Trading System - Multi-stage Docker Configuration

# Base image with Python 3.11
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for development server
EXPOSE 8000

# Command for development
CMD ["python", "-m", "src.main"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models

# Change ownership to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "src.main", "--prod"]

# Testing stage
FROM base as testing

# Install test dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source and test code
COPY . .

# Change ownership
RUN chown -R appuser:appuser /app
USER appuser

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src"]

# Data processor service
FROM base as data-processor

# Install additional data processing dependencies
RUN pip install \
    pandas==2.1.0 \
    numpy==1.24.3 \
    yfinance==0.2.18 \
    alpha-vantage==2.3.1 \
    ccxt==4.0.30

# Copy data processing code
COPY src/data/ ./src/data/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

RUN chown -R appuser:appuser /app
USER appuser

# Health check for data processor
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.data.market_data_manager import MarketDataManager; print('Data processor healthy')"

CMD ["python", "-m", "src.data.market_data_manager"]

# ML service
FROM base as ml-service

# Install ML dependencies
RUN pip install \
    tensorflow==2.13.0 \
    scikit-learn==1.3.0 \
    xgboost==1.7.6 \
    lightgbm==4.0.0

# Copy ML code
COPY src/ml/ ./src/ml/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

RUN chown -R appuser:appuser /app
USER appuser

# Health check for ML service
HEALTHCHECK --interval=120s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "from src.ml.model_manager import ModelManager; print('ML service healthy')"

CMD ["python", "-m", "src.ml.model_manager"]

# Risk management service
FROM base as risk-service

# Copy risk management code
COPY src/risk/ ./src/risk/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

RUN chown -R appuser:appuser /app
USER appuser

# Health check for risk service
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from src.risk.risk_manager import RiskManager; print('Risk service healthy')"

CMD ["python", "-m", "src.risk.risk_manager"]

# Execution service
FROM base as execution-service

# Copy execution code
COPY src/execution/ ./src/execution/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

RUN chown -R appuser:appuser /app
USER appuser

# Health check for execution service
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "from src.execution.order_manager import OrderManager; print('Execution service healthy')"

CMD ["python", "-m", "src.execution.order_manager"]

# Monitoring service
FROM base as monitoring-service

# Install monitoring dependencies
RUN pip install \
    prometheus-client==0.17.1 \
    grafana-api==1.0.3

# Copy monitoring code
COPY src/monitoring/ ./src/monitoring/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

RUN chown -R appuser:appuser /app
USER appuser

# Health check for monitoring service
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "from src.monitoring.system_monitor import SystemMonitor; print('Monitoring service healthy')"

# Expose monitoring port
EXPOSE 9090

CMD ["python", "-m", "src.monitoring.system_monitor"]

# Create directories
RUN mkdir -p logs data

# Create non-root user
RUN useradd --create-home --shell /bin/bash trader
RUN chown -R trader:trader /app
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "src/main.py"]