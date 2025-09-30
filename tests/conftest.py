"""
Pytest Configuration and Shared Fixtures

Provides common fixtures and configurations for all tests.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import os

# Pytest configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")  
    config.addinivalue_line("markers", "backtesting: Backtesting framework tests")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "simulation: Paper trading simulation tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "async: Asynchronous tests")

# Test data imports
from tests.fixtures.mock_data import (
    MockMarketData,
    MockPortfolioData,
    MockSignalData,
    MockOrderData
)

from tests.fixtures.test_config import TestConfigManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfigManager()


@pytest.fixture
def mock_market_data():
    """Provide mock market data."""
    return MockMarketData()


@pytest.fixture
def mock_portfolio_data():
    """Provide mock portfolio data."""
    return MockPortfolioData()


@pytest.fixture
def mock_signal_data():
    """Provide mock trading signals."""
    return MockSignalData()


@pytest.fixture
def mock_order_data():
    """Provide mock order data."""
    return MockOrderData()


@pytest.fixture
def temp_directory():
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        end=datetime.now(),
        freq='1h'
    )
    
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, len(dates))
    cumulative_changes = np.cumsum(price_changes)
    
    close_prices = base_price * (1 + cumulative_changes)
    
    # Generate OHLCV data
    data = {
        'timestamp': dates,
        'open': close_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': close_prices,
        'volume': np.random.randint(10000, 100000, len(dates))
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Generate sample feature data for ML testing."""
    df = sample_ohlcv_data.copy()
    
    # Add technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = 50 + np.random.normal(0, 20, len(df))
    df['macd'] = np.random.normal(0, 0.5, len(df))
    df['bb_upper'] = df['close'] * 1.02
    df['bb_lower'] = df['close'] * 0.98
    
    # Add returns
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Clean data
    df = df.dropna()
    
    return df


@pytest.fixture
def mock_broker_api():
    """Mock broker API for testing."""
    broker = Mock()
    broker.connect = AsyncMock(return_value=True)
    broker.disconnect = AsyncMock(return_value=True)
    broker.get_account_info = AsyncMock(return_value={
        'buying_power': 10000.0,
        'cash': 5000.0,
        'portfolio_value': 15000.0
    })
    broker.get_positions = AsyncMock(return_value=[])
    broker.submit_order = AsyncMock(return_value={'order_id': 'test_123'})
    broker.get_order_status = AsyncMock(return_value={'status': 'filled'})
    
    return broker


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = Mock()
    db.connect = AsyncMock(return_value=True)
    db.disconnect = AsyncMock(return_value=True)
    db.execute = AsyncMock(return_value=True)
    db.fetch_all = AsyncMock(return_value=[])
    db.fetch_one = AsyncMock(return_value=None)
    
    return db


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        'total_return': 0.15,
        'annual_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'volatility': 0.16,
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'trades_count': 150,
        'average_trade': 0.001
    }


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    
    return logger


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Set test environment variables
    monkeypatch.setenv('TESTING', 'true')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
    
    # Disable real API calls in tests
    monkeypatch.setenv('DISABLE_EXTERNAL_APIS', 'true')
    
    yield


@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30  # seconds


class TestBase:
    """Base class for all tests with common utilities."""
    
    @staticmethod
    def assert_near_equal(actual, expected, tolerance=0.001):
        """Assert two values are nearly equal within tolerance."""
        assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual}"
    
    @staticmethod
    def assert_dataframe_equal(df1, df2, check_dtype=False):
        """Assert two DataFrames are equal."""
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    
    @staticmethod
    def create_mock_async_context_manager(return_value=None):
        """Create a mock async context manager."""
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=return_value or mock)
        mock.__aexit__ = AsyncMock(return_value=None)
        return mock


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.asyncio
]