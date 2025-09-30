"""
Base Test Classes

Provides base classes and utilities for different types of tests.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class BaseTestCase:
    """Base test case with common utilities."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.start_time = datetime.now()
    
    def teardown_method(self):
        """Teardown method called after each test."""
        self.end_time = datetime.now()
        self.test_duration = self.end_time - self.start_time
    
    def assert_near_equal(self, actual: float, expected: float, tolerance: float = 0.001):
        """Assert two floats are nearly equal within tolerance."""
        assert abs(actual - expected) <= tolerance, (
            f"Expected {expected}, got {actual} (tolerance: {tolerance})"
        )
    
    def assert_percentage_near(self, actual: float, expected: float, percentage: float = 1.0):
        """Assert two values are within a percentage of each other."""
        tolerance = abs(expected) * (percentage / 100.0)
        self.assert_near_equal(actual, expected, tolerance)
    
    def assert_dataframe_structure(self, df: pd.DataFrame, expected_columns: List[str]):
        """Assert DataFrame has expected structure."""
        assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        
        for col in expected_columns:
            assert col in df.columns, f"Column '{col}' missing from DataFrame"
    
    def create_sample_ohlcv(self, days: int = 30, freq: str = '1h') -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq=freq
        )
        
        np.random.seed(42)
        base_price = 100.0
        
        # Generate price walk
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })


class AsyncTestCase(BaseTestCase):
    """Base test case for async testing."""
    
    async def setup_async(self):
        """Async setup method - override in subclasses."""
        pass
    
    async def teardown_async(self):
        """Async teardown method - override in subclasses."""
        pass
    
    def create_mock_async_method(self, return_value=None, side_effect=None):
        """Create a mock async method."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    async def wait_for_condition(self, condition_func, timeout: float = 5.0, 
                                interval: float = 0.1) -> bool:
        """Wait for a condition to become true."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False


class UnitTestCase(BaseTestCase):
    """Base class for unit tests."""
    
    def setup_method(self):
        """Setup for unit tests."""
        super().setup_method()
        self.mocks = {}
    
    def create_mock(self, name: str, **kwargs) -> Mock:
        """Create and register a mock object."""
        mock = Mock(**kwargs)
        self.mocks[name] = mock
        return mock
    
    def create_async_mock(self, name: str, **kwargs) -> AsyncMock:
        """Create and register an async mock object."""
        mock = AsyncMock(**kwargs)
        self.mocks[name] = mock
        return mock
    
    def assert_mock_called_with(self, mock_name: str, *args, **kwargs):
        """Assert a mock was called with specific arguments."""
        assert mock_name in self.mocks, f"Mock '{mock_name}' not found"
        self.mocks[mock_name].assert_called_with(*args, **kwargs)


class IntegrationTestCase(AsyncTestCase):
    """Base class for integration tests."""
    
    def setup_method(self):
        """Setup for integration tests."""
        super().setup_method()
        self.components = {}
        self.cleanup_tasks = []
    
    async def setup_component(self, name: str, component_class, config: Dict[str, Any]):
        """Setup a component for integration testing."""
        component = component_class(config)
        if hasattr(component, 'initialize'):
            await component.initialize()
        
        self.components[name] = component
        
        # Register cleanup if needed
        if hasattr(component, 'shutdown'):
            self.cleanup_tasks.append(component.shutdown)
    
    async def teardown_async(self):
        """Cleanup integration test components."""
        for cleanup_task in reversed(self.cleanup_tasks):
            try:
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    cleanup_task()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        self.components.clear()
        self.cleanup_tasks.clear()


class BacktestingTestCase(BaseTestCase):
    """Base class for backtesting tests."""
    
    def setup_method(self):
        """Setup for backtesting tests."""
        super().setup_method()
        self.backtest_config = {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 12, 31),
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005
        }
    
    def create_backtest_data(self, symbols: List[str], start_date: datetime, 
                           end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Create sample data for backtesting."""
        data = {}
        
        for symbol in symbols:
            # Generate realistic price data
            dates = pd.date_range(start=start_date, end=end_date, freq='1D')
            np.random.seed(hash(symbol) % 2**32)
            
            base_price = np.random.uniform(50, 200)
            returns = np.random.normal(0, 0.02, len(dates))
            prices = base_price * np.cumprod(1 + returns)
            
            data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'close': prices,
                'volume': np.random.randint(10000, 100000, len(dates))
            })
        
        return data
    
    def assert_backtest_results(self, results: Dict[str, Any]):
        """Assert backtesting results have expected structure."""
        required_fields = [
            'total_return', 'annual_return', 'sharpe_ratio', 
            'max_drawdown', 'win_rate', 'total_trades'
        ]
        
        for field in required_fields:
            assert field in results, f"Missing required field: {field}"
            assert isinstance(results[field], (int, float)), f"Field {field} should be numeric"


class PerformanceTestCase(BaseTestCase):
    """Base class for performance tests."""
    
    def setup_method(self):
        """Setup for performance tests."""
        super().setup_method()
        self.performance_thresholds = {
            'max_execution_time': 1.0,  # seconds
            'max_memory_usage': 100,    # MB
            'min_throughput': 1000      # operations per second
        }
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure function execution time."""
        start_time = datetime.now()
        
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return result, execution_time
    
    def assert_execution_time(self, execution_time: float, max_time: float = None):
        """Assert execution time is within acceptable limits."""
        threshold = max_time or self.performance_thresholds['max_execution_time']
        assert execution_time <= threshold, (
            f"Execution time {execution_time:.3f}s exceeded threshold {threshold}s"
        )
    
    async def measure_throughput(self, async_func, operations: int, *args, **kwargs):
        """Measure throughput of an async function."""
        start_time = datetime.now()
        
        tasks = [async_func(*args, **kwargs) for _ in range(operations)]
        await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        throughput = operations / duration if duration > 0 else 0
        
        return throughput


class MockDataProvider:
    """Provider for mock data used in tests."""
    
    @staticmethod
    def create_market_data(symbol: str = 'AAPL', days: int = 30) -> pd.DataFrame:
        """Create mock market data."""
        return BaseTestCase().create_sample_ohlcv(days=days)
    
    @staticmethod
    def create_portfolio_data() -> Dict[str, Any]:
        """Create mock portfolio data."""
        return {
            'cash': 10000.0,
            'positions': {
                'AAPL': {'quantity': 10, 'avg_price': 150.0},
                'MSFT': {'quantity': 5, 'avg_price': 300.0}
            },
            'total_value': 25000.0,
            'unrealized_pnl': 2000.0
        }
    
    @staticmethod
    def create_signal_data() -> Dict[str, Any]:
        """Create mock trading signal data."""
        return {
            'symbol': 'AAPL',
            'signal_type': 'BUY',
            'strength': 0.75,
            'confidence': 0.82,
            'timestamp': datetime.now(),
            'features': {
                'rsi': 35.0,
                'macd': 0.5,
                'volume_ratio': 1.2
            }
        }
    
    @staticmethod
    def create_order_data() -> Dict[str, Any]:
        """Create mock order data."""
        return {
            'order_id': 'test_order_123',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'order_type': 'MARKET',
            'status': 'FILLED',
            'fill_price': 150.25,
            'timestamp': datetime.now()
        }


# Test decorators
def slow_test(func):
    """Mark a test as slow running."""
    return pytest.mark.slow(func)


def requires_api(func):
    """Mark a test as requiring external API access."""
    return pytest.mark.api(func)


def requires_database(func):
    """Mark a test as requiring database access."""
    return pytest.mark.database(func)


def requires_broker(func):
    """Mark a test as requiring broker API access."""
    return pytest.mark.broker(func)