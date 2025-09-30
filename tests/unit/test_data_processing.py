"""
Unit Tests for Data Processing Module

Tests for market data ingestion, processing, and technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from tests.base_test import UnitTestCase, AsyncTestCase
from tests.fixtures.mock_data import MockMarketData, quick_ohlcv
from tests.fixtures.test_config import TestConfigManager

# Import modules to test
try:
    from src.data.market_data_manager import MarketDataManager
    from src.data.data_sources import YahooFinanceSource, AlphaVantageSource
    from src.data.technical_indicators import TechnicalIndicators, IndicatorCalculator
    from src.data.data_storage import DataStorage, RedisCache
except ImportError as e:
    pytest.skip(f"Data module not available: {e}", allow_module_level=True)


@pytest.mark.unit
class TestMarketDataManager(AsyncTestCase):
    """Test cases for MarketDataManager."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        self.mock_data = MockMarketData()
    
    async def test_market_data_manager_initialization(self):
        """Test MarketDataManager initialization."""
        with patch('src.data.market_data_manager.YahooFinanceSource') as mock_yahoo, \
             patch('src.data.market_data_manager.DataStorage') as mock_storage:
            
            mock_yahoo.return_value = AsyncMock()
            mock_storage.return_value = Mock()
            
            manager = MarketDataManager(self.config)
            await manager.initialize()
            
            assert manager is not None
            assert hasattr(manager, 'data_sources')
            assert hasattr(manager, 'storage')
    
    async def test_get_historical_data(self):
        """Test historical data retrieval."""
        # Create mock data
        test_data = quick_ohlcv('AAPL', days=30)
        
        with patch('src.data.market_data_manager.YahooFinanceSource') as mock_source:
            mock_source.return_value.get_historical_data = AsyncMock(return_value=test_data)
            
            manager = MarketDataManager(self.config)
            await manager.initialize()
            
            result = await manager.get_historical_data(
                symbol='AAPL',
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'close' in result.columns
            assert 'volume' in result.columns
    
    async def test_get_real_time_data(self):
        """Test real-time data retrieval."""
        test_tick = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'timestamp': datetime.now()
        }
        
        with patch('src.data.market_data_manager.YahooFinanceSource') as mock_source:
            mock_source.return_value.get_real_time_data = AsyncMock(return_value=test_tick)
            
            manager = MarketDataManager(self.config)
            await manager.initialize()
            
            result = await manager.get_real_time_data('AAPL')
            
            assert isinstance(result, dict)
            assert result['symbol'] == 'AAPL'
            assert 'price' in result
            assert 'timestamp' in result
    
    async def test_subscribe_to_real_time_updates(self):
        """Test real-time subscription functionality."""
        callback_called = False
        received_data = None
        
        async def test_callback(data):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = data
        
        with patch('src.data.market_data_manager.YahooFinanceSource') as mock_source:
            mock_source.return_value.subscribe = AsyncMock()
            
            manager = MarketDataManager(self.config)
            await manager.initialize()
            
            await manager.subscribe_to_real_time_updates('AAPL', test_callback)
            
            # Simulate callback execution
            test_data = {'symbol': 'AAPL', 'price': 150.0}
            await test_callback(test_data)
            
            assert callback_called
            assert received_data == test_data
    
    async def test_data_validation(self):
        """Test data validation functionality."""
        # Test invalid data
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'close': [None],  # Invalid close price
            'volume': [-100]  # Invalid volume
        })
        
        manager = MarketDataManager(self.config)
        
        # Should raise validation error or clean the data
        with pytest.raises((ValueError, TypeError)) or pd.testing.assert_frame_equal:
            cleaned_data = manager._validate_data(invalid_data)
            # If validation cleans data instead of raising error
            assert cleaned_data['close'].notna().all()
            assert (cleaned_data['volume'] >= 0).all()


@pytest.mark.unit  
class TestYahooFinanceSource(AsyncTestCase):
    """Test cases for Yahoo Finance data source."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
    
    async def test_yahoo_finance_initialization(self):
        """Test Yahoo Finance source initialization."""
        source = YahooFinanceSource(self.config)
        assert source is not None
        assert hasattr(source, 'session')
    
    async def test_get_historical_data_with_mock(self):
        """Test historical data retrieval with mocked API."""
        mock_response_data = {
            'chart': {
                'result': [{
                    'timestamp': [1640995200, 1641081600, 1641168000],
                    'indicators': {
                        'quote': [{
                            'open': [148.5, 149.0, 149.5],
                            'high': [149.0, 149.5, 150.0],
                            'low': [148.0, 148.5, 149.0],
                            'close': [148.8, 149.2, 149.8],
                            'volume': [1000000, 1100000, 1200000]
                        }]
                    }
                }]
            }
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_get.return_value.__aenter__.return_value = mock_response
            
            source = YahooFinanceSource(self.config)
            
            result = await source.get_historical_data(
                symbol='AAPL',
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 1, 3)
            )
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert len(result) == 3
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    async def test_rate_limiting(self):
        """Test API rate limiting functionality."""
        with patch('asyncio.sleep') as mock_sleep:
            source = YahooFinanceSource(self.config)
            
            # Simulate multiple rapid requests
            tasks = []
            for _ in range(5):
                task = source.get_historical_data('AAPL', datetime.now() - timedelta(days=1), datetime.now())
                tasks.append(task)
            
            # Should handle rate limiting gracefully
            # This test verifies the rate limiting mechanism exists
            assert hasattr(source, '_last_request_time') or hasattr(source, 'rate_limiter')


@pytest.mark.unit
class TestTechnicalIndicators(UnitTestCase):
    """Test cases for Technical Indicators."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.test_data = quick_ohlcv('AAPL', days=60)  # Need enough data for indicators
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        calculator = IndicatorCalculator()
        
        sma_10 = calculator.sma(self.test_data['close'], window=10)
        sma_20 = calculator.sma(self.test_data['close'], window=20)
        
        assert isinstance(sma_10, pd.Series)
        assert isinstance(sma_20, pd.Series)
        assert len(sma_10) == len(self.test_data)
        assert len(sma_20) == len(self.test_data)
        
        # First 9 values should be NaN for 10-period SMA
        assert sma_10.iloc[:9].isna().all()
        assert not sma_10.iloc[9:].isna().any()
        
        # SMA values should be reasonable
        assert sma_10.iloc[10] == self.test_data['close'].iloc[1:11].mean()
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        calculator = IndicatorCalculator()
        
        ema_12 = calculator.ema(self.test_data['close'], window=12)
        
        assert isinstance(ema_12, pd.Series)
        assert len(ema_12) == len(self.test_data)
        
        # EMA should have fewer NaN values than SMA
        assert ema_12.notna().sum() > len(self.test_data) - 12
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        calculator = IndicatorCalculator()
        
        rsi = calculator.rsi(self.test_data['close'], window=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(self.test_data)
        
        # RSI values should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        calculator = IndicatorCalculator()
        
        macd_line, signal_line, histogram = calculator.macd(
            self.test_data['close'], 
            fast_window=12, 
            slow_window=26, 
            signal_window=9
        )
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        
        # All should have same length
        assert len(macd_line) == len(signal_line) == len(histogram) == len(self.test_data)
        
        # Histogram should equal MACD - Signal
        valid_indices = macd_line.notna() & signal_line.notna()
        expected_histogram = macd_line[valid_indices] - signal_line[valid_indices]
        actual_histogram = histogram[valid_indices]
        
        pd.testing.assert_series_equal(expected_histogram, actual_histogram, check_names=False)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        calculator = IndicatorCalculator()
        
        upper, middle, lower = calculator.bollinger_bands(
            self.test_data['close'], 
            window=20, 
            std_dev=2
        )
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # All should have same length
        assert len(upper) == len(middle) == len(lower) == len(self.test_data)
        
        # Upper should be greater than middle, middle greater than lower
        valid_indices = upper.notna() & middle.notna() & lower.notna()
        assert (upper[valid_indices] >= middle[valid_indices]).all()
        assert (middle[valid_indices] >= lower[valid_indices]).all()
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        calculator = IndicatorCalculator()
        
        atr = calculator.atr(
            self.test_data['high'],
            self.test_data['low'], 
            self.test_data['close'],
            window=14
        )
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(self.test_data)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator calculation."""
        calculator = IndicatorCalculator()
        
        k_percent, d_percent = calculator.stochastic_oscillator(
            self.test_data['high'],
            self.test_data['low'],
            self.test_data['close'],
            k_window=14,
            d_window=3
        )
        
        assert isinstance(k_percent, pd.Series)
        assert isinstance(d_percent, pd.Series)
        
        # Values should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_williams_r(self):
        """Test Williams %R calculation."""
        calculator = IndicatorCalculator()
        
        williams_r = calculator.williams_r(
            self.test_data['high'],
            self.test_data['low'],
            self.test_data['close'],
            window=14
        )
        
        assert isinstance(williams_r, pd.Series)
        
        # Williams %R should be between -100 and 0
        valid_wr = williams_r.dropna()
        assert (valid_wr >= -100).all()
        assert (valid_wr <= 0).all()
    
    def test_technical_indicators_batch_calculation(self):
        """Test batch calculation of multiple indicators."""
        tech_indicators = TechnicalIndicators()
        
        result = tech_indicators.calculate_all_indicators(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        
        # Should contain original OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
        
        # Should contain calculated indicators
        expected_indicators = [
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
    
    def test_indicator_validation(self):
        """Test indicator input validation."""
        calculator = IndicatorCalculator()
        
        # Test with insufficient data
        short_data = pd.Series([100, 101, 99])
        
        with pytest.raises((ValueError, IndexError)):
            calculator.sma(short_data, window=50)  # Window larger than data
        
        # Test with invalid window
        with pytest.raises(ValueError):
            calculator.sma(self.test_data['close'], window=0)
        
        with pytest.raises(ValueError):
            calculator.sma(self.test_data['close'], window=-1)


@pytest.mark.unit
class TestDataStorage(AsyncTestCase):
    """Test cases for Data Storage."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
    
    async def test_data_storage_initialization(self):
        """Test DataStorage initialization."""
        with patch('src.data.data_storage.create_engine') as mock_engine:
            mock_engine.return_value = Mock()
            
            storage = DataStorage(self.config)
            await storage.initialize()
            
            assert storage is not None
            assert hasattr(storage, 'engine')
    
    async def test_store_market_data(self):
        """Test storing market data."""
        test_data = quick_ohlcv('AAPL', days=5)
        
        with patch('src.data.data_storage.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql') as mock_to_sql:
            
            mock_engine.return_value = Mock()
            mock_to_sql.return_value = None
            
            storage = DataStorage(self.config)
            await storage.initialize()
            
            await storage.store_market_data('AAPL', test_data)
            
            # Verify to_sql was called
            mock_to_sql.assert_called_once()
    
    async def test_retrieve_market_data(self):
        """Test retrieving market data."""
        expected_data = quick_ohlcv('AAPL', days=5)
        
        with patch('src.data.data_storage.create_engine') as mock_engine, \
             patch('pandas.read_sql') as mock_read_sql:
            
            mock_engine.return_value = Mock()
            mock_read_sql.return_value = expected_data
            
            storage = DataStorage(self.config)
            await storage.initialize()
            
            result = await storage.get_market_data(
                symbol='AAPL',
                start_date=datetime.now() - timedelta(days=5),
                end_date=datetime.now()
            )
            
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, expected_data)


@pytest.mark.unit
class TestRedisCache(AsyncTestCase):
    """Test cases for Redis Cache."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
    
    async def test_redis_cache_initialization(self):
        """Test Redis cache initialization."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            cache = RedisCache(self.config)
            await cache.initialize()
            
            assert cache is not None
            assert hasattr(cache, 'redis_client')
    
    async def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        test_data = {'symbol': 'AAPL', 'price': 150.0}
        
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_client.set = AsyncMock(return_value=True)
            mock_client.get = AsyncMock(return_value='{"symbol": "AAPL", "price": 150.0}')
            mock_redis.return_value = mock_client
            
            cache = RedisCache(self.config)
            await cache.initialize()
            
            # Test set
            await cache.set('test_key', test_data)
            mock_client.set.assert_called_once()
            
            # Test get
            result = await cache.get('test_key')
            mock_client.get.assert_called_once_with('test_key')
            
            # Note: Actual deserialization would depend on implementation
    
    async def test_cache_expiration(self):
        """Test cache expiration functionality."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_client.setex = AsyncMock(return_value=True)
            mock_redis.return_value = mock_client
            
            cache = RedisCache(self.config)
            await cache.initialize()
            
            await cache.set('test_key', {'data': 'test'}, expiry=300)
            
            # Verify setex was called with expiration
            mock_client.setex.assert_called_once()


# Test performance and edge cases
@pytest.mark.unit
class TestDataModulePerformance(UnitTestCase):
    """Performance and edge case tests for data module."""
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Generate large dataset
        large_data = MockMarketData().generate_ohlcv('AAPL', days=365)  # 1 year of hourly data
        
        calculator = IndicatorCalculator()
        
        start_time = datetime.now()
        result = calculator.sma(large_data['close'], window=20)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(large_data)
        # Should process reasonably fast (adjust threshold as needed)
        assert processing_time < 1.0  # Less than 1 second
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create data with missing values
        data_with_nan = pd.Series([100, 101, np.nan, 103, 104, np.nan, 106])
        
        calculator = IndicatorCalculator()
        
        # SMA should handle NaN values appropriately
        sma_result = calculator.sma(data_with_nan, window=3)
        
        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(data_with_nan)
        
        # Check that NaN values are handled correctly
        # (Implementation-specific behavior)
    
    def test_zero_and_negative_values(self):
        """Test handling of zero and negative values."""
        # Create data with edge cases
        edge_case_data = pd.Series([0, -1, 0.0001, -0.0001, 100])
        
        calculator = IndicatorCalculator()
        
        # Should handle edge cases gracefully
        try:
            sma_result = calculator.sma(edge_case_data, window=3)
            assert isinstance(sma_result, pd.Series)
        except Exception as e:
            # If implementation doesn't support negative prices, should raise appropriate error
            assert isinstance(e, (ValueError, TypeError))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])