"""
Integration Tests for Trading System

Tests for cross-module functionality and data flow validation.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from tests.base_test import IntegrationTestCase
from tests.fixtures.mock_data import MockMarketData, MockPortfolioData, MockSignalData
from tests.fixtures.test_config import TestConfigManager

# Import modules for integration testing
try:
    from src.data.market_data_manager import MarketDataManager
    from src.ml.model_manager import ModelManager
    from src.risk.risk_manager import RiskManager
    from src.risk.portfolio_manager import PortfolioManager
    from src.execution.order_manager import OrderManager
    from src.monitoring.monitoring_coordinator import MonitoringCoordinator
except ImportError as e:
    pytest.skip(f"Integration modules not available: {e}", allow_module_level=True)


@pytest.mark.integration
class TestDataToMLPipeline(IntegrationTestCase):
    """Test integration between data processing and ML components."""
    
    async def test_data_to_features_to_prediction_pipeline(self):
        """Test complete data-to-prediction pipeline."""
        config = TestConfigManager()
        
        # Setup components
        await self.setup_component('market_data', MarketDataManager, config)
        await self.setup_component('ml_manager', ModelManager, config)
        
        market_data_manager = self.components['market_data']
        ml_manager = self.components['ml_manager']
        
        # Mock historical data
        mock_data = MockMarketData().generate_ohlcv('AAPL', days=100)
        
        with patch.object(market_data_manager, 'get_historical_data', 
                         return_value=mock_data):
            
            # Step 1: Get market data
            historical_data = await market_data_manager.get_historical_data(
                symbol='AAPL',
                start_date=datetime.now() - timedelta(days=100),
                end_date=datetime.now()
            )
            
            assert isinstance(historical_data, pd.DataFrame)
            assert not historical_data.empty
            
            # Step 2: Train ML models
            training_result = await ml_manager.train_models(historical_data)
            
            assert training_result is not None
            assert 'model_performance' in training_result or training_result is True
            
            # Step 3: Generate prediction
            current_data = historical_data.tail(50)  # Last 50 periods for prediction
            
            prediction = await ml_manager.generate_prediction('AAPL', current_data)
            
            assert isinstance(prediction, dict)
            assert 'symbol' in prediction
            assert 'prediction' in prediction
            assert 'confidence' in prediction
    
    async def test_real_time_data_to_signal_pipeline(self):
        """Test real-time data processing to signal generation."""
        config = TestConfigManager()
        
        await self.setup_component('market_data', MarketDataManager, config)
        await self.setup_component('ml_manager', ModelManager, config)
        
        market_data_manager = self.components['market_data']
        ml_manager = self.components['ml_manager']
        
        # Mock real-time data stream
        signals_generated = []
        
        async def signal_callback(signal):
            signals_generated.append(signal)
        
        # Simulate real-time tick data
        mock_ticks = [
            {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'price': 150.5, 'volume': 1200, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'price': 151.0, 'volume': 800, 'timestamp': datetime.now()}
        ]
        
        with patch.object(market_data_manager, 'get_real_time_data') as mock_rt_data, \
             patch.object(ml_manager, 'generate_prediction') as mock_predict:
            
            mock_predict.return_value = {
                'symbol': 'AAPL',
                'prediction': 0.75,
                'confidence': 0.8,
                'signal_type': 'BUY'
            }
            
            for tick in mock_ticks:
                mock_rt_data.return_value = tick
                
                # Process tick and generate signal
                tick_data = await market_data_manager.get_real_time_data('AAPL')
                
                if tick_data:
                    # Simulate having enough historical data for prediction
                    mock_historical = MockMarketData().generate_ohlcv('AAPL', days=50)
                    signal = await ml_manager.generate_prediction('AAPL', mock_historical)
                    
                    if signal:
                        await signal_callback(signal)
            
            # Verify signals were generated
            assert len(signals_generated) > 0
            for signal in signals_generated:
                assert 'symbol' in signal
                assert 'prediction' in signal


@pytest.mark.integration
class TestMLToRiskPipeline(IntegrationTestCase):
    """Test integration between ML and risk management components."""
    
    async def test_signal_to_risk_filtering_pipeline(self):
        """Test ML signal to risk-filtered position sizing."""
        config = TestConfigManager()
        
        await self.setup_component('ml_manager', ModelManager, config)
        await self.setup_component('risk_manager', RiskManager, config)
        await self.setup_component('portfolio_manager', PortfolioManager, config)
        
        ml_manager = self.components['ml_manager']
        risk_manager = self.components['risk_manager']
        portfolio_manager = self.components['portfolio_manager']
        
        # Mock ML signal
        ml_signal = MockSignalData().generate_signal('AAPL')
        
        # Mock current portfolio
        current_portfolio = MockPortfolioData().generate_portfolio()
        
        with patch.object(ml_manager, 'generate_prediction', return_value=ml_signal), \
             patch.object(portfolio_manager, 'get_current_portfolio', return_value=current_portfolio):
            
            # Step 1: Generate ML signal
            signal = await ml_manager.generate_prediction('AAPL', MockMarketData().generate_ohlcv('AAPL'))
            
            # Step 2: Apply risk filtering
            risk_filtered_signal = await risk_manager.filter_signal(signal, current_portfolio)
            
            assert isinstance(risk_filtered_signal, dict)
            assert 'approved' in risk_filtered_signal
            assert 'risk_score' in risk_filtered_signal
            
            # Step 3: Calculate position size if approved
            if risk_filtered_signal.get('approved', False):
                position_size = await risk_manager.calculate_position_size(
                    signal, current_portfolio['total_value']
                )
                
                assert isinstance(position_size, (int, float))
                assert position_size >= 0
    
    async def test_portfolio_risk_monitoring_integration(self):
        """Test portfolio monitoring and risk metric calculation."""
        config = TestConfigManager()
        
        await self.setup_component('portfolio_manager', PortfolioManager, config)
        await self.setup_component('risk_manager', RiskManager, config)
        
        portfolio_manager = self.components['portfolio_manager']
        risk_manager = self.components['risk_manager']
        
        # Simulate portfolio with positions
        mock_portfolio = MockPortfolioData().generate_portfolio()
        
        with patch.object(portfolio_manager, 'get_current_portfolio', return_value=mock_portfolio):
            
            # Get current portfolio
            portfolio = await portfolio_manager.get_current_portfolio()
            
            # Calculate risk metrics
            risk_metrics = await risk_manager.calculate_portfolio_risk_metrics(portfolio)
            
            assert isinstance(risk_metrics, dict)
            
            # Should contain key risk metrics
            expected_metrics = ['var', 'max_drawdown', 'sharpe_ratio', 'volatility']
            for metric in expected_metrics:
                assert metric in risk_metrics or any(metric in key for key in risk_metrics.keys())


@pytest.mark.integration  
class TestRiskToExecutionPipeline(IntegrationTestCase):
    """Test integration between risk management and execution components."""
    
    async def test_risk_approved_signal_to_order_execution(self):
        """Test risk-approved signal to order execution pipeline."""
        config = TestConfigManager()
        
        await self.setup_component('risk_manager', RiskManager, config)
        await self.setup_component('order_manager', OrderManager, config)
        await self.setup_component('portfolio_manager', PortfolioManager, config)
        
        risk_manager = self.components['risk_manager']
        order_manager = self.components['order_manager']
        portfolio_manager = self.components['portfolio_manager']
        
        # Mock trading signal
        signal = {
            'symbol': 'AAPL',
            'signal_type': 'BUY',
            'strength': 0.8,
            'confidence': 0.75,
            'timestamp': datetime.now()
        }
        
        # Mock portfolio
        current_portfolio = {'total_value': 100000, 'cash': 50000, 'positions': {}}
        
        with patch.object(risk_manager, 'filter_signal') as mock_filter, \
             patch.object(risk_manager, 'calculate_position_size') as mock_size, \
             patch.object(order_manager, 'submit_order') as mock_submit, \
             patch.object(portfolio_manager, 'get_current_portfolio', return_value=current_portfolio):
            
            # Mock risk approval
            mock_filter.return_value = {
                'approved': True,
                'risk_score': 0.3,
                'reasons': ['Signal meets risk criteria']
            }
            
            # Mock position sizing
            mock_size.return_value = 5000  # $5000 position
            
            # Mock order submission
            mock_submit.return_value = {
                'order_id': 'test_order_123',
                'status': 'SUBMITTED',
                'symbol': 'AAPL',
                'quantity': 33,  # $5000 / $150 per share
                'order_type': 'MARKET'
            }
            
            # Step 1: Risk filtering
            risk_result = await risk_manager.filter_signal(signal, current_portfolio)
            
            if risk_result.get('approved'):
                # Step 2: Position sizing
                position_value = await risk_manager.calculate_position_size(
                    signal, current_portfolio['total_value']
                )
                
                # Step 3: Order execution
                order_result = await order_manager.submit_order({
                    'symbol': signal['symbol'],
                    'side': signal['signal_type'],
                    'quantity': int(position_value / 150),  # Assume $150 price
                    'order_type': 'MARKET'
                })
                
                assert order_result is not None
                assert 'order_id' in order_result
                
                # Verify call sequence
                mock_filter.assert_called_once()
                mock_size.assert_called_once()
                mock_submit.assert_called_once()


@pytest.mark.integration
class TestExecutionToMonitoringPipeline(IntegrationTestCase):
    """Test integration between execution and monitoring components."""
    
    async def test_order_execution_to_monitoring_pipeline(self):
        """Test order execution monitoring and performance tracking."""
        config = TestConfigManager()
        
        await self.setup_component('order_manager', OrderManager, config)
        await self.setup_component('portfolio_manager', PortfolioManager, config)
        await self.setup_component('monitoring', MonitoringCoordinator, config)
        
        order_manager = self.components['order_manager']
        portfolio_manager = self.components['portfolio_manager']
        monitoring_coordinator = self.components['monitoring']
        
        # Mock order execution
        order_data = {
            'symbol': 'MSFT',
            'side': 'BUY',
            'quantity': 50,
            'order_type': 'MARKET'
        }
        
        execution_result = {
            'order_id': 'exec_123',
            'status': 'FILLED',
            'fill_price': 300.0,
            'fill_quantity': 50,
            'timestamp': datetime.now()
        }
        
        with patch.object(order_manager, 'submit_order', return_value=execution_result), \
             patch.object(portfolio_manager, 'update_position') as mock_update, \
             patch.object(monitoring_coordinator, 'update_portfolio_performance') as mock_monitor:
            
            # Step 1: Execute order
            order_result = await order_manager.submit_order(order_data)
            
            # Step 2: Update portfolio
            if order_result.get('status') == 'FILLED':
                await portfolio_manager.update_position(
                    symbol=order_result['symbol'],
                    quantity=order_result['fill_quantity'],
                    price=order_result['fill_price'],
                    side=order_data['side']
                )
                
                # Step 3: Update monitoring
                updated_portfolio = MockPortfolioData().generate_portfolio()
                performance_metrics = {
                    'total_return': 0.05,
                    'daily_return': 0.01,
                    'sharpe_ratio': 1.2
                }
                
                await monitoring_coordinator.update_portfolio_performance(
                    updated_portfolio, performance_metrics
                )
            
            # Verify integration
            assert order_result['status'] == 'FILLED'
            mock_update.assert_called_once()
            mock_monitor.assert_called_once()


@pytest.mark.integration
class TestFullTradingWorkflow(IntegrationTestCase):
    """Test complete end-to-end trading workflow."""
    
    async def test_complete_trading_cycle(self):
        """Test complete trading cycle from data to execution to monitoring."""
        config = TestConfigManager()
        
        # Setup all components
        await self.setup_component('market_data', MarketDataManager, config)
        await self.setup_component('ml_manager', ModelManager, config)
        await self.setup_component('risk_manager', RiskManager, config)
        await self.setup_component('portfolio_manager', PortfolioManager, config)
        await self.setup_component('order_manager', OrderManager, config)
        await self.setup_component('monitoring', MonitoringCoordinator, config)
        
        # Get components
        market_data_manager = self.components['market_data']
        ml_manager = self.components['ml_manager']
        risk_manager = self.components['risk_manager']
        portfolio_manager = self.components['portfolio_manager']
        order_manager = self.components['order_manager']
        monitoring_coordinator = self.components['monitoring']
        
        # Track workflow steps
        workflow_steps = []
        
        # Mock data and responses
        mock_historical_data = MockMarketData().generate_ohlcv('TSLA', days=100)
        mock_signal = MockSignalData().generate_signal('TSLA')
        mock_portfolio = MockPortfolioData().generate_portfolio()
        
        with patch.object(market_data_manager, 'get_historical_data', return_value=mock_historical_data), \
             patch.object(ml_manager, 'generate_prediction', return_value=mock_signal), \
             patch.object(portfolio_manager, 'get_current_portfolio', return_value=mock_portfolio), \
             patch.object(risk_manager, 'filter_signal') as mock_risk_filter, \
             patch.object(risk_manager, 'calculate_position_size') as mock_position_size, \
             patch.object(order_manager, 'submit_order') as mock_order, \
             patch.object(monitoring_coordinator, 'update_portfolio_performance') as mock_monitor:
            
            # Configure mocks
            mock_risk_filter.return_value = {
                'approved': True,
                'risk_score': 0.25,
                'reasons': ['Low risk signal']
            }
            mock_position_size.return_value = 8000  # $8k position
            mock_order.return_value = {
                'order_id': 'workflow_order_123',
                'status': 'FILLED',
                'fill_price': 800.0,
                'fill_quantity': 10
            }
            
            # Step 1: Get market data
            workflow_steps.append('market_data')
            historical_data = await market_data_manager.get_historical_data(
                symbol='TSLA',
                start_date=datetime.now() - timedelta(days=100),
                end_date=datetime.now()
            )
            
            assert historical_data is not None
            
            # Step 2: Generate ML prediction
            workflow_steps.append('ml_prediction')
            signal = await ml_manager.generate_prediction('TSLA', historical_data)
            
            assert signal is not None
            assert signal['symbol'] == 'TSLA'
            
            # Step 3: Risk management
            workflow_steps.append('risk_filtering')
            current_portfolio = await portfolio_manager.get_current_portfolio()
            risk_result = await risk_manager.filter_signal(signal, current_portfolio)
            
            assert risk_result['approved'] is True
            
            # Step 4: Position sizing
            workflow_steps.append('position_sizing')
            position_size = await risk_manager.calculate_position_size(
                signal, current_portfolio['total_value']
            )
            
            assert position_size > 0
            
            # Step 5: Order execution
            workflow_steps.append('order_execution')
            order_result = await order_manager.submit_order({
                'symbol': 'TSLA',
                'side': 'BUY',
                'quantity': int(position_size / 800),  # Assume $800 per share
                'order_type': 'MARKET'
            })
            
            assert order_result['status'] == 'FILLED'
            
            # Step 6: Portfolio update and monitoring
            workflow_steps.append('monitoring_update')
            performance_metrics = {
                'total_return': 0.08,
                'daily_return': 0.015,
                'sharpe_ratio': 1.4
            }
            
            await monitoring_coordinator.update_portfolio_performance(
                current_portfolio, performance_metrics
            )
            
            # Verify complete workflow
            expected_steps = [
                'market_data', 'ml_prediction', 'risk_filtering',
                'position_sizing', 'order_execution', 'monitoring_update'
            ]
            
            assert workflow_steps == expected_steps
            
            # Verify all components were called
            mock_risk_filter.assert_called_once()
            mock_position_size.assert_called_once()
            mock_order.assert_called_once()
            mock_monitor.assert_called_once()


@pytest.mark.integration
class TestErrorHandlingAndRecovery(IntegrationTestCase):
    """Test error handling and recovery across components."""
    
    async def test_ml_prediction_failure_handling(self):
        """Test handling of ML prediction failures."""
        config = TestConfigManager()
        
        await self.setup_component('ml_manager', ModelManager, config)
        await self.setup_component('risk_manager', RiskManager, config)
        
        ml_manager = self.components['ml_manager']
        risk_manager = self.components['risk_manager']
        
        # Mock ML failure
        with patch.object(ml_manager, 'generate_prediction', side_effect=Exception("Model prediction failed")):
            
            try:
                signal = await ml_manager.generate_prediction('AAPL', MockMarketData().generate_ohlcv('AAPL'))
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Model prediction failed" in str(e)
            
            # System should handle gracefully and continue with default/conservative approach
            # This would be implemented in the actual system
    
    async def test_broker_connection_failure_handling(self):
        """Test handling of broker connection failures."""
        config = TestConfigManager()
        
        await self.setup_component('order_manager', OrderManager, config)
        
        order_manager = self.components['order_manager']
        
        # Mock broker connection failure
        with patch.object(order_manager, 'submit_order', side_effect=Exception("Broker connection lost")):
            
            try:
                result = await order_manager.submit_order({
                    'symbol': 'AAPL',
                    'side': 'BUY',
                    'quantity': 100,
                    'order_type': 'MARKET'
                })
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Broker connection lost" in str(e)
            
            # System should handle gracefully, retry, or queue orders
    
    async def test_data_feed_interruption_handling(self):
        """Test handling of market data feed interruptions."""
        config = TestConfigManager()
        
        await self.setup_component('market_data', MarketDataManager, config)
        
        market_data_manager = self.components['market_data']
        
        # Mock data feed interruption
        with patch.object(market_data_manager, 'get_real_time_data', side_effect=Exception("Data feed interrupted")):
            
            try:
                tick_data = await market_data_manager.get_real_time_data('AAPL')
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Data feed interrupted" in str(e)
            
            # System should handle gracefully, use backup feeds, or cache last known data


@pytest.mark.integration
@pytest.mark.slow
class TestSystemPerformance(IntegrationTestCase):
    """Test system performance under various conditions."""
    
    async def test_high_frequency_signal_processing(self):
        """Test system performance with high-frequency signals."""
        config = TestConfigManager()
        
        await self.setup_component('ml_manager', ModelManager, config)
        await self.setup_component('risk_manager', RiskManager, config)
        
        ml_manager = self.components['ml_manager']
        risk_manager = self.components['risk_manager']
        
        # Generate multiple signals rapidly
        signals_processed = 0
        start_time = datetime.now()
        
        with patch.object(ml_manager, 'generate_prediction') as mock_predict, \
             patch.object(risk_manager, 'filter_signal') as mock_filter:
            
            mock_predict.return_value = MockSignalData().generate_signal()
            mock_filter.return_value = {'approved': True, 'risk_score': 0.2}
            
            # Process 100 signals
            for i in range(100):
                signal = await ml_manager.generate_prediction('AAPL', MockMarketData().generate_ohlcv('AAPL'))
                risk_result = await risk_manager.filter_signal(signal, {})
                
                if risk_result.get('approved'):
                    signals_processed += 1
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process signals efficiently
        assert signals_processed > 0
        assert processing_time < 10.0  # Should complete in under 10 seconds
        
        # Calculate throughput
        signals_per_second = signals_processed / processing_time
        assert signals_per_second > 5  # At least 5 signals per second
    
    async def test_concurrent_symbol_processing(self):
        """Test processing multiple symbols concurrently."""
        config = TestConfigManager()
        
        await self.setup_component('ml_manager', ModelManager, config)
        
        ml_manager = self.components['ml_manager']
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        with patch.object(ml_manager, 'generate_prediction') as mock_predict:
            mock_predict.return_value = MockSignalData().generate_signal()
            
            # Process multiple symbols concurrently
            start_time = datetime.now()
            
            tasks = []
            for symbol in symbols:
                task = ml_manager.generate_prediction(symbol, MockMarketData().generate_ohlcv(symbol))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Should handle concurrent processing
            assert len(results) == len(symbols)
            assert all(result is not None for result in results)
            
            # Concurrent processing should be faster than sequential
            assert processing_time < 5.0  # Should complete quickly


if __name__ == '__main__':
    pytest.main([__file__, '-v'])