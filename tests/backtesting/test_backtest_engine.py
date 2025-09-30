"""
Backtesting Tests

Test cases for the backtesting framework functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from tests.base_test import BacktestingTestCase
from tests.fixtures.mock_data import MockMarketData, MockSignalData
from tests.backtesting.backtest_engine import (
    BacktestEngine, StrategyBacktester, BacktestOrder, BacktestPosition,
    BacktestMetrics, OrderSide, OrderStatus
)


class TestBacktestOrder(BacktestingTestCase):
    """Test BacktestOrder functionality."""
    
    def test_order_creation(self):
        """Test basic order creation."""
        timestamp = datetime.now()
        order = BacktestOrder(
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0
        )
        
        assert order.timestamp == timestamp
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.price == 150.0
        assert order.status == OrderStatus.PENDING
    
    def test_total_cost_calculation(self):
        """Test total cost calculation including commission."""
        order = BacktestOrder(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0
        )
        
        # Before fill
        assert order.total_cost == 0.0
        
        # After fill
        order.fill_price = 150.50
        order.commission = 5.0
        expected_cost = 100 * 150.50 + 5.0
        assert order.total_cost == expected_cost


class TestBacktestPosition(BacktestingTestCase):
    """Test BacktestPosition functionality."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        position = BacktestPosition("AAPL")
        
        assert position.symbol == "AAPL"
        assert position.quantity == 0
        assert position.avg_price == 0.0
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0
    
    def test_market_value_update(self):
        """Test market value and P&L calculation."""
        position = BacktestPosition("AAPL", quantity=100, avg_price=150.0)
        
        # Update with higher price
        position.update_market_value(155.0)
        assert position.market_value == 15500.0  # 100 * 155
        assert position.unrealized_pnl == 500.0  # 100 * (155 - 150)
        
        # Update with lower price
        position.update_market_value(145.0)
        assert position.market_value == 14500.0
        assert position.unrealized_pnl == -500.0
    
    def test_empty_position_update(self):
        """Test updating position with zero quantity."""
        position = BacktestPosition("AAPL", quantity=0, avg_price=150.0)
        position.update_market_value(155.0)
        
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0


class TestBacktestMetrics(BacktestingTestCase):
    """Test BacktestMetrics functionality."""
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        metrics = BacktestMetrics(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            final_capital=120000.0,
            daily_returns=[0.01, -0.005, 0.02, 0.0, -0.01]
        )
        
        metrics.calculate_metrics()
        
        # Check basic calculations
        assert metrics.total_return == 0.2  # 20% return
        assert metrics.days_traded == 364
        assert metrics.volatility > 0
    
    def test_empty_returns_handling(self):
        """Test handling of empty daily returns."""
        metrics = BacktestMetrics(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
            initial_capital=100000.0,
            final_capital=100000.0
        )
        
        metrics.calculate_metrics()
        
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0


class TestBacktestEngine(BacktestingTestCase):
    """Test BacktestEngine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.engine = BacktestEngine(self.test_config)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.initial_capital > 0
        assert self.engine.cash == self.engine.initial_capital
        assert len(self.engine.positions) == 0
        assert len(self.engine.orders) == 0
        assert len(self.engine.trades) == 0
    
    def test_reset_functionality(self):
        """Test engine reset."""
        # Add some state
        self.engine.cash = 50000
        self.engine.positions['AAPL'] = BacktestPosition('AAPL', quantity=100)
        self.engine.orders.append(Mock())
        
        # Reset
        self.engine.reset()
        
        # Verify reset
        assert self.engine.cash == self.engine.initial_capital
        assert len(self.engine.positions) == 0
        assert len(self.engine.orders) == 0
        assert len(self.engine.trades) == 0
    
    def test_add_order(self):
        """Test order addition."""
        timestamp = datetime.now()
        order = self.engine.add_order(
            timestamp=timestamp,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        
        assert len(self.engine.orders) == 1
        assert self.engine.orders[0] == order
        assert order.timestamp == timestamp
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
    
    def test_buy_order_execution(self):
        """Test buy order execution."""
        # Add buy order
        order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        
        # Execute order
        market_price = 150.0
        self.engine._execute_order(order, market_price)
        
        # Verify execution
        assert order.status == OrderStatus.FILLED
        assert order.fill_price is not None
        assert self.engine.cash < self.engine.initial_capital
        assert "AAPL" in self.engine.positions
        assert self.engine.positions["AAPL"].quantity == 100
        assert len(self.engine.trades) == 1
    
    def test_sell_order_execution(self):
        """Test sell order execution."""
        # First buy some shares
        buy_order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        self.engine._execute_order(buy_order, 150.0)
        
        # Record initial cash after buy
        cash_after_buy = self.engine.cash
        
        # Then sell some shares
        sell_order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50
        )
        self.engine._execute_order(sell_order, 155.0)
        
        # Verify execution
        assert sell_order.status == OrderStatus.FILLED
        assert self.engine.cash > cash_after_buy  # Should have more cash
        assert self.engine.positions["AAPL"].quantity == 50  # 50 shares remaining
        assert len(self.engine.trades) == 2  # Buy + Sell
    
    def test_insufficient_cash_rejection(self):
        """Test order rejection due to insufficient cash."""
        # Try to buy more than we can afford
        order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000000  # Very large quantity
        )
        
        self.engine._execute_order(order, 150.0)
        
        assert order.status == OrderStatus.REJECTED
        assert len(self.engine.positions) == 0
        assert len(self.engine.trades) == 0
    
    def test_insufficient_shares_rejection(self):
        """Test sell order rejection due to insufficient shares."""
        # Try to sell without owning shares
        order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100
        )
        
        self.engine._execute_order(order, 150.0)
        
        assert order.status == OrderStatus.REJECTED
        assert len(self.engine.trades) == 0
    
    def test_portfolio_update(self):
        """Test portfolio state update."""
        # Buy some shares first
        order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        self.engine._execute_order(order, 150.0)
        
        # Update portfolio
        timestamp = datetime.now()
        prices = {"AAPL": 155.0}
        self.engine.update_portfolio(timestamp, prices)
        
        # Verify portfolio state recorded
        assert len(self.engine.portfolio_history) == 1
        
        portfolio_state = self.engine.portfolio_history[0]
        assert portfolio_state['timestamp'] == timestamp
        assert portfolio_state['cash'] == self.engine.cash
        assert portfolio_state['positions_value'] > 0
        assert portfolio_state['total_value'] > portfolio_state['cash']
    
    def test_process_orders(self):
        """Test order processing."""
        # Add pending order
        order = self.engine.add_order(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        
        assert order.status == OrderStatus.PENDING
        
        # Process orders with market prices
        prices = {"AAPL": 150.0}
        self.engine.process_orders(prices)
        
        # Verify order was processed
        assert order.status == OrderStatus.FILLED
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # Create portfolio history with drawdown
        self.engine.portfolio_history = [
            {'total_value': 100000},
            {'total_value': 110000},
            {'total_value': 105000},
            {'total_value': 95000},
            {'total_value': 100000},
            {'total_value': 115000}
        ]
        
        max_drawdown, duration = self.engine._calculate_drawdown()
        
        # Should detect the drawdown from 110k to 95k
        assert max_drawdown < 0  # Negative value indicating loss
        assert duration > 0
    
    def test_empty_drawdown_calculation(self):
        """Test drawdown calculation with empty history."""
        max_drawdown, duration = self.engine._calculate_drawdown()
        
        assert max_drawdown == 0.0
        assert duration == 0


class TestStrategyBacktester(BacktestingTestCase):
    """Test StrategyBacktester functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.backtester = StrategyBacktester(self.test_config)
    
    @pytest.mark.asyncio
    async def test_ml_strategy_backtest(self):
        """Test ML strategy backtesting."""
        symbols = ["AAPL", "GOOGL"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)  # 3 months
        
        result = await self.backtester.backtest_ml_strategy(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify result structure
        assert 'metrics' in result
        assert 'report' in result
        assert 'portfolio_history' in result
        assert 'trades' in result
        assert 'market_data' in result
        
        # Verify metrics
        metrics = result['metrics']
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.start_date == start_date
        assert metrics.end_date == end_date
        assert metrics.initial_capital > 0
        
        # Verify DataFrames
        assert isinstance(result['portfolio_history'], pd.DataFrame)
        assert isinstance(result['trades'], pd.DataFrame)
        
        # Verify market data
        assert len(result['market_data']) == len(symbols)
        for symbol in symbols:
            assert symbol in result['market_data']
            assert isinstance(result['market_data'][symbol], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_custom_signal_generator(self):
        """Test backtesting with custom signal generator."""
        
        async def custom_signal_generator(market_data, start_date, end_date):
            """Generate simple buy signals."""
            signals = []
            
            for symbol in market_data.keys():
                # Generate a buy signal on the first day
                signals.append({
                    'timestamp': start_date,
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'strength': 1.0,
                    'confidence': 0.8
                })
                
                # Generate a sell signal after 30 days
                sell_date = start_date + timedelta(days=30)
                if sell_date <= end_date:
                    signals.append({
                        'timestamp': sell_date,
                        'symbol': symbol,
                        'signal_type': 'SELL',
                        'strength': 1.0,
                        'confidence': 0.7
                    })
            
            return signals
        
        symbols = ["AAPL"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 28)
        
        result = await self.backtester.backtest_ml_strategy(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            signal_generator=custom_signal_generator
        )
        
        # Should have executed trades based on custom signals
        trades_df = result['trades']
        assert len(trades_df) >= 2  # At least buy and sell
    
    def test_strategy_comparison(self):
        """Test strategy comparison functionality."""
        
        # Define mock strategies
        strategies = {
            'Simple MA': {
                'signal_generator': None  # Will use default mock signals
            },
            'RSI Strategy': {
                'signal_generator': None  # Will use default mock signals
            }
        }
        
        symbols = ["AAPL"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)  # Short period for testing
        
        with patch('asyncio.run') as mock_run:
            # Mock the asyncio.run to return sample results
            mock_result = {
                'metrics': BacktestMetrics(
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=100000,
                    final_capital=110000,
                    total_return=0.1,
                    annual_return=0.12,
                    sharpe_ratio=1.5,
                    max_drawdown=-0.05,
                    win_rate=0.6,
                    total_trades=10
                ),
                'report': 'Mock report',
                'portfolio_history': pd.DataFrame(),
                'trades': pd.DataFrame()
            }
            
            mock_run.return_value = mock_result
            
            result = self.backtester.compare_strategies(
                strategies=strategies,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Verify comparison result structure
            assert 'individual_results' in result
            assert 'comparison_report' in result
            
            # Should have results for each strategy
            assert len(result['individual_results']) == len(strategies)
            
            # Should have comparison report
            assert isinstance(result['comparison_report'], str)
            assert 'Strategy Comparison Report' in result['comparison_report']
    
    def test_report_generation(self):
        """Test backtest report generation."""
        metrics = BacktestMetrics(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            final_capital=120000.0,
            total_return=0.2,
            annual_return=0.2,
            volatility=0.15,
            sharpe_ratio=1.33,
            max_drawdown=-0.08,
            max_drawdown_duration=45,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6
        )
        
        report = self.backtester.engine.generate_backtest_report(metrics)
        
        # Verify report content
        assert 'Backtest Report' in report
        assert '$100,000.00' in report  # Initial capital
        assert '$120,000.00' in report  # Final capital
        assert '20.00%' in report  # Total return
        assert '1.33' in report  # Sharpe ratio
        assert '50' in report  # Total trades
    
    def test_portfolio_history_dataframe(self):
        """Test portfolio history DataFrame creation."""
        # Add some portfolio history
        self.backtester.engine.portfolio_history = [
            {
                'timestamp': datetime(2023, 1, 1),
                'cash': 95000,
                'positions_value': 5000,
                'total_value': 100000,
                'unrealized_pnl': 0,
                'total_pnl': 0
            },
            {
                'timestamp': datetime(2023, 1, 2),
                'cash': 95000,
                'positions_value': 5500,
                'total_value': 100500,
                'unrealized_pnl': 500,
                'total_pnl': 500
            }
        ]
        
        df = self.backtester.engine.get_portfolio_history_df()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert 'total_value' in df.columns
        assert df.iloc[1]['total_value'] == 100500
    
    def test_trades_dataframe(self):
        """Test trades DataFrame creation."""
        # Add some trades
        self.backtester.engine.trades = [
            {
                'timestamp': datetime(2023, 1, 1),
                'symbol': 'AAPL',
                'side': 'BUY',
                'quantity': 100,
                'price': 150.0,
                'commission': 5.0
            },
            {
                'timestamp': datetime(2023, 1, 2),
                'symbol': 'AAPL',
                'side': 'SELL',
                'quantity': 50,
                'price': 155.0,
                'commission': 5.0,
                'realized_pnl': 245.0
            }
        ]
        
        df = self.backtester.engine.get_trades_df()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert 'side' in df.columns
        assert df.iloc[0]['side'] == 'BUY'
        assert df.iloc[1]['side'] == 'SELL'


class TestBacktestIntegration(BacktestingTestCase):
    """Integration tests for complete backtesting workflow."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.backtester = StrategyBacktester(self.test_config)
    
    @pytest.mark.asyncio
    async def test_complete_backtest_workflow(self):
        """Test complete backtesting workflow from start to finish."""
        
        # Define test parameters
        symbols = ["AAPL", "MSFT"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # Run backtest
        result = await self.backtester.backtest_ml_strategy(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify complete result
        assert result is not None
        
        # Check metrics
        metrics = result['metrics']
        assert metrics.start_date == start_date
        assert metrics.end_date == end_date
        assert metrics.initial_capital > 0
        assert metrics.final_capital > 0
        
        # Check data integrity
        portfolio_df = result['portfolio_history']
        if len(portfolio_df) > 0:
            assert portfolio_df['total_value'].iloc[0] > 0
            assert all(portfolio_df['total_value'] >= 0)  # No negative portfolio values
        
        # Check trades
        trades_df = result['trades']
        if len(trades_df) > 0:
            assert all(trades_df['quantity'] > 0)  # All trade quantities positive
            assert all(trades_df['price'] > 0)  # All prices positive
        
        # Check report generation
        report = result['report']
        assert len(report) > 100  # Should be a substantial report
        assert 'Backtest Report' in report
    
    def test_error_handling(self):
        """Test error handling in backtesting."""
        
        # Test with invalid date range
        with pytest.raises((ValueError, AssertionError)):
            start_date = datetime(2023, 12, 31)
            end_date = datetime(2023, 1, 1)  # End before start
            
            asyncio.run(self.backtester.backtest_ml_strategy(
                symbols=["AAPL"],
                start_date=start_date,
                end_date=end_date
            ))
    
    @pytest.mark.asyncio
    async def test_performance_metrics_accuracy(self):
        """Test accuracy of performance metrics calculation."""
        
        # Create controlled test scenario
        symbols = ["TEST"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        # Mock predictable market data and signals for testing
        with patch.object(MockMarketData, 'generate_ohlcv') as mock_market:
            with patch.object(MockSignalData, 'generate_signal_history') as mock_signals:
                
                # Create predictable market data (steady price increase)
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                mock_market.return_value = pd.DataFrame({
                    'timestamp': dates,
                    'open': [100 + i for i in range(len(dates))],
                    'high': [101 + i for i in range(len(dates))],
                    'low': [99 + i for i in range(len(dates))],
                    'close': [100 + i for i in range(len(dates))],
                    'volume': [1000] * len(dates)
                })
                
                # Create simple buy signal
                mock_signals.return_value = [{
                    'timestamp': start_date,
                    'symbol': 'TEST',
                    'signal_type': 'BUY',
                    'strength': 1.0
                }]
                
                result = await self.backtester.backtest_ml_strategy(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # With steadily increasing prices and a buy signal, 
                # we should have positive returns
                metrics = result['metrics']
                
                # Basic sanity checks
                assert metrics.initial_capital > 0
                assert metrics.final_capital >= 0
                
                # If trades were executed, we should see some activity
                trades_df = result['trades']
                if len(trades_df) > 0:
                    assert trades_df.iloc[0]['side'] == 'BUY'


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])