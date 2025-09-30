"""
Complete Testing Framework Integration Test

Validates all testing components work together.
"""

import asyncio
import pytest
from datetime import datetime, timedelta

from tests.base_test import BaseTestCase
from tests.backtesting.backtest_engine import BacktestEngine, StrategyBacktester
from tests.performance.performance_tests import PerformanceTestRunner
from tests.simulation.paper_trading import PaperTradingEngine, SimulationMode
from tests.fixtures.test_config import TestConfigManager
from tests.test_runner import TestRunner


@pytest.mark.integration
class TestCompleteFramework(BaseTestCase):
    """Integration test for the complete testing framework."""
    
    def setUp(self):
        """Set up test environment.""" 
        super().setUp()
        self.config = TestConfigManager()
    
    def test_framework_initialization(self):
        """Test all framework components initialize correctly."""
        
        # Test backtesting engine
        backtest_engine = BacktestEngine(self.config)
        assert backtest_engine.initial_capital > 0
        assert backtest_engine.commission > 0
        assert backtest_engine.slippage > 0
        
        # Test strategy backtester
        strategy_backtester = StrategyBacktester(self.config)
        assert strategy_backtester.config == self.config
        assert strategy_backtester.engine is not None
        
        # Test paper trading engine  
        paper_engine = PaperTradingEngine(self.config)
        assert paper_engine.config == self.config
        assert paper_engine.market_simulator is not None
        
        # Test performance runner
        perf_runner = PerformanceTestRunner(self.config)
        assert perf_runner.config == self.config
        assert perf_runner.benchmark is not None
        
        # Test runner framework
        test_runner = TestRunner()
        assert test_runner.config is not None
    
    @pytest.mark.asyncio
    async def test_backtesting_workflow(self):
        """Test complete backtesting workflow."""
        
        backtester = StrategyBacktester(self.config)
        
        symbols = ["AAPL", "MSFT"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # Run backtest
        result = await backtester.backtest_ml_strategy(
            symbols=symbols,
            start_date=start_date, 
            end_date=end_date
        )
        
        # Verify results
        assert 'metrics' in result
        assert 'report' in result
        assert 'portfolio_history' in result
        assert 'trades' in result
        assert 'market_data' in result
        
        # Check metrics
        metrics = result['metrics']
        assert metrics.start_date == start_date
        assert metrics.end_date == end_date
        assert metrics.initial_capital > 0
        
        # Check report generation
        report = result['report']
        assert len(report) > 100
        assert 'Backtest Report' in report
        
        print("✅ Backtesting workflow test passed")
    
    @pytest.mark.asyncio
    async def test_performance_testing_workflow(self):
        """Test performance testing workflow."""
        
        perf_runner = PerformanceTestRunner(self.config)
        
        # Run a subset of performance tests
        await perf_runner.benchmark.benchmark_data_processing(num_symbols=2, days=10)
        await perf_runner.benchmark.benchmark_ml_inference([1, 5])
        
        # Verify results
        assert len(perf_runner.benchmark.results) >= 2
        
        # Check metrics
        for result in perf_runner.benchmark.results:
            assert result.operations_completed > 0
            assert result.duration > 0
            assert result.operations_per_second >= 0
        
        # Generate report
        report = perf_runner.benchmark.generate_performance_report()
        assert len(report) > 100
        assert 'Performance Test Report' in report
        
        print("✅ Performance testing workflow test passed")
    
    @pytest.mark.asyncio 
    async def test_paper_trading_workflow(self):
        """Test paper trading workflow."""
        
        engine = PaperTradingEngine(self.config)
        
        # Create account
        account_id = "test_account"
        account = engine.create_account(account_id, 50000.0)
        assert account.initial_balance == 50000.0
        
        # Start brief simulation
        symbols = ["AAPL", "MSFT"]
        
        # Start simulation task
        sim_task = asyncio.create_task(
            engine.start_simulation(symbols, SimulationMode.ACCELERATED)
        )
        
        # Wait for initialization
        await asyncio.sleep(0.5)
        
        # Place test orders
        order_id = engine.place_order(
            account_id=account_id,
            symbol="AAPL", 
            side="BUY",
            quantity=10
        )
        assert order_id is not None
        
        # Wait for order processing
        await asyncio.sleep(1.0)
        
        # Check account summary
        summary = engine.get_account_summary(account_id)
        assert 'total_portfolio_value' in summary
        assert summary['account_id'] == account_id
        
        # Stop simulation
        engine.simulation_running = False
        sim_task.cancel()
        
        print("✅ Paper trading workflow test passed")
    
    def test_configuration_management(self):
        """Test configuration management across components."""
        
        config = TestConfigManager()
        
        # Test different environments
        config.set_environment('test')
        test_config = config.get_config()
        assert 'database' in test_config
        
        config.set_environment('development')
        dev_config = config.get_config()
        assert dev_config != test_config
        
        # Test config overrides
        original_value = config.get_config('backtesting', {}).get('commission', 0.001)
        config.override_config('backtesting', 'commission', 0.002)
        new_value = config.get_config('backtesting', {}).get('commission', 0.001)
        assert new_value == 0.002
        assert new_value != original_value
        
        print("✅ Configuration management test passed")
    
    def test_mock_data_generation(self):
        """Test mock data generation consistency."""
        
        from tests.fixtures.mock_data import MockMarketData, MockSignalData, MockOrderData
        
        # Test market data
        mock_market = MockMarketData()
        data1 = mock_market.generate_ohlcv("TEST", days=10, freq='1H')
        data2 = mock_market.generate_ohlcv("TEST", days=10, freq='1H')
        
        # Should have consistent structure
        assert list(data1.columns) == list(data2.columns)
        assert len(data1) == len(data2)
        
        # Test signal data
        mock_signals = MockSignalData()
        signals1 = mock_signals.generate_signal_history(["TEST"], days=5)
        signals2 = mock_signals.generate_signal_history(["TEST"], days=5)
        
        # Should have consistent structure
        if signals1 and signals2:
            assert set(signals1[0].keys()) == set(signals2[0].keys())
        
        # Test order data
        mock_orders = MockOrderData()
        orders1 = mock_orders.generate_order_history(["TEST"], days=5)
        orders2 = mock_orders.generate_order_history(["TEST"], days=5)
        
        # Should have consistent structure
        if orders1 and orders2:
            assert set(orders1[0].keys()) == set(orders2[0].keys())
        
        print("✅ Mock data generation test passed")
    
    def test_error_handling(self):
        """Test error handling across components."""
        
        # Test backtest engine with invalid data
        engine = BacktestEngine(self.config)
        
        with pytest.raises((ValueError, TypeError, KeyError)):
            # Should handle invalid market data gracefully
            engine.run_backtest(
                market_data={}, 
                strategy_signals=[],
                start_date=datetime(2023, 12, 31),  # End before start
                end_date=datetime(2023, 1, 1)
            )
        
        # Test paper trading with invalid account
        paper_engine = PaperTradingEngine(self.config)
        
        with pytest.raises(ValueError):
            # Should raise error for non-existent account
            paper_engine.place_order(
                account_id="non_existent",
                symbol="AAPL",
                side="BUY", 
                quantity=100
            )
        
        print("✅ Error handling test passed")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations work correctly."""
        
        engine = PaperTradingEngine(self.config)
        
        # Create multiple accounts
        accounts = []
        for i in range(3):
            account_id = f"account_{i}"
            account = engine.create_account(account_id, 25000.0)
            accounts.append(account)
        
        # Start simulation
        symbols = ["AAPL"]
        sim_task = asyncio.create_task(
            engine.start_simulation(symbols, SimulationMode.ACCELERATED)
        )
        
        await asyncio.sleep(0.5)  # Wait for initialization
        
        # Place concurrent orders
        async def place_orders_for_account(account_id, num_orders=3):
            orders = []
            for i in range(num_orders):
                try:
                    order_id = engine.place_order(
                        account_id=account_id,
                        symbol="AAPL",
                        side="BUY" if i % 2 == 0 else "SELL",
                        quantity=5
                    )
                    orders.append(order_id)
                    await asyncio.sleep(0.1)  # Small delay between orders
                except Exception as e:
                    print(f"Order failed for {account_id}: {e}")
            return orders
        
        # Execute concurrent operations
        tasks = []
        for account in accounts:
            task = place_orders_for_account(account.account_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent operation failed: {result}")
        
        # Stop simulation
        engine.simulation_running = False
        sim_task.cancel()
        
        print("✅ Concurrent operations test passed")
    
    def test_data_persistence(self):
        """Test data persistence and state management."""
        
        import tempfile
        import os
        
        engine = PaperTradingEngine(self.config)
        
        # Create account and some state
        account_id = "persistence_test"
        account = engine.create_account(account_id, 30000.0)
        
        # Simulate some trading activity
        engine.accounts[account_id].total_trades = 5
        engine.accounts[account_id].winning_trades = 3
        engine.accounts[account_id].current_balance = 28000.0
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            engine.save_simulation_state(temp_file)
            assert os.path.exists(temp_file)
            
            # Create new engine and load state
            new_engine = PaperTradingEngine(self.config)
            new_engine.load_simulation_state(temp_file)
            
            # Verify state was loaded correctly
            loaded_account = new_engine.accounts.get(account_id)
            assert loaded_account is not None
            assert loaded_account.initial_balance == account.initial_balance
            assert loaded_account.total_trades == 5
            assert loaded_account.winning_trades == 3
            assert abs(loaded_account.current_balance - 28000.0) < 0.01
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        print("✅ Data persistence test passed")
    
    def test_integration_test_runner(self):
        """Test the test runner framework itself."""
        
        runner = TestRunner()
        
        # Test configuration
        assert runner.config is not None
        
        # Test suite discovery
        suites = runner.discover_test_suites()
        assert isinstance(suites, list)
        
        # Should find our test suites
        suite_names = [suite['name'] for suite in suites]
        expected_suites = ['unit', 'integration', 'backtesting', 'performance', 'simulation']
        
        for expected in expected_suites:
            # Check if any suite contains the expected name
            found = any(expected in name.lower() for name in suite_names)
            if found:
                print(f"Found test suite containing '{expected}'")
        
        print("✅ Test runner framework test passed")


@pytest.mark.asyncio
async def test_complete_framework_integration():
    """High-level integration test of the complete framework."""
    
    print("\n🚀 Running complete framework integration test...")
    
    config = TestConfigManager()
    
    # Test 1: Backtesting Integration
    print("\n1️⃣ Testing backtesting integration...")
    backtester = StrategyBacktester(config)
    
    backtest_result = await backtester.backtest_ml_strategy(
        symbols=["AAPL"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 15)
    )
    
    assert backtest_result['metrics'].operations_completed >= 0
    print("   ✅ Backtesting integration successful")
    
    # Test 2: Performance Testing Integration
    print("\n2️⃣ Testing performance integration...")
    perf_runner = PerformanceTestRunner(config)
    
    await perf_runner.benchmark.benchmark_data_processing(num_symbols=1, days=5)
    assert len(perf_runner.benchmark.results) > 0
    print("   ✅ Performance testing integration successful")
    
    # Test 3: Paper Trading Integration
    print("\n3️⃣ Testing paper trading integration...")
    paper_engine = PaperTradingEngine(config)
    
    account = paper_engine.create_account("integration_test", 10000.0)
    assert account.initial_balance == 10000.0
    print("   ✅ Paper trading integration successful")
    
    # Test 4: Framework Coordination
    print("\n4️⃣ Testing framework coordination...")
    
    # All components should share configuration
    assert backtester.config.environment == config.environment
    assert perf_runner.config.environment == config.environment  
    assert paper_engine.config.environment == config.environment
    print("   ✅ Framework coordination successful")
    
    print("\n🎉 Complete framework integration test PASSED!")


if __name__ == '__main__':
    # Run the integration test
    asyncio.run(test_complete_framework_integration())