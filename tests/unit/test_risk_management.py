"""
Unit Tests for Risk Management Module

Tests for risk management, position sizing, and portfolio management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from tests.base_test import UnitTestCase, AsyncTestCase
from tests.fixtures.mock_data import MockPortfolioData, MockMarketData
from tests.fixtures.test_config import TestConfigManager

# Import modules to test
try:
    from src.risk.risk_manager import RiskManager
    from src.risk.position_sizer import PositionSizer, KellyCriterion, VolatilityBased
    from src.risk.portfolio_manager import PortfolioManager, Position, Portfolio
    from src.risk.risk_metrics import RiskMetrics, DrawdownMonitor
    from src.risk.correlation_analyzer import CorrelationAnalyzer
except ImportError as e:
    pytest.skip(f"Risk module not available: {e}", allow_module_level=True)


@pytest.mark.unit
class TestRiskManager(AsyncTestCase):
    """Test cases for Risk Manager."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        self.risk_manager = RiskManager(self.config)
    
    async def test_risk_manager_initialization(self):
        """Test RiskManager initialization."""
        await self.risk_manager.initialize()
        
        assert self.risk_manager is not None
        assert hasattr(self.risk_manager, 'config')
        assert hasattr(self.risk_manager, 'position_limits')
        assert hasattr(self.risk_manager, 'risk_metrics')
    
    async def test_signal_filtering(self):
        """Test signal filtering based on risk criteria."""
        # Mock signal data
        signal = {
            'symbol': 'AAPL',
            'signal_type': 'BUY',
            'strength': 0.8,
            'confidence': 0.75,
            'timestamp': datetime.now()
        }
        
        # Mock portfolio data
        current_portfolio = {
            'total_value': 100000,
            'cash': 50000,
            'positions': {
                'AAPL': {'quantity': 100, 'market_value': 15000}
            }
        }
        
        # Test signal passes risk checks
        result = await self.risk_manager.filter_signal(signal, current_portfolio)
        
        assert isinstance(result, dict)
        assert 'approved' in result
        assert 'risk_score' in result
        assert 'reasons' in result
    
    async def test_position_size_limits(self):
        """Test position size limit enforcement."""
        signal = {
            'symbol': 'AAPL',
            'signal_type': 'BUY',
            'strength': 0.9
        }
        
        portfolio_value = 100000
        
        # Test maximum position size calculation
        max_position_value = await self.risk_manager.calculate_max_position_size(
            signal, portfolio_value
        )
        
        assert isinstance(max_position_value, float)
        assert max_position_value > 0
        
        # Should not exceed configured percentage (e.g., 5% of portfolio)
        max_allowed = portfolio_value * self.config.get_config('risk_management')['max_position_size']
        assert max_position_value <= max_allowed
    
    async def test_correlation_limits(self):
        """Test correlation-based risk limits."""
        # Mock existing positions with high correlation
        existing_positions = {
            'AAPL': {'quantity': 100, 'symbol': 'AAPL'},
            'MSFT': {'quantity': 50, 'symbol': 'MSFT'}  # Assume high tech correlation
        }
        
        # New signal for another tech stock
        new_signal = {
            'symbol': 'GOOGL',
            'signal_type': 'BUY',
            'strength': 0.8
        }
        
        # Should detect high correlation and potentially reduce position size
        correlation_check = await self.risk_manager.check_correlation_limits(
            new_signal, existing_positions
        )
        
        assert isinstance(correlation_check, dict)
        assert 'correlation_risk' in correlation_check
        assert 'adjustment_factor' in correlation_check
    
    async def test_volatility_adjustment(self):
        """Test position sizing based on volatility."""
        # High volatility asset
        high_vol_signal = {
            'symbol': 'CRYPTO_BTC',
            'volatility': 0.8,  # 80% annual volatility
            'signal_type': 'BUY'
        }
        
        # Low volatility asset
        low_vol_signal = {
            'symbol': 'BOND_ETF',
            'volatility': 0.1,  # 10% annual volatility
            'signal_type': 'BUY'
        }
        
        portfolio_value = 100000
        
        high_vol_size = await self.risk_manager.calculate_volatility_adjusted_size(
            high_vol_signal, portfolio_value
        )
        
        low_vol_size = await self.risk_manager.calculate_volatility_adjusted_size(
            low_vol_signal, portfolio_value
        )
        
        # High volatility should result in smaller position size
        assert high_vol_size < low_vol_size
    
    async def test_stop_loss_calculation(self):
        """Test stop-loss calculation."""
        position = {
            'symbol': 'AAPL',
            'quantity': 100,
            'avg_price': 150.0,
            'current_price': 148.0
        }
        
        stop_loss = await self.risk_manager.calculate_stop_loss(position)
        
        assert isinstance(stop_loss, dict)
        assert 'price' in stop_loss
        assert 'percentage' in stop_loss
        
        # Stop loss should be below current price for long position
        assert stop_loss['price'] < position['current_price']


@pytest.mark.unit
class TestPositionSizer(UnitTestCase):
    """Test cases for Position Sizer."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        self.position_sizer = PositionSizer(self.config)
    
    def test_position_sizer_initialization(self):
        """Test PositionSizer initialization."""
        assert self.position_sizer is not None
        assert hasattr(self.position_sizer, 'config')
        assert hasattr(self.position_sizer, 'sizing_methods')
    
    def test_fixed_percentage_sizing(self):
        """Test fixed percentage position sizing."""
        portfolio_value = 100000
        signal_strength = 0.8
        percentage = 0.05  # 5%
        
        position_size = self.position_sizer.calculate_fixed_percentage_size(
            portfolio_value, percentage, signal_strength
        )
        
        assert isinstance(position_size, float)
        assert position_size > 0
        
        # Should not exceed maximum percentage
        assert position_size <= portfolio_value * percentage
        
        # Stronger signals should get larger positions (up to max)
        weak_signal_size = self.position_sizer.calculate_fixed_percentage_size(
            portfolio_value, percentage, 0.3
        )
        
        assert position_size >= weak_signal_size
    
    def test_volatility_based_sizing(self):
        """Test volatility-based position sizing."""
        portfolio_value = 100000
        target_volatility = 0.02  # 2% portfolio volatility target
        
        # High volatility asset
        high_vol_size = self.position_sizer.calculate_volatility_based_size(
            portfolio_value, asset_volatility=0.4, target_volatility=target_volatility
        )
        
        # Low volatility asset
        low_vol_size = self.position_sizer.calculate_volatility_based_size(
            portfolio_value, asset_volatility=0.1, target_volatility=target_volatility
        )
        
        # High volatility should result in smaller position
        assert high_vol_size < low_vol_size
        
        # Both should contribute approximately target volatility to portfolio
        high_vol_contribution = (high_vol_size / portfolio_value) * 0.4
        low_vol_contribution = (low_vol_size / portfolio_value) * 0.1
        
        # Should be approximately equal to target volatility
        self.assert_near_equal(high_vol_contribution, target_volatility, tolerance=0.005)
        self.assert_near_equal(low_vol_contribution, target_volatility, tolerance=0.005)


@pytest.mark.unit  
class TestKellyCriterion(UnitTestCase):
    """Test cases for Kelly Criterion position sizing."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.kelly = KellyCriterion()
    
    def test_kelly_calculation_basic(self):
        """Test basic Kelly Criterion calculation."""
        win_probability = 0.6
        avg_win = 0.10  # 10% average win
        avg_loss = -0.05  # 5% average loss
        
        kelly_fraction = self.kelly.calculate_kelly_fraction(
            win_probability, avg_win, avg_loss
        )
        
        assert isinstance(kelly_fraction, float)
        assert 0 <= kelly_fraction <= 1  # Should be between 0 and 1 for positive expectancy
    
    def test_kelly_with_negative_expectancy(self):
        """Test Kelly Criterion with negative expectancy."""
        win_probability = 0.3  # Low win rate
        avg_win = 0.05
        avg_loss = -0.10  # Large losses
        
        kelly_fraction = self.kelly.calculate_kelly_fraction(
            win_probability, avg_win, avg_loss
        )
        
        # Negative expectancy should result in zero position
        assert kelly_fraction == 0
    
    def test_kelly_position_sizing(self):
        """Test Kelly-based position sizing."""
        portfolio_value = 100000
        
        # Historical performance data
        historical_returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.015]
        
        position_size = self.kelly.calculate_position_size(
            portfolio_value, historical_returns
        )
        
        assert isinstance(position_size, float)
        assert position_size >= 0
        assert position_size <= portfolio_value


@pytest.mark.unit
class TestPortfolioManager(AsyncTestCase):
    """Test cases for Portfolio Manager."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        self.portfolio_manager = PortfolioManager(self.config)
        
        # Mock portfolio data
        self.mock_portfolio_data = MockPortfolioData().generate_portfolio()
    
    async def test_portfolio_manager_initialization(self):
        """Test PortfolioManager initialization."""
        await self.portfolio_manager.initialize()
        
        assert self.portfolio_manager is not None
        assert hasattr(self.portfolio_manager, 'current_portfolio')
        assert hasattr(self.portfolio_manager, 'position_history')
    
    async def test_add_position(self):
        """Test adding a new position."""
        position = Position(
            symbol='AAPL',
            quantity=100,
            avg_price=150.0,
            timestamp=datetime.now()
        )
        
        await self.portfolio_manager.add_position(position)
        
        # Verify position was added
        current_portfolio = await self.portfolio_manager.get_current_portfolio()
        assert 'AAPL' in current_portfolio.positions
        assert current_portfolio.positions['AAPL'].quantity == 100
    
    async def test_update_position(self):
        """Test updating an existing position."""
        # Add initial position
        initial_position = Position(
            symbol='MSFT',
            quantity=50,
            avg_price=300.0,
            timestamp=datetime.now()
        )
        await self.portfolio_manager.add_position(initial_position)
        
        # Add more shares (should update average price)
        additional_shares = Position(
            symbol='MSFT',
            quantity=50,
            avg_price=320.0,
            timestamp=datetime.now()
        )
        await self.portfolio_manager.add_position(additional_shares)
        
        # Check updated position
        portfolio = await self.portfolio_manager.get_current_portfolio()
        msft_position = portfolio.positions['MSFT']
        
        assert msft_position.quantity == 100
        # Average price should be weighted average: (50*300 + 50*320) / 100 = 310
        self.assert_near_equal(msft_position.avg_price, 310.0, tolerance=0.01)
    
    async def test_remove_position(self):
        """Test removing a position."""
        # Add position
        position = Position(
            symbol='GOOGL',
            quantity=25,
            avg_price=2500.0,
            timestamp=datetime.now()
        )
        await self.portfolio_manager.add_position(position)
        
        # Remove position
        await self.portfolio_manager.remove_position('GOOGL', 25)
        
        # Verify position is removed
        portfolio = await self.portfolio_manager.get_current_portfolio()
        assert 'GOOGL' not in portfolio.positions or portfolio.positions['GOOGL'].quantity == 0
    
    async def test_portfolio_value_calculation(self):
        """Test portfolio total value calculation."""
        # Add multiple positions
        positions = [
            Position('AAPL', 100, 150.0, datetime.now()),
            Position('MSFT', 50, 300.0, datetime.now()),
            Position('GOOGL', 10, 2500.0, datetime.now())
        ]
        
        for position in positions:
            await self.portfolio_manager.add_position(position)
        
        # Mock current prices
        current_prices = {'AAPL': 155.0, 'MSFT': 310.0, 'GOOGL': 2600.0}
        
        with patch.object(self.portfolio_manager, 'get_current_prices', 
                         return_value=current_prices):
            
            portfolio = await self.portfolio_manager.get_current_portfolio()
            
            # Expected value: 100*155 + 50*310 + 10*2600 = 15500 + 15500 + 26000 = 57000
            expected_value = 57000
            self.assert_near_equal(portfolio.total_value, expected_value, tolerance=100)
    
    async def test_portfolio_pnl_calculation(self):
        """Test P&L calculation."""
        position = Position('TSLA', 20, 800.0, datetime.now())
        await self.portfolio_manager.add_position(position)
        
        # Mock current price (10% gain)
        current_prices = {'TSLA': 880.0}
        
        with patch.object(self.portfolio_manager, 'get_current_prices',
                         return_value=current_prices):
            
            portfolio = await self.portfolio_manager.get_current_portfolio()
            
            # Expected P&L: 20 * (880 - 800) = 1600
            expected_pnl = 1600
            self.assert_near_equal(portfolio.total_pnl, expected_pnl, tolerance=10)


@pytest.mark.unit
class TestRiskMetrics(UnitTestCase):
    """Test cases for Risk Metrics."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.risk_metrics = RiskMetrics()
        
        # Generate sample return data
        np.random.seed(42)  
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.risk_metrics.calculate_sharpe_ratio(
            self.returns, risk_free_rate=0.02
        )
        
        assert isinstance(sharpe, float)
        # Reasonable range for Sharpe ratio
        assert -3 <= sharpe <= 5
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create cumulative returns with known drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        
        max_dd, max_dd_duration = self.risk_metrics.calculate_max_drawdown(
            cumulative_returns
        )
        
        assert isinstance(max_dd, float)
        assert isinstance(max_dd_duration, int)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd_duration >= 0
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        var_95 = self.risk_metrics.calculate_var(self.returns, confidence_level=0.95)
        var_99 = self.risk_metrics.calculate_var(self.returns, confidence_level=0.99)
        
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_95 <= 0  # VaR should be negative (loss)
        assert var_99 <= var_95  # 99% VaR should be more negative than 95% VaR
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation."""
        cvar = self.risk_metrics.calculate_cvar(self.returns, confidence_level=0.95)
        var = self.risk_metrics.calculate_var(self.returns, confidence_level=0.95)
        
        assert isinstance(cvar, float)
        assert cvar <= var  # CVaR should be more negative than VaR
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        daily_vol = self.risk_metrics.calculate_volatility(self.returns, period='daily')
        annual_vol = self.risk_metrics.calculate_volatility(self.returns, period='annual')
        
        assert isinstance(daily_vol, float)
        assert isinstance(annual_vol, float)
        assert daily_vol > 0
        assert annual_vol > 0
        
        # Annual volatility should be approximately daily_vol * sqrt(252)
        expected_annual = daily_vol * np.sqrt(252)
        self.assert_near_equal(annual_vol, expected_annual, tolerance=0.001)
    
    def test_beta_calculation(self):
        """Test beta calculation."""
        # Generate market returns (benchmark)
        market_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        
        beta = self.risk_metrics.calculate_beta(self.returns, market_returns)
        
        assert isinstance(beta, float)
        # Beta typically ranges from 0 to 2 for most stocks
        assert -2 <= beta <= 3


@pytest.mark.unit
class TestDrawdownMonitor(UnitTestCase):
    """Test cases for Drawdown Monitor."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.drawdown_monitor = DrawdownMonitor()
    
    def test_drawdown_tracking(self):
        """Test real-time drawdown tracking."""
        # Simulate portfolio values with drawdown
        portfolio_values = [100000, 102000, 98000, 95000, 97000, 105000, 103000]
        
        for i, value in enumerate(portfolio_values):
            self.drawdown_monitor.update(value, datetime.now() + timedelta(days=i))
        
        current_dd = self.drawdown_monitor.get_current_drawdown()
        max_dd = self.drawdown_monitor.get_max_drawdown()
        
        assert isinstance(current_dd, float)
        assert isinstance(max_dd, float)
        assert current_dd <= 0
        assert max_dd <= 0
        assert max_dd <= current_dd  # Max drawdown should be worst case
    
    def test_drawdown_alerts(self):
        """Test drawdown alert thresholds."""
        alert_threshold = -0.10  # 10% drawdown alert
        self.drawdown_monitor.set_alert_threshold(alert_threshold)
        
        # Simulate large drawdown
        initial_value = 100000
        drawdown_value = 88000  # 12% drawdown
        
        self.drawdown_monitor.update(initial_value, datetime.now())
        alert_triggered = self.drawdown_monitor.update(
            drawdown_value, datetime.now() + timedelta(days=1)
        )
        
        # Should trigger alert for exceeding 10% threshold
        assert alert_triggered


@pytest.mark.unit
class TestCorrelationAnalyzer(UnitTestCase):
    """Test cases for Correlation Analyzer."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.correlation_analyzer = CorrelationAnalyzer()
    
    def test_correlation_calculation(self):
        """Test correlation calculation between assets."""
        # Generate correlated return data
        np.random.seed(42)
        returns_a = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Create correlated series
        noise = np.random.normal(0, 0.01, 100)
        returns_b = 0.7 * returns_a + 0.3 * noise  # 70% correlation
        
        correlation = self.correlation_analyzer.calculate_correlation(
            returns_a, returns_b
        )
        
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1
        # Should be moderately high correlation
        assert correlation > 0.5
    
    def test_portfolio_correlation_matrix(self):
        """Test portfolio correlation matrix calculation."""
        # Mock return data for multiple assets
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        returns_data = {}
        
        np.random.seed(42)
        for symbol in symbols:
            returns_data[symbol] = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        returns_df = pd.DataFrame(returns_data)
        
        corr_matrix = self.correlation_analyzer.calculate_correlation_matrix(returns_df)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (len(symbols), len(symbols))
        
        # Diagonal should be 1 (perfect self-correlation)
        for symbol in symbols:
            self.assert_near_equal(corr_matrix.loc[symbol, symbol], 1.0, tolerance=0.001)
    
    def test_diversification_score(self):
        """Test portfolio diversification score."""
        # Low diversification (high correlation)
        high_corr_matrix = pd.DataFrame({
            'A': [1.0, 0.9, 0.8],
            'B': [0.9, 1.0, 0.85],
            'C': [0.8, 0.85, 1.0]
        }, index=['A', 'B', 'C'])
        
        weights = [0.4, 0.3, 0.3]
        
        low_div_score = self.correlation_analyzer.calculate_diversification_score(
            high_corr_matrix, weights
        )
        
        # High diversification (low correlation)  
        low_corr_matrix = pd.DataFrame({
            'A': [1.0, 0.1, 0.2],
            'B': [0.1, 1.0, 0.15],
            'C': [0.2, 0.15, 1.0]
        }, index=['A', 'B', 'C'])
        
        high_div_score = self.correlation_analyzer.calculate_diversification_score(
            low_corr_matrix, weights
        )
        
        assert isinstance(low_div_score, float)
        assert isinstance(high_div_score, float)
        
        # Higher diversification should have higher score
        assert high_div_score > low_div_score


if __name__ == '__main__':
    pytest.main([__file__, '-v'])