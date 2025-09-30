"""
Comprehensive Backtesting Framework

Historical simulation and strategy validation system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path

from tests.fixtures.mock_data import MockMarketData, MockSignalData, MockOrderData
from tests.fixtures.test_config import TestConfigManager


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class BacktestOrder:
    """Backtesting order representation."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: int
    price: Optional[float] = None
    order_type: str = "MARKET"
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Calculate total order cost including fees."""
        if self.fill_price is None:
            return 0.0
        
        base_cost = self.quantity * self.fill_price
        return base_cost + self.commission


@dataclass
class BacktestPosition:
    """Backtesting position representation."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_market_value(self, current_price: float):
        """Update position market value and unrealized P&L."""
        if self.quantity != 0:
            self.market_value = self.quantity * current_price
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:
            self.market_value = 0.0
            self.unrealized_pnl = 0.0


@dataclass
class BacktestMetrics:
    """Backtesting performance metrics."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Performance ratios
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    @property
    def days_traded(self) -> int:
        """Calculate number of days in backtest."""
        return (self.end_date - self.start_date).days
    
    def calculate_metrics(self):
        """Calculate derived metrics."""
        if self.initial_capital > 0:
            self.total_return = (self.final_capital - self.initial_capital) / self.initial_capital
            
            if self.days_traded > 0:
                self.annual_return = (1 + self.total_return) ** (365 / self.days_traded) - 1
        
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            self.volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            
            if self.volatility > 0:
                self.sharpe_ratio = (self.annual_return - 0.02) / self.volatility  # Assume 2% risk-free rate
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades


class BacktestEngine:
    """Comprehensive backtesting engine."""
    
    def __init__(self, config: TestConfigManager):
        self.config = config
        self.initial_capital = config.get_initial_capital()
        self.commission = config.get_config('backtesting').get('commission', 0.001)
        self.slippage = config.get_config('backtesting').get('slippage', 0.0005)
        
        # Backtest state
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.orders = []
        self.trades = []
        
        # Current state
        self.current_date = None
        self.current_prices = {}
        
    def reset(self):
        """Reset backtest state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.orders = []
        self.trades = []
        self.current_date = None
        self.current_prices = {}
    
    def add_order(self, timestamp: datetime, symbol: str, side: OrderSide, 
                  quantity: int, order_type: str = "MARKET", price: float = None) -> BacktestOrder:
        """Add order to backtest."""
        order = BacktestOrder(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
        
        self.orders.append(order)
        return order
    
    def process_orders(self, current_prices: Dict[str, float]):
        """Process pending orders."""
        for order in self.orders:
            if order.status == OrderStatus.PENDING and order.symbol in current_prices:
                self._execute_order(order, current_prices[order.symbol])
    
    def _execute_order(self, order: BacktestOrder, market_price: float):
        """Execute a single order."""
        # Determine fill price
        if order.order_type == "MARKET":
            # Apply slippage for market orders
            slippage_factor = self.slippage if order.side == OrderSide.BUY else -self.slippage
            fill_price = market_price * (1 + slippage_factor)
        else:
            # Limit orders fill at limit price if market allows
            fill_price = order.price if order.price else market_price
        
        # Calculate commission
        commission = max(1.0, order.quantity * fill_price * self.commission)
        
        # Check if we have enough cash/shares
        total_cost = order.quantity * fill_price + commission
        
        if order.side == OrderSide.BUY:
            if self.cash >= total_cost:
                self._execute_buy_order(order, fill_price, commission)
            else:
                order.status = OrderStatus.REJECTED
                return
        else:  # SELL
            position = self.positions.get(order.symbol, BacktestPosition(order.symbol))
            if position.quantity >= order.quantity:
                self._execute_sell_order(order, fill_price, commission)
            else:
                order.status = OrderStatus.REJECTED
                return
        
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.commission = commission
        order.slippage = abs(fill_price - market_price) / market_price
    
    def _execute_buy_order(self, order: BacktestOrder, fill_price: float, commission: float):
        """Execute buy order."""
        total_cost = order.quantity * fill_price + commission
        
        # Update cash
        self.cash -= total_cost
        
        # Update position
        if order.symbol not in self.positions:
            self.positions[order.symbol] = BacktestPosition(order.symbol)
        
        position = self.positions[order.symbol]
        
        # Calculate new average price
        old_value = position.quantity * position.avg_price
        new_value = order.quantity * fill_price
        total_quantity = position.quantity + order.quantity
        
        if total_quantity > 0:
            position.avg_price = (old_value + new_value) / total_quantity
        
        position.quantity = total_quantity
        
        # Record trade
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': 'BUY',
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission
        })
    
    def _execute_sell_order(self, order: BacktestOrder, fill_price: float, commission: float):
        """Execute sell order."""
        position = self.positions[order.symbol]
        
        # Calculate realized P&L
        cost_basis = order.quantity * position.avg_price
        proceeds = order.quantity * fill_price - commission
        realized_pnl = proceeds - cost_basis
        
        # Update cash
        self.cash += proceeds
        
        # Update position
        position.quantity -= order.quantity
        position.realized_pnl += realized_pnl
        
        # Record trade
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': 'SELL',
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'realized_pnl': realized_pnl
        })
    
    def update_portfolio(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Update portfolio state."""
        self.current_date = timestamp
        self.current_prices = current_prices
        
        # Process any pending orders
        self.process_orders(current_prices)
        
        # Update position market values
        total_market_value = 0
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.quantity != 0:
                position.update_market_value(current_prices[symbol])
                total_market_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
        
        # Calculate total portfolio value
        total_portfolio_value = self.cash + total_market_value
        
        # Record portfolio state
        portfolio_state = {
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': total_market_value,
            'total_value': total_portfolio_value,
            'unrealized_pnl': total_unrealized_pnl,
            'total_pnl': total_unrealized_pnl + sum(p.realized_pnl for p in self.positions.values())
        }
        
        self.portfolio_history.append(portfolio_state)
    
    def run_backtest(self, market_data: Dict[str, pd.DataFrame], 
                    strategy_signals: List[Dict[str, Any]], 
                    start_date: datetime, end_date: datetime) -> BacktestMetrics:
        """Run complete backtest."""
        
        print(f"🔄 Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Reset state
        self.reset()
        
        # Create combined timeline
        all_dates = set()
        for symbol_data in market_data.values():
            symbol_dates = symbol_data['timestamp'].dt.date
            all_dates.update(symbol_dates)
        
        # Sort dates in backtest range
        sorted_dates = sorted([d for d in all_dates if start_date.date() <= d <= end_date.date()])
        
        # Process each day
        for date in sorted_dates:
            current_datetime = datetime.combine(date, datetime.min.time())
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol, data in market_data.items():
                day_data = data[data['timestamp'].dt.date == date]
                if not day_data.empty:
                    # Use closing price for end-of-day valuation
                    current_prices[symbol] = day_data['close'].iloc[-1]
            
            # Process strategy signals for this date
            day_signals = [s for s in strategy_signals 
                          if s['timestamp'].date() == date]
            
            for signal in day_signals:
                if signal['signal_type'] in ['BUY', 'SELL']:
                    # Calculate position size (simplified)
                    portfolio_value = self.cash + sum(
                        p.market_value for p in self.positions.values()
                    )
                    
                    if signal['signal_type'] == 'BUY':
                        # Risk 2% of portfolio per trade
                        position_value = portfolio_value * 0.02 * signal.get('strength', 1.0)
                        
                        if signal['symbol'] in current_prices:
                            quantity = int(position_value / current_prices[signal['symbol']])
                            
                            if quantity > 0:
                                self.add_order(
                                    timestamp=current_datetime,
                                    symbol=signal['symbol'],
                                    side=OrderSide.BUY,
                                    quantity=quantity
                                )
                    
                    elif signal['signal_type'] == 'SELL':
                        # Sell existing position
                        if signal['symbol'] in self.positions:
                            position = self.positions[signal['symbol']]
                            if position.quantity > 0:
                                # Sell portion based on signal strength
                                sell_quantity = int(position.quantity * signal.get('strength', 1.0))
                                
                                if sell_quantity > 0:
                                    self.add_order(
                                        timestamp=current_datetime,
                                        symbol=signal['symbol'],
                                        side=OrderSide.SELL,
                                        quantity=sell_quantity
                                    )
            
            # Update portfolio at end of day
            if current_prices:
                self.update_portfolio(current_datetime, current_prices)
        
        # Calculate final metrics
        return self._calculate_final_metrics(start_date, end_date)
    
    def _calculate_final_metrics(self, start_date: datetime, end_date: datetime) -> BacktestMetrics:
        """Calculate final backtest metrics."""
        
        # Get final portfolio value
        final_value = self.portfolio_history[-1]['total_value'] if self.portfolio_history else self.initial_capital
        
        # Calculate daily returns
        daily_returns = []
        if len(self.portfolio_history) > 1:
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1]['total_value']
                curr_value = self.portfolio_history[i]['total_value']
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    daily_returns.append(daily_return)
        
        # Calculate drawdown
        max_drawdown, max_dd_duration = self._calculate_drawdown()
        
        # Count trades
        buy_trades = [t for t in self.trades if t['side'] == 'BUY']
        sell_trades = [t for t in self.trades if t['side'] == 'SELL']
        
        winning_trades = len([t for t in sell_trades if t.get('realized_pnl', 0) > 0])
        losing_trades = len([t for t in sell_trades if t.get('realized_pnl', 0) < 0])
        
        # Create metrics object
        metrics = BacktestMetrics(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_value,
            daily_returns=daily_returns,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(sell_trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )
        
        # Calculate derived metrics
        metrics.calculate_metrics()
        
        return metrics
    
    def _calculate_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if not self.portfolio_history:
            return 0.0, 0
        
        values = [h['total_value'] for h in self.portfolio_history]
        
        # Calculate running maximum and drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        
        # Calculate maximum drawdown duration
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_drawdown, max_duration
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        return pd.DataFrame(self.portfolio_history)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades history as DataFrame."""
        return pd.DataFrame(self.trades)
    
    def generate_backtest_report(self, metrics: BacktestMetrics) -> str:
        """Generate comprehensive backtest report."""
        
        report = f"""
# Backtest Report

## Summary
- **Period**: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}
- **Duration**: {metrics.days_traded} days
- **Initial Capital**: ${metrics.initial_capital:,.2f}
- **Final Capital**: ${metrics.final_capital:,.2f}

## Returns
- **Total Return**: {metrics.total_return:.2%}
- **Annual Return**: {metrics.annual_return:.2%}
- **Volatility**: {metrics.volatility:.2%}

## Risk Metrics
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Maximum Drawdown**: {metrics.max_drawdown:.2%}
- **Max Drawdown Duration**: {metrics.max_drawdown_duration} days

## Trading Statistics
- **Total Trades**: {metrics.total_trades}
- **Winning Trades**: {metrics.winning_trades}
- **Losing Trades**: {metrics.losing_trades}
- **Win Rate**: {metrics.win_rate:.2%}

## Performance Ratios
- **Profit Factor**: {metrics.profit_factor:.2f}
- **Calmar Ratio**: {metrics.calmar_ratio:.2f}
- **Sortino Ratio**: {metrics.sortino_ratio:.2f}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        return report.strip()


class StrategyBacktester:
    """High-level strategy backtesting interface."""
    
    def __init__(self, config: TestConfigManager):
        self.config = config
        self.engine = BacktestEngine(config)
        
    async def backtest_ml_strategy(self, symbols: List[str], 
                                 start_date: datetime, end_date: datetime,
                                 signal_generator=None) -> Dict[str, Any]:
        """Backtest ML-based trading strategy."""
        
        # Generate market data
        market_data = {}
        mock_data = MockMarketData()
        
        for symbol in symbols:
            data = mock_data.generate_ohlcv(
                symbol=symbol,
                days=(end_date - start_date).days,
                freq='1D'
            )
            # Adjust timestamps to match backtest period
            data['timestamp'] = pd.date_range(start=start_date, end=end_date, periods=len(data))
            market_data[symbol] = data
        
        # Generate trading signals
        if signal_generator is None:
            # Use mock signals
            signals = []
            mock_signal_gen = MockSignalData()
            
            for symbol in symbols:
                symbol_signals = mock_signal_gen.generate_signal_history(
                    [symbol], 
                    days=(end_date - start_date).days
                )
                signals.extend(symbol_signals)
        else:
            signals = await signal_generator(market_data, start_date, end_date)
        
        # Run backtest
        metrics = self.engine.run_backtest(market_data, signals, start_date, end_date)
        
        # Generate report
        report = self.engine.generate_backtest_report(metrics)
        
        return {
            'metrics': metrics,
            'report': report,
            'portfolio_history': self.engine.get_portfolio_history_df(),
            'trades': self.engine.get_trades_df(),
            'market_data': market_data
        }
    
    def compare_strategies(self, strategies: Dict[str, Any], 
                          symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Compare multiple trading strategies."""
        
        results = {}
        
        for strategy_name, strategy_config in strategies.items():
            print(f"Testing strategy: {strategy_name}")
            
            # Run backtest for this strategy
            result = asyncio.run(self.backtest_ml_strategy(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                signal_generator=strategy_config.get('signal_generator')
            ))
            
            results[strategy_name] = result
        
        # Generate comparison report
        comparison_report = self._generate_strategy_comparison(results)
        
        return {
            'individual_results': results,
            'comparison_report': comparison_report
        }
    
    def _generate_strategy_comparison(self, results: Dict[str, Dict]) -> str:
        """Generate strategy comparison report."""
        
        comparison_data = []
        
        for strategy_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annual_return:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Total Trades': metrics.total_trades
            })
        
        # Convert to DataFrame for better formatting
        comparison_df = pd.DataFrame(comparison_data)
        
        report = f"""
# Strategy Comparison Report

## Performance Summary

{comparison_df.to_string(index=False)}

## Key Insights

"""
        
        # Find best performing strategies
        best_return = max(results.values(), key=lambda x: x['metrics'].total_return)
        best_sharpe = max(results.values(), key=lambda x: x['metrics'].sharpe_ratio)
        lowest_drawdown = min(results.values(), key=lambda x: abs(x['metrics'].max_drawdown))
        
        # Add insights to report
        for strategy_name, result in results.items():
            if result == best_return:
                report += f"- **Best Total Return**: {strategy_name} ({result['metrics'].total_return:.2%})\\n"
            if result == best_sharpe:
                report += f"- **Best Risk-Adjusted Return**: {strategy_name} (Sharpe: {result['metrics'].sharpe_ratio:.2f})\\n"
            if result == lowest_drawdown:
                report += f"- **Lowest Drawdown**: {strategy_name} ({result['metrics'].max_drawdown:.2%})\\n"
        
        return report