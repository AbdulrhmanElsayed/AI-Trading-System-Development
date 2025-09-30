"""
Portfolio Manager

Manages portfolio state, positions, and performance tracking.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict, replace
import json

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import Portfolio, Position, PerformanceMetrics, MarketData


class PortfolioManager:
    """Manages portfolio positions and performance."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("PortfolioManager")
        
        # Portfolio configuration
        self.initial_capital = config.get('trading.initial_capital', 100000.0)
        self.base_currency = config.get('trading.base_currency', 'USD')
        
        # Current portfolio state
        self.portfolio = Portfolio(cash=self.initial_capital)
        
        # Performance tracking
        self.performance_history = []
        self.daily_returns = []
        self.trade_history = []
        
        # Benchmarking
        self.benchmark_symbol = config.get('trading.benchmark_symbol', 'SPY')
        self.benchmark_returns = []
        
        # Portfolio snapshots for analysis
        self.portfolio_snapshots = []
        self.last_snapshot_time = datetime.now()
    
    async def initialize(self):
        """Initialize portfolio manager."""
        self.logger.info(
            f"Portfolio Manager initialized with ${self.initial_capital:,.2f} initial capital"
        )
        
        # Load any existing portfolio state
        await self._load_portfolio_state()
        
        # Take initial snapshot
        await self._take_portfolio_snapshot()
    
    async def update_position(self, trade_data: Dict[str, Any]):
        """Update portfolio based on executed trade."""
        try:
            symbol = trade_data['symbol']
            quantity = float(trade_data['quantity'])
            price = float(trade_data['price'])
            side = trade_data['side']  # 'buy' or 'sell'
            commission = float(trade_data.get('commission', 0))
            timestamp = trade_data.get('timestamp', datetime.now())
            
            # Calculate trade value
            trade_value = quantity * price
            total_cost = trade_value + commission
            
            # Update position
            if side.lower() == 'buy':
                await self._add_to_position(symbol, quantity, price, total_cost, timestamp)
            else:  # sell
                await self._reduce_position(symbol, quantity, price, total_cost, timestamp)
            
            # Record trade
            self.trade_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'total_cost': total_cost
            })
            
            # Update portfolio timestamp
            self.portfolio.timestamp = timestamp
            
            self.logger.info(
                f"Portfolio updated: {side} {quantity} {symbol} @ ${price:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise
    
    async def _add_to_position(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        total_cost: float,
        timestamp: datetime
    ):
        """Add to existing position or create new one."""
        if symbol in self.portfolio.positions:
            # Update existing position
            current_position = self.portfolio.positions[symbol]
            
            # Calculate new average price
            current_value = current_position.quantity * current_position.average_price
            new_total_quantity = current_position.quantity + quantity
            new_total_value = current_value + total_cost
            new_average_price = new_total_value / new_total_quantity
            
            # Update position
            self.portfolio.positions[symbol] = replace(
                current_position,
                quantity=new_total_quantity,
                average_price=new_average_price,
                timestamp=timestamp
            )
        else:
            # Create new position
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=total_cost / quantity,  # Include commission in cost basis
                market_price=price,
                timestamp=timestamp
            )
        
        # Reduce available cash
        self.portfolio.cash -= total_cost
    
    async def _reduce_position(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        total_cost: float,
        timestamp: datetime
    ):
        """Reduce existing position."""
        if symbol not in self.portfolio.positions:
            self.logger.warning(f"Attempted to sell non-existent position: {symbol}")
            return
        
        current_position = self.portfolio.positions[symbol]
        
        if quantity > current_position.quantity:
            self.logger.warning(
                f"Attempted to sell more than available: {quantity} > {current_position.quantity}"
            )
            quantity = current_position.quantity  # Sell all available
        
        # Calculate realized P&L
        cost_basis = quantity * current_position.average_price
        sale_proceeds = (quantity * price) - (total_cost - (quantity * price))  # Remove commission
        realized_pnl = sale_proceeds - cost_basis
        
        # Update position
        remaining_quantity = current_position.quantity - quantity
        
        if remaining_quantity <= 0.000001:  # Essentially zero
            # Close position completely
            del self.portfolio.positions[symbol]
        else:
            # Partially reduce position
            self.portfolio.positions[symbol] = replace(
                current_position,
                quantity=remaining_quantity,
                realized_pnl=current_position.realized_pnl + realized_pnl,
                timestamp=timestamp
            )
        
        # Add proceeds to cash
        net_proceeds = (quantity * price) - (total_cost - (quantity * price))
        self.portfolio.cash += net_proceeds
    
    async def update_market_prices(self, market_data: Dict[str, MarketData]):
        """Update market prices for all positions."""
        try:
            for symbol, position in self.portfolio.positions.items():
                if symbol in market_data:
                    new_price = market_data[symbol].price
                    
                    # Calculate unrealized P&L
                    market_value = position.quantity * new_price
                    cost_basis = position.quantity * position.average_price
                    unrealized_pnl = market_value - cost_basis
                    
                    # Update position
                    self.portfolio.positions[symbol] = replace(
                        position,
                        market_price=new_price,
                        unrealized_pnl=unrealized_pnl,
                        timestamp=datetime.now()
                    )
            
            # Update portfolio timestamp
            self.portfolio.timestamp = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating market prices: {e}")
    
    async def get_current_portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        return self.portfolio
    
    async def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            # Calculate total portfolio value
            total_value = self.portfolio.total_value
            total_pnl = self.portfolio.total_pnl
            
            if not self.daily_returns:
                return PerformanceMetrics()
            
            # Convert to numpy array for calculations
            returns = np.array(self.daily_returns)
            
            # Basic metrics
            total_return = (total_value - self.initial_capital) / self.initial_capital
            
            # Annualized return (assuming 252 trading days)
            if len(returns) > 0:
                days = len(returns)
                annualized_return = (1 + total_return) ** (252 / days) - 1
            else:
                annualized_return = 0.0
            
            # Volatility (annualized)
            if len(returns) > 1:
                daily_vol = np.std(returns, ddof=1)
                volatility = daily_vol * np.sqrt(252)
            else:
                volatility = 0.0
            
            # Sharpe Ratio
            risk_free_rate = self.config.get('risk_management.risk_free_rate', 0.02)
            if volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Maximum Drawdown
            max_drawdown = await self._calculate_max_drawdown()
            
            # Win Rate and other trade statistics
            trade_stats = await self._calculate_trade_statistics()
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=trade_stats['win_rate'],
                profit_factor=trade_stats['profit_factor'],
                total_trades=trade_stats['total_trades'],
                winning_trades=trade_stats['winning_trades'],
                losing_trades=trade_stats['losing_trades'],
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio snapshots."""
        if not self.portfolio_snapshots:
            return 0.0
        
        values = [snapshot['total_value'] for snapshot in self.portfolio_snapshots]
        
        if not values:
            return 0.0
        
        # Calculate running maximum
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate trade-based statistics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Group trades by symbol to calculate P&L per trade
        symbol_trades = {}
        for trade in self.trade_history:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        gross_profits = 0.0
        gross_losses = 0.0
        
        # Analyze completed trades (buy then sell pairs)
        for symbol, trades in symbol_trades.items():
            position = 0
            cost_basis = 0
            
            for trade in sorted(trades, key=lambda x: x['timestamp']):
                if trade['side'].lower() == 'buy':
                    # Add to position
                    new_cost = position * cost_basis + trade['quantity'] * trade['price']
                    position += trade['quantity']
                    cost_basis = new_cost / position if position > 0 else 0
                else:  # sell
                    if position > 0:
                        # Calculate P&L for this sale
                        sell_quantity = min(trade['quantity'], position)
                        sale_proceeds = sell_quantity * trade['price']
                        sale_cost = sell_quantity * cost_basis
                        pnl = sale_proceeds - sale_cost - trade['commission']
                        
                        total_trades += 1
                        
                        if pnl > 0:
                            winning_trades += 1
                            gross_profits += pnl
                        else:
                            losing_trades += 1
                            gross_losses += abs(pnl)
                        
                        position -= sell_quantity
        
        # Calculate statistics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    async def update_daily_returns(self):
        """Update daily returns calculation."""
        try:
            current_value = self.portfolio.total_value
            
            if not self.daily_returns:
                # First day
                self.daily_returns.append(0.0)
                self.last_portfolio_value = current_value
                return
            
            # Calculate daily return
            if hasattr(self, 'last_portfolio_value') and self.last_portfolio_value > 0:
                daily_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
                self.daily_returns.append(daily_return)
            
            self.last_portfolio_value = current_value
            
            # Keep only recent returns to avoid memory issues
            if len(self.daily_returns) > 1000:
                self.daily_returns = self.daily_returns[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error updating daily returns: {e}")
    
    async def _take_portfolio_snapshot(self):
        """Take a snapshot of current portfolio state."""
        try:
            snapshot = {
                'timestamp': datetime.now(),
                'total_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'positions_count': len(self.portfolio.positions),
                'total_pnl': self.portfolio.total_pnl,
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl
                    }
                    for symbol, pos in self.portfolio.positions.items()
                }
            }
            
            self.portfolio_snapshots.append(snapshot)
            self.last_snapshot_time = datetime.now()
            
            # Keep only recent snapshots
            if len(self.portfolio_snapshots) > 10000:
                self.portfolio_snapshots = self.portfolio_snapshots[-10000:]
            
        except Exception as e:
            self.logger.error(f"Error taking portfolio snapshot: {e}")
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        summary = {
            'total_positions': len(self.portfolio.positions),
            'total_market_value': sum(pos.market_value for pos in self.portfolio.positions.values()),
            'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.portfolio.positions.values()),
            'total_realized_pnl': sum(pos.realized_pnl for pos in self.portfolio.positions.values()),
            'positions': []
        }
        
        for symbol, position in self.portfolio.positions.items():
            pos_dict = asdict(position)
            pos_dict['pnl_percent'] = (
                position.unrealized_pnl / (position.quantity * position.average_price)
                if position.quantity > 0 and position.average_price > 0 else 0.0
            )
            summary['positions'].append(pos_dict)
        
        # Sort by market value descending
        summary['positions'].sort(key=lambda x: x['market_value'], reverse=True)
        
        return summary
    
    async def _load_portfolio_state(self):
        """Load portfolio state from storage (placeholder)."""
        # This would load from database or file
        # For now, use initial state
        pass
    
    async def _save_portfolio_state(self):
        """Save portfolio state to storage (placeholder)."""
        # This would save to database or file
        # For now, just log
        self.logger.info(f"Portfolio state: Value=${self.portfolio.total_value:,.2f}")
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics."""
        return {
            'initial_capital': self.initial_capital,
            'current_value': self.portfolio.total_value,
            'cash_balance': self.portfolio.cash,
            'total_pnl': self.portfolio.total_pnl,
            'total_return_pct': (
                (self.portfolio.total_value - self.initial_capital) / self.initial_capital * 100
            ),
            'positions_count': len(self.portfolio.positions),
            'trades_count': len(self.trade_history),
            'snapshots_count': len(self.portfolio_snapshots),
            'last_update': self.portfolio.timestamp.isoformat() if self.portfolio.timestamp else None
        }