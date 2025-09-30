"""
Execution Engine

Main execution engine that coordinates order management and broker interactions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import MarketData, TradingSignal
from src.execution.order_manager import OrderManager, Order, OrderType, OrderSide, OrderStatus, TimeInForce
from src.execution.broker_interface import BrokerManager, AccountInfo


class ExecutionEngine:
    """Main execution engine for trading operations."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("ExecutionEngine")
        
        # Core components
        self.order_manager = OrderManager(config)
        self.broker_manager = BrokerManager(config)
        
        # Execution configuration
        self.execution_enabled = config.get('execution.enabled', False)
        self.paper_trading_mode = config.get('execution.paper_trading', True)
        self.max_orders_per_symbol = config.get('execution.max_orders_per_symbol', 5)
        self.min_order_value = config.get('execution.min_order_value', 100)  # $100
        
        # Position management
        self.max_position_value = config.get('execution.max_position_value', 50000)  # $50K
        self.position_sizing_method = config.get('execution.position_sizing_method', 'fixed_dollar')
        self.default_position_size = config.get('execution.default_position_size', 1000)  # $1K
        
        # Risk controls
        self.enable_risk_checks = config.get('execution.enable_risk_checks', True)
        self.max_daily_loss = config.get('execution.max_daily_loss', 5000)  # $5K
        self.max_drawdown_percent = config.get('execution.max_drawdown_percent', 0.10)  # 10%
        
        # State tracking
        self.account_info: Optional[AccountInfo] = None
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.execution_paused = False
        
        # Callbacks
        self.signal_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        
        # Performance tracking
        self.execution_metrics = {
            'signals_received': 0,
            'orders_created': 0,
            'orders_executed': 0,
            'total_volume': 0.0,
            'avg_execution_latency': 0.0,
            'last_update': datetime.now()
        }
    
    async def initialize(self):
        """Initialize execution engine."""
        self.logger.info("Initializing Execution Engine")
        
        # Initialize components
        await self.order_manager.initialize()
        await self.broker_manager.initialize()
        
        # Set up order manager callbacks
        self.order_manager.add_fill_callback(self._on_order_fill)
        
        # Load account information
        await self._update_account_info()
        
        # Start monitoring tasks
        asyncio.create_task(self._account_monitoring_loop())
        asyncio.create_task(self._position_monitoring_loop())
        
        self.logger.info(f"Execution Engine initialized (Paper Trading: {self.paper_trading_mode})")
    
    async def process_trading_signal(self, signal: TradingSignal) -> bool:
        """Process trading signal and generate orders."""
        try:
            self.execution_metrics['signals_received'] += 1
            
            if not self.execution_enabled or self.execution_paused:
                self.logger.debug(f"Execution disabled/paused, ignoring signal for {signal.symbol}")
                return False
            
            # Run signal callbacks
            for callback in self.signal_callbacks:
                await callback(signal)
            
            # Pre-execution risk checks
            if self.enable_risk_checks:
                if not await self._pre_execution_risk_check(signal):
                    return False
            
            # Generate orders based on signal
            orders = await self._generate_orders_from_signal(signal)
            
            if not orders:
                self.logger.debug(f"No orders generated for signal: {signal.symbol}")
                return False
            
            # Submit orders
            success = True
            for order in orders:
                if await self._submit_order_with_checks(order):
                    self.execution_metrics['orders_created'] += 1
                else:
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing trading signal: {e}")
            return False
    
    async def _generate_orders_from_signal(self, signal: TradingSignal) -> List[Order]:
        """Generate orders based on trading signal."""
        orders = []
        
        try:
            # Determine order side
            if signal.action == 'buy':
                side = OrderSide.BUY
            elif signal.action == 'sell':
                side = OrderSide.SELL
            else:
                self.logger.warning(f"Unknown signal action: {signal.action}")
                return orders
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            
            if position_size <= 0:
                self.logger.debug(f"Position size is zero for {signal.symbol}")
                return orders
            
            # Determine order type and price
            order_type = OrderType.MARKET
            price = None
            
            if signal.entry_price and abs(signal.entry_price - signal.current_price) / signal.current_price > 0.002:
                # Use limit order if entry price is significantly different from current price
                order_type = OrderType.LIMIT
                price = signal.entry_price
            
            # Create main order
            main_order = self.order_manager.create_order(
                symbol=signal.symbol,
                side=side,
                quantity=position_size,
                order_type=order_type,
                price=price,
                time_in_force=TimeInForce.DAY,
                metadata={
                    'signal_id': signal.id if hasattr(signal, 'id') else None,
                    'signal_confidence': signal.confidence,
                    'strategy': signal.strategy if hasattr(signal, 'strategy') else 'unknown'
                }
            )
            
            orders.append(main_order)
            
            # Add stop loss and take profit orders if specified
            if signal.stop_loss or signal.take_profit:
                orders.extend(await self._create_bracket_orders(main_order, signal))
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error generating orders from signal: {e}")
            return []
    
    async def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on configuration and risk management."""
        try:
            if self.position_sizing_method == 'fixed_dollar':
                # Fixed dollar amount
                position_value = self.default_position_size
                
                if signal.current_price > 0:
                    return position_value / signal.current_price
                else:
                    return 0.0
            
            elif self.position_sizing_method == 'percent_portfolio':
                # Percentage of portfolio
                if not self.account_info:
                    return 0.0
                
                portfolio_percent = self.config.get('execution.portfolio_percent_per_trade', 0.02)  # 2%
                position_value = self.account_info.portfolio_value * portfolio_percent
                
                if signal.current_price > 0:
                    return position_value / signal.current_price
                else:
                    return 0.0
            
            elif self.position_sizing_method == 'volatility_adjusted':
                # Adjust based on volatility (simplified)
                base_size = self.default_position_size
                
                # Get volatility from signal metadata if available
                volatility = getattr(signal, 'volatility', 0.20)  # Default 20%
                
                # Reduce size for high volatility assets
                volatility_adjustment = min(1.0, 0.20 / volatility)
                adjusted_value = base_size * volatility_adjustment
                
                if signal.current_price > 0:
                    return adjusted_value / signal.current_price
                else:
                    return 0.0
            
            else:
                # Default to fixed dollar
                if signal.current_price > 0:
                    return self.default_position_size / signal.current_price
                else:
                    return 0.0
                    
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _create_bracket_orders(self, main_order: Order, signal: TradingSignal) -> List[Order]:
        """Create stop loss and take profit orders."""
        bracket_orders = []
        
        try:
            if signal.stop_loss:
                # Create stop loss order
                stop_side = OrderSide.SELL if main_order.side == OrderSide.BUY else OrderSide.BUY
                
                stop_order = self.order_manager.create_order(
                    symbol=main_order.symbol,
                    side=stop_side,
                    quantity=main_order.quantity,
                    order_type=OrderType.STOP,
                    stop_price=signal.stop_loss,
                    time_in_force=TimeInForce.GTC,  # Good till cancelled
                    metadata={
                        'bracket_type': 'stop_loss',
                        'parent_order': main_order.id
                    }
                )
                
                stop_order.parent_order_id = main_order.id
                main_order.child_order_ids.append(stop_order.id)
                bracket_orders.append(stop_order)
            
            if signal.take_profit:
                # Create take profit order
                profit_side = OrderSide.SELL if main_order.side == OrderSide.BUY else OrderSide.BUY
                
                profit_order = self.order_manager.create_order(
                    symbol=main_order.symbol,
                    side=profit_side,
                    quantity=main_order.quantity,
                    order_type=OrderType.LIMIT,
                    price=signal.take_profit,
                    time_in_force=TimeInForce.GTC,
                    metadata={
                        'bracket_type': 'take_profit',
                        'parent_order': main_order.id
                    }
                )
                
                profit_order.parent_order_id = main_order.id
                main_order.child_order_ids.append(profit_order.id)
                bracket_orders.append(profit_order)
            
            return bracket_orders
            
        except Exception as e:
            self.logger.error(f"Error creating bracket orders: {e}")
            return []
    
    async def _submit_order_with_checks(self, order: Order) -> bool:
        """Submit order with additional checks."""
        try:
            # Check order limits per symbol
            active_orders = self.order_manager.get_active_orders(order.symbol)
            if len(active_orders) >= self.max_orders_per_symbol:
                self.logger.warning(
                    f"Too many active orders for {order.symbol}: {len(active_orders)}/{self.max_orders_per_symbol}"
                )
                return False
            
            # Check minimum order value
            estimated_value = order.quantity * (order.price or 100)  # Rough estimate
            if estimated_value < self.min_order_value:
                self.logger.warning(f"Order value ${estimated_value:.2f} below minimum ${self.min_order_value}")
                return False
            
            # Submit to order manager
            if not await self.order_manager.submit_order(order):
                return False
            
            # Submit to broker
            if not await self.broker_manager.submit_order(order):
                # Cancel the order if broker submission failed
                await self.order_manager.cancel_order(order.id)
                return False
            
            self.logger.info(f"Successfully submitted order {order.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return False
    
    async def _pre_execution_risk_check(self, signal: TradingSignal) -> bool:
        """Perform pre-execution risk checks."""
        try:
            # Check if execution is paused
            if self.execution_paused:
                return False
            
            # Check daily loss limit
            if abs(self.daily_pnl) > self.max_daily_loss:
                self.logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
                self.execution_paused = True
                return False
            
            # Check account buying power
            if self.account_info:
                position_value = self.default_position_size  # Estimate
                if position_value > self.account_info.buying_power:
                    self.logger.warning(f"Insufficient buying power: ${self.account_info.buying_power:.2f}")
                    return False
            
            # Check maximum position value
            current_position = self.current_positions.get(signal.symbol, {})
            current_value = current_position.get('market_value', 0)
            
            if current_value > self.max_position_value:
                self.logger.warning(f"Position value limit exceeded for {signal.symbol}: ${current_value:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-execution risk check: {e}")
            return False
    
    async def _on_order_fill(self, order: Order, fill):
        """Handle order fill notifications."""
        try:
            self.logger.info(
                f"Order fill: {order.symbol} {order.side.value} {fill.quantity} @ ${fill.price:.4f}"
            )
            
            # Update execution metrics
            self.execution_metrics['orders_executed'] += 1
            self.execution_metrics['total_volume'] += fill.quantity * fill.price
            
            # Update positions
            await self._update_positions()
            
            # Calculate P&L
            await self._update_daily_pnl(order, fill)
            
            # Run execution callbacks
            for callback in self.execution_callbacks:
                await callback(order, fill)
            
        except Exception as e:
            self.logger.error(f"Error handling order fill: {e}")
    
    async def _update_account_info(self):
        """Update account information from broker."""
        try:
            self.account_info = await self.broker_manager.get_account_info()
            
            if self.account_info:
                self.logger.debug(f"Account updated: ${self.account_info.portfolio_value:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating account info: {e}")
    
    async def _update_positions(self):
        """Update current positions from broker."""
        try:
            positions = await self.broker_manager.get_positions()
            
            self.current_positions = {}
            for pos in positions:
                symbol = pos.get('symbol')
                if symbol:
                    self.current_positions[symbol] = pos
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _update_daily_pnl(self, order: Order, fill):
        """Update daily P&L calculation."""
        try:
            # Simplified P&L calculation
            if order.side == OrderSide.SELL:
                # For sell orders, we realize P&L
                # This is a simplified calculation - real implementation would track cost basis
                pass
            
        except Exception as e:
            self.logger.error(f"Error updating daily P&L: {e}")
    
    async def _account_monitoring_loop(self):
        """Monitor account status periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                await self._update_account_info()
                
            except Exception as e:
                self.logger.error(f"Error in account monitoring loop: {e}")
    
    async def _position_monitoring_loop(self):
        """Monitor positions periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                await self._update_positions()
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring loop: {e}")
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all active orders, optionally filtered by symbol."""
        try:
            active_orders = self.order_manager.get_active_orders(symbol)
            cancelled_count = 0
            
            for order in active_orders:
                if await self.order_manager.cancel_order(order.id):
                    if order.broker_order_id:
                        await self.broker_manager.cancel_order(order.broker_order_id)
                    cancelled_count += 1
            
            self.logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return 0
    
    def pause_execution(self):
        """Pause trade execution."""
        self.execution_paused = True
        self.logger.warning("Execution paused")
    
    def resume_execution(self):
        """Resume trade execution."""
        self.execution_paused = False
        self.logger.info("Execution resumed")
    
    def add_signal_callback(self, callback: Callable):
        """Add callback for trading signals."""
        self.signal_callbacks.append(callback)
    
    def add_execution_callback(self, callback: Callable):
        """Add callback for order executions."""
        self.execution_callbacks.append(callback)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution status."""
        return {
            'execution_enabled': self.execution_enabled,
            'paper_trading': self.paper_trading_mode,
            'execution_paused': self.execution_paused,
            'account_info': {
                'portfolio_value': self.account_info.portfolio_value if self.account_info else 0,
                'buying_power': self.account_info.buying_power if self.account_info else 0,
                'cash': self.account_info.cash if self.account_info else 0
            } if self.account_info else {},
            'positions': len(self.current_positions),
            'active_orders': len(self.order_manager.get_active_orders()),
            'daily_pnl': self.daily_pnl,
            'broker_status': self.broker_manager.get_broker_status(),
            'execution_metrics': self.execution_metrics,
            'order_summary': self.order_manager.get_execution_summary()
        }
    
    async def shutdown(self):
        """Shutdown execution engine."""
        self.logger.info("Shutting down Execution Engine")
        
        # Cancel all active orders
        await self.cancel_all_orders()
        
        # Shutdown broker connections
        await self.broker_manager.shutdown()
        
        self.logger.info("Execution Engine shutdown complete")