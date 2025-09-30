"""
Order Manager

Manages order lifecycle, execution logic, and order routing.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import MarketData


class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "good_till_cancelled"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


@dataclass
class Order:
    """Order data structure."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    broker_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For bracket orders
    child_order_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_buy_order(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                              OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED


@dataclass
class Fill:
    """Order fill/execution data."""
    id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    broker_fill_id: Optional[str] = None


class OrderManager:
    """Manages order lifecycle and execution."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("OrderManager")
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_history: deque = deque(maxlen=10000)
        
        # Execution tracking
        self.pending_orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        
        # Configuration
        self.max_order_age_minutes = config.get('execution.max_order_age_minutes', 1440)  # 24 hours
        self.order_timeout_seconds = config.get('execution.order_timeout_seconds', 30)
        self.enable_bracket_orders = config.get('execution.enable_bracket_orders', True)
        
        # Callbacks and hooks
        self.pre_order_hooks: List[Callable] = []
        self.post_order_hooks: List[Callable] = []
        self.fill_callbacks: List[Callable] = []
        
        # Risk controls
        self.max_position_size = config.get('execution.max_position_size', 1000000)  # $1M
        self.max_order_size = config.get('execution.max_order_size', 100000)  # $100K
        self.daily_order_limit = config.get('execution.daily_order_limit', 1000)
        
        # Order tracking
        self.daily_order_count = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Performance metrics
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_fill_time': 0.0,
            'total_commission': 0.0,
            'last_update': datetime.now()
        }
    
    async def initialize(self):
        """Initialize order manager."""
        self.logger.info("Order Manager initialized")
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_orders())
        
        # Start daily reset task
        asyncio.create_task(self._daily_reset_task())
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Create a new order."""
        
        # Generate unique order ID
        order_id = f"ord_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        # Validate order parameters
        self._validate_order_params(symbol, side, quantity, order_type, price, stop_price)
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            metadata=metadata or {}
        )
        
        # Store order
        self.orders[order_id] = order
        self.pending_orders[order_id] = order
        
        self.logger.info(f"Created order {order_id}: {side.value} {quantity} {symbol}")
        
        return order
    
    def _validate_order_params(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float],
        stop_price: Optional[float]
    ):
        """Validate order parameters."""
        
        # Basic validations
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"{order_type.value} orders require a price")
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and stop_price is None:
            raise ValueError(f"{order_type.value} orders require a stop price")
        
        if price is not None and price <= 0:
            raise ValueError("Order price must be positive")
        
        if stop_price is not None and stop_price <= 0:
            raise ValueError("Stop price must be positive")
        
        # Risk controls
        estimated_value = quantity * (price or 100)  # Rough estimate if no price
        
        if estimated_value > self.max_order_size:
            raise ValueError(f"Order size ${estimated_value:,.2f} exceeds maximum ${self.max_order_size:,.2f}")
        
        # Daily order limit
        self._check_daily_limits()
    
    def _check_daily_limits(self):
        """Check daily order limits."""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now.date() > self.daily_reset_time.date():
            self.daily_order_count = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if self.daily_order_count >= self.daily_order_limit:
            raise ValueError(f"Daily order limit of {self.daily_order_limit} exceeded")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order for execution."""
        try:
            # Run pre-order hooks
            for hook in self.pre_order_hooks:
                await hook(order)
            
            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            # Move to active orders
            if order.id in self.pending_orders:
                del self.pending_orders[order.id]
            self.active_orders[order.id] = order
            
            # Increment daily counter
            self.daily_order_count += 1
            
            self.logger.info(f"Submitted order {order.id}")
            
            # Run post-order hooks
            for hook in self.post_order_hooks:
                await hook(order)
            
            self.execution_stats['total_orders'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if not order.is_active:
                self.logger.warning(f"Order {order_id} is not active (status: {order.status.value})")
                return False
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            self.logger.info(f"Cancelled order {order_id}")
            
            self.execution_stats['cancelled_orders'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def process_fill(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        commission: float = 0.0,
        broker_fill_id: Optional[str] = None
    ) -> bool:
        """Process a fill for an order."""
        try:
            if order_id not in self.orders:
                self.logger.error(f"Fill received for unknown order {order_id}")
                return False
            
            order = self.orders[order_id]
            
            # Validate fill
            if fill_quantity <= 0:
                self.logger.error(f"Invalid fill quantity: {fill_quantity}")
                return False
            
            if order.filled_quantity + fill_quantity > order.quantity + 0.000001:  # Small tolerance
                self.logger.error(
                    f"Fill quantity {fill_quantity} would overfill order {order_id} "
                    f"(filled: {order.filled_quantity}, total: {order.quantity})"
                )
                return False
            
            # Create fill record
            fill = Fill(
                id=f"fill_{uuid.uuid4().hex[:8]}",
                order_id=order_id,
                symbol=order.symbol,
                quantity=fill_quantity,
                price=fill_price,
                commission=commission,
                timestamp=datetime.now(),
                broker_fill_id=broker_fill_id
            )
            
            self.fills.append(fill)
            
            # Update order
            order.filled_quantity += fill_quantity
            order.commission += commission
            
            # Calculate weighted average fill price
            if order.filled_price is None:
                order.filled_price = fill_price
            else:
                total_filled_value = (order.filled_quantity - fill_quantity) * order.filled_price
                total_filled_value += fill_quantity * fill_price
                order.filled_price = total_filled_value / order.filled_quantity
            
            # Update order status
            if order.filled_quantity >= order.quantity - 0.000001:  # Complete fill
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                
                # Remove from active orders
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                
                self.execution_stats['filled_orders'] += 1
                
                self.logger.info(f"Order {order_id} completely filled at ${fill_price:.4f}")
                
            else:  # Partial fill
                order.status = OrderStatus.PARTIALLY_FILLED
                self.logger.info(
                    f"Order {order_id} partially filled: {fill_quantity} at ${fill_price:.4f} "
                    f"(total filled: {order.filled_quantity}/{order.quantity})"
                )
            
            # Update execution statistics
            self.execution_stats['total_commission'] += commission
            self._update_avg_fill_time(order)
            
            # Notify fill callbacks
            for callback in self.fill_callbacks:
                await callback(order, fill)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing fill for order {order_id}: {e}")
            return False
    
    def _update_avg_fill_time(self, order: Order):
        """Update average fill time statistics."""
        if order.submitted_at and order.filled_at:
            fill_time = (order.filled_at - order.submitted_at).total_seconds()
            
            # Update moving average
            current_avg = self.execution_stats['avg_fill_time']
            filled_orders = self.execution_stats['filled_orders']
            
            if filled_orders <= 1:
                self.execution_stats['avg_fill_time'] = fill_time
            else:
                # Weighted moving average
                self.execution_stats['avg_fill_time'] = (
                    (current_avg * (filled_orders - 1) + fill_time) / filled_orders
                )
    
    def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: Optional[float] = None,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> List[Order]:
        """Create a bracket order (entry + profit target + stop loss)."""
        
        if not self.enable_bracket_orders:
            raise ValueError("Bracket orders are not enabled")
        
        orders = []
        
        # Create entry order
        entry_order = self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT if entry_price else OrderType.MARKET,
            price=entry_price,
            time_in_force=time_in_force,
            metadata={'bracket_type': 'entry'}
        )
        orders.append(entry_order)
        
        # Create profit target order
        if profit_target:
            profit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            profit_order = self.create_order(
                symbol=symbol,
                side=profit_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=profit_target,
                time_in_force=time_in_force,
                metadata={'bracket_type': 'profit_target', 'parent_order': entry_order.id}
            )
            profit_order.parent_order_id = entry_order.id
            entry_order.child_order_ids.append(profit_order.id)
            orders.append(profit_order)
        
        # Create stop loss order
        if stop_loss:
            stop_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            stop_order = self.create_order(
                symbol=symbol,
                side=stop_side,
                quantity=quantity,
                order_type=OrderType.STOP,
                stop_price=stop_loss,
                time_in_force=time_in_force,
                metadata={'bracket_type': 'stop_loss', 'parent_order': entry_order.id}
            )
            stop_order.parent_order_id = entry_order.id
            entry_order.child_order_ids.append(stop_order.id)
            orders.append(stop_order)
        
        self.logger.info(f"Created bracket order for {symbol}: {len(orders)} orders")
        
        return orders
    
    async def _cleanup_expired_orders(self):
        """Cleanup expired orders periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                now = datetime.now()
                expired_orders = []
                
                for order_id, order in self.active_orders.items():
                    # Check age-based expiration
                    age_minutes = (now - order.created_at).total_seconds() / 60
                    if age_minutes > self.max_order_age_minutes:
                        expired_orders.append(order_id)
                        continue
                    
                    # Check day orders at market close
                    if order.time_in_force == TimeInForce.DAY:
                        # Simplified market hours check (9:30 AM - 4:00 PM EST)
                        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                        if now > market_close:
                            expired_orders.append(order_id)
                
                # Cancel expired orders
                for order_id in expired_orders:
                    order = self.active_orders[order_id]
                    order.status = OrderStatus.EXPIRED
                    del self.active_orders[order_id]
                    self.logger.info(f"Expired order {order_id}")
                
                if expired_orders:
                    self.logger.info(f"Cleaned up {len(expired_orders)} expired orders")
            
            except Exception as e:
                self.logger.error(f"Error in order cleanup: {e}")
    
    async def _daily_reset_task(self):
        """Reset daily counters at midnight."""
        while True:
            try:
                # Wait until next midnight
                now = datetime.now()
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                sleep_seconds = (tomorrow - now).total_seconds()
                
                await asyncio.sleep(sleep_seconds)
                
                # Reset daily counters
                self.daily_order_count = 0
                self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                self.logger.info("Daily order counters reset")
                
            except Exception as e:
                self.logger.error(f"Error in daily reset task: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get list of active orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        return orders
    
    def get_order_fills(self, order_id: str) -> List[Fill]:
        """Get fills for a specific order."""
        return [fill for fill in self.fills if fill.order_id == order_id]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution performance summary."""
        
        # Calculate fill rate
        total_orders = self.execution_stats['total_orders']
        fill_rate = (self.execution_stats['filled_orders'] / total_orders * 100) if total_orders > 0 else 0
        
        # Calculate rejection rate
        rejection_rate = (self.execution_stats['rejected_orders'] / total_orders * 100) if total_orders > 0 else 0
        
        return {
            'total_orders': total_orders,
            'active_orders': len(self.active_orders),
            'filled_orders': self.execution_stats['filled_orders'],
            'cancelled_orders': self.execution_stats['cancelled_orders'],
            'rejected_orders': self.execution_stats['rejected_orders'],
            'fill_rate': f"{fill_rate:.2f}%",
            'rejection_rate': f"{rejection_rate:.2f}%",
            'avg_fill_time_seconds': round(self.execution_stats['avg_fill_time'], 2),
            'total_commission': self.execution_stats['total_commission'],
            'daily_order_count': self.daily_order_count,
            'total_fills': len(self.fills),
            'last_update': self.execution_stats['last_update'].isoformat()
        }
    
    def add_pre_order_hook(self, hook: Callable):
        """Add a pre-order validation hook."""
        self.pre_order_hooks.append(hook)
    
    def add_post_order_hook(self, hook: Callable):
        """Add a post-order execution hook."""
        self.post_order_hooks.append(hook)
    
    def add_fill_callback(self, callback: Callable):
        """Add a fill notification callback."""
        self.fill_callbacks.append(callback)
    
    def export_order_data(self) -> Dict[str, Any]:
        """Export order data for analysis."""
        return {
            'orders': [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'type': order.order_type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'status': order.status.value,
                    'filled_quantity': order.filled_quantity,
                    'filled_price': order.filled_price,
                    'commission': order.commission,
                    'created_at': order.created_at.isoformat(),
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None
                }
                for order in self.orders.values()
            ],
            'fills': [
                {
                    'id': fill.id,
                    'order_id': fill.order_id,
                    'symbol': fill.symbol,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'commission': fill.commission,
                    'timestamp': fill.timestamp.isoformat()
                }
                for fill in self.fills
            ],
            'execution_stats': self.execution_stats
        }