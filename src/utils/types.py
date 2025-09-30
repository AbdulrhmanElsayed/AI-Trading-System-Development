"""
Common data structures and types used across the trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class MarketDataType(Enum):
    """Market data types."""
    PRICE = "PRICE"
    VOLUME = "VOLUME"
    NEWS = "NEWS"
    SENTIMENT = "SENTIMENT"
    TECHNICAL = "TECHNICAL"


@dataclass
class MarketData:
    """Market data container."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Trading signal container."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    signal_type: SignalType = SignalType.HOLD
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quantity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Order container."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    signal_type: SignalType = SignalType.HOLD
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position container."""
    symbol: str = ""
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """Calculate market value of position."""
        return self.quantity * self.market_price
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class Portfolio:
    """Portfolio container."""
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Calculate total portfolio P&L."""
        return sum(pos.total_pnl for pos in self.positions.values())


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskMetrics:
    """Risk metrics container."""
    var_95: float = 0.0  # 95% Value at Risk
    var_99: float = 0.0  # 99% Value at Risk
    expected_shortfall: float = 0.0
    beta: float = 0.0
    correlation_spy: float = 0.0
    position_concentration: float = 0.0
    leverage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)