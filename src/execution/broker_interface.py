"""
Broker Interface

Manages connections and interactions with multiple brokers.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
from enum import Enum

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.execution.order_manager import Order, OrderStatus, OrderType, OrderSide, Fill


class BrokerType(Enum):
    """Supported broker types."""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    BINANCE = "binance"
    PAPER_TRADING = "paper_trading"


@dataclass
class BrokerConfig:
    """Broker configuration."""
    broker_type: BrokerType
    api_key: str
    api_secret: str
    base_url: str
    paper_trading: bool = True
    enabled: bool = True
    rate_limit_per_minute: int = 200
    timeout_seconds: int = 30


@dataclass
class AccountInfo:
    """Account information from broker."""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    day_trading_buying_power: float
    pattern_day_trader: bool
    currency: str = "USD"
    timestamp: datetime = None


class BrokerInterface(ABC):
    """Abstract base class for broker interfaces."""
    
    def __init__(self, config: BrokerConfig, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.session = None
        self.connected = False
        
        # Rate limiting
        self.request_times = []
        self.rate_limit_lock = asyncio.Lock()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker API."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        """Submit order to broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at broker."""
        pass
    
    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> Optional[OrderStatus]:
        """Get order status from broker."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        pass
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits."""
        async with self.rate_limit_lock:
            now = datetime.now()
            
            # Remove requests older than 1 minute
            minute_ago = now - timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > minute_ago]
            
            # Check if we're at the limit
            if len(self.request_times) >= self.config.rate_limit_per_minute:
                sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Record this request
            self.request_times.append(now)


class AlpacaInterface(BrokerInterface):
    """Alpaca broker interface."""
    
    def __init__(self, config: BrokerConfig, logger: TradingLogger):
        super().__init__(config, logger)
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.api_secret
        }
    
    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                headers=self.headers
            )
            
            # Test connection
            async with self.session.get(f"{self.config.base_url}/v2/account") as response:
                if response.status == 200:
                    self.connected = True
                    self.logger.info("Connected to Alpaca")
                    return True
                else:
                    self.logger.error(f"Failed to connect to Alpaca: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error connecting to Alpaca: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca API."""
        if self.session:
            await self.session.close()
        self.connected = False
        self.logger.info("Disconnected from Alpaca")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order to Alpaca."""
        try:
            await self._rate_limit_check()
            
            # Convert our order to Alpaca format
            alpaca_order = self._convert_to_alpaca_order(order)
            
            async with self.session.post(
                f"{self.config.base_url}/v2/orders",
                json=alpaca_order
            ) as response:
                
                if response.status == 201:
                    response_data = await response.json()
                    order.broker_order_id = response_data.get("id")
                    self.logger.info(f"Submitted order {order.id} to Alpaca: {order.broker_order_id}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to submit order to Alpaca: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error submitting order to Alpaca: {e}")
            return False
    
    def _convert_to_alpaca_order(self, order: Order) -> Dict[str, Any]:
        """Convert internal order to Alpaca format."""
        alpaca_order = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": self._convert_order_type(order.order_type),
            "time_in_force": self._convert_time_in_force(order.time_in_force),
            "client_order_id": order.id
        }
        
        if order.price:
            alpaca_order["limit_price"] = str(order.price)
        
        if order.stop_price:
            alpaca_order["stop_price"] = str(order.stop_price)
        
        return alpaca_order
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Alpaca format."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop"
        }
        return mapping.get(order_type, "market")
    
    def _convert_time_in_force(self, tif) -> str:
        """Convert time in force to Alpaca format."""
        return tif.value.lower()
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at Alpaca."""
        try:
            await self._rate_limit_check()
            
            async with self.session.delete(
                f"{self.config.base_url}/v2/orders/{broker_order_id}"
            ) as response:
                
                if response.status == 204:
                    self.logger.info(f"Cancelled Alpaca order {broker_order_id}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to cancel Alpaca order: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error cancelling Alpaca order: {e}")
            return False
    
    async def get_order_status(self, broker_order_id: str) -> Optional[OrderStatus]:
        """Get order status from Alpaca."""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(
                f"{self.config.base_url}/v2/orders/{broker_order_id}"
            ) as response:
                
                if response.status == 200:
                    order_data = await response.json()
                    return self._convert_alpaca_status(order_data.get("status"))
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting Alpaca order status: {e}")
            return None
    
    def _convert_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to internal format."""
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """Get Alpaca account information."""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(f"{self.config.base_url}/v2/account") as response:
                if response.status == 200:
                    account_data = await response.json()
                    
                    return AccountInfo(
                        account_id=account_data.get("account_number"),
                        buying_power=float(account_data.get("buying_power", 0)),
                        cash=float(account_data.get("cash", 0)),
                        portfolio_value=float(account_data.get("portfolio_value", 0)),
                        day_trading_buying_power=float(account_data.get("daytrading_buying_power", 0)),
                        pattern_day_trader=account_data.get("pattern_day_trader", False),
                        currency=account_data.get("currency", "USD"),
                        timestamp=datetime.now()
                    )
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting Alpaca account info: {e}")
            return None
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get Alpaca positions."""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(f"{self.config.base_url}/v2/positions") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error getting Alpaca positions: {e}")
            return []


class PaperTradingInterface(BrokerInterface):
    """Paper trading (simulation) broker interface."""
    
    def __init__(self, config: BrokerConfig, logger: TradingLogger):
        super().__init__(config, logger)
        
        # Simulated account state
        self.account = AccountInfo(
            account_id="PAPER_ACCOUNT",
            buying_power=100000.0,
            cash=100000.0,
            portfolio_value=100000.0,
            day_trading_buying_power=400000.0,
            pattern_day_trader=False,
            timestamp=datetime.now()
        )
        
        # Simulated order and position tracking
        self.simulated_orders = {}
        self.simulated_positions = {}
        
        # Market data simulation
        self.market_prices = {}
    
    async def connect(self) -> bool:
        """Connect to paper trading (always succeeds)."""
        self.connected = True
        self.logger.info("Connected to Paper Trading")
        return True
    
    async def disconnect(self):
        """Disconnect from paper trading."""
        self.connected = False
        self.logger.info("Disconnected from Paper Trading")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order to paper trading."""
        try:
            # Generate simulated broker order ID
            order.broker_order_id = f"PAPER_{order.id}"
            
            # Store order
            self.simulated_orders[order.broker_order_id] = {
                "order": order,
                "status": OrderStatus.ACCEPTED,
                "created_at": datetime.now()
            }
            
            # Simulate order processing
            asyncio.create_task(self._simulate_order_execution(order))
            
            self.logger.info(f"Submitted paper order {order.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting paper order: {e}")
            return False
    
    async def _simulate_order_execution(self, order: Order):
        """Simulate order execution with random delays and partial fills."""
        try:
            await asyncio.sleep(0.1 + (hash(order.id) % 10) / 10)  # Random delay 0.1-1.1s
            
            # Get simulated market price
            market_price = self._get_simulated_price(order.symbol)
            
            # Determine if order should fill based on type
            should_fill = False
            fill_price = market_price
            
            if order.order_type == OrderType.MARKET:
                should_fill = True
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.price >= market_price:
                    should_fill = True
                    fill_price = order.price
                elif order.side == OrderSide.SELL and order.price <= market_price:
                    should_fill = True
                    fill_price = order.price
            
            if should_fill:
                # Simulate partial or complete fill
                fill_quantity = order.quantity
                if hash(order.id) % 10 == 0:  # 10% chance of partial fill
                    fill_quantity = order.quantity * 0.5
                
                # Update simulated account
                self._update_simulated_account(order, fill_quantity, fill_price)
                
                # Mark order as filled (this would normally be done by order manager)
                if order.broker_order_id in self.simulated_orders:
                    sim_order = self.simulated_orders[order.broker_order_id]
                    if fill_quantity >= order.quantity:
                        sim_order["status"] = OrderStatus.FILLED
                    else:
                        sim_order["status"] = OrderStatus.PARTIALLY_FILLED
            
        except Exception as e:
            self.logger.error(f"Error simulating order execution: {e}")
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated market price for symbol."""
        # Use cached price or generate random price
        if symbol not in self.market_prices:
            base_price = hash(symbol) % 1000 + 50  # Random price between $50-$1050
            self.market_prices[symbol] = float(base_price)
        
        # Add small random movement
        current_price = self.market_prices[symbol]
        movement = (hash(str(datetime.now().timestamp())) % 201 - 100) / 10000  # ±1%
        new_price = current_price * (1 + movement)
        self.market_prices[symbol] = new_price
        
        return new_price
    
    def _update_simulated_account(self, order: Order, fill_quantity: float, fill_price: float):
        """Update simulated account state after fill."""
        trade_value = fill_quantity * fill_price
        commission = 0.0  # No commission in paper trading
        
        if order.side == OrderSide.BUY:
            # Reduce cash
            self.account.cash -= trade_value + commission
            
            # Add to position
            if order.symbol in self.simulated_positions:
                pos = self.simulated_positions[order.symbol]
                total_quantity = pos["quantity"] + fill_quantity
                total_value = pos["quantity"] * pos["avg_price"] + trade_value
                pos["quantity"] = total_quantity
                pos["avg_price"] = total_value / total_quantity
            else:
                self.simulated_positions[order.symbol] = {
                    "quantity": fill_quantity,
                    "avg_price": fill_price
                }
        
        else:  # SELL
            # Add cash
            self.account.cash += trade_value - commission
            
            # Reduce position
            if order.symbol in self.simulated_positions:
                pos = self.simulated_positions[order.symbol]
                pos["quantity"] -= fill_quantity
                
                if pos["quantity"] <= 0:
                    del self.simulated_positions[order.symbol]
        
        # Update portfolio value
        portfolio_value = self.account.cash
        for symbol, pos in self.simulated_positions.items():
            market_price = self._get_simulated_price(symbol)
            portfolio_value += pos["quantity"] * market_price
        
        self.account.portfolio_value = portfolio_value
        self.account.buying_power = self.account.cash  # Simplified
        self.account.timestamp = datetime.now()
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel paper order."""
        if broker_order_id in self.simulated_orders:
            self.simulated_orders[broker_order_id]["status"] = OrderStatus.CANCELLED
            self.logger.info(f"Cancelled paper order {broker_order_id}")
            return True
        return False
    
    async def get_order_status(self, broker_order_id: str) -> Optional[OrderStatus]:
        """Get paper order status."""
        if broker_order_id in self.simulated_orders:
            return self.simulated_orders[broker_order_id]["status"]
        return None
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """Get paper account information."""
        return self.account
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get paper positions."""
        positions = []
        for symbol, pos in self.simulated_positions.items():
            market_price = self._get_simulated_price(symbol)
            positions.append({
                "symbol": symbol,
                "qty": pos["quantity"],
                "avg_entry_price": pos["avg_price"],
                "market_value": pos["quantity"] * market_price,
                "unrealized_pl": pos["quantity"] * (market_price - pos["avg_price"])
            })
        return positions


class BrokerManager:
    """Manages multiple broker connections."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("BrokerManager")
        
        # Broker interfaces
        self.brokers: Dict[BrokerType, BrokerInterface] = {}
        self.primary_broker: Optional[BrokerInterface] = None
        
        # Connection status
        self.connection_status: Dict[BrokerType, bool] = {}
        
        # Load broker configurations
        self.broker_configs = self._load_broker_configs()
    
    def _load_broker_configs(self) -> Dict[BrokerType, BrokerConfig]:
        """Load broker configurations from config."""
        configs = {}
        
        broker_settings = self.config.get('brokers', {})
        
        for broker_name, settings in broker_settings.items():
            try:
                broker_type = BrokerType(broker_name.lower())
                
                config = BrokerConfig(
                    broker_type=broker_type,
                    api_key=settings.get('api_key', ''),
                    api_secret=settings.get('api_secret', ''),
                    base_url=settings.get('base_url', ''),
                    paper_trading=settings.get('paper_trading', True),
                    enabled=settings.get('enabled', False),
                    rate_limit_per_minute=settings.get('rate_limit_per_minute', 200),
                    timeout_seconds=settings.get('timeout_seconds', 30)
                )
                
                configs[broker_type] = config
                
            except ValueError:
                self.logger.warning(f"Unknown broker type: {broker_name}")
        
        # Always add paper trading
        if BrokerType.PAPER_TRADING not in configs:
            configs[BrokerType.PAPER_TRADING] = BrokerConfig(
                broker_type=BrokerType.PAPER_TRADING,
                api_key="paper",
                api_secret="paper",
                base_url="",
                paper_trading=True,
                enabled=True
            )
        
        return configs
    
    async def initialize(self):
        """Initialize broker manager and connect to brokers."""
        self.logger.info("Initializing Broker Manager")
        
        # Create broker interfaces
        for broker_type, config in self.broker_configs.items():
            if not config.enabled:
                continue
            
            if broker_type == BrokerType.ALPACA:
                interface = AlpacaInterface(config, self.logger)
            elif broker_type == BrokerType.PAPER_TRADING:
                interface = PaperTradingInterface(config, self.logger)
            else:
                self.logger.warning(f"Broker interface not implemented: {broker_type.value}")
                continue
            
            self.brokers[broker_type] = interface
            
            # Try to connect
            connected = await interface.connect()
            self.connection_status[broker_type] = connected
            
            if connected and self.primary_broker is None:
                self.primary_broker = interface
                self.logger.info(f"Set {broker_type.value} as primary broker")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order to primary broker."""
        if not self.primary_broker:
            self.logger.error("No primary broker available")
            return False
        
        return await self.primary_broker.submit_order(order)
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at primary broker."""
        if not self.primary_broker:
            return False
        
        return await self.primary_broker.cancel_order(broker_order_id)
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """Get account info from primary broker."""
        if not self.primary_broker:
            return None
        
        return await self.primary_broker.get_account_info()
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from primary broker."""
        if not self.primary_broker:
            return []
        
        return await self.primary_broker.get_positions()
    
    def switch_broker(self, broker_type: BrokerType) -> bool:
        """Switch to different broker as primary."""
        if broker_type in self.brokers and self.connection_status.get(broker_type, False):
            self.primary_broker = self.brokers[broker_type]
            self.logger.info(f"Switched to {broker_type.value} as primary broker")
            return True
        
        self.logger.error(f"Cannot switch to {broker_type.value}: not connected")
        return False
    
    def get_broker_status(self) -> Dict[str, Any]:
        """Get status of all brokers."""
        status = {}
        
        for broker_type, connected in self.connection_status.items():
            status[broker_type.value] = {
                'connected': connected,
                'is_primary': self.primary_broker and broker_type in self.brokers and 
                            self.brokers[broker_type] == self.primary_broker,
                'paper_trading': self.broker_configs.get(broker_type, BrokerConfig(
                    BrokerType.PAPER_TRADING, "", "", "")).paper_trading
            }
        
        return status
    
    async def shutdown(self):
        """Shutdown all broker connections."""
        self.logger.info("Shutting down broker connections")
        
        for broker in self.brokers.values():
            await broker.disconnect()
        
        self.brokers.clear()
        self.connection_status.clear()
        self.primary_broker = None