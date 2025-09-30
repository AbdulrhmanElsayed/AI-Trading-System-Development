"""
Paper Trading Simulation

Realistic trading simulation without real money.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from pathlib import Path

from tests.fixtures.mock_data import MockMarketData, MockSignalData, MockOrderData
from tests.fixtures.test_config import TestConfigManager


class SimulationMode(Enum):
    """Paper trading simulation modes."""
    REAL_TIME = "real_time"
    ACCELERATED = "accelerated" 
    BACKTEST = "backtest"


class OrderType(Enum):
    """Order types for paper trading."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status in paper trading."""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    
    # Price fields
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Fill information
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    fills: List[Dict] = field(default_factory=list)
    
    # Metadata
    user_id: str = "default"
    strategy_id: Optional[str] = None
    notes: str = ""
    
    @property
    def remaining_quantity(self) -> int:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    def add_fill(self, quantity: int, price: float, timestamp: datetime):
        """Add a fill to this order."""
        fill = {
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'fill_id': str(uuid.uuid4())
        }
        
        self.fills.append(fill)
        self.filled_quantity += quantity
        
        # Update average fill price
        if self.fills:
            total_value = sum(f['quantity'] * f['price'] for f in self.fills)
            self.avg_fill_price = total_value / self.filled_quantity
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class PaperPosition:
    """Paper trading position."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Cost basis
    total_cost: float = 0.0
    
    def update_position(self, trade_quantity: int, trade_price: float, trade_cost: float = 0.0):
        """Update position from a trade."""
        if self.quantity == 0:
            # New position
            self.quantity = trade_quantity
            self.avg_price = trade_price
            self.total_cost = abs(trade_quantity) * trade_price + trade_cost
        else:
            if (self.quantity > 0 and trade_quantity > 0) or (self.quantity < 0 and trade_quantity < 0):
                # Adding to existing position
                old_value = self.quantity * self.avg_price
                new_value = trade_quantity * trade_price
                total_quantity = self.quantity + trade_quantity
                
                if total_quantity != 0:
                    self.avg_price = (old_value + new_value) / total_quantity
                
                self.quantity = total_quantity
                self.total_cost += abs(trade_quantity) * trade_price + trade_cost
            else:
                # Reducing or closing position
                close_quantity = min(abs(self.quantity), abs(trade_quantity))
                realized_pnl_per_share = trade_price - self.avg_price
                
                if self.quantity < 0:  # Short position
                    realized_pnl_per_share = self.avg_price - trade_price
                
                # Calculate realized P&L
                trade_realized_pnl = close_quantity * realized_pnl_per_share - trade_cost
                self.realized_pnl += trade_realized_pnl
                
                # Update quantity
                if abs(trade_quantity) >= abs(self.quantity):
                    # Position closed or flipped
                    remaining_quantity = abs(trade_quantity) - abs(self.quantity)
                    self.quantity = remaining_quantity * (1 if trade_quantity > 0 else -1)
                    
                    if remaining_quantity > 0:
                        # New position in opposite direction
                        self.avg_price = trade_price
                        self.total_cost = remaining_quantity * trade_price + trade_cost
                    else:
                        # Position closed
                        self.avg_price = 0.0
                        self.total_cost = 0.0
                else:
                    # Partial close
                    if self.quantity > 0:
                        self.quantity -= abs(trade_quantity)
                    else:
                        self.quantity += abs(trade_quantity)
        
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
    
    def update_market_value(self, current_price: float):
        """Update unrealized P&L with current market price."""
        if self.quantity != 0:
            if self.quantity > 0:  # Long position
                self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
            else:  # Short position
                self.unrealized_pnl = abs(self.quantity) * (self.avg_price - current_price)
        else:
            self.unrealized_pnl = 0.0
        
        self.total_pnl = self.realized_pnl + self.unrealized_pnl


@dataclass
class PaperAccount:
    """Paper trading account."""
    account_id: str
    initial_balance: float
    current_balance: float
    
    # Trading limits
    max_position_size: float = 0.1  # 10% of account per position
    max_daily_loss: float = 0.05   # 5% max daily loss
    
    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    
    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    
    def can_place_order(self, order: PaperOrder, current_price: float) -> Tuple[bool, str]:
        """Check if order can be placed."""
        
        # Check account balance for buy orders
        if order.side == "BUY":
            estimated_cost = order.quantity * current_price * 1.01  # Add buffer for slippage
            
            if estimated_cost > self.current_balance:
                return False, "Insufficient balance"
        
        # Check daily loss limit
        if self.daily_pnl < -abs(self.initial_balance * self.max_daily_loss):
            return False, "Daily loss limit exceeded"
        
        # Check position size limit
        position_value = order.quantity * current_price
        max_position_value = self.current_balance * self.max_position_size
        
        if position_value > max_position_value:
            return False, f"Position size exceeds limit (max: {max_position_value:.2f})"
        
        return True, "OK"
    
    def process_trade(self, order: PaperOrder, fill_price: float, commission: float = 0.0):
        """Process a completed trade."""
        
        trade_value = order.filled_quantity * fill_price
        
        if order.side == "BUY":
            self.current_balance -= (trade_value + commission)
        else:  # SELL
            self.current_balance += (trade_value - commission)
        
        # Update statistics
        self.total_trades += 1
        self.daily_trades += 1


class MarketSimulator:
    """Simulates market conditions for paper trading."""
    
    def __init__(self, config: TestConfigManager):
        self.config = config
        self.mock_data = MockMarketData()
        self.current_prices = {}
        self.price_history = {}
        self.volatility_factor = 1.0
        
    async def start_market_simulation(self, symbols: List[str], 
                                    mode: SimulationMode = SimulationMode.REAL_TIME):
        """Start market price simulation."""
        
        print(f"🔄 Starting market simulation for {len(symbols)} symbols in {mode.value} mode")
        
        # Initialize prices for all symbols
        for symbol in symbols:
            base_price = 50 + hash(symbol) % 200  # $50-250 base price
            self.current_prices[symbol] = base_price
            self.price_history[symbol] = [(datetime.now(), base_price)]
        
        if mode == SimulationMode.REAL_TIME:
            await self._real_time_simulation(symbols)
        elif mode == SimulationMode.ACCELERATED:
            await self._accelerated_simulation(symbols)
        else:  # BACKTEST
            await self._backtest_simulation(symbols)
    
    async def _real_time_simulation(self, symbols: List[str]):
        """Real-time market simulation."""
        
        while True:
            await asyncio.sleep(1.0)  # Update every second
            
            timestamp = datetime.now()
            
            for symbol in symbols:
                # Generate realistic price movement
                current_price = self.current_prices[symbol]
                
                # Random walk with volatility
                price_change_pct = np.random.normal(0, 0.001 * self.volatility_factor)  # 0.1% std dev
                new_price = current_price * (1 + price_change_pct)
                
                # Ensure positive price
                new_price = max(0.01, new_price)
                
                self.current_prices[symbol] = new_price
                self.price_history[symbol].append((timestamp, new_price))
                
                # Keep history manageable
                if len(self.price_history[symbol]) > 1000:
                    self.price_history[symbol] = self.price_history[symbol][-500:]
    
    async def _accelerated_simulation(self, symbols: List[str]):
        """Accelerated simulation (faster than real-time)."""
        
        for minute in range(1440):  # Simulate 24 hours in minutes
            timestamp = datetime.now() + timedelta(minutes=minute)
            
            for symbol in symbols:
                current_price = self.current_prices[symbol]
                
                # More volatile in accelerated mode
                price_change_pct = np.random.normal(0, 0.005 * self.volatility_factor)
                new_price = current_price * (1 + price_change_pct)
                new_price = max(0.01, new_price)
                
                self.current_prices[symbol] = new_price
                self.price_history[symbol].append((timestamp, new_price))
            
            await asyncio.sleep(0.01)  # 10ms per minute = 24 hours in 14.4 seconds
    
    async def _backtest_simulation(self, symbols: List[str]):
        """Historical backtest simulation."""
        
        # Generate historical data for backtesting
        for symbol in symbols:
            historical_data = self.mock_data.generate_ohlcv(
                symbol=symbol, 
                days=365, 
                freq='1H'
            )
            
            for _, row in historical_data.iterrows():
                timestamp = row['timestamp']
                price = row['close']
                
                self.current_prices[symbol] = price
                self.price_history[symbol].append((timestamp, price))
                
                await asyncio.sleep(0.001)  # Fast simulation
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        return self.current_prices.get(symbol)
    
    def get_bid_ask_spread(self, symbol: str) -> Tuple[float, float]:
        """Get simulated bid/ask prices."""
        
        mid_price = self.current_prices.get(symbol, 100.0)
        spread_pct = 0.001  # 0.1% spread
        
        bid = mid_price * (1 - spread_pct / 2)
        ask = mid_price * (1 + spread_pct / 2)
        
        return bid, ask


class PaperTradingEngine:
    """Main paper trading engine."""
    
    def __init__(self, config: TestConfigManager):
        self.config = config
        self.market_simulator = MarketSimulator(config)
        
        # Trading state
        self.accounts = {}
        self.orders = {}
        self.positions = {}
        self.trade_history = []
        
        # Execution settings
        self.commission_rate = config.get_config('paper_trading').get('commission_rate', 0.001)
        self.slippage_factor = config.get_config('paper_trading').get('slippage_factor', 0.0005)
        
        # Simulation control
        self.simulation_running = False
        
    def create_account(self, account_id: str, initial_balance: float) -> PaperAccount:
        """Create a new paper trading account."""
        
        account = PaperAccount(
            account_id=account_id,
            initial_balance=initial_balance,
            current_balance=initial_balance
        )
        
        self.accounts[account_id] = account
        self.positions[account_id] = {}
        
        print(f"✅ Created paper trading account: {account_id} with ${initial_balance:,.2f}")
        
        return account
    
    async def start_simulation(self, symbols: List[str], 
                             mode: SimulationMode = SimulationMode.ACCELERATED):
        """Start the paper trading simulation."""
        
        self.simulation_running = True
        
        # Start market simulation
        market_task = asyncio.create_task(
            self.market_simulator.start_market_simulation(symbols, mode)
        )
        
        # Start order processing
        processing_task = asyncio.create_task(self._process_orders_loop())
        
        try:
            await asyncio.gather(market_task, processing_task)
        except KeyboardInterrupt:
            print("\n🛑 Stopping paper trading simulation...")
            self.simulation_running = False
    
    def place_order(self, account_id: str, symbol: str, side: str, quantity: int,
                   order_type: OrderType = OrderType.MARKET, 
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   strategy_id: Optional[str] = None) -> str:
        """Place a paper trading order."""
        
        # Validate account
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        current_price = self.market_simulator.get_current_price(symbol)
        
        if current_price is None:
            raise ValueError(f"No market data available for {symbol}")
        
        # Check if order can be placed
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            user_id=account_id,
            strategy_id=strategy_id
        )
        
        can_place, reason = account.can_place_order(order, current_price)
        
        if not can_place:
            order.status = OrderStatus.REJECTED
            order.notes = reason
            print(f"❌ Order rejected: {reason}")
            return order.order_id
        
        # Store order
        self.orders[order.order_id] = order
        
        print(f"📝 Order placed: {side} {quantity} {symbol} @ {order_type.value}")
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            print(f"🚫 Order cancelled: {order_id}")
            return True
        
        return False
    
    async def _process_orders_loop(self):
        """Process orders continuously."""
        
        while self.simulation_running:
            await asyncio.sleep(0.1)  # Process orders every 100ms
            
            for order in list(self.orders.values()):
                if order.is_active:
                    await self._try_fill_order(order)
    
    async def _try_fill_order(self, order: PaperOrder):
        """Attempt to fill an order."""
        
        current_price = self.market_simulator.get_current_price(order.symbol)
        
        if current_price is None:
            return
        
        # Determine if order should fill
        should_fill = False
        fill_price = current_price
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
            # Apply slippage for market orders
            if order.side == "BUY":
                fill_price = current_price * (1 + self.slippage_factor)
            else:
                fill_price = current_price * (1 - self.slippage_factor)
                
        elif order.order_type == OrderType.LIMIT:
            if order.side == "BUY" and current_price <= order.limit_price:
                should_fill = True
                fill_price = order.limit_price
            elif order.side == "SELL" and current_price >= order.limit_price:
                should_fill = True
                fill_price = order.limit_price
                
        elif order.order_type == OrderType.STOP:
            if order.side == "BUY" and current_price >= order.stop_price:
                should_fill = True
                fill_price = current_price
            elif order.side == "SELL" and current_price <= order.stop_price:
                should_fill = True
                fill_price = current_price
        
        if should_fill:
            await self._fill_order(order, fill_price)
    
    async def _fill_order(self, order: PaperOrder, fill_price: float):
        """Fill an order completely."""
        
        timestamp = datetime.now()
        commission = order.quantity * fill_price * self.commission_rate
        
        # Add fill to order
        order.add_fill(order.remaining_quantity, fill_price, timestamp)
        
        # Update account
        account = self.accounts[order.user_id]
        account.process_trade(order, fill_price, commission)
        
        # Update position
        await self._update_position(order, fill_price, commission)
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'order_id': order.order_id,
            'account_id': order.user_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'strategy_id': order.strategy_id
        }
        
        self.trade_history.append(trade_record)
        
        print(f"✅ Order filled: {order.side} {order.quantity} {order.symbol} @ ${fill_price:.2f}")
    
    async def _update_position(self, order: PaperOrder, fill_price: float, commission: float):
        """Update position after trade execution."""
        
        account_positions = self.positions[order.user_id]
        
        if order.symbol not in account_positions:
            account_positions[order.symbol] = PaperPosition(order.symbol)
        
        position = account_positions[order.symbol]
        
        # Update position based on trade
        trade_quantity = order.quantity if order.side == "BUY" else -order.quantity
        position.update_position(trade_quantity, fill_price, commission)
        
        # Update unrealized P&L with current market price
        current_price = self.market_simulator.get_current_price(order.symbol)
        if current_price:
            position.update_market_value(current_price)
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get account summary with positions and P&L."""
        
        if account_id not in self.accounts:
            return {}
        
        account = self.accounts[account_id]
        positions = self.positions.get(account_id, {})
        
        # Calculate total portfolio value
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in positions.values())
        
        # Update positions with current market prices
        for symbol, position in positions.items():
            current_price = self.market_simulator.get_current_price(symbol)
            if current_price:
                position.update_market_value(current_price)
        
        positions_value = sum(
            abs(pos.quantity) * self.market_simulator.get_current_price(pos.symbol)
            for pos in positions.values()
            if self.market_simulator.get_current_price(pos.symbol) is not None
        )
        
        total_portfolio_value = account.current_balance + positions_value + total_unrealized_pnl
        
        return {
            'account_id': account_id,
            'initial_balance': account.initial_balance,
            'current_balance': account.current_balance,
            'positions_value': positions_value,
            'total_portfolio_value': total_portfolio_value,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_return_pct': ((total_portfolio_value - account.initial_balance) / account.initial_balance) * 100,
            'total_trades': account.total_trades,
            'winning_trades': account.winning_trades,
            'losing_trades': account.losing_trades,
            'positions': {symbol: asdict(pos) for symbol, pos in positions.items()},
            'active_orders': [asdict(order) for order in self.orders.values() if order.user_id == account_id and order.is_active]
        }
    
    def save_simulation_state(self, filepath: str):
        """Save current simulation state to file."""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'accounts': {aid: asdict(account) for aid, account in self.accounts.items()},
            'positions': {aid: {sym: asdict(pos) for sym, pos in positions.items()} 
                         for aid, positions in self.positions.items()},
            'orders': {oid: asdict(order) for oid, order in self.orders.items()},
            'trade_history': self.trade_history,
            'current_prices': self.market_simulator.current_prices
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"💾 Simulation state saved to {filepath}")
    
    def load_simulation_state(self, filepath: str):
        """Load simulation state from file."""
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Reconstruct accounts
        for aid, account_data in state['accounts'].items():
            self.accounts[aid] = PaperAccount(**account_data)
        
        # Reconstruct positions  
        for aid, positions_data in state['positions'].items():
            self.positions[aid] = {}
            for sym, pos_data in positions_data.items():
                self.positions[aid][sym] = PaperPosition(**pos_data)
        
        # Reconstruct orders
        for oid, order_data in state['orders'].items():
            order_data['order_type'] = OrderType(order_data['order_type'])
            order_data['status'] = OrderStatus(order_data['status'])
            order_data['timestamp'] = datetime.fromisoformat(order_data['timestamp'])
            self.orders[oid] = PaperOrder(**order_data)
        
        # Restore other state
        self.trade_history = state['trade_history']
        self.market_simulator.current_prices = state['current_prices']
        
        print(f"📂 Simulation state loaded from {filepath}")


# Example usage and testing
async def run_paper_trading_demo():
    """Demonstrate paper trading functionality."""
    
    config = TestConfigManager()
    engine = PaperTradingEngine(config)
    
    # Create test account
    account_id = "demo_account"
    account = engine.create_account(account_id, 100000.0)  # $100k starting balance
    
    # Define symbols to trade
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    print("🚀 Starting paper trading demo...")
    
    # Start simulation in background
    simulation_task = asyncio.create_task(
        engine.start_simulation(symbols, SimulationMode.ACCELERATED)
    )
    
    # Wait a bit for market to initialize
    await asyncio.sleep(1.0)
    
    # Place some demo orders
    print("\n📝 Placing demo orders...")
    
    # Market buy order
    order1 = engine.place_order(
        account_id=account_id,
        symbol="AAPL",
        side="BUY", 
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    # Limit sell order
    current_googl = engine.market_simulator.get_current_price("GOOGL")
    if current_googl:
        order2 = engine.place_order(
            account_id=account_id,
            symbol="GOOGL",
            side="BUY",
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=current_googl * 0.98  # 2% below current price
        )
    
    # Let simulation run for a bit
    print("\n⏳ Running simulation for 10 seconds...")
    await asyncio.sleep(10.0)
    
    # Check account status
    print("\n📊 Account Summary:")
    summary = engine.get_account_summary(account_id)
    
    print(f"Portfolio Value: ${summary['total_portfolio_value']:,.2f}")
    print(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_return_pct']:.2f}%)")
    print(f"Cash Balance: ${summary['current_balance']:,.2f}")
    print(f"Positions Value: ${summary['positions_value']:,.2f}")
    print(f"Total Trades: {summary['total_trades']}")
    
    # Show positions
    if summary['positions']:
        print("\n📈 Current Positions:")
        for symbol, position in summary['positions'].items():
            if position['quantity'] != 0:
                print(f"  {symbol}: {position['quantity']} shares @ ${position['avg_price']:.2f}")
                print(f"    Unrealized P&L: ${position['unrealized_pnl']:.2f}")
    
    # Save simulation state
    engine.save_simulation_state("demo_simulation.json")
    
    # Stop simulation
    engine.simulation_running = False
    simulation_task.cancel()
    
    print("\n✅ Paper trading demo completed!")


if __name__ == '__main__':
    # Need to import numpy here to avoid import issues
    import numpy as np
    
    asyncio.run(run_paper_trading_demo())