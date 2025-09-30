"""
Mock Data Generators

Provides realistic mock data for testing all components of the trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class MockTrade:
    """Mock trade data."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0


class MockMarketData:
    """Generates realistic mock market data for testing."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH']
    
    def generate_ohlcv(self, symbol: str = 'AAPL', days: int = 30, 
                      freq: str = '1h', base_price: float = 150.0) -> pd.DataFrame:
        """Generate OHLCV data with realistic price movements."""
        
        # Create date range
        if freq == '1h':
            periods = days * 24
        elif freq == '1D':
            periods = days
        elif freq == '1min':
            periods = days * 24 * 60
        else:
            periods = days
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=periods,
            freq=freq
        )
        
        # Generate realistic price walk
        # Use geometric Brownian motion
        dt = 1 / (365 * 24) if freq == '1h' else 1 / 365  # Time step
        mu = 0.1  # Expected annual return
        sigma = 0.2  # Annual volatility
        
        # Generate random returns
        returns = self.rng.normal(
            mu * dt,
            sigma * np.sqrt(dt),
            len(dates)
        )
        
        # Calculate prices
        price_multipliers = np.exp(np.cumsum(returns))
        prices = base_price * price_multipliers
        
        # Generate OHLC from close prices
        noise_factor = 0.001
        
        # Open prices (previous close + small gap)
        open_prices = np.roll(prices, 1)
        open_prices[0] = base_price
        open_prices *= (1 + self.rng.normal(0, noise_factor, len(open_prices)))
        
        # High prices (max of open/close + positive noise)
        high_prices = np.maximum(open_prices, prices)
        high_prices *= (1 + np.abs(self.rng.normal(0, noise_factor * 2, len(high_prices))))
        
        # Low prices (min of open/close - positive noise) 
        low_prices = np.minimum(open_prices, prices)
        low_prices *= (1 - np.abs(self.rng.normal(0, noise_factor * 2, len(low_prices))))
        
        # Volume (log-normal distribution)
        base_volume = 1000000
        volumes = self.rng.lognormal(
            mean=np.log(base_volume),
            sigma=0.5,
            size=len(dates)
        ).astype(int)
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': volumes
        })
    
    def generate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data."""
        df = ohlcv_data.copy()
        
        # Simple moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def generate_multi_symbol_data(self, symbols: List[str] = None, 
                                 days: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate data for multiple symbols."""
        if symbols is None:
            symbols = self.symbols[:5]  # Use first 5 symbols
        
        data = {}
        for symbol in symbols:
            # Use different base prices for different symbols
            base_price = self.rng.uniform(50, 300)
            ohlcv = self.generate_ohlcv(symbol, days, base_price=base_price)
            data[symbol] = self.generate_technical_indicators(ohlcv)
        
        return data
    
    def generate_real_time_tick(self, symbol: str = 'AAPL', 
                              last_price: float = 150.0) -> Dict[str, Any]:
        """Generate a single real-time tick."""
        # Small price movement
        price_change = self.rng.normal(0, last_price * 0.0005)
        new_price = max(0.01, last_price + price_change)
        
        return {
            'symbol': symbol,
            'price': round(new_price, 2),
            'volume': self.rng.randint(100, 1000),
            'timestamp': datetime.now(),
            'bid': round(new_price - 0.01, 2),
            'ask': round(new_price + 0.01, 2),
            'bid_size': self.rng.randint(100, 500),
            'ask_size': self.rng.randint(100, 500)
        }


class MockPortfolioData:
    """Generates mock portfolio and position data."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def generate_portfolio(self, initial_cash: float = 100000.0,
                          symbols: List[str] = None) -> Dict[str, Any]:
        """Generate mock portfolio data."""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        positions = {}
        total_position_value = 0.0
        
        for symbol in symbols:
            if self.rng.random() > 0.3:  # 70% chance of having position
                quantity = self.rng.randint(10, 100)
                avg_price = self.rng.uniform(100, 300)
                current_price = avg_price * (1 + self.rng.normal(0, 0.05))
                
                position_value = quantity * current_price
                unrealized_pnl = quantity * (current_price - avg_price)
                
                positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': round(avg_price, 2),
                    'current_price': round(current_price, 2),
                    'market_value': round(position_value, 2),
                    'unrealized_pnl': round(unrealized_pnl, 2),
                    'unrealized_pnl_percent': round(unrealized_pnl / (quantity * avg_price) * 100, 2)
                }
                
                total_position_value += position_value
        
        # Calculate remaining cash (assuming some was used for positions)
        used_cash = sum(pos['quantity'] * pos['avg_price'] for pos in positions.values())
        remaining_cash = max(0, initial_cash - used_cash)
        
        total_value = remaining_cash + total_position_value
        total_pnl = sum(pos['unrealized_pnl'] for pos in positions.values())
        
        return {
            'timestamp': datetime.now(),
            'cash': round(remaining_cash, 2),
            'positions': positions,
            'total_value': round(total_value, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_percent': round(total_pnl / initial_cash * 100, 2) if initial_cash > 0 else 0.0,
            'buying_power': round(remaining_cash * 4, 2),  # Assume 4:1 margin
            'day_pnl': round(self.rng.normal(0, total_value * 0.01), 2)
        }
    
    def generate_position_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Generate position history for a symbol."""
        history = []
        current_quantity = 0
        current_avg_price = 0
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            # Random trade activity
            if self.rng.random() > 0.7:  # 30% chance of trade
                trade_quantity = self.rng.choice([-50, -25, -10, 10, 25, 50])
                trade_price = self.rng.uniform(100, 200)
                
                if current_quantity + trade_quantity >= 0:
                    # Update position
                    if trade_quantity > 0:  # Buy
                        total_cost = current_quantity * current_avg_price + trade_quantity * trade_price
                        current_quantity += trade_quantity
                        current_avg_price = total_cost / current_quantity if current_quantity > 0 else 0
                    else:  # Sell
                        current_quantity += trade_quantity  # trade_quantity is negative
                        if current_quantity <= 0:
                            current_quantity = 0
                            current_avg_price = 0
            
            # Current market price
            market_price = current_avg_price * (1 + self.rng.normal(0, 0.02)) if current_avg_price > 0 else 100
            
            history.append({
                'date': date,
                'symbol': symbol,
                'quantity': current_quantity,
                'avg_price': round(current_avg_price, 2),
                'market_price': round(market_price, 2),
                'market_value': round(current_quantity * market_price, 2),
                'unrealized_pnl': round(current_quantity * (market_price - current_avg_price), 2)
            })
        
        return history


class MockSignalData:
    """Generates mock trading signals."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def generate_signal(self, symbol: str = 'AAPL') -> Dict[str, Any]:
        """Generate a single trading signal."""
        signal_type = self.rng.choice([SignalType.BUY, SignalType.SELL, SignalType.HOLD])
        
        # Generate signal strength (0 to 1)
        strength = self.rng.uniform(0.3, 1.0)
        
        # Generate confidence (0 to 1)
        confidence = self.rng.uniform(0.5, 1.0)
        
        # Generate feature values that led to this signal
        features = {
            'rsi': self.rng.uniform(20, 80),
            'macd': self.rng.normal(0, 0.5),
            'bb_position': self.rng.uniform(-1, 1),  # -1 = lower band, 1 = upper band
            'volume_ratio': self.rng.uniform(0.5, 2.0),
            'momentum': self.rng.normal(0, 0.02),
            'volatility': self.rng.uniform(0.1, 0.4),
            'trend_strength': self.rng.uniform(0, 1)
        }
        
        # Generate model predictions if ensemble
        model_predictions = {
            'lstm_prob': self.rng.uniform(0, 1),
            'xgboost_prob': self.rng.uniform(0, 1),
            'random_forest_prob': self.rng.uniform(0, 1),
            'ensemble_prob': self.rng.uniform(0, 1)
        }
        
        return {
            'signal_id': f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.rng.integers(1000, 9999)}",
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal_type': signal_type.value,
            'strength': round(strength, 3),
            'confidence': round(confidence, 3),
            'features': features,
            'model_predictions': model_predictions,
            'metadata': {
                'model_version': 'v1.2.3',
                'data_quality_score': self.rng.uniform(0.8, 1.0),
                'market_regime': self.rng.choice(['trending', 'ranging', 'volatile'])
            }
        }
    
    def generate_signal_history(self, symbols: List[str], days: int = 7) -> List[Dict[str, Any]]:
        """Generate historical signals for multiple symbols."""
        signals = []
        
        for day in range(days):
            for symbol in symbols:
                # Generate 0-5 signals per symbol per day
                num_signals = self.rng.integers(0, 6)
                
                for _ in range(num_signals):
                    signal = self.generate_signal(symbol)
                    # Adjust timestamp to past dates
                    signal['timestamp'] = datetime.now() - timedelta(
                        days=days-day,
                        hours=self.rng.integers(0, 24),
                        minutes=self.rng.integers(0, 60)
                    )
                    signals.append(signal)
        
        # Sort by timestamp
        signals.sort(key=lambda x: x['timestamp'])
        return signals


class MockOrderData:
    """Generates mock order and execution data."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def generate_order(self, symbol: str = 'AAPL', side: str = 'BUY') -> Dict[str, Any]:
        """Generate a mock order."""
        order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.rng.integers(1000, 9999)}"
        
        quantity = self.rng.integers(10, 500)
        price = self.rng.uniform(100, 300)
        
        order_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        order_type = self.rng.choice(order_types)
        
        status = self.rng.choice([
            OrderStatus.PENDING,
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.CANCELLED
        ], p=[0.1, 0.7, 0.1, 0.1])
        
        # Fill details based on status
        fill_quantity = 0
        fill_price = 0
        
        if status == OrderStatus.FILLED:
            fill_quantity = quantity
            # Add realistic slippage
            slippage = self.rng.normal(0, 0.001) * price
            fill_price = price + slippage
        elif status == OrderStatus.PARTIALLY_FILLED:
            fill_quantity = self.rng.integers(1, quantity)
            slippage = self.rng.normal(0, 0.001) * price
            fill_price = price + slippage
        
        return {
            'order_id': order_id,
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'quantity': quantity,
            'price': round(price, 2) if order_type != 'MARKET' else None,
            'status': status.value,
            'fill_quantity': fill_quantity,
            'fill_price': round(fill_price, 2) if fill_price > 0 else None,
            'commission': round(max(1.0, quantity * 0.005), 2),
            'time_in_force': 'DAY',
            'broker': 'TEST_BROKER'
        }
    
    def generate_execution_report(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution report for an order."""
        latency_ms = self.rng.integers(50, 500)  # 50-500ms execution latency
        
        return {
            'execution_id': f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.rng.integers(1000, 9999)}",
            'order_id': order['order_id'],
            'timestamp': datetime.now(),
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['fill_quantity'],
            'price': order['fill_price'],
            'commission': order['commission'],
            'execution_latency_ms': latency_ms,
            'venue': self.rng.choice(['NYSE', 'NASDAQ', 'ARCA', 'BATS']),
            'market_impact_bps': round(self.rng.uniform(-2, 5), 2)  # Market impact in basis points
        }
    
    def generate_order_book(self, symbol: str = 'AAPL', 
                           center_price: float = 150.0, levels: int = 10) -> Dict[str, Any]:
        """Generate mock order book data."""
        
        bids = []
        asks = []
        
        spread = self.rng.uniform(0.01, 0.05)
        best_bid = center_price - spread/2
        best_ask = center_price + spread/2
        
        for i in range(levels):
            # Bid side (decreasing prices)
            bid_price = best_bid - i * 0.01
            bid_size = self.rng.integers(100, 1000)
            bids.append({'price': round(bid_price, 2), 'size': bid_size})
            
            # Ask side (increasing prices)
            ask_price = best_ask + i * 0.01
            ask_size = self.rng.integers(100, 1000)
            asks.append({'price': round(ask_price, 2), 'size': ask_size})
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'bids': bids,
            'asks': asks,
            'spread': round(spread, 4),
            'mid_price': round((best_bid + best_ask) / 2, 2)
        }
    
    def generate_trade_history(self, symbol: str = 'AAPL', days: int = 30) -> List[MockTrade]:
        """Generate trade execution history."""
        trades = []
        
        for day in range(days):
            # Generate 0-10 trades per day
            num_trades = self.rng.integers(0, 11)
            
            for _ in range(num_trades):
                trade_time = datetime.now() - timedelta(
                    days=days-day,
                    hours=self.rng.integers(9, 16),  # Trading hours
                    minutes=self.rng.integers(0, 60)
                )
                
                trade = MockTrade(
                    trade_id=f"trade_{trade_time.strftime('%Y%m%d_%H%M%S')}_{self.rng.integers(1000, 9999)}",
                    symbol=symbol,
                    side=self.rng.choice(['BUY', 'SELL']),
                    quantity=self.rng.integers(10, 200),
                    price=round(self.rng.uniform(100, 200), 2),
                    timestamp=trade_time,
                    commission=self.rng.uniform(1.0, 10.0)
                )
                trades.append(trade)
        
        # Sort by timestamp
        trades.sort(key=lambda x: x.timestamp)
        return trades


# Convenience functions for quick test data generation
def quick_ohlcv(symbol: str = 'AAPL', days: int = 30) -> pd.DataFrame:
    """Quick OHLCV data generation."""
    return MockMarketData().generate_ohlcv(symbol, days)

def quick_portfolio(cash: float = 100000.0) -> Dict[str, Any]:
    """Quick portfolio data generation."""
    return MockPortfolioData().generate_portfolio(cash)

def quick_signal(symbol: str = 'AAPL') -> Dict[str, Any]:
    """Quick signal data generation."""
    return MockSignalData().generate_signal(symbol)

def quick_order(symbol: str = 'AAPL', side: str = 'BUY') -> Dict[str, Any]:
    """Quick order data generation."""
    return MockOrderData().generate_order(symbol, side)