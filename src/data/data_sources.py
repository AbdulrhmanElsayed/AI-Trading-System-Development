"""
Data Sources Module

Contains implementations for various market data providers.
"""

import asyncio
import aiohttp
import yfinance as yf
import ccxt.async_support as ccxt
import json
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import MarketData


class BaseDataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger(self.__class__.__name__)
        self.session = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize the data source."""
        pass
    
    @abstractmethod
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for a symbol."""
        pass
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()


class YFinanceDataSource(BaseDataSource):
    """Yahoo Finance data source."""
    
    async def initialize(self):
        """Initialize Yahoo Finance data source."""
        self.session = aiohttp.ClientSession()
        self.logger.info("YFinance data source initialized")
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data from Yahoo Finance."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get current price and info
            info = await loop.run_in_executor(None, lambda: ticker.info)
            hist = await loop.run_in_executor(
                None, lambda: ticker.history(period='1d', interval='1m').tail(1)
            )
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(latest['Close']),
                volume=float(latest['Volume']),
                open=float(latest['Open']),
                high=float(latest['High']),
                low=float(latest['Low']),
                close=float(latest['Close']),
                bid=float(info.get('bid', 0)) if info.get('bid') else None,
                ask=float(info.get('ask', 0)) if info.get('ask') else None
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching YFinance data for {symbol}: {e}")
            return None


class AlphaVantageDataSource(BaseDataSource):
    """Alpha Vantage data source."""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.api_key = config.get('apis.data_apis.alpha_vantage.api_key')
        self.base_url = "https://www.alphavantage.co/query"
    
    async def initialize(self):
        """Initialize Alpha Vantage data source."""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        self.session = aiohttp.ClientSession()
        self.logger.info("Alpha Vantage data source initialized")
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                
                if 'Global Quote' not in data:
                    return None
                
                quote = data['Global Quote']
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=float(quote['05. price']),
                    volume=float(quote['06. volume']),
                    open=float(quote['02. open']),
                    high=float(quote['03. high']),
                    low=float(quote['04. low']),
                    close=float(quote['08. previous close'])
                )
                
                return market_data
                
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None


class BinanceDataSource(BaseDataSource):
    """Binance cryptocurrency data source."""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.api_key = config.get('apis.brokers.binance.api_key')
        self.secret_key = config.get('apis.brokers.binance.secret_key')
        self.testnet = config.get('apis.brokers.binance.testnet', True)
        self.exchange = None
        self.websocket_callbacks = []
    
    async def initialize(self):
        """Initialize Binance data source."""
        try:
            sandbox = self.testnet
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'sandbox': sandbox,
                'enableRateLimit': True,
            })
            
            # Test connection
            await self.exchange.load_markets()
            self.logger.info("Binance data source initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance: {e}")
            raise
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data from Binance."""
        try:
            # Convert symbol format (e.g., BTC/USD -> BTCUSDT)
            binance_symbol = symbol.replace('/', '')
            
            ticker = await self.exchange.fetch_ticker(binance_symbol)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(ticker['last']),
                volume=float(ticker['baseVolume']),
                bid=float(ticker['bid']) if ticker['bid'] else None,
                ask=float(ticker['ask']) if ticker['ask'] else None,
                high=float(ticker['high']),
                low=float(ticker['low']),
                open=float(ticker['open']),
                close=float(ticker['close'])
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Binance data for {symbol}: {e}")
            return None
    
    async def start_websocket(self, callback):
        """Start websocket connection for real-time data."""
        try:
            symbols = self.config.get('data_sources.crypto.symbols', [])
            streams = []
            
            for symbol in symbols:
                # Convert to Binance format
                binance_symbol = symbol.replace('/', '').lower()
                streams.append(f"{binance_symbol}@ticker")
            
            if not streams:
                return
            
            stream_url = f"wss://stream.binance.com:9443/ws/{'+'.join(streams)}"
            
            async with websockets.connect(stream_url) as websocket:
                self.logger.info("Binance websocket connected")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Parse Binance ticker message
                        if 's' in data:  # Symbol field
                            symbol_formatted = f"{data['s'][:3]}/{data['s'][3:]}"
                            
                            websocket_data = {
                                'symbol': symbol_formatted,
                                'price': data['c'],  # Close price
                                'volume': data['v'],  # Volume
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            await callback(websocket_data)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing websocket message: {e}")
                        
        except Exception as e:
            self.logger.error(f"Websocket error: {e}")
    
    async def cleanup(self):
        """Cleanup Binance resources."""
        if self.exchange:
            await self.exchange.close()
        await super().cleanup()


class NewsDataSource(BaseDataSource):
    """News and sentiment data source."""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.newsapi_key = config.get('apis.data_apis.newsapi.api_key')
        self.base_url = "https://newsapi.org/v2/everything"
    
    async def initialize(self):
        """Initialize news data source."""
        if not self.newsapi_key:
            self.logger.warning("NewsAPI key not configured - news data will be unavailable")
            return
        
        self.session = aiohttp.ClientSession()
        self.logger.info("News data source initialized")
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """News doesn't provide traditional market data."""
        return None
    
    async def get_news_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news and sentiment for a symbol."""
        try:
            if not self.newsapi_key or not self.session:
                return []
            
            # Build search query
            query = f"{symbol} stock OR {symbol} price"
            if '/' in symbol:  # Crypto pair
                base_currency = symbol.split('/')[0]
                query = f"{base_currency} cryptocurrency OR {base_currency} crypto"
            
            params = {
                'q': query,
                'apiKey': self.newsapi_key,
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'language': 'en'
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                
                if data.get('status') != 'ok':
                    return []
                
                articles = []
                for article in data.get('articles', []):
                    # Simple sentiment analysis (can be enhanced with ML)
                    sentiment_score = self._analyze_sentiment(article['title'])
                    
                    articles.append({
                        'symbol': symbol,
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'published_at': article['publishedAt'],
                        'source': article['source']['name'],
                        'sentiment_score': sentiment_score,
                        'timestamp': datetime.now().isoformat()
                    })
                
                return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (placeholder for more sophisticated analysis)."""
        positive_words = ['up', 'rise', 'gain', 'bull', 'positive', 'growth', 'profit']
        negative_words = ['down', 'fall', 'loss', 'bear', 'negative', 'decline', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / (positive_count + negative_count)