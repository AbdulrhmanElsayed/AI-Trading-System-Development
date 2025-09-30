"""
Market Data Manager

Handles real-time and historical market data ingestion from multiple sources.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import websockets
import json
from dataclasses import asdict

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import MarketData, MarketDataType
from src.data.data_sources import (
    YFinanceDataSource,
    AlphaVantageDataSource,
    BinanceDataSource,
    NewsDataSource
)
from src.data.technical_indicators import TechnicalIndicators
from src.data.data_storage import DataStorage


class MarketDataManager:
    """Centralized market data management."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("MarketDataManager")
        
        # Data sources
        self.data_sources = {}
        self.websocket_connections = {}
        
        # Components
        self.technical_indicators = None
        self.data_storage = None
        
        # State
        self.is_running = False
        self.latest_data = {}
        self.subscribers = []
        
        # Configuration
        self.update_interval = config.get('trading.trading_frequency_seconds', 60)
        self.symbols = self._get_all_symbols()
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols from configuration."""
        symbols = []
        
        # Add stock symbols
        stock_symbols = self.config.get('data_sources.stocks.symbols', [])
        symbols.extend(stock_symbols)
        
        # Add crypto symbols
        crypto_symbols = self.config.get('data_sources.crypto.symbols', [])
        symbols.extend(crypto_symbols)
        
        # Add forex symbols
        forex_symbols = self.config.get('data_sources.forex.symbols', [])
        symbols.extend(forex_symbols)
        
        return symbols
    
    async def initialize(self):
        """Initialize market data manager."""
        try:
            self.logger.info("Initializing Market Data Manager...")
            
            # Initialize technical indicators
            self.technical_indicators = TechnicalIndicators()
            
            # Initialize data storage
            self.data_storage = DataStorage(self.config)
            await self.data_storage.initialize()
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            self.logger.info("Market Data Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Market Data Manager: {e}")
            raise
    
    async def _initialize_data_sources(self):
        """Initialize all configured data sources."""
        # Stock data sources
        if self.config.get('data_sources.stocks.primary') == 'yfinance':
            self.data_sources['stocks'] = YFinanceDataSource(self.config)
        elif self.config.get('data_sources.stocks.primary') == 'alpha_vantage':
            self.data_sources['stocks'] = AlphaVantageDataSource(self.config)
        
        # Crypto data sources
        crypto_exchanges = self.config.get('data_sources.crypto.exchanges', [])
        if 'binance' in crypto_exchanges:
            self.data_sources['crypto'] = BinanceDataSource(self.config)
        
        # News data source
        self.data_sources['news'] = NewsDataSource(self.config)
        
        # Initialize all sources
        for source_name, source in self.data_sources.items():
            try:
                await source.initialize()
                self.logger.info(f"Initialized {source_name} data source")
            except Exception as e:
                self.logger.error(f"Failed to initialize {source_name}: {e}")
    
    async def start(self):
        """Start market data collection."""
        try:
            if self.is_running:
                return
            
            self.logger.info("Starting market data collection...")
            self.is_running = True
            
            # Start data collection tasks
            tasks = [
                asyncio.create_task(self._collect_market_data()),
                asyncio.create_task(self._collect_news_data()),
                asyncio.create_task(self._websocket_data_handler())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in market data collection: {e}")
            raise
    
    async def stop(self):
        """Stop market data collection."""
        self.logger.info("Stopping market data collection...")
        self.is_running = False
        
        # Close websocket connections
        for connection in self.websocket_connections.values():
            if connection and not connection.closed:
                await connection.close()
        
        # Stop data sources
        for source in self.data_sources.values():
            if hasattr(source, 'stop'):
                await source.stop()
    
    async def _collect_market_data(self):
        """Main market data collection loop."""
        while self.is_running:
            try:
                # Collect data for all symbols
                for symbol in self.symbols:
                    await self._process_symbol_data(symbol)
                
                # Wait before next collection
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting market data: {e}")
                await asyncio.sleep(10)  # Brief pause before retry
    
    async def _process_symbol_data(self, symbol: str):
        """Process data for a specific symbol."""
        try:
            # Determine data source based on symbol type
            source_type = self._get_source_type(symbol)
            data_source = self.data_sources.get(source_type)
            
            if not data_source:
                return
            
            # Get latest market data
            market_data = await data_source.get_latest_data(symbol)
            
            if market_data:
                # Calculate technical indicators
                enriched_data = await self._enrich_with_indicators(market_data)
                
                # Store data
                await self.data_storage.store_market_data(enriched_data)
                
                # Update latest data cache
                self.latest_data[symbol] = enriched_data
                
                # Notify subscribers
                await self._notify_subscribers(enriched_data)
                
        except Exception as e:
            self.logger.error(f"Error processing data for {symbol}: {e}")
    
    def _get_source_type(self, symbol: str) -> str:
        """Determine the appropriate data source for a symbol."""
        if '/' in symbol:  # Crypto or forex pair
            if symbol in self.config.get('data_sources.crypto.symbols', []):
                return 'crypto'
            else:
                return 'forex'
        else:  # Stock symbol
            return 'stocks'
    
    async def _enrich_with_indicators(self, market_data: MarketData) -> MarketData:
        """Enrich market data with technical indicators."""
        try:
            # Get historical data for indicator calculation
            historical_data = await self.data_storage.get_historical_data(
                market_data.symbol,
                days=100  # Enough for most indicators
            )
            
            if len(historical_data) < 20:  # Minimum for basic indicators
                return market_data
            
            # Calculate indicators
            indicators = self.technical_indicators.calculate_all(historical_data)
            
            # Add indicators to metadata
            market_data.metadata['indicators'] = indicators
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error enriching data with indicators: {e}")
            return market_data
    
    async def _collect_news_data(self):
        """Collect news and sentiment data."""
        while self.is_running:
            try:
                news_source = self.data_sources.get('news')
                if news_source:
                    # Collect news for each symbol
                    for symbol in self.symbols:
                        news_data = await news_source.get_news_sentiment(symbol)
                        if news_data:
                            await self.data_storage.store_sentiment_data(news_data)
                
                # News updates less frequently
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting news data: {e}")
                await asyncio.sleep(60)
    
    async def _websocket_data_handler(self):
        """Handle real-time websocket data streams."""
        while self.is_running:
            try:
                # Setup websocket connections for real-time data
                crypto_source = self.data_sources.get('crypto')
                if crypto_source and hasattr(crypto_source, 'start_websocket'):
                    await crypto_source.start_websocket(self._handle_websocket_message)
                
                await asyncio.sleep(1)  # Prevent tight loop
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in websocket handler: {e}")
                await asyncio.sleep(10)
    
    async def _handle_websocket_message(self, message: Dict[str, Any]):
        """Handle incoming websocket messages."""
        try:
            # Parse websocket message and create MarketData object
            # This is specific to each exchange's message format
            if 'symbol' in message and 'price' in message:
                market_data = MarketData(
                    symbol=message['symbol'],
                    timestamp=datetime.utcnow(),
                    price=float(message['price']),
                    volume=float(message.get('volume', 0))
                )
                
                # Update latest data and notify subscribers
                self.latest_data[market_data.symbol] = market_data
                await self._notify_subscribers(market_data)
                
        except Exception as e:
            self.logger.error(f"Error handling websocket message: {e}")
    
    async def _notify_subscribers(self, market_data: MarketData):
        """Notify all subscribers of new market data."""
        for subscriber in self.subscribers:
            try:
                await subscriber(market_data)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback):
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from market data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def get_latest_data(self, symbol: Optional[str] = None) -> Dict[str, MarketData]:
        """Get latest market data."""
        if symbol:
            return {symbol: self.latest_data.get(symbol)}
        return self.latest_data.copy()
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d'
    ) -> List[MarketData]:
        """Get historical market data."""
        return await self.data_storage.get_historical_data(
            symbol, start_date, end_date, timeframe
        )
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market data collection status."""
        return {
            'is_running': self.is_running,
            'symbols_tracked': len(self.symbols),
            'data_sources': list(self.data_sources.keys()),
            'last_update': datetime.utcnow().isoformat(),
            'websocket_connections': len(self.websocket_connections)
        }