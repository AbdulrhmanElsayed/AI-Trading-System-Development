"""
Data Storage Module

Handles persistent storage for market data, indicators, and system state.
"""

import asyncio
import asyncpg
import redis.asyncio as redis
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import MarketData


class DataStorage:
    """Unified data storage management."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("DataStorage")
        
        # Database connections
        self.postgres_pool = None
        self.timescale_pool = None
        self.redis_client = None
        
        # Connection settings
        self.postgres_config = config.get_section('database.main')
        self.timescale_config = config.get_section('database.timeseries')
        self.redis_config = config.get_section('database.cache')
    
    async def initialize(self):
        """Initialize all database connections."""
        try:
            self.logger.info("Initializing database connections...")
            
            # Initialize PostgreSQL connection pool
            await self._init_postgres()
            
            # Initialize TimescaleDB connection pool
            await self._init_timescale()
            
            # Initialize Redis connection
            await self._init_redis()
            
            self.logger.info("All database connections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize databases: {e}")
            raise
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection pool."""
        try:
            dsn = (
                f"postgresql://{self.postgres_config['user']}:"
                f"{self.postgres_config['password']}@"
                f"{self.postgres_config['host']}:"
                f"{self.postgres_config['port']}/"
                f"{self.postgres_config['database']}"
            )
            
            self.postgres_pool = await asyncpg.create_pool(
                dsn,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            self.logger.info("PostgreSQL connection pool created")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _init_timescale(self):
        """Initialize TimescaleDB connection pool."""
        try:
            dsn = (
                f"postgresql://{self.timescale_config['user']}:"
                f"{self.timescale_config['password']}@"
                f"{self.timescale_config['host']}:"
                f"{self.timescale_config['port']}/"
                f"{self.timescale_config['database']}"
            )
            
            self.timescale_pool = await asyncpg.create_pool(
                dsn,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            self.logger.info("TimescaleDB connection pool created")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['database'],
                password=self.redis_config.get('password'),
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def store_market_data(self, market_data: MarketData):
        """Store market data in TimescaleDB."""
        try:
            if not self.timescale_pool:
                return
            
            query = """
                INSERT INTO ohlcv_data (
                    timestamp, symbol, open_price, high_price, low_price, 
                    close_price, volume, timeframe
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (timestamp, symbol, timeframe) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume
            """
            
            async with self.timescale_pool.acquire() as conn:
                await conn.execute(
                    query,
                    market_data.timestamp,
                    market_data.symbol,
                    market_data.open or market_data.price,
                    market_data.high or market_data.price,
                    market_data.low or market_data.price,
                    market_data.close or market_data.price,
                    market_data.volume or 0,
                    '1m'  # Default timeframe
                )
            
            # Also cache in Redis for quick access
            await self._cache_latest_data(market_data)
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    async def _cache_latest_data(self, market_data: MarketData):
        """Cache latest market data in Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"latest:{market_data.symbol}"
            data = asdict(market_data)
            data['timestamp'] = data['timestamp'].isoformat()
            
            await self.redis_client.setex(
                key,
                300,  # 5 minutes TTL
                json.dumps(data)
            )
            
        except Exception as e:
            self.logger.error(f"Error caching market data: {e}")
    
    async def store_technical_indicators(self, symbol: str, indicators: Dict[str, Any]):
        """Store technical indicators in TimescaleDB."""
        try:
            if not self.timescale_pool or not indicators:
                return
            
            async with self.timescale_pool.acquire() as conn:
                for indicator_name, value in indicators.items():
                    if value is not None:
                        query = """
                            INSERT INTO technical_indicators (
                                timestamp, symbol, indicator_name, value
                            ) VALUES ($1, $2, $3, $4)
                        """
                        
                        await conn.execute(
                            query,
                            datetime.now(),
                            symbol,
                            indicator_name,
                            float(value)
                        )
            
        except Exception as e:
            self.logger.error(f"Error storing technical indicators: {e}")
    
    async def store_sentiment_data(self, sentiment_data: List[Dict[str, Any]]):
        """Store sentiment data in TimescaleDB."""
        try:
            if not self.timescale_pool or not sentiment_data:
                return
            
            query = """
                INSERT INTO sentiment_data (
                    timestamp, symbol, source, headline, sentiment_score, confidence, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            async with self.timescale_pool.acquire() as conn:
                for data in sentiment_data:
                    await conn.execute(
                        query,
                        datetime.now(),
                        data.get('symbol'),
                        data.get('source'),
                        data.get('title', '')[:500],  # Truncate long headlines
                        data.get('sentiment_score', 0.0),
                        data.get('confidence', 0.0),
                        json.dumps(data)
                    )
            
        except Exception as e:
            self.logger.error(f"Error storing sentiment data: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = '1m',
        days: Optional[int] = None
    ) -> List[MarketData]:
        """Get historical market data."""
        try:
            if not self.timescale_pool:
                return []
            
            # Set default date range
            if days:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
            elif not start_date:
                start_date = datetime.now() - timedelta(days=30)
            
            if not end_date:
                end_date = datetime.now()
            
            query = """
                SELECT timestamp, symbol, open_price, high_price, low_price, 
                       close_price, volume
                FROM ohlcv_data
                WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                      AND timeframe = $4
                ORDER BY timestamp ASC
            """
            
            async with self.timescale_pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, start_date, end_date, timeframe)
                
                market_data_list = []
                for row in rows:
                    market_data = MarketData(
                        symbol=row['symbol'],
                        timestamp=row['timestamp'],
                        price=float(row['close_price']),
                        volume=float(row['volume']),
                        open=float(row['open_price']),
                        high=float(row['high_price']),
                        low=float(row['low_price']),
                        close=float(row['close_price'])
                    )
                    market_data_list.append(market_data)
                
                return market_data_list
        
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return []
    
    async def get_latest_cached_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data from Redis."""
        try:
            if not self.redis_client:
                return None
            
            key = f"latest:{symbol}"
            data = await self.redis_client.get(key)
            
            if data:
                parsed_data = json.loads(data)
                parsed_data['timestamp'] = datetime.fromisoformat(parsed_data['timestamp'])
                
                return MarketData(**parsed_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached data: {e}")
            return None
    
    async def get_technical_indicators(
        self,
        symbol: str,
        indicator_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get technical indicators for a symbol."""
        try:
            if not self.timescale_pool:
                return {}
            
            if not start_date:
                start_date = datetime.now() - timedelta(days=7)
            
            where_clause = "WHERE symbol = $1 AND timestamp >= $2"
            params = [symbol, start_date]
            
            if indicator_names:
                where_clause += " AND indicator_name = ANY($3)"
                params.append(indicator_names)
            
            query = f"""
                SELECT timestamp, indicator_name, value
                FROM technical_indicators
                {where_clause}
                ORDER BY timestamp ASC, indicator_name
            """
            
            async with self.timescale_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                indicators = {}
                for row in rows:
                    indicator_name = row['indicator_name']
                    if indicator_name not in indicators:
                        indicators[indicator_name] = []
                    
                    indicators[indicator_name].append({
                        'timestamp': row['timestamp'],
                        'value': float(row['value'])
                    })
                
                return indicators
        
        except Exception as e:
            self.logger.error(f"Error getting technical indicators: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup database connections."""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.timescale_pool:
                await self.timescale_pool.close()
            
            if self.redis_client:
                await self.redis_client.aclose()
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'postgres_connected': self.postgres_pool is not None,
            'timescale_connected': self.timescale_pool is not None,
            'redis_connected': self.redis_client is not None,
            'data_points': 0,
            'indicators': 0,
            'sentiment_records': 0
        }
        
        try:
            if self.timescale_pool:
                async with self.timescale_pool.acquire() as conn:
                    # Count data points
                    result = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_data")
                    stats['data_points'] = result or 0
                    
                    # Count indicators
                    result = await conn.fetchval("SELECT COUNT(*) FROM technical_indicators")
                    stats['indicators'] = result or 0
                    
                    # Count sentiment records
                    result = await conn.fetchval("SELECT COUNT(*) FROM sentiment_data")
                    stats['sentiment_records'] = result or 0
        
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
        
        return stats