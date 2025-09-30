"""
Data Processing Module

Handles all data-related operations including:
- Market data ingestion from multiple sources
- Technical indicator calculations
- Data storage and caching
- Real-time data streaming
"""

from .market_data_manager import MarketDataManager
from .data_sources import (
    BaseDataSource,
    YFinanceDataSource,
    AlphaVantageDataSource,
    BinanceDataSource,
    NewsDataSource
)
from .technical_indicators import TechnicalIndicators
from .data_storage import DataStorage

__all__ = [
    'MarketDataManager',
    'BaseDataSource',
    'YFinanceDataSource',
    'AlphaVantageDataSource',
    'BinanceDataSource',
    'NewsDataSource',
    'TechnicalIndicators',
    'DataStorage'
]