"""
Feature Engineering Module

Creates and transforms features for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timedelta

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class FeatureEngineer:
    """Feature engineering for trading ML models."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("FeatureEngineer")
        
        # Feature configuration
        self.lookback_periods = [5, 10, 20, 50]
        self.technical_indicators_enabled = config.get('ml.features.technical_indicators', True)
        self.sentiment_enabled = config.get('ml.features.sentiment_scores', True)
        self.volume_enabled = config.get('ml.features.volume_analysis', True)
        self.price_patterns_enabled = config.get('ml.features.price_patterns', True)
        
        # Normalization
        self.normalize_features = True
        self.feature_scalers = {}
    
    async def initialize(self):
        """Initialize feature engineer."""
        self.logger.info("Feature Engineer initialized")
    
    async def engineer_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Engineer features from raw market data."""
        try:
            if data.empty or len(data) < 20:
                self.logger.warning("Insufficient data for feature engineering")
                return None
            
            # Ensure data is sorted by timestamp
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
            
            # Start with base features
            features = data.copy()
            
            # Add price-based features
            features = await self._add_price_features(features)
            
            # Add volume features
            if self.volume_enabled:
                features = await self._add_volume_features(features)
            
            # Add technical indicators
            if self.technical_indicators_enabled:
                features = await self._add_technical_indicators(features)
            
            # Add time-based features
            features = await self._add_time_features(features)
            
            # Add lagged features
            features = await self._add_lagged_features(features)
            
            # Add rolling statistics
            features = await self._add_rolling_features(features)
            
            # Add price patterns
            if self.price_patterns_enabled:
                features = await self._add_price_patterns(features)
            
            # Create target variable (future returns)
            features = await self._create_target_variable(features)
            
            # Clean and normalize
            features = await self._clean_and_normalize(features)
            
            self.logger.info(f"Engineered {len(features.columns)} features from {len(data)} samples")
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return None
    
    async def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price ratios
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Price position within high-low range
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        data['price_position'] = data['price_position'].fillna(0.5)
        
        # Volatility proxy
        data['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        return data
    
    async def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in data.columns:
            return data
        
        # Volume ratios
        data['volume_sma_5'] = data['volume'].rolling(5).mean()
        data['volume_sma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # Price-volume relationship
        data['price_volume'] = data['close'] * data['volume']
        data['volume_weighted_price'] = (
            data['price_volume'].rolling(20).sum() / 
            data['volume'].rolling(20).sum()
        )
        
        # On-balance volume
        data['obv_change'] = np.where(
            data['close'] > data['close'].shift(1), 
            data['volume'], 
            -data['volume']
        )
        data['obv'] = data['obv_change'].cumsum()
        
        return data
    
    async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(data) > period:
                data[f'sma_{period}'] = data['close'].rolling(period).mean()
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                data[f'price_sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
        
        # RSI
        data['rsi'] = await self._calculate_rsi(data['close'])
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        data['bb_upper'] = sma_20 + (2 * std_20)
        data['bb_lower'] = sma_20 - (2 * std_20)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma_20
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        return data
    
    async def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in data.columns:
            return data
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract time components
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['quarter'] = data['timestamp'].dt.quarter
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Market session indicators
        data['is_market_open'] = (
            (data['hour'] >= 9) & (data['hour'] < 16) & 
            (data['day_of_week'] < 5)
        ).astype(int)
        
        return data
    
    async def _add_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        lag_columns = ['close', 'volume', 'returns', 'rsi']
        lag_periods = [1, 3, 5]
        
        for col in lag_columns:
            if col in data.columns:
                for lag in lag_periods:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return data
    
    async def _add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        windows = [5, 10, 20]
        
        for window in windows:
            if len(data) > window:
                # Rolling statistics for returns
                if 'returns' in data.columns:
                    data[f'returns_mean_{window}'] = data['returns'].rolling(window).mean()
                    data[f'returns_std_{window}'] = data['returns'].rolling(window).std()
                    data[f'returns_skew_{window}'] = data['returns'].rolling(window).skew()
                
                # Rolling statistics for volume
                if 'volume' in data.columns:
                    data[f'volume_mean_{window}'] = data['volume'].rolling(window).mean()
                    data[f'volume_std_{window}'] = data['volume'].rolling(window).std()
                
                # Rolling min/max
                data[f'close_min_{window}'] = data['close'].rolling(window).min()
                data[f'close_max_{window}'] = data['close'].rolling(window).max()
                data[f'close_position_{window}'] = (
                    (data['close'] - data[f'close_min_{window}']) / 
                    (data[f'close_max_{window}'] - data[f'close_min_{window}'])
                )
        
        return data
    
    async def _add_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        # Momentum patterns
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        # Support and resistance
        data['support_20'] = data['low'].rolling(20).min()
        data['resistance_20'] = data['high'].rolling(20).max()
        data['support_distance'] = (data['close'] - data['support_20']) / data['close']
        data['resistance_distance'] = (data['resistance_20'] - data['close']) / data['close']
        
        # Trend strength
        data['trend_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
        data['trend_20'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
        
        # Volatility regime
        data['volatility_5'] = data['returns'].rolling(5).std()
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_regime'] = data['volatility_5'] / data['volatility_20']
        
        return data
    
    async def _create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for supervised learning."""
        # Future return prediction
        forward_periods = [1, 3, 5]
        
        for period in forward_periods:
            data[f'future_return_{period}'] = (
                data['close'].shift(-period) / data['close'] - 1
            )
        
        # Classification targets
        threshold = 0.01  # 1% threshold
        data['target_class'] = 1  # Hold
        data.loc[data['future_return_1'] > threshold, 'target_class'] = 2  # Buy
        data.loc[data['future_return_1'] < -threshold, 'target_class'] = 0  # Sell
        
        return data
    
    async def _clean_and_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features."""
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with forward fill, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        missing_ratios = data.isnull().sum() / len(data)
        cols_to_keep = missing_ratios[missing_ratios < missing_threshold].index
        data = data[cols_to_keep]
        
        # Remove rows with any remaining NaN values
        data = data.dropna()
        
        # Normalize numerical features (except target variables)
        if self.normalize_features:
            exclude_cols = [
                'timestamp', 'symbol', 'target_class',
                'future_return_1', 'future_return_3', 'future_return_5'
            ]
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
            
            # Simple normalization (min-max scaling)
            for col in cols_to_normalize:
                if col in data.columns:
                    col_min = data[col].min()
                    col_max = data[col].max()
                    
                    if col_max != col_min:
                        data[col] = (data[col] - col_min) / (col_max - col_min)
                    else:
                        data[col] = 0.5  # If all values are the same
        
        return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder)."""
        # This would be populated by models during training
        return {}
    
    def get_feature_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about engineered features."""
        if data.empty:
            return {}
        
        return {
            'total_features': len(data.columns),
            'sample_count': len(data),
            'feature_types': {
                'numeric': len(data.select_dtypes(include=[np.number]).columns),
                'categorical': len(data.select_dtypes(include=['object']).columns),
                'datetime': len(data.select_dtypes(include=['datetime64']).columns)
            },
            'missing_values': data.isnull().sum().sum(),
            'feature_names': list(data.columns)
        }