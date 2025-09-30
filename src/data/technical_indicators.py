"""
Technical Indicators Module

Calculates various technical indicators for market analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import asdict

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from src.utils.logger import TradingLogger
from src.utils.types import MarketData


class TechnicalIndicators:
    """Calculate technical indicators for market data."""
    
    def __init__(self):
        self.logger = TradingLogger("TechnicalIndicators")
        
        if not TALIB_AVAILABLE:
            self.logger.warning("TA-Lib not available - using numpy implementations")
    
    def calculate_all(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Calculate all available technical indicators."""
        if len(market_data) < 20:
            return {}
        
        # Convert to pandas DataFrame
        df = self._to_dataframe(market_data)
        
        indicators = {}
        
        try:
            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(df))
            
            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(df))
            
            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(df))
            
            # Support/Resistance
            indicators.update(self._calculate_support_resistance(df))
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'open': md.open or md.price,
                'high': md.high or md.price,
                'low': md.low or md.price,
                'close': md.close or md.price,
                'volume': md.volume or 0
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend-following indicators."""
        indicators = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        try:
            # Moving Averages
            if TALIB_AVAILABLE:
                indicators['sma_20'] = float(talib.SMA(close, timeperiod=20)[-1])
                indicators['sma_50'] = float(talib.SMA(close, timeperiod=50)[-1])
                indicators['ema_12'] = float(talib.EMA(close, timeperiod=12)[-1])
                indicators['ema_26'] = float(talib.EMA(close, timeperiod=26)[-1])
            else:
                indicators['sma_20'] = float(np.mean(close[-20:]))
                indicators['sma_50'] = float(np.mean(close[-50:]))
                indicators['ema_12'] = self._calculate_ema(close, 12)
                indicators['ema_26'] = self._calculate_ema(close, 26)
            
            # MACD
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(close)
                indicators['macd'] = float(macd[-1])
                indicators['macd_signal'] = float(macd_signal[-1])
                indicators['macd_histogram'] = float(macd_hist[-1])
            else:
                ema_12 = indicators['ema_12']
                ema_26 = indicators['ema_26']
                indicators['macd'] = ema_12 - ema_26
                indicators['macd_signal'] = self._calculate_ema([indicators['macd']], 9)
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                indicators['bb_upper'] = float(bb_upper[-1])
                indicators['bb_middle'] = float(bb_middle[-1])
                indicators['bb_lower'] = float(bb_lower[-1])
            else:
                sma_20 = indicators['sma_20']
                std_20 = float(np.std(close[-20:]))
                indicators['bb_upper'] = sma_20 + (2 * std_20)
                indicators['bb_middle'] = sma_20
                indicators['bb_lower'] = sma_20 - (2 * std_20)
            
            # ADX (Average Directional Index)
            if TALIB_AVAILABLE and len(close) >= 14:
                indicators['adx'] = float(talib.ADX(high, low, close, timeperiod=14)[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {e}")
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators."""
        indicators = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        try:
            # RSI (Relative Strength Index)
            if TALIB_AVAILABLE:
                indicators['rsi'] = float(talib.RSI(close, timeperiod=14)[-1])
            else:
                indicators['rsi'] = self._calculate_rsi(close, 14)
            
            # Stochastic Oscillator
            if TALIB_AVAILABLE:
                slowk, slowd = talib.STOCH(high, low, close)
                indicators['stoch_k'] = float(slowk[-1])
                indicators['stoch_d'] = float(slowd[-1])
            else:
                stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14)
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
            
            # Williams %R
            if TALIB_AVAILABLE:
                indicators['williams_r'] = float(talib.WILLR(high, low, close, timeperiod=14)[-1])
            else:
                indicators['williams_r'] = self._calculate_williams_r(high, low, close, 14)
            
            # ROC (Rate of Change)
            if len(close) >= 12:
                if TALIB_AVAILABLE:
                    indicators['roc'] = float(talib.ROC(close, timeperiod=12)[-1])
                else:
                    indicators['roc'] = ((close[-1] - close[-12]) / close[-12]) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility indicators."""
        indicators = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        try:
            # Average True Range (ATR)
            if TALIB_AVAILABLE and len(close) >= 14:
                indicators['atr'] = float(talib.ATR(high, low, close, timeperiod=14)[-1])
            else:
                indicators['atr'] = self._calculate_atr(high, low, close, 14)
            
            # Volatility (Standard Deviation)
            if len(close) >= 20:
                indicators['volatility'] = float(np.std(close[-20:]))
            
            # Bollinger Band Width
            if 'bb_upper' in indicators and 'bb_lower' in indicators and 'bb_middle' in indicators:
                indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume indicators."""
        indicators = {}
        
        close = df['close'].values
        volume = df['volume'].values
        
        try:
            # On-Balance Volume (OBV)
            if TALIB_AVAILABLE:
                indicators['obv'] = float(talib.OBV(close, volume)[-1])
            else:
                indicators['obv'] = self._calculate_obv(close, volume)
            
            # Volume Moving Average
            if len(volume) >= 20:
                indicators['volume_sma'] = float(np.mean(volume[-20:]))
                indicators['volume_ratio'] = float(volume[-1] / indicators['volume_sma'])
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
        
        return indicators
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        indicators = {}
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Pivot Points
            if len(high) >= 3:
                pivot = (high[-2] + low[-2] + close[-2]) / 3
                indicators['pivot_point'] = float(pivot)
                indicators['resistance_1'] = float(2 * pivot - low[-2])
                indicators['support_1'] = float(2 * pivot - high[-2])
                indicators['resistance_2'] = float(pivot + (high[-2] - low[-2]))
                indicators['support_2'] = float(pivot - (high[-2] - low[-2]))
            
            # Recent highs and lows
            if len(high) >= 20:
                indicators['recent_high'] = float(np.max(high[-20:]))
                indicators['recent_low'] = float(np.min(low[-20:]))
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
        
        return indicators
    
    # Helper methods for manual calculations when TA-Lib is not available
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return float(ema)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int):
        """Calculate Stochastic Oscillator."""
        if len(close) < period:
            return 50.0, 50.0
        
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simple moving average for %D
        d_percent = k_percent  # Simplified
        
        return float(k_percent), float(d_percent)
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Williams %R."""
        if len(close) < period:
            return -50.0
        
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - close[-1]) / (highest_high - lowest_low)) * -100
        return float(williams_r)
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return 0.0
        
        # True Range calculation
        tr_list = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_list.append(max(tr1, tr2, tr3))
        
        # Average of last 'period' true ranges
        atr = np.mean(tr_list[-period:])
        return float(atr)
    
    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate On-Balance Volume."""
        obv = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
        return float(obv)