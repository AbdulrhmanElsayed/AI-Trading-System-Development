"""
Position Sizing Algorithms

Various position sizing methods for optimal capital allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import TradingSignal, Portfolio, RiskMetrics


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    FIXED = "fixed"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"


class PositionSizer:
    """Position sizing calculator with multiple algorithms."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("PositionSizer")
        
        # Configuration
        self.method = PositionSizingMethod(
            config.get('risk_management.position_sizing.method', 'kelly_criterion')
        )
        self.max_position_size = config.get('trading.max_position_size', 0.1)
        self.max_leverage = config.get('risk_management.position_sizing.max_leverage', 2.0)
        self.base_currency = config.get('trading.base_currency', 'USD')
        
        # Risk parameters
        self.risk_free_rate = config.get('risk_management.risk_free_rate', 0.02)
        self.confidence_threshold = config.get('ml.min_signal_confidence', 0.6)
        
        # Historical performance tracking for Kelly Criterion
        self.win_rate_history = {}
        self.avg_win_loss_ratio = {}
        
        # Volatility tracking
        self.volatility_cache = {}
        
    async def initialize(self):
        """Initialize position sizer."""
        self.logger.info(f"Position sizer initialized with method: {self.method.value}")
    
    async def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio: Portfolio,
        risk_metrics: RiskMetrics
    ) -> float:
        """Calculate position size based on configured method."""
        try:
            if signal.confidence < self.confidence_threshold:
                self.logger.debug(f"Signal confidence too low: {signal.confidence}")
                return 0.0
            
            # Get base position size using selected method
            if self.method == PositionSizingMethod.FIXED:
                base_size = await self._fixed_position_size(signal, portfolio)
            elif self.method == PositionSizingMethod.KELLY_CRITERION:
                base_size = await self._kelly_criterion_size(signal, portfolio)
            elif self.method == PositionSizingMethod.VOLATILITY_BASED:
                base_size = await self._volatility_based_size(signal, portfolio, risk_metrics)
            elif self.method == PositionSizingMethod.RISK_PARITY:
                base_size = await self._risk_parity_size(signal, portfolio, risk_metrics)
            elif self.method == PositionSizingMethod.OPTIMAL_F:
                base_size = await self._optimal_f_size(signal, portfolio)
            else:
                base_size = await self._fixed_position_size(signal, portfolio)
            
            # Apply position size constraints
            final_size = await self._apply_constraints(base_size, signal, portfolio)
            
            self.logger.debug(
                f"Position size calculated for {signal.symbol}: "
                f"Base: {base_size:.4f}, Final: {final_size:.4f}"
            )
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _fixed_position_size(self, signal: TradingSignal, portfolio: Portfolio) -> float:
        """Fixed percentage position sizing."""
        if not signal.price or portfolio.total_value <= 0:
            return 0.0
        
        # Fixed percentage of portfolio
        position_value = portfolio.total_value * self.max_position_size
        position_size = position_value / signal.price
        
        return position_size
    
    async def _kelly_criterion_size(self, signal: TradingSignal, portfolio: Portfolio) -> float:
        """Kelly Criterion position sizing."""
        try:
            # Get historical performance for this symbol
            win_rate = await self._get_win_rate(signal.symbol)
            avg_win_loss_ratio = await self._get_avg_win_loss_ratio(signal.symbol)
            
            if win_rate <= 0 or avg_win_loss_ratio <= 0:
                # Fallback to confidence-based sizing if no history
                kelly_fraction = signal.confidence * 0.5  # Conservative
            else:
                # Kelly formula: f = (bp - q) / b
                # where b = avg_win_loss_ratio, p = win_rate, q = loss_rate
                loss_rate = 1 - win_rate
                kelly_fraction = (avg_win_loss_ratio * win_rate - loss_rate) / avg_win_loss_ratio
                
                # Apply confidence adjustment
                kelly_fraction *= signal.confidence
            
            # Cap Kelly fraction for safety
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
            
            if portfolio.total_value <= 0 or not signal.price:
                return 0.0
            
            position_value = portfolio.total_value * kelly_fraction
            position_size = position_value / signal.price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in Kelly criterion calculation: {e}")
            return await self._fixed_position_size(signal, portfolio)
    
    async def _volatility_based_size(
        self, 
        signal: TradingSignal, 
        portfolio: Portfolio, 
        risk_metrics: RiskMetrics
    ) -> float:
        """Volatility-based position sizing (inverse volatility)."""
        try:
            # Get symbol volatility
            volatility = await self._get_symbol_volatility(signal.symbol)
            
            if volatility <= 0:
                return await self._fixed_position_size(signal, portfolio)
            
            # Target portfolio volatility
            target_volatility = self.config.get('risk_management.target_volatility', 0.15)
            
            # Position size inversely proportional to volatility
            # Higher volatility = smaller position
            volatility_adjustment = target_volatility / volatility
            
            # Apply confidence factor
            confidence_factor = signal.confidence
            
            # Base position size
            base_fraction = self.max_position_size * volatility_adjustment * confidence_factor
            base_fraction = max(0, min(base_fraction, self.max_position_size))
            
            if portfolio.total_value <= 0 or not signal.price:
                return 0.0
            
            position_value = portfolio.total_value * base_fraction
            position_size = position_value / signal.price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in volatility-based sizing: {e}")
            return await self._fixed_position_size(signal, portfolio)
    
    async def _risk_parity_size(
        self, 
        signal: TradingSignal, 
        portfolio: Portfolio, 
        risk_metrics: RiskMetrics
    ) -> float:
        """Risk parity position sizing."""
        try:
            # Get current portfolio risk contribution
            current_positions = len(portfolio.positions)
            
            if current_positions == 0:
                # First position - use fixed size
                return await self._fixed_position_size(signal, portfolio)
            
            # Target equal risk contribution from each position
            target_risk_per_position = 1.0 / (current_positions + 1)
            
            # Get symbol volatility and correlation
            symbol_volatility = await self._get_symbol_volatility(signal.symbol)
            
            if symbol_volatility <= 0:
                return await self._fixed_position_size(signal, portfolio)
            
            # Calculate position size for target risk contribution
            portfolio_volatility = risk_metrics.volatility or 0.15
            
            # Risk parity fraction
            risk_parity_fraction = (target_risk_per_position * portfolio_volatility) / symbol_volatility
            risk_parity_fraction = max(0, min(risk_parity_fraction, self.max_position_size))
            
            # Apply confidence adjustment
            risk_parity_fraction *= signal.confidence
            
            if portfolio.total_value <= 0 or not signal.price:
                return 0.0
            
            position_value = portfolio.total_value * risk_parity_fraction
            position_size = position_value / signal.price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in risk parity sizing: {e}")
            return await self._fixed_position_size(signal, portfolio)
    
    async def _optimal_f_size(self, signal: TradingSignal, portfolio: Portfolio) -> float:
        """Optimal F position sizing (Ralph Vince method)."""
        try:
            # Get historical returns for this symbol
            historical_returns = await self._get_historical_returns(signal.symbol)
            
            if not historical_returns or len(historical_returns) < 10:
                return await self._fixed_position_size(signal, portfolio)
            
            # Calculate Optimal F
            optimal_f = self._calculate_optimal_f(historical_returns)
            
            # Apply confidence and safety factors
            optimal_f *= signal.confidence
            optimal_f = max(0, min(optimal_f, self.max_position_size))
            
            if portfolio.total_value <= 0 or not signal.price:
                return 0.0
            
            position_value = portfolio.total_value * optimal_f
            position_size = position_value / signal.price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in Optimal F calculation: {e}")
            return await self._fixed_position_size(signal, portfolio)
    
    def _calculate_optimal_f(self, returns: List[float]) -> float:
        """Calculate Optimal F using Ralph Vince method."""
        if not returns:
            return 0.0
        
        # Find the largest loss
        largest_loss = min(returns) if returns else 0
        
        if largest_loss >= 0:
            return 0.1  # No losses in history, use conservative sizing
        
        # Test different f values to find optimal
        best_f = 0.0
        best_geomean = 0.0
        
        # Test f values from 0.01 to 0.50 in 0.01 increments
        for f in np.arange(0.01, 0.51, 0.01):
            # Calculate geometric mean of returns for this f
            hpr_values = []  # Holding Period Returns
            
            for ret in returns:
                hpr = 1 + (f * ret / abs(largest_loss))
                if hpr > 0:
                    hpr_values.append(hpr)
            
            if hpr_values:
                geomean = np.exp(np.mean(np.log(hpr_values)))
                
                if geomean > best_geomean:
                    best_geomean = geomean
                    best_f = f
        
        return best_f
    
    async def _apply_constraints(
        self, 
        base_size: float, 
        signal: TradingSignal, 
        portfolio: Portfolio
    ) -> float:
        """Apply various constraints to position size."""
        if base_size <= 0 or not signal.price:
            return 0.0
        
        # Maximum position size constraint
        max_value = portfolio.total_value * self.max_position_size
        max_size_by_value = max_value / signal.price
        
        # Apply maximum constraint
        constrained_size = min(base_size, max_size_by_value)
        
        # Minimum position size (avoid dust trades)
        min_value = self.config.get('trading.min_position_value', 100.0)
        min_size = min_value / signal.price
        
        if constrained_size < min_size:
            constrained_size = 0.0  # Too small, don't trade
        
        # Available cash constraint
        available_cash = portfolio.cash
        max_size_by_cash = (available_cash * 0.95) / signal.price  # Keep 5% cash buffer
        
        constrained_size = min(constrained_size, max_size_by_cash)
        
        # Round to reasonable precision
        constrained_size = round(constrained_size, 6)
        
        return constrained_size
    
    # Helper methods for data retrieval (placeholders for now)
    
    async def _get_win_rate(self, symbol: str) -> float:
        """Get historical win rate for a symbol."""
        # This would query historical trade results
        # For now, return a reasonable default
        return self.win_rate_history.get(symbol, 0.55)
    
    async def _get_avg_win_loss_ratio(self, symbol: str) -> float:
        """Get average win/loss ratio for a symbol."""
        # This would calculate from historical trades
        return self.avg_win_loss_ratio.get(symbol, 1.5)
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get recent volatility for a symbol."""
        # This would calculate from recent price data
        return self.volatility_cache.get(symbol, 0.20)
    
    async def _get_historical_returns(self, symbol: str) -> List[float]:
        """Get historical returns for a symbol."""
        # This would fetch from historical data
        # For now, return simulated returns
        np.random.seed(42)  # For consistent results
        returns = np.random.normal(0.001, 0.02, 100).tolist()
        return returns
    
    def update_performance_stats(self, symbol: str, was_profitable: bool, return_pct: float):
        """Update performance statistics for Kelly Criterion."""
        if symbol not in self.win_rate_history:
            self.win_rate_history[symbol] = 0.5
            self.avg_win_loss_ratio[symbol] = 1.0
        
        # Simple moving average update
        alpha = 0.1  # Learning rate
        
        if was_profitable:
            # Update win rate
            self.win_rate_history[symbol] = (
                self.win_rate_history[symbol] * (1 - alpha) + alpha
            )
        else:
            # Update loss rate
            self.win_rate_history[symbol] = (
                self.win_rate_history[symbol] * (1 - alpha)
            )
        
        # Update win/loss ratio (simplified)
        if return_pct != 0:
            current_ratio = self.avg_win_loss_ratio[symbol]
            new_ratio = abs(return_pct) if was_profitable else 1.0 / abs(return_pct)
            self.avg_win_loss_ratio[symbol] = current_ratio * (1 - alpha) + new_ratio * alpha
    
    def get_method(self) -> str:
        """Get current position sizing method."""
        return self.method.value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get position sizing statistics."""
        return {
            'method': self.method.value,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage,
            'symbols_tracked': len(self.win_rate_history),
            'avg_win_rate': np.mean(list(self.win_rate_history.values())) if self.win_rate_history else 0.0,
            'avg_win_loss_ratio': np.mean(list(self.avg_win_loss_ratio.values())) if self.avg_win_loss_ratio else 0.0
        }