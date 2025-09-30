"""
Risk Metrics Calculator

Calculates various risk metrics for portfolio and positions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math
from scipy import stats

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import Portfolio, Position, MarketData


@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    var_95: float = 0.0  # Value at Risk (95%)
    var_99: float = 0.0  # Value at Risk (99%)
    cvar_95: float = 0.0  # Conditional Value at Risk (95%)
    cvar_99: float = 0.0  # Conditional Value at Risk (99%)
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    timestamp: datetime = None


class RiskMetricsCalculator:
    """Calculates comprehensive risk metrics."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("RiskMetricsCalculator")
        
        # Risk parameters
        self.lookback_period = config.get('risk_management.lookback_period', 252)  # Days
        self.confidence_levels = [0.95, 0.99]
        self.risk_free_rate = config.get('risk_management.risk_free_rate', 0.02)
        
        # Historical data storage
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.portfolio_values = []
        
        # Correlation matrices
        self.correlation_matrix = None
        self.last_correlation_update = None
        
    def calculate_portfolio_metrics(
        self, 
        portfolio_returns: List[float],
        benchmark_returns: Optional[List[float]] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for portfolio."""
        try:
            if not portfolio_returns or len(portfolio_returns) < 2:
                return RiskMetrics(timestamp=datetime.now())
            
            returns = np.array(portfolio_returns)
            
            # Basic statistics
            volatility = np.std(returns, ddof=1) * np.sqrt(252)  # Annualized
            
            # Value at Risk (VaR)
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            cvar_99 = self._calculate_cvar(returns, 0.99)
            
            # Maximum Drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Downside deviation
            downside_deviation = self._calculate_downside_deviation(returns)
            
            # Sortino Ratio
            sortino_ratio = self._calculate_sortino_ratio(returns, downside_deviation)
            
            # Calmar Ratio
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Higher moments
            skewness = stats.skew(returns) if len(returns) > 2 else 0.0
            kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0.0
            
            # Market-related metrics (if benchmark provided)
            beta = 0.0
            alpha = 0.0
            tracking_error = 0.0
            information_ratio = 0.0
            
            if benchmark_returns and len(benchmark_returns) == len(portfolio_returns):
                beta = self._calculate_beta(returns, np.array(benchmark_returns))
                alpha = self._calculate_alpha(returns, np.array(benchmark_returns), beta)
                tracking_error = self._calculate_tracking_error(returns, np.array(benchmark_returns))
                information_ratio = self._calculate_information_ratio(
                    returns, np.array(benchmark_returns), tracking_error
                )
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                downside_deviation=downside_deviation,
                skewness=skewness,
                kurtosis=kurtosis,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return RiskMetrics(timestamp=datetime.now())
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk using historical simulation."""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Find the percentile
        index = int((1 - confidence_level) * len(sorted_returns))
        
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
        elif index < 0:
            index = 0
        
        var = -sorted_returns[index]  # VaR is positive for losses
        return var
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        
        # Calculate average of returns worse than VaR
        worse_returns = returns[returns <= -var]
        
        if len(worse_returns) == 0:
            return var
        
        cvar = -np.mean(worse_returns)
        return cvar
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0
        
        # Convert to cumulative returns (wealth index)
        cumulative_returns = np.cumprod(1 + np.array(returns))
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Return maximum drawdown (as positive value)
        max_dd = -np.min(drawdowns)
        return max_dd
    
    def _calculate_downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation (semi-standard deviation)."""
        if len(returns) == 0:
            return 0.0
        
        # Only consider returns below the mean (or risk-free rate)
        target_return = self.risk_free_rate / 252  # Daily risk-free rate
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_variance = np.mean((downside_returns - target_return) ** 2)
        downside_deviation = np.sqrt(downside_variance) * np.sqrt(252)  # Annualized
        
        return downside_deviation
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, downside_deviation: float) -> float:
        """Calculate Sortino ratio."""
        if downside_deviation == 0:
            return 0.0
        
        # Annualized excess return
        mean_return = np.mean(returns) * 252
        excess_return = mean_return - self.risk_free_rate
        
        sortino_ratio = excess_return / downside_deviation
        return sortino_ratio
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        
        # Annualized return
        mean_return = np.mean(returns) * 252
        
        calmar_ratio = mean_return / max_drawdown
        return calmar_ratio
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate portfolio beta relative to benchmark."""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            return 0.0
        
        beta = covariance / benchmark_variance
        return beta
    
    def _calculate_alpha(
        self, 
        portfolio_returns: np.ndarray, 
        benchmark_returns: np.ndarray, 
        beta: float
    ) -> float:
        """Calculate Jensen's alpha."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Annualized returns
        portfolio_return = np.mean(portfolio_returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        
        # Jensen's alpha formula
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        return alpha
    
    def _calculate_tracking_error(
        self, 
        portfolio_returns: np.ndarray, 
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate tracking error."""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        # Tracking error (annualized standard deviation of active returns)
        tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)
        return tracking_error
    
    def _calculate_information_ratio(
        self, 
        portfolio_returns: np.ndarray, 
        benchmark_returns: np.ndarray,
        tracking_error: float
    ) -> float:
        """Calculate information ratio."""
        if tracking_error == 0 or len(portfolio_returns) != len(benchmark_returns):
            return 0.0
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        # Annualized excess return
        excess_return = np.mean(active_returns) * 252
        
        information_ratio = excess_return / tracking_error
        return information_ratio
    
    def calculate_position_risk(self, position: Position, market_data: MarketData) -> Dict[str, float]:
        """Calculate risk metrics for individual position."""
        try:
            # Position value
            position_value = position.quantity * position.market_price
            
            # Delta (price sensitivity)
            delta = position.quantity  # For stocks, delta = quantity
            
            # Simple volatility estimate (would need historical data for better estimate)
            volatility = 0.20  # Default 20% annual volatility
            
            # Daily VaR (95% confidence)
            daily_volatility = volatility / np.sqrt(252)
            var_95 = position_value * 1.645 * daily_volatility  # 95% confidence
            var_99 = position_value * 2.326 * daily_volatility  # 99% confidence
            
            return {
                'position_value': position_value,
                'delta': delta,
                'var_95': var_95,
                'var_99': var_99,
                'volatility': volatility,
                'daily_volatility': daily_volatility,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'pnl_percent': (
                    position.unrealized_pnl / (position.quantity * position.average_price)
                    if position.quantity > 0 and position.average_price > 0 else 0.0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {position.symbol}: {e}")
            return {}
    
    def calculate_portfolio_var(
        self, 
        positions: Dict[str, Position],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate portfolio-level VaR considering correlations."""
        try:
            if not positions:
                return {'var_95': 0.0, 'var_99': 0.0, 'diversified_var_95': 0.0, 'diversified_var_99': 0.0}
            
            symbols = list(positions.keys())
            position_vars = []
            position_values = []
            
            # Calculate individual position VaRs
            for symbol, position in positions.items():
                # Simple VaR calculation (would use historical data in practice)
                position_value = position.quantity * position.market_price
                position_volatility = 0.20  # Default volatility
                daily_vol = position_volatility / np.sqrt(252)
                
                var_95 = position_value * 1.645 * daily_vol
                position_vars.append(var_95)
                position_values.append(position_value)
            
            # Undiversified VaR (sum of individual VaRs)
            undiversified_var_95 = sum(position_vars)
            undiversified_var_99 = sum(v * 2.326 / 1.645 for v in position_vars)
            
            # Diversified VaR (considering correlations)
            diversified_var_95 = undiversified_var_95
            diversified_var_99 = undiversified_var_99
            
            if correlation_matrix is not None and len(symbols) > 1:
                # Matrix calculation for diversified VaR
                weights = np.array(position_values) / sum(position_values)
                vol_vector = np.array([0.20] * len(symbols))  # Default volatilities
                
                portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(vol_vector, vol_vector), weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                total_value = sum(position_values)
                daily_portfolio_vol = portfolio_volatility / np.sqrt(252)
                
                diversified_var_95 = total_value * 1.645 * daily_portfolio_vol
                diversified_var_99 = total_value * 2.326 * daily_portfolio_vol
            
            return {
                'var_95': undiversified_var_95,
                'var_99': undiversified_var_99,
                'diversified_var_95': diversified_var_95,
                'diversified_var_99': diversified_var_99,
                'diversification_benefit': undiversified_var_95 - diversified_var_95
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return {'var_95': 0.0, 'var_99': 0.0, 'diversified_var_95': 0.0, 'diversified_var_99': 0.0}
    
    def calculate_stress_scenarios(self, positions: Dict[str, Position]) -> Dict[str, Dict[str, float]]:
        """Calculate portfolio performance under stress scenarios."""
        try:
            scenarios = {
                'market_crash_2008': {'equity_shock': -0.50, 'volatility_shock': 2.0},
                'black_monday_1987': {'equity_shock': -0.22, 'volatility_shock': 3.0},
                'covid_crash_2020': {'equity_shock': -0.35, 'volatility_shock': 4.0},
                'flash_crash_2010': {'equity_shock': -0.10, 'volatility_shock': 5.0},
                'mild_correction': {'equity_shock': -0.15, 'volatility_shock': 1.5}
            }
            
            results = {}
            
            for scenario_name, scenario in scenarios.items():
                total_pnl = 0.0
                total_value = 0.0
                
                for symbol, position in positions.items():
                    position_value = position.quantity * position.market_price
                    total_value += position_value
                    
                    # Apply shock to position
                    shocked_price = position.market_price * (1 + scenario['equity_shock'])
                    shocked_value = position.quantity * shocked_price
                    position_pnl = shocked_value - position_value
                    total_pnl += position_pnl
                
                results[scenario_name] = {
                    'total_pnl': total_pnl,
                    'pnl_percent': total_pnl / total_value if total_value > 0 else 0.0,
                    'shocked_value': total_value + total_pnl
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating stress scenarios: {e}")
            return {}
    
    def calculate_concentration_risk(self, positions: Dict[str, Position]) -> Dict[str, Any]:
        """Calculate portfolio concentration metrics."""
        try:
            if not positions:
                return {'herfindahl_index': 0.0, 'top_5_concentration': 0.0, 'max_position_weight': 0.0}
            
            # Calculate position weights
            total_value = sum(pos.quantity * pos.market_price for pos in positions.values())
            
            if total_value <= 0:
                return {'herfindahl_index': 0.0, 'top_5_concentration': 0.0, 'max_position_weight': 0.0}
            
            weights = []
            for position in positions.values():
                weight = (position.quantity * position.market_price) / total_value
                weights.append(weight)
            
            # Sort weights descending
            weights.sort(reverse=True)
            
            # Herfindahl-Hirschman Index
            hhi = sum(w ** 2 for w in weights)
            
            # Top 5 concentration
            top_5_concentration = sum(weights[:5])
            
            # Maximum position weight
            max_position_weight = weights[0] if weights else 0.0
            
            return {
                'herfindahl_index': hhi,
                'top_5_concentration': top_5_concentration,
                'max_position_weight': max_position_weight,
                'number_of_positions': len(positions),
                'effective_number_of_positions': 1 / hhi if hhi > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return {'herfindahl_index': 0.0, 'top_5_concentration': 0.0, 'max_position_weight': 0.0}
    
    def update_correlation_matrix(self, returns_data: Dict[str, List[float]]):
        """Update correlation matrix from returns data."""
        try:
            if not returns_data or len(returns_data) < 2:
                return
            
            # Convert to DataFrame for easier correlation calculation
            df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            self.correlation_matrix = df.corr().values
            self.last_correlation_update = datetime.now()
            
            self.logger.info(f"Updated correlation matrix for {len(returns_data)} assets")
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
    
    def get_risk_summary(self, metrics: RiskMetrics) -> Dict[str, Any]:
        """Get human-readable risk summary."""
        return {
            'risk_level': self._assess_risk_level(metrics),
            'key_metrics': {
                'volatility': f"{metrics.volatility:.2%}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'var_95': f"${metrics.var_95:,.2f}",
                'sharpe_ratio': f"{metrics.sortino_ratio:.2f}",
                'beta': f"{metrics.beta:.2f}"
            },
            'warnings': self._generate_risk_warnings(metrics),
            'recommendations': self._generate_risk_recommendations(metrics)
        }
    
    def _assess_risk_level(self, metrics: RiskMetrics) -> str:
        """Assess overall risk level."""
        if metrics.volatility > 0.30 or metrics.max_drawdown > 0.20:
            return "HIGH"
        elif metrics.volatility > 0.20 or metrics.max_drawdown > 0.10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_risk_warnings(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk warnings based on metrics."""
        warnings = []
        
        if metrics.max_drawdown > 0.15:
            warnings.append("High maximum drawdown detected")
        
        if metrics.volatility > 0.25:
            warnings.append("High portfolio volatility")
        
        if metrics.var_95 > 0:  # Would set appropriate thresholds
            warnings.append(f"Daily VaR (95%) exceeds ${metrics.var_95:,.0f}")
        
        if abs(metrics.skewness) > 2:
            warnings.append("High return skewness detected")
        
        if metrics.kurtosis > 5:
            warnings.append("High kurtosis indicates fat tail risk")
        
        return warnings
    
    def _generate_risk_recommendations(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if metrics.max_drawdown > 0.10:
            recommendations.append("Consider tightening stop-loss levels")
        
        if metrics.volatility > 0.20:
            recommendations.append("Consider diversifying portfolio further")
        
        if metrics.sortino_ratio < 1.0:
            recommendations.append("Review risk-adjusted returns")
        
        if metrics.beta > 1.5:
            recommendations.append("High market sensitivity - consider hedging")
        
        return recommendations