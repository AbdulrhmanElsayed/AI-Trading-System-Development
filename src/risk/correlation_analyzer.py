"""
Correlation Analyzer

Analyzes correlations between assets and manages correlation-based risk.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
from collections import defaultdict
import json

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import Portfolio, Position, MarketData


@dataclass
class CorrelationMetrics:
    """Correlation metrics data structure."""
    correlation_matrix: np.ndarray
    asset_names: List[str]
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    eigenvalues: List[float]
    principal_components: np.ndarray
    concentration_risk: float
    timestamp: datetime


@dataclass
class CorrelationAlert:
    """Correlation-based alert."""
    asset_pair: Tuple[str, str]
    correlation_value: float
    threshold: float
    alert_type: str  # 'high_correlation', 'correlation_breakdown', 'regime_change'
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high'


class CorrelationAnalyzer:
    """Analyzes asset correlations and manages correlation risk."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("CorrelationAnalyzer")
        
        # Configuration parameters
        self.lookback_period = config.get('risk_management.correlation.lookback_period', 60)  # Days
        self.min_observations = config.get('risk_management.correlation.min_observations', 30)
        self.update_frequency = config.get('risk_management.correlation.update_frequency', 3600)  # Seconds
        
        # Correlation thresholds
        self.high_correlation_threshold = config.get('risk_management.correlation.high_threshold', 0.8)
        self.breakdown_threshold = config.get('risk_management.correlation.breakdown_threshold', 0.3)
        self.regime_change_threshold = config.get('risk_management.correlation.regime_change_threshold', 0.4)
        
        # Data storage
        self.price_data = defaultdict(list)  # Asset price history
        self.return_data = defaultdict(list)  # Asset return history
        self.correlation_history = []  # Historical correlation matrices
        
        # Current state
        self.current_correlations = None
        self.last_update = None
        self.correlation_alerts = []
        
        # Market regime detection
        self.correlation_regimes = []
        self.current_regime = None
        
        # Risk metrics
        self.portfolio_correlation_risk = 0.0
        self.concentration_metrics = {}
        
    async def initialize(self):
        """Initialize correlation analyzer."""
        self.logger.info("Correlation Analyzer initialized")
        
        # Start periodic updates
        asyncio.create_task(self._periodic_update_loop())
    
    async def _periodic_update_loop(self):
        """Periodic correlation update loop."""
        while True:
            try:
                await asyncio.sleep(self.update_frequency)
                await self.update_correlations()
                
            except Exception as e:
                self.logger.error(f"Error in correlation update loop: {e}")
    
    def add_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Add price data for correlation calculation."""
        self.price_data[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent data
        cutoff_date = timestamp - timedelta(days=self.lookback_period * 2)
        self.price_data[symbol] = [
            entry for entry in self.price_data[symbol]
            if entry['timestamp'] > cutoff_date
        ]
        
        # Calculate returns if we have enough data
        if len(self.price_data[symbol]) >= 2:
            self._update_returns(symbol)
    
    def _update_returns(self, symbol: str):
        """Update return calculations for a symbol."""
        prices = self.price_data[symbol]
        
        if len(prices) < 2:
            return
        
        # Calculate return for the latest price
        current_price = prices[-1]['price']
        previous_price = prices[-2]['price']
        
        if previous_price > 0:
            return_value = (current_price - previous_price) / previous_price
            
            self.return_data[symbol].append({
                'return': return_value,
                'timestamp': prices[-1]['timestamp']
            })
            
            # Keep only recent returns
            cutoff_date = prices[-1]['timestamp'] - timedelta(days=self.lookback_period * 2)
            self.return_data[symbol] = [
                entry for entry in self.return_data[symbol]
                if entry['timestamp'] > cutoff_date
            ]
    
    async def update_correlations(self):
        """Update correlation matrix and related metrics."""
        try:
            # Get symbols with sufficient data
            valid_symbols = [
                symbol for symbol in self.return_data.keys()
                if len(self.return_data[symbol]) >= self.min_observations
            ]
            
            if len(valid_symbols) < 2:
                self.logger.warning("Insufficient data for correlation analysis")
                return
            
            # Create returns matrix
            returns_matrix = self._create_returns_matrix(valid_symbols)
            
            if returns_matrix is None:
                return
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(returns_matrix)
            
            # Calculate metrics
            metrics = self._calculate_correlation_metrics(correlation_matrix, valid_symbols)
            
            # Store current correlations
            self.current_correlations = metrics
            self.last_update = datetime.now()
            
            # Add to history
            self.correlation_history.append(metrics)
            
            # Keep only recent history
            if len(self.correlation_history) > 1000:
                self.correlation_history = self.correlation_history[-500:]
            
            # Check for alerts
            await self._check_correlation_alerts(metrics)
            
            # Detect regime changes
            await self._detect_regime_changes(metrics)
            
            # Update portfolio risk metrics
            await self._update_portfolio_correlation_risk()
            
            self.logger.debug(f"Updated correlations for {len(valid_symbols)} assets")
            
        except Exception as e:
            self.logger.error(f"Error updating correlations: {e}")
    
    def _create_returns_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Create aligned returns matrix for correlation calculation."""
        try:
            # Find common time periods
            all_timestamps = set()
            for symbol in symbols:
                timestamps = [entry['timestamp'] for entry in self.return_data[symbol]]
                all_timestamps.update(timestamps)
            
            # Sort timestamps
            sorted_timestamps = sorted(all_timestamps)
            
            # Take recent observations only
            recent_timestamps = sorted_timestamps[-self.lookback_period:]
            
            if len(recent_timestamps) < self.min_observations:
                return None
            
            # Create aligned returns matrix
            returns_matrix = []
            
            for symbol in symbols:
                symbol_returns = []
                symbol_data = {entry['timestamp']: entry['return'] for entry in self.return_data[symbol]}
                
                for timestamp in recent_timestamps:
                    if timestamp in symbol_data:
                        symbol_returns.append(symbol_data[timestamp])
                    else:
                        # Use previous value or 0 if no previous value
                        symbol_returns.append(0.0)
                
                returns_matrix.append(symbol_returns)
            
            return np.array(returns_matrix)
            
        except Exception as e:
            self.logger.error(f"Error creating returns matrix: {e}")
            return None
    
    def _calculate_correlation_metrics(
        self, 
        correlation_matrix: np.ndarray, 
        symbols: List[str]
    ) -> CorrelationMetrics:
        """Calculate comprehensive correlation metrics."""
        
        # Remove diagonal (self-correlations)
        n = correlation_matrix.shape[0]
        off_diagonal = correlation_matrix[np.triu_indices(n, k=1)]
        
        # Basic correlation statistics
        avg_correlation = np.mean(off_diagonal)
        max_correlation = np.max(off_diagonal)
        min_correlation = np.min(off_diagonal)
        
        # Principal Component Analysis
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Concentration risk (proportion of variance in first PC)
        total_variance = np.sum(eigenvalues)
        concentration_risk = eigenvalues[0] / total_variance if total_variance > 0 else 0.0
        
        return CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            asset_names=symbols,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            min_correlation=min_correlation,
            eigenvalues=eigenvalues.tolist(),
            principal_components=eigenvectors,
            concentration_risk=concentration_risk,
            timestamp=datetime.now()
        )
    
    async def _check_correlation_alerts(self, metrics: CorrelationMetrics):
        """Check for correlation-based alerts."""
        try:
            current_time = datetime.now()
            correlation_matrix = metrics.correlation_matrix
            symbols = metrics.asset_names
            
            # Check pairwise correlations
            n = len(symbols)
            for i in range(n):
                for j in range(i + 1, n):
                    correlation = correlation_matrix[i, j]
                    symbol_pair = (symbols[i], symbols[j])
                    
                    # High correlation alert
                    if correlation > self.high_correlation_threshold:
                        alert = CorrelationAlert(
                            asset_pair=symbol_pair,
                            correlation_value=correlation,
                            threshold=self.high_correlation_threshold,
                            alert_type='high_correlation',
                            timestamp=current_time,
                            severity='medium' if correlation < 0.9 else 'high'
                        )
                        await self._trigger_correlation_alert(alert)
                    
                    # Check for correlation breakdown (if we have historical data)
                    await self._check_correlation_breakdown(symbol_pair, correlation, current_time)
            
            # Check concentration risk
            if metrics.concentration_risk > 0.7:
                self.logger.warning(
                    f"High correlation concentration risk: {metrics.concentration_risk:.2%}"
                )
            
        except Exception as e:
            self.logger.error(f"Error checking correlation alerts: {e}")
    
    async def _check_correlation_breakdown(
        self, 
        symbol_pair: Tuple[str, str], 
        current_correlation: float,
        timestamp: datetime
    ):
        """Check for correlation breakdown scenarios."""
        try:
            # Look for historical correlation for this pair
            if len(self.correlation_history) < 10:
                return
            
            # Get historical correlations for this pair
            historical_correlations = []
            
            for hist_metrics in self.correlation_history[-10:]:
                if symbol_pair[0] in hist_metrics.asset_names and symbol_pair[1] in hist_metrics.asset_names:
                    i = hist_metrics.asset_names.index(symbol_pair[0])
                    j = hist_metrics.asset_names.index(symbol_pair[1])
                    hist_corr = hist_metrics.correlation_matrix[i, j]
                    historical_correlations.append(hist_corr)
            
            if not historical_correlations:
                return
            
            # Check for significant drop in correlation
            avg_historical = np.mean(historical_correlations)
            correlation_change = abs(current_correlation - avg_historical)
            
            if correlation_change > self.regime_change_threshold:
                alert = CorrelationAlert(
                    asset_pair=symbol_pair,
                    correlation_value=current_correlation,
                    threshold=self.regime_change_threshold,
                    alert_type='regime_change',
                    timestamp=timestamp,
                    severity='high'
                )
                await self._trigger_correlation_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking correlation breakdown: {e}")
    
    async def _trigger_correlation_alert(self, alert: CorrelationAlert):
        """Trigger correlation alert."""
        self.correlation_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.correlation_alerts) > 1000:
            self.correlation_alerts = self.correlation_alerts[-500:]
        
        self.logger.warning(
            f"Correlation alert: {alert.alert_type} for {alert.asset_pair[0]}-{alert.asset_pair[1]} "
            f"(correlation: {alert.correlation_value:.3f}, threshold: {alert.threshold:.3f})"
        )
    
    async def _detect_regime_changes(self, metrics: CorrelationMetrics):
        """Detect correlation regime changes."""
        try:
            if len(self.correlation_history) < 5:
                return
            
            # Compare current average correlation with recent history
            recent_avg_correlations = [
                hist.avg_correlation for hist in self.correlation_history[-5:]
            ]
            
            historical_avg = np.mean(recent_avg_correlations[:-1])
            current_avg = metrics.avg_correlation
            
            # Check for significant change
            change_magnitude = abs(current_avg - historical_avg)
            
            if change_magnitude > 0.2:  # 20% change threshold
                regime_type = "high_correlation" if current_avg > historical_avg else "low_correlation"
                
                new_regime = {
                    'type': regime_type,
                    'start_time': datetime.now(),
                    'avg_correlation': current_avg,
                    'change_from_previous': change_magnitude
                }
                
                self.correlation_regimes.append(new_regime)
                self.current_regime = new_regime
                
                self.logger.info(f"Correlation regime change detected: {regime_type}")
            
        except Exception as e:
            self.logger.error(f"Error detecting regime changes: {e}")
    
    async def _update_portfolio_correlation_risk(self):
        """Update portfolio-level correlation risk metrics."""
        try:
            if not self.current_correlations:
                return
            
            # Calculate portfolio correlation risk
            correlation_matrix = self.current_correlations.correlation_matrix
            n_assets = correlation_matrix.shape[0]
            
            if n_assets < 2:
                return
            
            # Average correlation (excluding diagonal)
            off_diagonal = correlation_matrix[np.triu_indices(n_assets, k=1)]
            avg_correlation = np.mean(off_diagonal)
            
            # Risk contribution from correlations
            # Higher average correlation means less diversification benefit
            self.portfolio_correlation_risk = avg_correlation
            
            # Calculate concentration metrics
            eigenvalues = self.current_correlations.eigenvalues
            
            if eigenvalues:
                self.concentration_metrics = {
                    'first_pc_contribution': eigenvalues[0] / sum(eigenvalues),
                    'top_3_pc_contribution': sum(eigenvalues[:3]) / sum(eigenvalues) if len(eigenvalues) >= 3 else 1.0,
                    'effective_number_of_assets': 1 / sum((ev / sum(eigenvalues)) ** 2 for ev in eigenvalues),
                    'diversification_ratio': n_assets / sum(eigenvalues) * eigenvalues[0] if eigenvalues[0] > 0 else 0
                }
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio correlation risk: {e}")
    
    def get_correlation_matrix(self, symbols: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Get current correlation matrix, optionally filtered by symbols."""
        if not self.current_correlations:
            return None
        
        if symbols is None:
            return self.current_correlations.correlation_matrix
        
        # Filter by requested symbols
        current_symbols = self.current_correlations.asset_names
        indices = []
        
        for symbol in symbols:
            if symbol in current_symbols:
                indices.append(current_symbols.index(symbol))
        
        if not indices:
            return None
        
        # Extract submatrix
        full_matrix = self.current_correlations.correlation_matrix
        return full_matrix[np.ix_(indices, indices)]
    
    def get_asset_correlations(self, symbol: str) -> Dict[str, float]:
        """Get correlations of a specific asset with all other assets."""
        if not self.current_correlations or symbol not in self.current_correlations.asset_names:
            return {}
        
        symbol_index = self.current_correlations.asset_names.index(symbol)
        correlation_row = self.current_correlations.correlation_matrix[symbol_index]
        
        correlations = {}
        for i, other_symbol in enumerate(self.current_correlations.asset_names):
            if other_symbol != symbol:
                correlations[other_symbol] = correlation_row[i]
        
        return correlations
    
    def get_highly_correlated_pairs(self, threshold: float = None) -> List[Tuple[str, str, float]]:
        """Get pairs of assets with high correlation."""
        if not self.current_correlations:
            return []
        
        if threshold is None:
            threshold = self.high_correlation_threshold
        
        high_corr_pairs = []
        correlation_matrix = self.current_correlations.correlation_matrix
        symbols = self.current_correlations.asset_names
        
        n = len(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > threshold:
                    high_corr_pairs.append((symbols[i], symbols[j], correlation))
        
        # Sort by absolute correlation descending
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def calculate_diversification_metrics(self, portfolio_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio diversification metrics given weights."""
        try:
            if not self.current_correlations:
                return {}
            
            symbols = list(portfolio_weights.keys())
            correlation_matrix = self.get_correlation_matrix(symbols)
            
            if correlation_matrix is None:
                return {}
            
            # Create weight vector in the same order as correlation matrix
            current_symbols = [s for s in self.current_correlations.asset_names if s in symbols]
            weights = np.array([portfolio_weights.get(symbol, 0.0) for symbol in current_symbols])
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
            
            # Individual asset variances (assuming unit variance for simplicity)
            individual_variances = np.ones(len(weights))
            
            # Diversification ratio
            weighted_avg_vol = np.dot(weights, individual_variances)
            portfolio_vol = np.sqrt(portfolio_variance)
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
            
            # Effective number of assets
            weight_concentration = np.sum(weights ** 2)
            effective_assets = 1 / weight_concentration if weight_concentration > 0 else 0
            
            return {
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': portfolio_vol,
                'diversification_ratio': diversification_ratio,
                'effective_number_of_assets': effective_assets,
                'weight_concentration': weight_concentration,
                'average_correlation': np.mean(correlation_matrix[np.triu_indices(len(correlation_matrix), k=1)])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification metrics: {e}")
            return {}
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get comprehensive correlation analysis summary."""
        if not self.current_correlations:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'num_assets': len(self.current_correlations.asset_names),
            'avg_correlation': self.current_correlations.avg_correlation,
            'max_correlation': self.current_correlations.max_correlation,
            'min_correlation': self.current_correlations.min_correlation,
            'concentration_risk': self.current_correlations.concentration_risk,
            'portfolio_correlation_risk': self.portfolio_correlation_risk,
            'concentration_metrics': self.concentration_metrics,
            'recent_alerts': len([a for a in self.correlation_alerts if a.timestamp > datetime.now() - timedelta(hours=24)]),
            'current_regime': self.current_regime,
            'highly_correlated_pairs': len(self.get_highly_correlated_pairs()),
            'eigenvalue_analysis': {
                'largest_eigenvalue': self.current_correlations.eigenvalues[0] if self.current_correlations.eigenvalues else 0,
                'eigenvalue_ratio': (
                    self.current_correlations.eigenvalues[0] / self.current_correlations.eigenvalues[1]
                    if len(self.current_correlations.eigenvalues) >= 2 else 1
                )
            }
        }
    
    def export_correlation_data(self) -> Dict[str, Any]:
        """Export correlation data for external analysis."""
        if not self.current_correlations:
            return {}
        
        return {
            'correlation_matrix': self.current_correlations.correlation_matrix.tolist(),
            'asset_names': self.current_correlations.asset_names,
            'eigenvalues': self.current_correlations.eigenvalues,
            'timestamp': self.current_correlations.timestamp.isoformat(),
            'alerts': [
                {
                    'asset_pair': alert.asset_pair,
                    'correlation_value': alert.correlation_value,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.correlation_alerts[-50:]  # Recent alerts only
            ]
        }