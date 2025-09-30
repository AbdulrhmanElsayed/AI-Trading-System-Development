"""
Risk Manager

Core risk management system for the trading platform.
Handles position sizing, portfolio risk assessment, and real-time monitoring.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import json

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import (
    TradingSignal, Position, Portfolio, RiskMetrics, 
    PerformanceMetrics, SignalType, MarketData
)
from src.risk.position_sizing import PositionSizer
from src.risk.portfolio_manager import PortfolioManager
from src.risk.risk_metrics import RiskCalculator
from src.risk.drawdown_monitor import DrawdownMonitor
from src.risk.correlation_analyzer import CorrelationAnalyzer


class RiskManager:
    """Central risk management coordinator."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("RiskManager")
        
        # Risk components
        self.position_sizer = PositionSizer(config)
        self.portfolio_manager = PortfolioManager(config)
        self.risk_calculator = RiskCalculator(config)
        self.drawdown_monitor = DrawdownMonitor(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        
        # Risk limits and parameters
        self.max_daily_loss = config.get('trading.max_daily_loss', 0.02)
        self.max_drawdown = config.get('trading.max_drawdown', 0.15)
        self.max_position_size = config.get('trading.max_position_size', 0.1)
        self.max_positions = config.get('trading.max_positions', 10)
        self.max_correlation = config.get('risk_management.limits.max_correlation', 0.7)
        self.max_sector_exposure = config.get('risk_management.limits.max_sector_exposure', 0.3)
        
        # Current state
        self.current_portfolio = Portfolio()
        self.current_risk_metrics = RiskMetrics()
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.is_emergency_stop = False
        
        # Performance tracking
        self.performance_history = []
        self.risk_alerts = []
        
        # Market data cache for risk calculations
        self.market_data_cache = {}
    
    async def initialize(self):
        """Initialize risk management system."""
        try:
            self.logger.info("Initializing Risk Management System...")
            
            # Initialize all components
            await self.position_sizer.initialize()
            await self.portfolio_manager.initialize()
            await self.risk_calculator.initialize()
            await self.drawdown_monitor.initialize()
            await self.correlation_analyzer.initialize()
            
            # Load initial portfolio state
            await self._load_portfolio_state()
            
            self.logger.info("Risk Management System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Manager: {e}")
            raise
    
    async def start(self):
        """Start risk management monitoring."""
        try:
            self.logger.info("Starting risk management monitoring...")
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._continuous_risk_monitoring()),
                asyncio.create_task(self._periodic_risk_assessment()),
                asyncio.create_task(self._drawdown_monitoring())
            ]
            
            # Store tasks to prevent garbage collection
            self._monitoring_tasks = tasks
            
            self.logger.info("Risk management monitoring started")
            
        except Exception as e:
            self.logger.error(f"Error starting risk management: {e}")
            raise
    
    async def stop(self):
        """Stop risk management system."""
        self.logger.info("Stopping Risk Management System...")
        
        # Cancel monitoring tasks
        if hasattr(self, '_monitoring_tasks'):
            for task in self._monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        # Save final state
        await self._save_portfolio_state()
        
        self.logger.info("Risk Management System stopped")
    
    async def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter trading signals through risk management checks."""
        try:
            if not signals:
                return []
            
            filtered_signals = []
            
            for signal in signals:
                # Check if signal passes risk filters
                risk_approval = await self._evaluate_signal_risk(signal)
                
                if risk_approval['approved']:
                    # Calculate position size
                    position_size = await self.position_sizer.calculate_position_size(
                        signal, self.current_portfolio, self.current_risk_metrics
                    )
                    
                    if position_size > 0:
                        # Update signal with position size
                        signal.quantity = position_size
                        signal.metadata['risk_approval'] = risk_approval
                        signal.metadata['position_size_method'] = self.position_sizer.get_method()
                        
                        filtered_signals.append(signal)
                        
                        self.logger.info(
                            f"Signal approved: {signal.symbol} {signal.signal_type.value} "
                            f"Size: {position_size:.4f} Confidence: {signal.confidence:.2f}"
                        )
                    else:
                        self.logger.warning(f"Signal rejected - zero position size: {signal.symbol}")
                else:
                    self.logger.warning(
                        f"Signal rejected: {signal.symbol} - {risk_approval['reason']}"
                    )
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return []
    
    async def _evaluate_signal_risk(self, signal: TradingSignal) -> Dict[str, Any]:
        """Evaluate if a signal meets risk criteria."""
        try:
            # Check emergency stop
            if self.is_emergency_stop:
                return {'approved': False, 'reason': 'Emergency stop active'}
            
            # Check daily loss limit
            if self.daily_pnl <= -abs(self.max_daily_loss):
                return {'approved': False, 'reason': 'Daily loss limit exceeded'}
            
            # Check maximum drawdown
            if self.current_drawdown >= self.max_drawdown:
                return {'approved': False, 'reason': 'Maximum drawdown exceeded'}
            
            # Check maximum positions
            if len(self.current_portfolio.positions) >= self.max_positions and signal.signal_type != SignalType.SELL:
                return {'approved': False, 'reason': 'Maximum positions limit reached'}
            
            # Check correlation limits
            correlation_check = await self._check_correlation_limits(signal)
            if not correlation_check['approved']:
                return correlation_check
            
            # Check sector exposure limits
            sector_check = await self._check_sector_limits(signal)
            if not sector_check['approved']:
                return sector_check
            
            # Check position concentration
            concentration_check = await self._check_concentration_limits(signal)
            if not concentration_check['approved']:
                return concentration_check
            
            # Check volatility limits
            volatility_check = await self._check_volatility_limits(signal)
            if not volatility_check['approved']:
                return volatility_check
            
            return {
                'approved': True, 
                'reason': 'All risk checks passed',
                'risk_score': await self._calculate_signal_risk_score(signal)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating signal risk: {e}")
            return {'approved': False, 'reason': f'Risk evaluation error: {e}'}
    
    async def _check_correlation_limits(self, signal: TradingSignal) -> Dict[str, Any]:
        """Check correlation limits with existing positions."""
        try:
            if not self.current_portfolio.positions or signal.signal_type == SignalType.SELL:
                return {'approved': True, 'reason': 'No correlation check needed'}
            
            # Get correlations with existing positions
            correlations = await self.correlation_analyzer.get_correlations(
                signal.symbol, 
                list(self.current_portfolio.positions.keys())
            )
            
            # Check if any correlation exceeds limit
            for other_symbol, correlation in correlations.items():
                if abs(correlation) > self.max_correlation:
                    return {
                        'approved': False, 
                        'reason': f'High correlation ({correlation:.2f}) with {other_symbol}'
                    }
            
            return {'approved': True, 'reason': 'Correlation limits satisfied'}
            
        except Exception as e:
            self.logger.error(f"Error checking correlation limits: {e}")
            return {'approved': True, 'reason': 'Correlation check failed, allowing signal'}
    
    async def _check_sector_limits(self, signal: TradingSignal) -> Dict[str, Any]:
        """Check sector exposure limits."""
        try:
            # Get sector for the signal symbol
            signal_sector = await self._get_symbol_sector(signal.symbol)
            
            if not signal_sector:
                return {'approved': True, 'reason': 'No sector information available'}
            
            # Calculate current sector exposure
            sector_exposure = await self._calculate_sector_exposure(signal_sector)
            
            # Check if adding this position would exceed sector limit
            if sector_exposure >= self.max_sector_exposure and signal.signal_type != SignalType.SELL:
                return {
                    'approved': False,
                    'reason': f'Sector {signal_sector} exposure limit exceeded ({sector_exposure:.2%})'
                }
            
            return {'approved': True, 'reason': 'Sector limits satisfied'}
            
        except Exception as e:
            self.logger.error(f"Error checking sector limits: {e}")
            return {'approved': True, 'reason': 'Sector check failed, allowing signal'}
    
    async def _check_concentration_limits(self, signal: TradingSignal) -> Dict[str, Any]:
        """Check position concentration limits."""
        try:
            # Check if single position would be too large
            estimated_position_value = signal.price * (signal.quantity or 1)
            portfolio_value = self.current_portfolio.total_value
            
            if portfolio_value > 0:
                concentration = estimated_position_value / portfolio_value
                
                if concentration > self.max_position_size and signal.signal_type != SignalType.SELL:
                    return {
                        'approved': False,
                        'reason': f'Position concentration too high ({concentration:.2%})'
                    }
            
            return {'approved': True, 'reason': 'Concentration limits satisfied'}
            
        except Exception as e:
            self.logger.error(f"Error checking concentration limits: {e}")
            return {'approved': True, 'reason': 'Concentration check failed, allowing signal'}
    
    async def _check_volatility_limits(self, signal: TradingSignal) -> Dict[str, Any]:
        """Check volatility-based risk limits."""
        try:
            # Get recent volatility for the symbol
            volatility = await self._get_symbol_volatility(signal.symbol)
            
            if volatility is None:
                return {'approved': True, 'reason': 'No volatility data available'}
            
            # Check if volatility is within acceptable range
            max_volatility = self.config.get('risk_management.limits.max_volatility', 0.5)
            
            if volatility > max_volatility:
                return {
                    'approved': False,
                    'reason': f'Volatility too high ({volatility:.2%})'
                }
            
            return {'approved': True, 'reason': 'Volatility limits satisfied'}
            
        except Exception as e:
            self.logger.error(f"Error checking volatility limits: {e}")
            return {'approved': True, 'reason': 'Volatility check failed, allowing signal'}
    
    async def _calculate_signal_risk_score(self, signal: TradingSignal) -> float:
        """Calculate risk score for a signal (0-1, higher = riskier)."""
        try:
            risk_factors = []
            
            # Confidence factor (lower confidence = higher risk)
            confidence_risk = 1.0 - signal.confidence
            risk_factors.append(confidence_risk * 0.3)
            
            # Volatility factor
            volatility = await self._get_symbol_volatility(signal.symbol) or 0.2
            volatility_risk = min(volatility / 0.5, 1.0)  # Normalize to 0-1
            risk_factors.append(volatility_risk * 0.3)
            
            # Position size factor
            if signal.quantity and signal.price:
                position_value = signal.quantity * signal.price
                portfolio_value = self.current_portfolio.total_value
                if portfolio_value > 0:
                    size_risk = position_value / portfolio_value
                    risk_factors.append(min(size_risk / self.max_position_size, 1.0) * 0.2)
            
            # Market conditions factor
            market_risk = await self._assess_market_conditions_risk()
            risk_factors.append(market_risk * 0.2)
            
            return sum(risk_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Medium risk as default
    
    async def update_portfolio(self, executed_trades: List[Dict[str, Any]]):
        """Update portfolio state after trade execution."""
        try:
            for trade in executed_trades:
                await self.portfolio_manager.update_position(trade)
            
            # Update current portfolio
            self.current_portfolio = await self.portfolio_manager.get_current_portfolio()
            
            # Recalculate risk metrics
            await self._update_risk_metrics()
            
            # Check for risk alerts
            await self._check_risk_alerts()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    async def _continuous_risk_monitoring(self):
        """Continuous risk monitoring loop."""
        while True:
            try:
                # Update current risk metrics
                await self._update_risk_metrics()
                
                # Check emergency conditions
                await self._check_emergency_conditions()
                
                # Update drawdown
                await self.drawdown_monitor.update_drawdown(self.current_portfolio)
                self.current_drawdown = self.drawdown_monitor.get_current_drawdown()
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in continuous risk monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _periodic_risk_assessment(self):
        """Periodic comprehensive risk assessment."""
        while True:
            try:
                # Full risk assessment every 5 minutes
                await asyncio.sleep(300)
                
                # Update correlations
                await self.correlation_analyzer.update_correlations(
                    list(self.current_portfolio.positions.keys())
                )
                
                # Calculate comprehensive risk metrics
                self.current_risk_metrics = await self.risk_calculator.calculate_portfolio_risk(
                    self.current_portfolio
                )
                
                # Log risk summary
                await self._log_risk_summary()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic risk assessment: {e}")
                await asyncio.sleep(300)
    
    async def _drawdown_monitoring(self):
        """Dedicated drawdown monitoring."""
        while True:
            try:
                # Check drawdown every minute
                await asyncio.sleep(60)
                
                current_dd = self.drawdown_monitor.get_current_drawdown()
                
                if current_dd >= self.max_drawdown * 0.8:  # 80% of max drawdown
                    self.logger.warning(f"Approaching maximum drawdown: {current_dd:.2%}")
                    await self._trigger_risk_alert("drawdown_warning", {
                        'current_drawdown': current_dd,
                        'max_drawdown': self.max_drawdown
                    })
                
                if current_dd >= self.max_drawdown:
                    self.logger.error(f"Maximum drawdown exceeded: {current_dd:.2%}")
                    await self._trigger_emergency_stop("Maximum drawdown exceeded")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in drawdown monitoring: {e}")
                await asyncio.sleep(60)
    
    # Helper methods (continued in next part due to length)
    async def _get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol (placeholder implementation)."""
        # This would integrate with a financial data provider
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF',
            'BTC/USD': 'Cryptocurrency', 'ETH/USD': 'Cryptocurrency'
        }
        return sector_map.get(symbol, 'Unknown')
    
    async def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a specific sector."""
        sector_value = 0.0
        total_value = self.current_portfolio.total_value
        
        for symbol, position in self.current_portfolio.positions.items():
            symbol_sector = await self._get_symbol_sector(symbol)
            if symbol_sector == sector:
                sector_value += position.market_value
        
        return sector_value / total_value if total_value > 0 else 0.0
    
    async def _get_symbol_volatility(self, symbol: str) -> Optional[float]:
        """Get recent volatility for a symbol."""
        # This would calculate from recent price data
        # For now, return a placeholder
        return 0.2  # 20% volatility
    
    async def _assess_market_conditions_risk(self) -> float:
        """Assess overall market conditions risk (0-1)."""
        # This would analyze market indicators like VIX, market breadth, etc.
        # For now, return medium risk
        return 0.5
    
    async def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        return {
            'emergency_stop': self.is_emergency_stop,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'portfolio_value': self.current_portfolio.total_value,
            'position_count': len(self.current_portfolio.positions),
            'risk_metrics': asdict(self.current_risk_metrics),
            'risk_alerts_count': len(self.risk_alerts),
            'last_update': datetime.now().isoformat()
        }
    
    # Additional helper methods would continue here...
    # (Implementation continues with remaining methods)