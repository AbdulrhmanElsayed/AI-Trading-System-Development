"""
Performance Monitor

Real-time performance tracking and analytics.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import Portfolio, Position, PerformanceMetrics


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    portfolio_value: float
    cash_balance: float
    total_pnl: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    active_positions: int
    
    
@dataclass
class AlertRule:
    """Performance alert rule configuration."""
    name: str
    metric: str  # 'drawdown', 'pnl', 'volatility', etc.
    threshold: float
    operator: str  # 'greater_than', 'less_than', 'equal_to'
    severity: str  # 'info', 'warning', 'critical'
    enabled: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None


class PerformanceMonitor:
    """Monitors trading performance in real-time."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("PerformanceMonitor")
        
        # Configuration
        self.monitoring_enabled = config.get('monitoring.performance.enabled', True)
        self.snapshot_interval = config.get('monitoring.performance.snapshot_interval', 60)  # seconds
        self.max_snapshots = config.get('monitoring.performance.max_snapshots', 10000)
        
        # Performance data storage
        self.performance_snapshots = deque(maxlen=self.max_snapshots)
        self.daily_performance = defaultdict(dict)
        self.monthly_performance = defaultdict(dict)
        
        # Real-time metrics
        self.current_metrics = {}
        self.benchmark_metrics = {}
        
        # Alert system
        self.alert_rules = self._setup_default_alerts()
        self.triggered_alerts = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_snapshot = None
        
        # Benchmark tracking
        self.benchmark_symbol = config.get('monitoring.benchmark_symbol', 'SPY')
        self.benchmark_data = deque(maxlen=1000)
        
    def _setup_default_alerts(self) -> List[AlertRule]:
        """Setup default performance alert rules."""
        return [
            AlertRule(
                name="High Drawdown Alert",
                metric="max_drawdown",
                threshold=0.10,  # 10%
                operator="greater_than",
                severity="critical"
            ),
            AlertRule(
                name="Daily Loss Alert", 
                metric="daily_pnl",
                threshold=-1000.0,  # $1000 loss
                operator="less_than",
                severity="warning"
            ),
            AlertRule(
                name="High Volatility Alert",
                metric="volatility",
                threshold=0.30,  # 30% annualized
                operator="greater_than", 
                severity="warning"
            ),
            AlertRule(
                name="Low Win Rate Alert",
                metric="win_rate",
                threshold=0.30,  # 30%
                operator="less_than",
                severity="info"
            ),
            AlertRule(
                name="Portfolio Value Alert",
                metric="portfolio_value",
                threshold=50000.0,  # $50K
                operator="less_than",
                severity="warning"
            )
        ]
    
    async def initialize(self):
        """Initialize performance monitor."""
        self.logger.info("Performance Monitor initialized")
        
        if self.monitoring_enabled:
            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._alert_task = asyncio.create_task(self._alert_monitoring_loop())
    
    async def _monitoring_loop(self):
        """Main performance monitoring loop."""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.snapshot_interval)
                await self._take_performance_snapshot()
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(self.snapshot_interval)
    
    async def _alert_monitoring_loop(self):
        """Alert monitoring loop."""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(30)  # Check alerts every 30 seconds
                await self._check_alert_rules()
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def update_portfolio_data(self, portfolio: Portfolio, performance_metrics: PerformanceMetrics):
        """Update portfolio and performance data."""
        try:
            # Update current metrics
            self.current_metrics = {
                'portfolio_value': portfolio.total_value,
                'cash_balance': portfolio.cash,
                'total_pnl': portfolio.total_pnl,
                'total_return': performance_metrics.total_return,
                'annualized_return': performance_metrics.annualized_return,
                'volatility': performance_metrics.volatility,
                'sharpe_ratio': performance_metrics.sharpe_ratio,
                'max_drawdown': performance_metrics.max_drawdown,
                'win_rate': performance_metrics.win_rate,
                'total_trades': performance_metrics.total_trades,
                'active_positions': len(portfolio.positions),
                'timestamp': datetime.now()
            }
            
            # Update daily performance tracking
            today = datetime.now().date()
            self.daily_performance[today] = self.current_metrics.copy()
            
            # Update monthly performance tracking  
            current_month = datetime.now().replace(day=1).date()
            self.monthly_performance[current_month] = self.current_metrics.copy()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio data: {e}")
    
    async def _take_performance_snapshot(self):
        """Take a performance snapshot."""
        try:
            if not self.current_metrics:
                return
            
            # Calculate daily return
            daily_return = 0.0
            if self.performance_snapshots:
                last_snapshot = self.performance_snapshots[-1]
                if last_snapshot.portfolio_value > 0:
                    daily_return = (
                        (self.current_metrics['portfolio_value'] - last_snapshot.portfolio_value) /
                        last_snapshot.portfolio_value
                    )
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                portfolio_value=self.current_metrics.get('portfolio_value', 0),
                cash_balance=self.current_metrics.get('cash_balance', 0),
                total_pnl=self.current_metrics.get('total_pnl', 0),
                daily_return=daily_return,
                volatility=self.current_metrics.get('volatility', 0),
                sharpe_ratio=self.current_metrics.get('sharpe_ratio', 0),
                max_drawdown=self.current_metrics.get('max_drawdown', 0),
                win_rate=self.current_metrics.get('win_rate', 0),
                total_trades=self.current_metrics.get('total_trades', 0),
                active_positions=self.current_metrics.get('active_positions', 0)
            )
            
            self.performance_snapshots.append(snapshot)
            self.last_snapshot = snapshot
            
            self.logger.debug(f"Performance snapshot: ${snapshot.portfolio_value:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error taking performance snapshot: {e}")
    
    async def _check_alert_rules(self):
        """Check all alert rules and trigger alerts."""
        if not self.current_metrics:
            return
        
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if rule.last_triggered:
                time_since_last = current_time - rule.last_triggered
                if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            # Get metric value
            metric_value = self.current_metrics.get(rule.metric)
            if metric_value is None:
                continue
            
            # Check threshold
            triggered = False
            if rule.operator == "greater_than" and metric_value > rule.threshold:
                triggered = True
            elif rule.operator == "less_than" and metric_value < rule.threshold:
                triggered = True
            elif rule.operator == "equal_to" and abs(metric_value - rule.threshold) < 0.001:
                triggered = True
            
            if triggered:
                await self._trigger_alert(rule, metric_value, current_time)
    
    async def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Trigger a performance alert."""
        alert = {
            'rule_name': rule.name,
            'metric': rule.metric,
            'value': value,
            'threshold': rule.threshold,
            'severity': rule.severity,
            'timestamp': timestamp,
            'message': f"{rule.name}: {rule.metric} = {value:.4f} (threshold: {rule.threshold:.4f})"
        }
        
        self.triggered_alerts.append(alert)
        rule.last_triggered = timestamp
        
        # Keep only recent alerts
        if len(self.triggered_alerts) > 1000:
            self.triggered_alerts = self.triggered_alerts[-500:]
        
        # Log alert
        log_method = getattr(self.logger, rule.severity, self.logger.info)
        log_method(alert['message'])
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.current_metrics:
            return {}
        
        return self.current_metrics.copy()
    
    def get_performance_history(self, hours: int = 24) -> List[PerformanceSnapshot]:
        """Get performance history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            snapshot for snapshot in self.performance_snapshots
            if snapshot.timestamp > cutoff_time
        ]
    
    def get_daily_performance(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """Get daily performance for specified days."""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        
        return {
            date.isoformat(): metrics
            for date, metrics in self.daily_performance.items()
            if date > cutoff_date
        }
    
    def get_monthly_performance(self, months: int = 12) -> Dict[str, Dict[str, Any]]:
        """Get monthly performance summary."""
        cutoff_date = datetime.now().replace(day=1).date() - timedelta(days=months*30)
        
        return {
            date.isoformat(): metrics
            for date, metrics in self.monthly_performance.items()
            if date > cutoff_date
        }
    
    def calculate_performance_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics."""
        if not self.performance_snapshots:
            return {}
        
        try:
            # Extract data arrays
            values = [s.portfolio_value for s in self.performance_snapshots]
            returns = [s.daily_return for s in self.performance_snapshots if s.daily_return != 0]
            
            if not values or not returns:
                return {}
            
            # Basic statistics
            stats = {
                'total_return': (values[-1] - values[0]) / values[0] if values[0] > 0 else 0,
                'avg_daily_return': np.mean(returns),
                'volatility': np.std(returns) * np.sqrt(252),  # Annualized
                'best_day': max(returns) if returns else 0,
                'worst_day': min(returns) if returns else 0,
                'positive_days': len([r for r in returns if r > 0]),
                'negative_days': len([r for r in returns if r < 0]),
                'max_portfolio_value': max(values),
                'min_portfolio_value': min(values),
                'current_portfolio_value': values[-1],
                'total_snapshots': len(self.performance_snapshots),
                'monitoring_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            }
            
            # Win rate
            if len(returns) > 0:
                stats['win_rate'] = stats['positive_days'] / len(returns)
            else:
                stats['win_rate'] = 0
            
            # Sharpe ratio
            if stats['volatility'] > 0:
                risk_free_rate = self.config.get('risk_management.risk_free_rate', 0.02)
                excess_return = stats['avg_daily_return'] * 252 - risk_free_rate
                stats['sharpe_ratio'] = excess_return / stats['volatility']
            else:
                stats['sharpe_ratio'] = 0
            
            # Maximum drawdown
            peak = values[0]
            max_dd = 0
            for value in values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            stats['max_drawdown'] = max_dd
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating performance stats: {e}")
            return {}
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.triggered_alerts
            if alert['timestamp'] > cutoff_time
        ]
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule by name."""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        self.logger.info(f"Removed alert rule: {rule_name}")
    
    def enable_alert_rule(self, rule_name: str):
        """Enable alert rule."""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.info(f"Enabled alert rule: {rule_name}")
                break
    
    def disable_alert_rule(self, rule_name: str):
        """Disable alert rule."""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.info(f"Disabled alert rule: {rule_name}")
                break
    
    def get_alert_rules_config(self) -> List[Dict[str, Any]]:
        """Get alert rules configuration."""
        return [
            {
                'name': rule.name,
                'metric': rule.metric,
                'threshold': rule.threshold,
                'operator': rule.operator,
                'severity': rule.severity,
                'enabled': rule.enabled,
                'cooldown_minutes': rule.cooldown_minutes,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            }
            for rule in self.alert_rules
        ]
    
    def export_performance_data(self) -> Dict[str, Any]:
        """Export performance data for analysis."""
        return {
            'current_metrics': self.current_metrics,
            'performance_stats': self.calculate_performance_stats(),
            'snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'portfolio_value': s.portfolio_value,
                    'daily_return': s.daily_return,
                    'volatility': s.volatility,
                    'sharpe_ratio': s.sharpe_ratio,
                    'max_drawdown': s.max_drawdown,
                    'win_rate': s.win_rate,
                    'total_trades': s.total_trades,
                    'active_positions': s.active_positions
                }
                for s in list(self.performance_snapshots)[-100:]  # Last 100 snapshots
            ],
            'recent_alerts': self.get_recent_alerts(24),
            'alert_rules': self.get_alert_rules_config(),
            'daily_performance': self.get_daily_performance(30),
            'monthly_performance': self.get_monthly_performance(12)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary."""
        stats = self.calculate_performance_stats()
        current = self.get_current_performance()
        alerts = self.get_recent_alerts(24)
        
        return {
            'current_value': current.get('portfolio_value', 0),
            'total_return': stats.get('total_return', 0),
            'daily_return': current.get('daily_return', 0),
            'sharpe_ratio': stats.get('sharpe_ratio', 0),
            'max_drawdown': stats.get('max_drawdown', 0),
            'win_rate': stats.get('win_rate', 0),
            'total_trades': current.get('total_trades', 0),
            'active_positions': current.get('active_positions', 0),
            'recent_alerts_count': len(alerts),
            'critical_alerts_count': len([a for a in alerts if a['severity'] == 'critical']),
            'monitoring_status': 'active' if self.monitoring_enabled else 'inactive',
            'last_update': current.get('timestamp', datetime.now()).isoformat() if current.get('timestamp') else None
        }
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_enabled = False
        if hasattr(self, '_monitoring_task'):
            self._monitoring_task.cancel()
        if hasattr(self, '_alert_task'):
            self._alert_task.cancel()
        self.logger.info("Performance monitoring stopped")