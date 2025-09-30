"""
Monitoring Coordinator

Coordinates all monitoring components and provides unified monitoring interface.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.alerting_system import AlertingSystem, AlertSeverity
from src.monitoring.system_health import SystemHealthMonitor
from src.monitoring.dashboard import DashboardManager


class MonitoringCoordinator:
    """Central coordinator for all monitoring components."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("MonitoringCoordinator")
        
        # Monitoring components
        self.performance_monitor = PerformanceMonitor(config)
        self.alerting_system = AlertingSystem(config)
        self.system_health_monitor = SystemHealthMonitor(config)
        self.dashboard_manager = DashboardManager(config)
        
        # Component status
        self.components_initialized = False
        self.monitoring_active = False
        
        # Integration with other systems
        self.portfolio_manager = None
        self.risk_manager = None
        self.execution_engine = None
        
        # Event handlers
        self.event_handlers = {}
        
    async def initialize(self):
        """Initialize all monitoring components."""
        try:
            self.logger.info("Initializing Monitoring Coordinator")
            
            # Initialize individual components
            await self.performance_monitor.initialize()
            await self.alerting_system.initialize()
            await self.system_health_monitor.initialize()
            await self.dashboard_manager.initialize()
            
            # Setup component integrations
            self._setup_component_integrations()
            
            # Register system health checkers
            self._register_health_checkers()
            
            # Setup alert context providers
            self._setup_alert_context_providers()
            
            self.components_initialized = True
            self.monitoring_active = True
            
            self.logger.info("Monitoring Coordinator initialized successfully")
            
            # Start monitoring coordination tasks
            asyncio.create_task(self._coordination_loop())
            
        except Exception as e:
            self.logger.error(f"Error initializing Monitoring Coordinator: {e}")
            raise
    
    def _setup_component_integrations(self):
        """Setup integrations between monitoring components."""
        
        # System health alerts -> Alerting system
        self.system_health_monitor.add_alert_callback(self._handle_system_health_alert)
        
        # Register data sources with dashboard
        self.dashboard_manager.register_data_source('performance_monitor', self.performance_monitor)
        self.dashboard_manager.register_data_source('system_health', self.system_health_monitor)
        self.dashboard_manager.register_data_source('alerting_system', self.alerting_system)
    
    def _register_health_checkers(self):
        """Register health checkers for system components."""
        
        # Performance monitor health check
        self.system_health_monitor.register_health_checker(
            'performance_monitor',
            lambda: {'status': 'healthy' if self.performance_monitor.monitoring_enabled else 'offline'}
        )
        
        # Alerting system health check
        self.system_health_monitor.register_health_checker(
            'alerting_system',
            lambda: {'status': 'healthy' if self.alerting_system.alerting_enabled else 'offline'}
        )
        
        # Dashboard health check
        self.system_health_monitor.register_health_checker(
            'dashboard',
            lambda: {
                'status': 'healthy' if self.dashboard_manager.dashboard_server.websocket_server else 'offline',
                'metadata': {'connected_clients': len(self.dashboard_manager.dashboard_server.websocket_clients)}
            }
        )
    
    def _setup_alert_context_providers(self):
        """Setup context providers for the alerting system."""
        
        # Performance context provider
        self.alerting_system.register_context_provider(
            'performance',
            lambda: self.performance_monitor.get_current_performance()
        )
        
        # System health context provider
        self.alerting_system.register_context_provider(
            'system_health',
            lambda: self.system_health_monitor.get_current_metrics()
        )
    
    async def _coordination_loop(self):
        """Main coordination loop for monitoring."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Coordinate every 30 seconds
                
                # Collect data from all sources
                await self._coordinate_data_collection()
                
                # Evaluate alerts
                await self._coordinate_alert_evaluation()
                
                # Update dashboard
                await self._coordinate_dashboard_updates()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring coordination loop: {e}")
                await asyncio.sleep(30)
    
    async def _coordinate_data_collection(self):
        """Coordinate data collection across all monitoring components."""
        try:
            # Get current system state
            current_context = {}
            
            # Add performance data
            performance_data = self.performance_monitor.get_current_performance()
            if performance_data:
                current_context.update(performance_data)
            
            # Add system health data
            system_metrics = self.system_health_monitor.get_current_metrics()
            if system_metrics:
                current_context.update({
                    'cpu_usage': system_metrics.cpu_usage,
                    'memory_usage': system_metrics.memory_usage,
                    'disk_usage': system_metrics.disk_usage,
                    'system_load': system_metrics.system_load
                })
            
            # Add external system data if available
            if self.portfolio_manager:
                try:
                    portfolio = await self.portfolio_manager.get_current_portfolio()
                    current_context.update({
                        'portfolio_value': portfolio.total_value,
                        'cash_balance': portfolio.cash,
                        'total_pnl': portfolio.total_pnl,
                        'active_positions': len(portfolio.positions)
                    })
                except Exception as e:
                    self.logger.error(f"Error getting portfolio data: {e}")
            
            # Store context for other components to use
            self.current_context = current_context
            
        except Exception as e:
            self.logger.error(f"Error coordinating data collection: {e}")
    
    async def _coordinate_alert_evaluation(self):
        """Coordinate alert evaluation across all systems."""
        try:
            if hasattr(self, 'current_context'):
                await self.alerting_system.evaluate_alerts(self.current_context)
                
        except Exception as e:
            self.logger.error(f"Error coordinating alert evaluation: {e}")
    
    async def _coordinate_dashboard_updates(self):
        """Coordinate dashboard data updates."""
        try:
            # Update dashboard with latest data
            if hasattr(self, 'current_context'):
                
                # Update portfolio data
                portfolio_data = {
                    k: v for k, v in self.current_context.items()
                    if k in ['portfolio_value', 'cash_balance', 'total_pnl', 'active_positions']
                }
                if portfolio_data:
                    await self.dashboard_manager.dashboard_server.update_portfolio_data(portfolio_data)
                
                # Update performance data
                performance_data = self.performance_monitor.get_current_performance()
                if performance_data:
                    await self.dashboard_manager.dashboard_server.update_performance_data(performance_data)
                
                # Update system health data
                health_summary = self.system_health_monitor.get_system_summary()
                await self.dashboard_manager.dashboard_server.update_system_health_data(health_summary)
                
                # Update alerts data
                recent_alerts = self.alerting_system.get_alert_history(24)
                alerts_data = [
                    {
                        'id': alert.id,
                        'title': alert.title,
                        'message': alert.message,
                        'severity': alert.severity.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts
                ]
                await self.dashboard_manager.dashboard_server.update_alerts_data(alerts_data)
            
        except Exception as e:
            self.logger.error(f"Error coordinating dashboard updates: {e}")
    
    async def _handle_system_health_alert(self, alert: Dict[str, Any]):
        """Handle system health alerts."""
        try:
            # Create alert context
            alert_context = {
                'alert_type': 'system_health',
                'metric': alert['type'],
                'value': alert['value'],
                'threshold': alert['threshold'],
                **getattr(self, 'current_context', {})
            }
            
            # Map severity
            severity_mapping = {
                'critical': AlertSeverity.CRITICAL,
                'warning': AlertSeverity.WARNING,
                'info': AlertSeverity.INFO
            }
            severity = severity_mapping.get(alert['severity'], AlertSeverity.WARNING)
            
            # Trigger alert through alerting system
            await self.alerting_system.evaluate_alerts(alert_context)
            
        except Exception as e:
            self.logger.error(f"Error handling system health alert: {e}")
    
    def register_portfolio_manager(self, portfolio_manager):
        """Register portfolio manager for monitoring integration."""
        self.portfolio_manager = portfolio_manager
        self.logger.info("Portfolio manager registered with monitoring")
        
        # Add portfolio-specific health checker
        self.system_health_monitor.register_health_checker(
            'portfolio_manager',
            self._check_portfolio_manager_health
        )
    
    def register_risk_manager(self, risk_manager):
        """Register risk manager for monitoring integration."""
        self.risk_manager = risk_manager
        self.logger.info("Risk manager registered with monitoring")
        
        # Add risk-specific health checker
        self.system_health_monitor.register_health_checker(
            'risk_manager',
            self._check_risk_manager_health
        )
    
    def register_execution_engine(self, execution_engine):
        """Register execution engine for monitoring integration."""
        self.execution_engine = execution_engine
        self.logger.info("Execution engine registered with monitoring")
        
        # Add execution-specific health checker  
        self.system_health_monitor.register_health_checker(
            'execution_engine',
            self._check_execution_engine_health
        )
    
    async def _check_portfolio_manager_health(self) -> Dict[str, Any]:
        """Check portfolio manager health."""
        try:
            if not self.portfolio_manager:
                return {'status': 'offline', 'error': 'Portfolio manager not registered'}
            
            # Try to get portfolio data
            portfolio = await self.portfolio_manager.get_current_portfolio()
            
            return {
                'status': 'healthy',
                'metadata': {
                    'portfolio_value': portfolio.total_value,
                    'positions_count': len(portfolio.positions),
                    'last_update': portfolio.timestamp.isoformat() if portfolio.timestamp else None
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _check_risk_manager_health(self) -> Dict[str, Any]:
        """Check risk manager health."""
        try:
            if not self.risk_manager:
                return {'status': 'offline', 'error': 'Risk manager not registered'}
            
            # Check if risk manager is operational
            return {
                'status': 'healthy',
                'metadata': {
                    'monitoring_enabled': getattr(self.risk_manager, 'monitoring_enabled', False)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _check_execution_engine_health(self) -> Dict[str, Any]:
        """Check execution engine health."""
        try:
            if not self.execution_engine:
                return {'status': 'offline', 'error': 'Execution engine not registered'}
            
            # Get execution status
            status = self.execution_engine.get_execution_status()
            
            return {
                'status': 'healthy' if status.get('execution_enabled') else 'warning',
                'metadata': {
                    'execution_enabled': status.get('execution_enabled'),
                    'active_orders': status.get('active_orders', 0),
                    'paper_trading': status.get('paper_trading', True)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def update_portfolio_performance(self, portfolio, performance_metrics):
        """Update portfolio and performance data."""
        try:
            await self.performance_monitor.update_portfolio_data(portfolio, performance_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio performance: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            'coordinator': {
                'initialized': self.components_initialized,
                'active': self.monitoring_active,
                'registered_systems': {
                    'portfolio_manager': self.portfolio_manager is not None,
                    'risk_manager': self.risk_manager is not None,
                    'execution_engine': self.execution_engine is not None
                }
            },
            'performance_monitor': self.performance_monitor.get_performance_summary(),
            'alerting_system': self.alerting_system.get_alert_summary(),
            'system_health': self.system_health_monitor.get_system_summary(),
            'dashboard': self.dashboard_manager.get_dashboard_status()
        }
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return self.dashboard_manager.get_dashboard_url()
    
    async def send_test_alert(self, severity: AlertSeverity = AlertSeverity.INFO):
        """Send a test alert to verify alerting system."""
        test_context = {
            'test_alert': True,
            'timestamp': datetime.now().isoformat(),
            **getattr(self, 'current_context', {})
        }
        
        await self.alerting_system.evaluate_alerts(test_context)
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for monitoring events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Added event handler for: {event_type}")
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit monitoring event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def shutdown(self):
        """Shutdown all monitoring components."""
        self.logger.info("Shutting down Monitoring Coordinator")
        
        self.monitoring_active = False
        
        # Shutdown individual components
        self.performance_monitor.stop_monitoring()
        self.system_health_monitor.stop_monitoring()
        await self.dashboard_manager.shutdown()
        
        self.logger.info("Monitoring Coordinator shutdown complete")