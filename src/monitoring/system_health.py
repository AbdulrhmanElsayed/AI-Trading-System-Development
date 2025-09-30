"""
System Health Monitor

Monitors system health, resource usage, and component status.
"""

import asyncio
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_sent: int
    network_received: int
    open_files: int
    active_threads: int
    system_load: float


@dataclass
class ComponentHealth:
    """Individual component health status."""
    name: str
    status: ComponentStatus
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class SystemHealthMonitor:
    """Monitors overall system health and performance."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("SystemHealthMonitor")
        
        # Configuration
        self.monitoring_enabled = config.get('monitoring.health.enabled', True)
        self.check_interval = config.get('monitoring.health.check_interval', 30)  # seconds
        self.max_metrics_history = config.get('monitoring.health.max_history', 2880)  # 24h at 30s intervals
        
        # Thresholds
        self.cpu_warning_threshold = config.get('monitoring.health.cpu_warning', 80.0)  # %
        self.cpu_critical_threshold = config.get('monitoring.health.cpu_critical', 95.0)  # %
        self.memory_warning_threshold = config.get('monitoring.health.memory_warning', 80.0)  # %
        self.memory_critical_threshold = config.get('monitoring.health.memory_critical', 95.0)  # %
        self.disk_warning_threshold = config.get('monitoring.health.disk_warning', 85.0)  # %
        self.disk_critical_threshold = config.get('monitoring.health.disk_critical', 95.0)  # %
        
        # Data storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Component health checkers
        self.health_checkers: Dict[str, Callable] = {}
        
        # System information
        self.system_info = self._get_system_info()
        
        # Process monitoring
        self.process = psutil.Process()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        try:
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'hostname': platform.node(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()),
                'disk_partitions': [
                    {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype
                    }
                    for partition in psutil.disk_partitions()
                ]
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {}
    
    async def initialize(self):
        """Initialize system health monitor."""
        self.logger.info("System Health Monitor initialized")
        
        if self.monitoring_enabled:
            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._component_check_task = asyncio.create_task(self._component_check_loop())
    
    async def _monitoring_loop(self):
        """Main system metrics monitoring loop."""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.check_interval)
                await self._collect_system_metrics()
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _component_check_loop(self):
        """Component health check loop."""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(60)  # Check components every minute
                await self._check_component_health()
                
            except Exception as e:
                self.logger.error(f"Error in component check loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_received = network.bytes_recv
            
            # Process-specific metrics
            open_files = len(self.process.open_files())
            active_threads = self.process.num_threads()
            
            # System load
            try:
                system_load = psutil.getloadavg()[0]  # 1-minute load average
            except AttributeError:
                # Windows doesn't have getloadavg
                system_load = cpu_usage / 100.0
            
            # Create metrics object
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_sent=network_sent,
                network_received=network_received,
                open_files=open_files,
                active_threads=active_threads,
                system_load=system_load
            )
            
            # Store metrics
            self.system_metrics_history.append(metrics)
            
            # Trim history
            if len(self.system_metrics_history) > self.max_metrics_history:
                self.system_metrics_history = self.system_metrics_history[-self.max_metrics_history//2:]
            
            # Check thresholds and trigger alerts
            await self._check_system_thresholds(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds."""
        alerts = []
        
        # CPU usage alerts
        if metrics.cpu_usage >= self.cpu_critical_threshold:
            alerts.append({
                'type': 'cpu_critical',
                'message': f"Critical CPU usage: {metrics.cpu_usage:.1f}%",
                'severity': 'critical',
                'value': metrics.cpu_usage,
                'threshold': self.cpu_critical_threshold
            })
        elif metrics.cpu_usage >= self.cpu_warning_threshold:
            alerts.append({
                'type': 'cpu_warning',
                'message': f"High CPU usage: {metrics.cpu_usage:.1f}%",
                'severity': 'warning',
                'value': metrics.cpu_usage,
                'threshold': self.cpu_warning_threshold
            })
        
        # Memory usage alerts
        if metrics.memory_usage >= self.memory_critical_threshold:
            alerts.append({
                'type': 'memory_critical',
                'message': f"Critical memory usage: {metrics.memory_usage:.1f}%",
                'severity': 'critical',
                'value': metrics.memory_usage,
                'threshold': self.memory_critical_threshold
            })
        elif metrics.memory_usage >= self.memory_warning_threshold:
            alerts.append({
                'type': 'memory_warning',
                'message': f"High memory usage: {metrics.memory_usage:.1f}%",
                'severity': 'warning',
                'value': metrics.memory_usage,
                'threshold': self.memory_warning_threshold
            })
        
        # Disk usage alerts
        if metrics.disk_usage >= self.disk_critical_threshold:
            alerts.append({
                'type': 'disk_critical',
                'message': f"Critical disk usage: {metrics.disk_usage:.1f}%",
                'severity': 'critical',
                'value': metrics.disk_usage,
                'threshold': self.disk_critical_threshold
            })
        elif metrics.disk_usage >= self.disk_warning_threshold:
            alerts.append({
                'type': 'disk_warning',
                'message': f"High disk usage: {metrics.disk_usage:.1f}%",
                'severity': 'warning',
                'value': metrics.disk_usage,
                'threshold': self.disk_warning_threshold
            })
        
        # Trigger alerts
        for alert in alerts:
            await self._trigger_system_alert(alert)
    
    async def _trigger_system_alert(self, alert: Dict[str, Any]):
        """Trigger system health alert."""
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Log the alert
        log_method = getattr(self.logger, alert['severity'], self.logger.info)
        log_method(f"System Health Alert: {alert['message']}")
    
    async def _check_component_health(self):
        """Check health of all registered components."""
        for component_name, checker in self.health_checkers.items():
            try:
                start_time = datetime.now()
                
                # Run health check
                if asyncio.iscoroutinefunction(checker):
                    result = await checker()
                else:
                    result = checker()
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000  # ms
                
                # Parse result
                if isinstance(result, bool):
                    status = ComponentStatus.HEALTHY if result else ComponentStatus.ERROR
                    error_message = None
                    metadata = None
                elif isinstance(result, dict):
                    status = ComponentStatus(result.get('status', 'healthy'))
                    error_message = result.get('error')
                    metadata = result.get('metadata')
                else:
                    status = ComponentStatus.ERROR
                    error_message = f"Invalid health check result: {result}"
                    metadata = None
                
                # Update component health
                self.component_health[component_name] = ComponentHealth(
                    name=component_name,
                    status=status,
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    error_message=error_message,
                    metadata=metadata
                )
                
            except Exception as e:
                # Health check failed
                self.component_health[component_name] = ComponentHealth(
                    name=component_name,
                    status=ComponentStatus.ERROR,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
                
                self.logger.error(f"Health check failed for {component_name}: {e}")
    
    def register_health_checker(self, component_name: str, checker: Callable):
        """Register a health checker for a component."""
        self.health_checkers[component_name] = checker
        self.logger.info(f"Registered health checker for: {component_name}")
    
    def unregister_health_checker(self, component_name: str):
        """Unregister a health checker."""
        if component_name in self.health_checkers:
            del self.health_checkers[component_name]
            if component_name in self.component_health:
                del self.component_health[component_name]
            self.logger.info(f"Unregistered health checker for: {component_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for system health alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get system metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_component_health_status(self) -> Dict[str, ComponentHealth]:
        """Get health status of all components."""
        return self.component_health.copy()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        current_metrics = self.get_current_metrics()
        
        # Component health summary
        component_summary = {}
        healthy_components = 0
        warning_components = 0
        error_components = 0
        offline_components = 0
        
        for component in self.component_health.values():
            if component.status == ComponentStatus.HEALTHY:
                healthy_components += 1
            elif component.status == ComponentStatus.WARNING:
                warning_components += 1
            elif component.status == ComponentStatus.ERROR:
                error_components += 1
            elif component.status == ComponentStatus.OFFLINE:
                offline_components += 1
        
        component_summary = {
            'total': len(self.component_health),
            'healthy': healthy_components,
            'warning': warning_components,
            'error': error_components,
            'offline': offline_components
        }
        
        # Overall health status
        if error_components > 0 or offline_components > 0:
            overall_status = ComponentStatus.ERROR
        elif warning_components > 0:
            overall_status = ComponentStatus.WARNING
        else:
            overall_status = ComponentStatus.HEALTHY
        
        return {
            'overall_status': overall_status.value,
            'system_info': self.system_info,
            'current_metrics': {
                'cpu_usage': current_metrics.cpu_usage if current_metrics else 0,
                'memory_usage': current_metrics.memory_usage if current_metrics else 0,
                'disk_usage': current_metrics.disk_usage if current_metrics else 0,
                'system_load': current_metrics.system_load if current_metrics else 0,
                'open_files': current_metrics.open_files if current_metrics else 0,
                'active_threads': current_metrics.active_threads if current_metrics else 0,
                'timestamp': current_metrics.timestamp.isoformat() if current_metrics else None
            },
            'component_summary': component_summary,
            'monitoring_enabled': self.monitoring_enabled,
            'uptime_hours': (
                (datetime.now() - self.system_info.get('boot_time', datetime.now())).total_seconds() / 3600
                if self.system_info.get('boot_time') else 0
            ),
            'metrics_collected': len(self.system_metrics_history)
        }
    
    def get_resource_trends(self, hours: int = 6) -> Dict[str, List[float]]:
        """Get resource usage trends over time."""
        metrics = self.get_metrics_history(hours)
        
        if not metrics:
            return {}
        
        return {
            'timestamps': [m.timestamp.isoformat() for m in metrics],
            'cpu_usage': [m.cpu_usage for m in metrics],
            'memory_usage': [m.memory_usage for m in metrics],
            'disk_usage': [m.disk_usage for m in metrics],
            'system_load': [m.system_load for m in metrics],
            'network_sent': [m.network_sent for m in metrics],
            'network_received': [m.network_received for m in metrics]
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.system_metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(1)  # Last hour
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        load_values = [m.system_load for m in recent_metrics]
        
        return {
            'cpu_stats': {
                'current': cpu_values[-1],
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_stats': {
                'current': memory_values[-1],
                'average': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'load_stats': {
                'current': load_values[-1],
                'average': sum(load_values) / len(load_values),
                'max': max(load_values),
                'min': min(load_values)
            },
            'sample_count': len(recent_metrics),
            'sample_period_minutes': (
                (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60
                if len(recent_metrics) > 1 else 0
            )
        }
    
    def export_health_data(self) -> Dict[str, Any]:
        """Export system health data."""
        return {
            'system_summary': self.get_system_summary(),
            'component_health': {
                name: {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'response_time_ms': health.response_time_ms,
                    'error_message': health.error_message,
                    'metadata': health.metadata
                }
                for name, health in self.component_health.items()
            },
            'resource_trends': self.get_resource_trends(6),
            'performance_stats': self.get_performance_stats(),
            'thresholds': {
                'cpu_warning': self.cpu_warning_threshold,
                'cpu_critical': self.cpu_critical_threshold,
                'memory_warning': self.memory_warning_threshold,
                'memory_critical': self.memory_critical_threshold,
                'disk_warning': self.disk_warning_threshold,
                'disk_critical': self.disk_critical_threshold
            }
        }
    
    def stop_monitoring(self):
        """Stop system health monitoring."""
        self.monitoring_enabled = False
        
        if hasattr(self, '_monitoring_task'):
            self._monitoring_task.cancel()
        if hasattr(self, '_component_check_task'):
            self._component_check_task.cancel()
        
        self.logger.info("System health monitoring stopped")