"""
Drawdown Monitor

Monitors portfolio drawdowns and triggers alerts/actions when limits are exceeded.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import asyncio
from enum import Enum

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import Portfolio, Position


class DrawdownSeverity(Enum):
    """Drawdown severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DrawdownEvent:
    """Drawdown event data structure."""
    start_time: datetime
    end_time: Optional[datetime] = None
    peak_value: float = 0.0
    trough_value: float = 0.0
    max_drawdown: float = 0.0
    duration_days: int = 0
    recovery_time_days: Optional[int] = None
    severity: DrawdownSeverity = DrawdownSeverity.LOW
    is_active: bool = True
    triggered_actions: List[str] = field(default_factory=list)


@dataclass
class DrawdownAlert:
    """Drawdown alert configuration."""
    threshold: float
    severity: DrawdownSeverity
    action: str
    enabled: bool = True
    cooldown_hours: int = 1
    last_triggered: Optional[datetime] = None


class DrawdownMonitor:
    """Monitors portfolio drawdowns and manages risk responses."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("DrawdownMonitor")
        
        # Monitoring configuration
        self.monitoring_enabled = config.get('risk_management.drawdown_monitoring.enabled', True)
        self.check_interval = config.get('risk_management.drawdown_monitoring.check_interval', 60)  # seconds
        
        # Drawdown thresholds and actions
        self.alerts = self._setup_drawdown_alerts()
        
        # State tracking
        self.portfolio_values = []
        self.current_drawdown_event = None
        self.historical_events = []
        self.peak_value = 0.0
        self.last_check_time = datetime.now()
        
        # Emergency controls
        self.emergency_stop_triggered = False
        self.max_daily_drawdown = config.get('risk_management.max_daily_drawdown', 0.05)
        self.max_total_drawdown = config.get('risk_management.max_total_drawdown', 0.20)
        
        # Callbacks for actions
        self.action_callbacks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.monitoring_stats = {
            'checks_performed': 0,
            'alerts_triggered': 0,
            'events_detected': 0,
            'actions_executed': 0,
            'last_update': None
        }
        
    def _setup_drawdown_alerts(self) -> List[DrawdownAlert]:
        """Setup drawdown alert configurations."""
        default_alerts = [
            DrawdownAlert(threshold=0.02, severity=DrawdownSeverity.LOW, action="log_warning"),
            DrawdownAlert(threshold=0.05, severity=DrawdownSeverity.MEDIUM, action="reduce_positions"),
            DrawdownAlert(threshold=0.10, severity=DrawdownSeverity.HIGH, action="close_losing_positions"),
            DrawdownAlert(threshold=0.15, severity=DrawdownSeverity.CRITICAL, action="emergency_stop"),
        ]
        
        # Load custom alerts from config if available
        custom_alerts = self.config.get('risk_management.drawdown_alerts', [])
        
        if custom_alerts:
            alerts = []
            for alert_config in custom_alerts:
                alerts.append(DrawdownAlert(
                    threshold=alert_config['threshold'],
                    severity=DrawdownSeverity(alert_config['severity']),
                    action=alert_config['action'],
                    enabled=alert_config.get('enabled', True),
                    cooldown_hours=alert_config.get('cooldown_hours', 1)
                ))
            return alerts
        
        return default_alerts
    
    async def start_monitoring(self):
        """Start continuous drawdown monitoring."""
        if not self.monitoring_enabled:
            self.logger.info("Drawdown monitoring is disabled")
            return
        
        self.logger.info("Starting drawdown monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._check_drawdown()
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in drawdown monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_drawdown(self):
        """Check current drawdown levels and trigger alerts."""
        try:
            if not self.portfolio_values:
                return
            
            current_value = self.portfolio_values[-1]['value']
            current_time = self.portfolio_values[-1]['timestamp']
            
            # Update peak value
            if current_value > self.peak_value:
                self.peak_value = current_value
                
                # End current drawdown event if recovering
                if self.current_drawdown_event and self.current_drawdown_event.is_active:
                    await self._end_drawdown_event(current_time, recovered=True)
            
            # Calculate current drawdown
            if self.peak_value > 0:
                current_drawdown = (self.peak_value - current_value) / self.peak_value
            else:
                current_drawdown = 0.0
            
            # Start new drawdown event if threshold exceeded
            if current_drawdown > 0.01 and not self.current_drawdown_event:
                await self._start_drawdown_event(current_time, self.peak_value, current_value)
            
            # Update existing drawdown event
            elif self.current_drawdown_event and self.current_drawdown_event.is_active:
                await self._update_drawdown_event(current_time, current_value, current_drawdown)
            
            # Check alert thresholds
            await self._check_alert_thresholds(current_drawdown, current_time)
            
            # Check emergency conditions
            await self._check_emergency_conditions(current_drawdown)
            
            self.monitoring_stats['checks_performed'] += 1
            self.monitoring_stats['last_update'] = current_time
            self.last_check_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {e}")
    
    async def _start_drawdown_event(self, start_time: datetime, peak_value: float, current_value: float):
        """Start a new drawdown event."""
        initial_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0.0
        
        self.current_drawdown_event = DrawdownEvent(
            start_time=start_time,
            peak_value=peak_value,
            trough_value=current_value,
            max_drawdown=initial_drawdown,
            severity=self._assess_drawdown_severity(initial_drawdown)
        )
        
        self.monitoring_stats['events_detected'] += 1
        
        self.logger.warning(
            f"Drawdown event started: {initial_drawdown:.2%} from peak ${peak_value:,.2f}"
        )
    
    async def _update_drawdown_event(
        self, 
        current_time: datetime, 
        current_value: float, 
        current_drawdown: float
    ):
        """Update existing drawdown event."""
        if not self.current_drawdown_event:
            return
        
        # Update trough value if this is a new low
        if current_value < self.current_drawdown_event.trough_value:
            self.current_drawdown_event.trough_value = current_value
        
        # Update maximum drawdown
        if current_drawdown > self.current_drawdown_event.max_drawdown:
            self.current_drawdown_event.max_drawdown = current_drawdown
            
            # Update severity
            new_severity = self._assess_drawdown_severity(current_drawdown)
            if new_severity != self.current_drawdown_event.severity:
                self.current_drawdown_event.severity = new_severity
                self.logger.warning(
                    f"Drawdown severity escalated to {new_severity.value}: {current_drawdown:.2%}"
                )
        
        # Update duration
        duration = current_time - self.current_drawdown_event.start_time
        self.current_drawdown_event.duration_days = duration.days
    
    async def _end_drawdown_event(self, end_time: datetime, recovered: bool = False):
        """End the current drawdown event."""
        if not self.current_drawdown_event:
            return
        
        self.current_drawdown_event.end_time = end_time
        self.current_drawdown_event.is_active = False
        
        if recovered:
            recovery_time = end_time - self.current_drawdown_event.start_time
            self.current_drawdown_event.recovery_time_days = recovery_time.days
        
        # Add to historical events
        self.historical_events.append(self.current_drawdown_event)
        
        self.logger.info(
            f"Drawdown event ended: Max DD {self.current_drawdown_event.max_drawdown:.2%}, "
            f"Duration {self.current_drawdown_event.duration_days} days"
        )
        
        self.current_drawdown_event = None
    
    async def _check_alert_thresholds(self, current_drawdown: float, current_time: datetime):
        """Check if any alert thresholds are triggered."""
        for alert in self.alerts:
            if not alert.enabled:
                continue
            
            # Check if threshold is exceeded
            if current_drawdown >= alert.threshold:
                
                # Check cooldown period
                if alert.last_triggered:
                    time_since_last = current_time - alert.last_triggered
                    if time_since_last < timedelta(hours=alert.cooldown_hours):
                        continue
                
                # Trigger alert
                await self._trigger_alert(alert, current_drawdown, current_time)
    
    async def _trigger_alert(self, alert: DrawdownAlert, current_drawdown: float, trigger_time: datetime):
        """Trigger a drawdown alert and execute associated action."""
        try:
            self.logger.warning(
                f"Drawdown alert triggered: {alert.severity.value} - {current_drawdown:.2%} "
                f"exceeds {alert.threshold:.2%} threshold"
            )
            
            # Execute associated action
            await self._execute_action(alert.action, current_drawdown, trigger_time)
            
            # Update alert state
            alert.last_triggered = trigger_time
            
            # Record triggered action in current event
            if self.current_drawdown_event:
                self.current_drawdown_event.triggered_actions.append(alert.action)
            
            self.monitoring_stats['alerts_triggered'] += 1
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    async def _execute_action(self, action: str, drawdown: float, trigger_time: datetime):
        """Execute drawdown response action."""
        try:
            self.logger.info(f"Executing drawdown action: {action}")
            
            # Check if custom callback is registered
            if action in self.action_callbacks:
                await self.action_callbacks[action](drawdown, trigger_time)
                self.monitoring_stats['actions_executed'] += 1
                return
            
            # Built-in actions
            if action == "log_warning":
                self.logger.warning(f"Drawdown warning: {drawdown:.2%}")
                
            elif action == "reduce_positions":
                self.logger.warning(f"Drawdown action: Reduce positions ({drawdown:.2%} DD)")
                # Would integrate with position manager to reduce position sizes
                
            elif action == "close_losing_positions":
                self.logger.critical(f"Drawdown action: Close losing positions ({drawdown:.2%} DD)")
                # Would integrate with execution engine to close losing positions
                
            elif action == "emergency_stop":
                self.logger.critical(f"EMERGENCY STOP: {drawdown:.2%} drawdown")
                await self._trigger_emergency_stop()
                
            else:
                self.logger.warning(f"Unknown drawdown action: {action}")
            
            self.monitoring_stats['actions_executed'] += 1
            
        except Exception as e:
            self.logger.error(f"Error executing action '{action}': {e}")
    
    async def _check_emergency_conditions(self, current_drawdown: float):
        """Check emergency stop conditions."""
        # Check total drawdown limit
        if current_drawdown >= self.max_total_drawdown and not self.emergency_stop_triggered:
            self.logger.critical(
                f"Emergency stop triggered: Total drawdown {current_drawdown:.2%} "
                f"exceeds limit {self.max_total_drawdown:.2%}"
            )
            await self._trigger_emergency_stop()
            return
        
        # Check daily drawdown limit
        daily_drawdown = await self._calculate_daily_drawdown()
        if daily_drawdown >= self.max_daily_drawdown and not self.emergency_stop_triggered:
            self.logger.critical(
                f"Emergency stop triggered: Daily drawdown {daily_drawdown:.2%} "
                f"exceeds limit {self.max_daily_drawdown:.2%}"
            )
            await self._trigger_emergency_stop()
    
    async def _trigger_emergency_stop(self):
        """Trigger emergency stop procedures."""
        if self.emergency_stop_triggered:
            return
        
        self.emergency_stop_triggered = True
        
        # Execute emergency stop callback if registered
        if "emergency_stop" in self.action_callbacks:
            await self.action_callbacks["emergency_stop"](None, datetime.now())
        
        self.logger.critical("EMERGENCY STOP ACTIVATED - All trading halted")
    
    async def _calculate_daily_drawdown(self) -> float:
        """Calculate drawdown for the current day."""
        if not self.portfolio_values:
            return 0.0
        
        today = datetime.now().date()
        
        # Find today's values
        today_values = [
            entry for entry in self.portfolio_values
            if entry['timestamp'].date() == today
        ]
        
        if not today_values:
            return 0.0
        
        # Find today's peak and current value
        today_peak = max(entry['value'] for entry in today_values)
        current_value = today_values[-1]['value']
        
        if today_peak > 0:
            daily_drawdown = (today_peak - current_value) / today_peak
        else:
            daily_drawdown = 0.0
        
        return daily_drawdown
    
    def _assess_drawdown_severity(self, drawdown: float) -> DrawdownSeverity:
        """Assess drawdown severity level."""
        if drawdown >= 0.15:
            return DrawdownSeverity.CRITICAL
        elif drawdown >= 0.10:
            return DrawdownSeverity.HIGH
        elif drawdown >= 0.05:
            return DrawdownSeverity.MEDIUM
        else:
            return DrawdownSeverity.LOW
    
    def update_portfolio_value(self, value: float, timestamp: Optional[datetime] = None):
        """Update portfolio value for drawdown calculation."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.portfolio_values.append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only recent values to prevent memory issues
        if len(self.portfolio_values) > 10000:
            self.portfolio_values = self.portfolio_values[-5000:]
    
    def register_action_callback(self, action: str, callback: Callable):
        """Register callback function for drawdown action."""
        self.action_callbacks[action] = callback
        self.logger.info(f"Registered callback for action: {action}")
    
    def get_current_drawdown_info(self) -> Dict[str, Any]:
        """Get current drawdown information."""
        if not self.portfolio_values or self.peak_value == 0:
            return {
                'current_drawdown': 0.0,
                'peak_value': 0.0,
                'current_value': 0.0,
                'is_in_drawdown': False,
                'active_event': None
            }
        
        current_value = self.portfolio_values[-1]['value']
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        
        return {
            'current_drawdown': current_drawdown,
            'peak_value': self.peak_value,
            'current_value': current_value,
            'is_in_drawdown': self.current_drawdown_event is not None,
            'active_event': self.current_drawdown_event,
            'daily_drawdown': await self._calculate_daily_drawdown()
        }
    
    def get_drawdown_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drawdown statistics."""
        if not self.historical_events:
            return {
                'total_events': 0,
                'avg_max_drawdown': 0.0,
                'avg_duration_days': 0.0,
                'avg_recovery_days': 0.0,
                'worst_drawdown': 0.0,
                'longest_drawdown_days': 0,
                'severity_distribution': {}
            }
        
        # Calculate statistics from historical events
        max_drawdowns = [event.max_drawdown for event in self.historical_events]
        durations = [event.duration_days for event in self.historical_events]
        recovery_times = [
            event.recovery_time_days for event in self.historical_events
            if event.recovery_time_days is not None
        ]
        
        # Severity distribution
        severity_counts = {}
        for event in self.historical_events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_events': len(self.historical_events),
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0.0,
            'avg_duration_days': np.mean(durations) if durations else 0.0,
            'avg_recovery_days': np.mean(recovery_times) if recovery_times else 0.0,
            'worst_drawdown': max(max_drawdowns) if max_drawdowns else 0.0,
            'longest_drawdown_days': max(durations) if durations else 0,
            'severity_distribution': severity_counts,
            'current_event_active': self.current_drawdown_event is not None,
            'emergency_stop_active': self.emergency_stop_triggered,
            'monitoring_stats': self.monitoring_stats
        }
    
    def reset_emergency_stop(self):
        """Reset emergency stop flag (use with caution)."""
        if self.emergency_stop_triggered:
            self.emergency_stop_triggered = False
            self.logger.warning("Emergency stop flag reset - Trading may resume")
        
    def stop_monitoring(self):
        """Stop drawdown monitoring."""
        self.monitoring_enabled = False
        self.logger.info("Drawdown monitoring stopped")
    
    def get_alert_configurations(self) -> List[Dict[str, Any]]:
        """Get current alert configurations."""
        return [
            {
                'threshold': alert.threshold,
                'severity': alert.severity.value,
                'action': alert.action,
                'enabled': alert.enabled,
                'cooldown_hours': alert.cooldown_hours,
                'last_triggered': alert.last_triggered.isoformat() if alert.last_triggered else None
            }
            for alert in self.alerts
        ]