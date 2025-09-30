# Milestone 6: Monitoring & Dashboard System

## Overview

This milestone implements a comprehensive monitoring and dashboard system for the AI Trading Platform, providing real-time oversight, alerting capabilities, and web-based visualization of system performance and health.

## Architecture Components

### 1. Performance Monitor (`performance_monitor.py`)
**Purpose**: Real-time tracking of trading performance and portfolio metrics

**Key Features**:
- Real-time performance calculation (returns, Sharpe ratio, drawdown, volatility)
- Configurable alert rules for performance thresholds
- Performance history tracking and analysis
- Automated performance reporting

**Core Classes**:
- `PerformanceMonitor`: Main monitoring engine
- `PerformanceSnapshot`: Data structure for performance metrics
- `AlertRule`: Configuration for performance-based alerts

### 2. Alerting System (`alerting_system.py`)
**Purpose**: Multi-channel alerting and notification system

**Key Features**:
- Multiple notification channels (Email, Slack, Discord, Webhook, Console)
- Flexible alert rule engine with condition evaluation
- Rate limiting to prevent alert spam
- Alert history tracking and management
- Context-aware alert messages

**Core Classes**:
- `AlertingSystem`: Central alerting coordinator
- `Alert`: Individual alert data structure
- `AlertSeverity`: Alert severity levels (INFO, WARNING, CRITICAL)
- `AlertChannel`: Notification channel configurations

### 3. System Health Monitor (`system_health.py`)
**Purpose**: System resource monitoring and component health tracking

**Key Features**:
- Real-time system metrics (CPU, memory, disk usage)
- Component health checks for trading system modules
- Threshold-based health alerting
- System performance analytics
- Resource utilization trending

**Core Classes**:
- `SystemHealthMonitor`: Health monitoring engine
- `SystemMetrics`: System resource metrics
- `ComponentHealth`: Individual component health status

### 4. Dashboard System (`dashboard.py`)
**Purpose**: Real-time web dashboard with WebSocket support

**Key Features**:
- Real-time data visualization through WebSockets
- Interactive portfolio and performance charts
- System health status displays
- Alert notifications and history
- Responsive web interface

**Core Classes**:
- `DashboardServer`: WebSocket server for real-time data
- `DashboardManager`: Dashboard coordination and management

### 5. Monitoring Coordinator (`monitoring_coordinator.py`)
**Purpose**: Central coordination of all monitoring components

**Key Features**:
- Unified monitoring interface
- Component integration and data flow coordination
- Event handling and distribution
- System-wide monitoring orchestration
- Integration with trading system modules

**Core Classes**:
- `MonitoringCoordinator`: Central monitoring coordinator

### 6. Configuration Templates (`config_templates.py`)
**Purpose**: Pre-defined configurations for monitoring components

**Key Features**:
- Default monitoring configurations
- Alert rule templates
- Dashboard layout templates
- Easy customization and deployment

## Key Features

### Real-Time Monitoring
- **Portfolio Performance**: Live tracking of returns, P&L, and risk metrics
- **System Health**: CPU, memory, disk usage, and component status
- **Execution Monitoring**: Order execution, latency, and broker connectivity
- **Alert Management**: Real-time notifications across multiple channels

### Advanced Alerting
- **Multi-Channel Notifications**: Email, Slack, Discord, webhooks, console
- **Smart Rate Limiting**: Prevents alert spam while ensuring critical alerts
- **Context-Aware Messages**: Dynamic alert content based on system state
- **Severity-Based Routing**: Different channels for different alert levels

### Web Dashboard
- **Real-Time Updates**: WebSocket-based live data streaming
- **Interactive Charts**: Performance charts, system metrics visualization
- **Alert Dashboard**: Real-time alert notifications and history
- **System Status**: Component health and connectivity status

### Integration Capabilities
- **Trading System Integration**: Portfolio manager, risk manager, execution engine
- **Data Flow Coordination**: Automated data collection and distribution
- **Event-Driven Architecture**: Responsive to system events and changes
- **Extensible Design**: Easy to add new monitoring components

## Configuration

### Basic Setup
```python
from src.monitoring import MonitoringCoordinator, MonitoringConfig
from src.utils.config import ConfigManager

# Load configuration
config = ConfigManager()
monitoring_config = MonitoringConfig.get_complete_monitoring_config()
config.config.update(monitoring_config)

# Initialize monitoring
coordinator = MonitoringCoordinator(config)
await coordinator.initialize()
```

### Alert Configuration
```python
# Configure email alerts
email_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'trading.bot@example.com',
    'sender_password': 'your_password',
    'recipients': ['trader@example.com']
}

# Add custom alert rules
await coordinator.alerting_system.add_alert_rule(
    'portfolio_loss_alert',
    {
        'conditions': [{'field': 'daily_return', 'operator': 'lt', 'value': -0.05}],
        'severity': 'critical',
        'title': 'Portfolio Loss Alert',
        'message': 'Daily return is {daily_return:.2%}',
        'channels': ['email', 'console']
    }
)
```

### Dashboard Setup
```python
# Start dashboard server
dashboard_url = coordinator.get_dashboard_url()
print(f"Dashboard available at: {dashboard_url}")

# Access dashboard at http://127.0.0.1:8080
```

## Usage Examples

### Basic Monitoring Setup
```python
import asyncio
from src.monitoring import MonitoringCoordinator, MonitoringConfig

async def setup_monitoring():
    # Initialize coordinator
    coordinator = MonitoringCoordinator(config)
    await coordinator.initialize()
    
    # Register trading components
    coordinator.register_portfolio_manager(portfolio_manager)
    coordinator.register_risk_manager(risk_manager)
    coordinator.register_execution_engine(execution_engine)
    
    # Start monitoring
    print(f"Dashboard: {coordinator.get_dashboard_url()}")
    return coordinator

# Run monitoring
coordinator = await setup_monitoring()
```

### Custom Alert Rules
```python
# Performance-based alerts
performance_alerts = {
    'daily_loss_5_percent': {
        'conditions': [{'field': 'daily_return', 'operator': 'lt', 'value': -0.05}],
        'severity': 'warning',
        'title': 'Daily Loss Alert',
        'channels': ['email', 'console']
    },
    'max_drawdown_alert': {
        'conditions': [{'field': 'max_drawdown', 'operator': 'lt', 'value': -0.15}],
        'severity': 'critical',
        'title': 'Maximum Drawdown Alert',
        'channels': ['email', 'slack', 'console']
    }
}

# Add alerts to system
for name, config in performance_alerts.items():
    await coordinator.alerting_system.add_alert_rule(name, config)
```

### System Health Monitoring
```python
# Get current system health
health_status = coordinator.system_health_monitor.get_system_summary()
print(f"System Status: {health_status}")

# Check component health
component_health = coordinator.system_health_monitor.get_component_health('portfolio_manager')
print(f"Portfolio Manager Health: {component_health}")
```

## Dashboard Features

### Main Dashboard Sections
1. **Portfolio Overview**: Current portfolio value, P&L, returns, Sharpe ratio
2. **Performance Charts**: Historical performance, daily returns, drawdown charts
3. **System Health**: CPU/memory usage, component status indicators
4. **Recent Alerts**: Real-time alert notifications and history
5. **Active Positions**: Current portfolio positions and allocations

### Real-Time Updates
- WebSocket connection provides live data updates
- Automatic refresh of charts and metrics
- Real-time alert notifications
- System status indicators

### Customization
- Configurable dashboard layouts
- Custom chart types and metrics
- Adjustable update intervals
- Theme and styling options

## Alert Channels

### Email Alerts
- SMTP server configuration
- HTML formatted messages
- Attachment support for reports
- Rate limiting protection

### Slack Integration
- Webhook-based notifications
- Channel-specific routing
- Rich message formatting
- Bot integration support

### Discord Integration
- Webhook notifications
- Server and channel routing
- Embed message support
- Real-time notifications

### Webhook Alerts
- Custom webhook endpoints
- JSON payload delivery
- Custom headers support
- Retry mechanism

## Monitoring Metrics

### Performance Metrics
- **Returns**: Daily, weekly, monthly, total returns
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility
- **Portfolio Metrics**: Total value, P&L, cash balance
- **Execution Metrics**: Order success rate, latency, slippage

### System Metrics
- **Resource Usage**: CPU, memory, disk utilization
- **Network Metrics**: Latency, connectivity status
- **Component Health**: Service availability, response times
- **Error Rates**: System errors, execution failures

### Trading Metrics
- **Position Metrics**: Active positions, exposure, concentration
- **Risk Metrics**: VaR, position sizing compliance
- **Execution Quality**: Fill rates, execution time, market impact
- **Broker Connectivity**: Connection status, API limits

## Integration Guide

### Portfolio Manager Integration
```python
# Register portfolio manager
coordinator.register_portfolio_manager(portfolio_manager)

# Automatic portfolio monitoring
# - Performance calculation
# - Risk metric updates
# - Alert evaluation
```

### Risk Manager Integration
```python
# Register risk manager
coordinator.register_risk_manager(risk_manager)

# Automatic risk monitoring
# - Position limit compliance
# - Risk threshold monitoring
# - Risk alert generation
```

### Execution Engine Integration
```python
# Register execution engine
coordinator.register_execution_engine(execution_engine)

# Automatic execution monitoring
# - Order status tracking
# - Execution latency monitoring
# - Broker connectivity status
```

## Testing and Validation

### Unit Tests
- Individual component testing
- Mock data simulation
- Alert rule validation
- Dashboard functionality testing

### Integration Tests
- End-to-end monitoring workflows
- Cross-component data flow validation
- Alert delivery testing
- Dashboard real-time updates

### Performance Tests
- System resource impact assessment
- High-frequency data handling
- Concurrent alert processing
- WebSocket connection scaling

## Deployment Considerations

### Resource Requirements
- **CPU**: Moderate usage for real-time processing
- **Memory**: ~200-500MB depending on data retention
- **Network**: WebSocket connections for dashboard
- **Storage**: Alert history and metrics retention

### Security Considerations
- Dashboard access control
- Alert channel authentication
- Sensitive data handling
- Network security for WebSocket connections

### Scalability Features
- Horizontal scaling support
- Database backend for metrics storage
- Load balancing for dashboard access
- Distributed monitoring coordination

## Configuration Files

### Main Configuration
```yaml
monitoring:
  performance_monitor:
    update_interval: 60
    metrics_retention_days: 90
    
  alerting_system:
    rate_limits:
      email: {max_per_hour: 10, max_per_day: 50}
      slack: {max_per_hour: 20, max_per_day: 100}
    
  dashboard:
    host: "127.0.0.1"
    port: 8080
    auto_refresh_interval: 5000
    
  system_health:
    monitoring_interval: 30
    thresholds:
      cpu_usage: {warning: 70, critical: 90}
      memory_usage: {warning: 80, critical: 95}
```

## API Reference

### MonitoringCoordinator
- `initialize()`: Initialize all monitoring components
- `register_portfolio_manager()`: Register portfolio integration
- `register_risk_manager()`: Register risk management integration
- `get_monitoring_status()`: Get comprehensive system status
- `send_test_alert()`: Send test notification

### PerformanceMonitor
- `update_portfolio_data()`: Update portfolio performance data
- `get_current_performance()`: Get current performance metrics
- `get_performance_summary()`: Get performance analysis summary

### AlertingSystem
- `add_alert_rule()`: Add custom alert rule
- `evaluate_alerts()`: Evaluate alert conditions
- `get_alert_history()`: Retrieve alert history

### SystemHealthMonitor
- `get_current_metrics()`: Get current system metrics
- `get_system_summary()`: Get system health summary
- `register_health_checker()`: Add custom health check

## Troubleshooting

### Common Issues
1. **Dashboard not accessible**: Check firewall settings and port availability
2. **Alerts not sending**: Verify channel configurations (SMTP, webhooks)
3. **High resource usage**: Adjust monitoring intervals and data retention
4. **WebSocket connection issues**: Check network connectivity and browser settings

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger('MonitoringCoordinator').setLevel(logging.DEBUG)

# Monitoring status check
status = coordinator.get_monitoring_status()
print(f"Debug Status: {status}")
```

### Log Analysis
- Check monitoring component logs for errors
- Verify alert delivery in channel logs
- Monitor WebSocket connection logs
- Review system health metrics for bottlenecks

## Next Steps

With the monitoring and dashboard system complete, the next milestones are:

1. **Milestone 7**: Testing Framework - Comprehensive unit and integration tests
2. **Milestone 8**: Deployment & Production Setup - Docker containerization and deployment

The monitoring system provides the foundation for production operations, ensuring system reliability, performance tracking, and proactive issue detection.