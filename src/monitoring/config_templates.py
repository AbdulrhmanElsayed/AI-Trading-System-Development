"""
Monitoring Configuration Templates

Default configurations for all monitoring components.
"""

from datetime import timedelta
from typing import Dict, Any, List


class MonitoringConfig:
    """Default monitoring configuration."""
    
    @staticmethod
    def get_performance_monitor_config() -> Dict[str, Any]:
        """Get performance monitor configuration."""
        return {
            'update_interval': 60,  # seconds
            'alert_rules': {
                'daily_return_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'daily_return',
                            'operator': 'lt',
                            'value': -0.05  # Alert if daily return < -5%
                        }
                    ],
                    'severity': 'warning',
                    'title': 'Daily Loss Alert',
                    'message': 'Daily return is below -5%: {daily_return:.2%}'
                },
                'drawdown_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'max_drawdown',
                            'operator': 'lt',
                            'value': -0.15  # Alert if drawdown > 15%
                        }
                    ],
                    'severity': 'critical',
                    'title': 'Maximum Drawdown Alert',
                    'message': 'Maximum drawdown exceeded 15%: {max_drawdown:.2%}'
                },
                'sharpe_ratio_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'sharpe_ratio',
                            'operator': 'lt',
                            'value': 0.5  # Alert if Sharpe ratio < 0.5
                        }
                    ],
                    'severity': 'warning',
                    'title': 'Low Sharpe Ratio Alert',
                    'message': 'Sharpe ratio is below 0.5: {sharpe_ratio:.2f}'
                },
                'portfolio_value_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'portfolio_value',
                            'operator': 'lt',
                            'value': 8000  # Alert if portfolio < $8,000
                        }
                    ],
                    'severity': 'critical',
                    'title': 'Portfolio Value Alert',
                    'message': 'Portfolio value dropped below $8,000: ${portfolio_value:,.2f}'
                }
            },
            'metrics_retention_days': 90
        }
    
    @staticmethod
    def get_alerting_system_config() -> Dict[str, Any]:
        """Get alerting system configuration."""
        return {
            'channels': {
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': 'trading.bot@example.com',
                    'sender_password': '',  # Set via environment variable
                    'recipients': ['trader@example.com']
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': '',
                    'channel': '#trading-alerts',
                    'username': 'TradingBot'
                },
                'discord': {
                    'enabled': False,
                    'webhook_url': '',
                    'username': 'TradingBot'
                },
                'webhook': {
                    'enabled': False,
                    'url': '',
                    'headers': {
                        'Content-Type': 'application/json'
                    }
                },
                'console': {
                    'enabled': True,
                    'log_level': 'INFO'
                }
            },
            'alert_rules': {
                'system_health_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'cpu_usage',
                            'operator': 'gt',
                            'value': 90.0
                        }
                    ],
                    'severity': 'critical',
                    'title': 'High CPU Usage Alert',
                    'message': 'CPU usage is at {cpu_usage:.1f}%',
                    'channels': ['email', 'console']
                },
                'memory_usage_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'memory_usage',
                            'operator': 'gt',
                            'value': 85.0
                        }
                    ],
                    'severity': 'warning',
                    'title': 'High Memory Usage Alert',
                    'message': 'Memory usage is at {memory_usage:.1f}%',
                    'channels': ['console']
                },
                'portfolio_performance_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'daily_return',
                            'operator': 'lt',
                            'value': -0.03
                        }
                    ],
                    'severity': 'warning',
                    'title': 'Daily Performance Alert',
                    'message': 'Daily return is {daily_return:.2%}',
                    'channels': ['email', 'console']
                },
                'execution_error_alert': {
                    'enabled': True,
                    'conditions': [
                        {
                            'field': 'execution_errors',
                            'operator': 'gt',
                            'value': 5
                        }
                    ],
                    'severity': 'critical',
                    'title': 'Execution Error Alert',
                    'message': 'Multiple execution errors detected: {execution_errors}',
                    'channels': ['email', 'slack', 'console']
                }
            },
            'rate_limits': {
                'email': {
                    'max_per_hour': 10,
                    'max_per_day': 50
                },
                'slack': {
                    'max_per_hour': 20,
                    'max_per_day': 100
                },
                'console': {
                    'max_per_hour': 100,
                    'max_per_day': 1000
                }
            }
        }
    
    @staticmethod
    def get_system_health_config() -> Dict[str, Any]:
        """Get system health monitor configuration."""
        return {
            'monitoring_interval': 30,  # seconds
            'thresholds': {
                'cpu_usage': {
                    'warning': 70.0,
                    'critical': 90.0
                },
                'memory_usage': {
                    'warning': 80.0,
                    'critical': 95.0
                },
                'disk_usage': {
                    'warning': 80.0,
                    'critical': 95.0
                },
                'system_load': {
                    'warning': 4.0,
                    'critical': 8.0
                },
                'network_latency': {
                    'warning': 100.0,  # ms
                    'critical': 500.0
                }
            },
            'retention_hours': 168,  # 7 days
            'component_health_checks': [
                'portfolio_manager',
                'risk_manager',
                'execution_engine',
                'performance_monitor',
                'alerting_system',
                'dashboard'
            ]
        }
    
    @staticmethod
    def get_dashboard_config() -> Dict[str, Any]:
        """Get dashboard configuration."""
        return {
            'server': {
                'host': '127.0.0.1',
                'port': 8080,
                'ssl_enabled': False,
                'ssl_cert_path': '',
                'ssl_key_path': ''
            },
            'websocket': {
                'ping_interval': 20,
                'ping_timeout': 10,
                'max_message_size': 1048576  # 1MB
            },
            'dashboard': {
                'title': 'AI Trading System Dashboard',
                'theme': 'dark',
                'auto_refresh_interval': 5000,  # ms
                'chart_retention_points': 100,
                'timezone': 'UTC'
            },
            'data_sources': {
                'portfolio': {
                    'update_interval': 10  # seconds
                },
                'performance': {
                    'update_interval': 60  # seconds
                },
                'system_health': {
                    'update_interval': 30  # seconds
                },
                'alerts': {
                    'update_interval': 15  # seconds
                }
            }
        }
    
    @staticmethod
    def get_integration_config() -> Dict[str, Any]:
        """Get integration configuration."""
        return {
            'coordinator': {
                'coordination_interval': 30,  # seconds
                'health_check_interval': 60,  # seconds
                'context_update_interval': 10  # seconds
            },
            'data_flow': {
                'performance_data_interval': 60,
                'system_metrics_interval': 30,
                'dashboard_update_interval': 5,
                'alert_evaluation_interval': 15
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay': 5,  # seconds
                'circuit_breaker_threshold': 5,
                'circuit_breaker_timeout': 300  # seconds
            }
        }
    
    @staticmethod
    def get_complete_monitoring_config() -> Dict[str, Any]:
        """Get complete monitoring system configuration."""
        return {
            'performance_monitor': MonitoringConfig.get_performance_monitor_config(),
            'alerting_system': MonitoringConfig.get_alerting_system_config(),
            'system_health': MonitoringConfig.get_system_health_config(),
            'dashboard': MonitoringConfig.get_dashboard_config(),
            'integration': MonitoringConfig.get_integration_config()
        }


class AlertTemplates:
    """Pre-defined alert templates."""
    
    PORTFOLIO_ALERTS = {
        'daily_loss_5_percent': {
            'enabled': True,
            'conditions': [
                {'field': 'daily_return', 'operator': 'lt', 'value': -0.05}
            ],
            'severity': 'warning',
            'title': 'Daily Loss Alert (5%)',
            'message': 'Daily return is {daily_return:.2%}',
            'channels': ['email', 'console']
        },
        'daily_loss_10_percent': {
            'enabled': True,
            'conditions': [
                {'field': 'daily_return', 'operator': 'lt', 'value': -0.10}
            ],
            'severity': 'critical',
            'title': 'Major Daily Loss Alert (10%)',
            'message': 'CRITICAL: Daily return is {daily_return:.2%}',
            'channels': ['email', 'slack', 'console']
        },
        'portfolio_value_drop': {
            'enabled': True,
            'conditions': [
                {'field': 'portfolio_value', 'operator': 'lt', 'value': 5000}
            ],
            'severity': 'critical',
            'title': 'Portfolio Value Critical',
            'message': 'Portfolio value dropped to ${portfolio_value:,.2f}',
            'channels': ['email', 'slack', 'console']
        },
        'max_drawdown_exceeded': {
            'enabled': True,
            'conditions': [
                {'field': 'max_drawdown', 'operator': 'lt', 'value': -0.20}
            ],
            'severity': 'critical',
            'title': 'Maximum Drawdown Exceeded',
            'message': 'Drawdown exceeded 20%: {max_drawdown:.2%}',
            'channels': ['email', 'slack', 'console']
        }
    }
    
    SYSTEM_ALERTS = {
        'high_cpu_usage': {
            'enabled': True,
            'conditions': [
                {'field': 'cpu_usage', 'operator': 'gt', 'value': 85.0}
            ],
            'severity': 'warning',
            'title': 'High CPU Usage',
            'message': 'CPU usage is at {cpu_usage:.1f}%',
            'channels': ['console']
        },
        'critical_cpu_usage': {
            'enabled': True,
            'conditions': [
                {'field': 'cpu_usage', 'operator': 'gt', 'value': 95.0}
            ],
            'severity': 'critical',
            'title': 'Critical CPU Usage',
            'message': 'CRITICAL: CPU usage is at {cpu_usage:.1f}%',
            'channels': ['email', 'console']
        },
        'high_memory_usage': {
            'enabled': True,
            'conditions': [
                {'field': 'memory_usage', 'operator': 'gt', 'value': 85.0}
            ],
            'severity': 'warning',
            'title': 'High Memory Usage',
            'message': 'Memory usage is at {memory_usage:.1f}%',
            'channels': ['console']
        },
        'disk_space_low': {
            'enabled': True,
            'conditions': [
                {'field': 'disk_usage', 'operator': 'gt', 'value': 90.0}
            ],
            'severity': 'warning',
            'title': 'Low Disk Space',
            'message': 'Disk usage is at {disk_usage:.1f}%',
            'channels': ['email', 'console']
        }
    }
    
    EXECUTION_ALERTS = {
        'order_failure': {
            'enabled': True,
            'conditions': [
                {'field': 'failed_orders', 'operator': 'gt', 'value': 3}
            ],
            'severity': 'warning',
            'title': 'Multiple Order Failures',
            'message': 'Multiple orders failed: {failed_orders}',
            'channels': ['email', 'console']
        },
        'broker_connection_lost': {
            'enabled': True,
            'conditions': [
                {'field': 'broker_connected', 'operator': 'eq', 'value': False}
            ],
            'severity': 'critical',
            'title': 'Broker Connection Lost',
            'message': 'Lost connection to broker',
            'channels': ['email', 'slack', 'console']
        },
        'execution_latency_high': {
            'enabled': True,
            'conditions': [
                {'field': 'execution_latency', 'operator': 'gt', 'value': 5000}  # ms
            ],
            'severity': 'warning',
            'title': 'High Execution Latency',
            'message': 'Execution latency is {execution_latency:.0f}ms',
            'channels': ['console']
        }
    }
    
    @staticmethod
    def get_all_alert_templates() -> Dict[str, Dict]:
        """Get all predefined alert templates."""
        return {
            **AlertTemplates.PORTFOLIO_ALERTS,
            **AlertTemplates.SYSTEM_ALERTS,
            **AlertTemplates.EXECUTION_ALERTS
        }


class DashboardTemplates:
    """Dashboard configuration templates."""
    
    @staticmethod
    def get_trading_dashboard_layout() -> Dict[str, Any]:
        """Get trading dashboard layout configuration."""
        return {
            'layout': {
                'sections': [
                    {
                        'id': 'portfolio_overview',
                        'title': 'Portfolio Overview',
                        'type': 'grid',
                        'position': {'row': 1, 'col': 1, 'width': 12, 'height': 4},
                        'widgets': [
                            {'type': 'metric', 'metric': 'portfolio_value', 'title': 'Portfolio Value'},
                            {'type': 'metric', 'metric': 'daily_pnl', 'title': 'Daily P&L'},
                            {'type': 'metric', 'metric': 'total_return', 'title': 'Total Return'},
                            {'type': 'metric', 'metric': 'sharpe_ratio', 'title': 'Sharpe Ratio'}
                        ]
                    },
                    {
                        'id': 'performance_charts',
                        'title': 'Performance Charts',
                        'type': 'chart_grid',
                        'position': {'row': 2, 'col': 1, 'width': 8, 'height': 6},
                        'charts': [
                            {'type': 'line', 'metric': 'portfolio_value', 'title': 'Portfolio Value Over Time'},
                            {'type': 'line', 'metric': 'daily_returns', 'title': 'Daily Returns'}
                        ]
                    },
                    {
                        'id': 'system_health',
                        'title': 'System Health',
                        'type': 'status_grid',
                        'position': {'row': 2, 'col': 9, 'width': 4, 'height': 6},
                        'widgets': [
                            {'type': 'gauge', 'metric': 'cpu_usage', 'title': 'CPU Usage'},
                            {'type': 'gauge', 'metric': 'memory_usage', 'title': 'Memory Usage'},
                            {'type': 'status', 'metric': 'system_status', 'title': 'System Status'}
                        ]
                    },
                    {
                        'id': 'recent_alerts',
                        'title': 'Recent Alerts',
                        'type': 'list',
                        'position': {'row': 3, 'col': 1, 'width': 12, 'height': 3},
                        'data_source': 'alerts',
                        'max_items': 10
                    }
                ]
            },
            'styling': {
                'theme': 'dark',
                'primary_color': '#007acc',
                'success_color': '#28a745',
                'warning_color': '#ffc107',
                'danger_color': '#dc3545',
                'font_family': 'Arial, sans-serif'
            }
        }