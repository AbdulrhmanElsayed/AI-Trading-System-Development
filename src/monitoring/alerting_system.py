"""
Alerting System

Comprehensive alerting and notification system for trading events.
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    throttle_minutes: int = 60
    max_alerts_per_hour: int = 10
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    channels_sent: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False


class AlertingSystem:
    """Main alerting system coordinator."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("AlertingSystem")
        
        # Configuration
        self.alerting_enabled = config.get('monitoring.alerting.enabled', True)
        self.max_alerts_history = config.get('monitoring.alerting.max_history', 10000)
        
        # Channel configurations
        self.email_config = config.get('monitoring.alerting.email', {})
        self.slack_config = config.get('monitoring.alerting.slack', {})
        self.discord_config = config.get('monitoring.alerting.discord', {})
        self.webhook_config = config.get('monitoring.alerting.webhook', {})
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.alert_counters: Dict[str, Dict[str, int]] = {}  # rule_id -> {hour: count}
        
        # Context providers (functions that provide context for alerts)
        self.context_providers: Dict[str, Callable] = {}
        
        # Setup default rules
        self._setup_default_alert_rules()
        
        # Rate limiting
        self.last_sent_times: Dict[str, datetime] = {}
        
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                id="system_error",
                name="System Error Alert",
                description="Triggered when system errors occur",
                condition="error_count > 0",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG]
            ),
            AlertRule(
                id="high_drawdown",
                name="High Drawdown Alert",
                description="Portfolio drawdown exceeds threshold",
                condition="drawdown > 0.10",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.LOG]
            ),
            AlertRule(
                id="large_loss",
                name="Large Loss Alert", 
                description="Single trade or daily loss exceeds threshold",
                condition="daily_pnl < -1000",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG]
            ),
            AlertRule(
                id="connection_loss",
                name="Broker Connection Lost",
                description="Lost connection to broker",
                condition="broker_connected == False",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.LOG]
            ),
            AlertRule(
                id="model_performance",
                name="Model Performance Degradation",
                description="ML model performance below threshold",
                condition="model_accuracy < 0.55",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG]
            ),
            AlertRule(
                id="risk_limit_breach",
                name="Risk Limit Breach",
                description="Risk limits exceeded",
                condition="position_risk > max_risk_limit",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.LOG]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    async def initialize(self):
        """Initialize alerting system."""
        self.logger.info("Alerting System initialized")
        
        # Test connections to external services
        await self._test_alert_channels()
    
    async def _test_alert_channels(self):
        """Test connections to configured alert channels."""
        if self.slack_config.get('webhook_url'):
            await self._test_slack_connection()
        
        if self.discord_config.get('webhook_url'):
            await self._test_discord_connection()
        
        if self.email_config.get('enabled'):
            await self._test_email_connection()
    
    async def _test_slack_connection(self):
        """Test Slack connection."""
        try:
            webhook_url = self.slack_config.get('webhook_url')
            if not webhook_url:
                return
            
            test_payload = {
                "text": "Trading System Alert Test - Slack connection successful",
                "username": "TradingBot",
                "icon_emoji": ":robot_face:"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=test_payload, timeout=10) as response:
                    if response.status == 200:
                        self.logger.info("Slack connection test successful")
                    else:
                        self.logger.warning(f"Slack connection test failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Slack connection test error: {e}")
    
    async def _test_discord_connection(self):
        """Test Discord connection.""" 
        try:
            webhook_url = self.discord_config.get('webhook_url')
            if not webhook_url:
                return
            
            test_payload = {
                "content": "Trading System Alert Test - Discord connection successful"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=test_payload, timeout=10) as response:
                    if response.status == 204:  # Discord returns 204 for success
                        self.logger.info("Discord connection test successful")
                    else:
                        self.logger.warning(f"Discord connection test failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Discord connection test error: {e}")
    
    async def _test_email_connection(self):
        """Test email connection."""
        try:
            if not self.email_config.get('enabled'):
                return
            
            # Simple SMTP connection test
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            
            if username and password:
                server.login(username, password)
                self.logger.info("Email connection test successful")
            
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Email connection test error: {e}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule."""
        self.alert_rules[rule.id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def enable_alert_rule(self, rule_id: str):
        """Enable alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            self.logger.info(f"Enabled alert rule: {rule_id}")
    
    def disable_alert_rule(self, rule_id: str):
        """Disable alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            self.logger.info(f"Disabled alert rule: {rule_id}")
    
    def register_context_provider(self, name: str, provider: Callable):
        """Register a context provider function."""
        self.context_providers[name] = provider
        self.logger.info(f"Registered context provider: {name}")
    
    async def evaluate_alerts(self, context: Dict[str, Any]):
        """Evaluate all alert rules against current context."""
        if not self.alerting_enabled:
            return
        
        # Merge context with context providers
        full_context = context.copy()
        for name, provider in self.context_providers.items():
            try:
                provider_context = await provider() if asyncio.iscoroutinefunction(provider) else provider()
                full_context[name] = provider_context
            except Exception as e:
                self.logger.error(f"Error getting context from provider {name}: {e}")
        
        # Evaluate each rule
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check rate limiting
                if not self._check_rate_limit(rule):
                    continue
                
                # Evaluate condition
                if self._evaluate_condition(rule.condition, full_context):
                    await self._trigger_alert(rule, full_context)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_id}: {e}")
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate alert condition safely."""
        try:
            # Create safe evaluation environment
            safe_globals = {
                '__builtins__': {},
                'abs': abs,
                'max': max,
                'min': min,
                'len': len,
                'sum': sum,
                'any': any,
                'all': all
            }
            
            # Add context variables
            safe_globals.update(context)
            
            # Evaluate condition
            result = eval(condition, safe_globals)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _check_rate_limit(self, rule: AlertRule) -> bool:
        """Check if alert rule is within rate limits."""
        now = datetime.now()
        rule_id = rule.id
        
        # Check throttle period
        if rule_id in self.last_sent_times:
            time_since_last = now - self.last_sent_times[rule_id]
            if time_since_last.total_seconds() < rule.throttle_minutes * 60:
                return False
        
        # Check hourly limit
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        if rule_id not in self.alert_counters:
            self.alert_counters[rule_id] = {}
        
        hour_key = current_hour.isoformat()
        current_count = self.alert_counters[rule_id].get(hour_key, 0)
        
        if current_count >= rule.max_alerts_per_hour:
            return False
        
        return True
    
    async def _trigger_alert(self, rule: AlertRule, context: Dict[str, Any]):
        """Trigger an alert."""
        try:
            # Create alert instance
            alert = Alert(
                id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.id}",
                rule_id=rule.id,
                title=rule.name,
                message=self._format_alert_message(rule, context),
                severity=rule.severity,
                timestamp=datetime.now(),
                context=context.copy()
            )
            
            # Send to channels
            for channel in rule.channels:
                try:
                    success = await self._send_to_channel(alert, channel)
                    if success:
                        alert.channels_sent.append(channel)
                except Exception as e:
                    self.logger.error(f"Error sending alert to {channel.value}: {e}")
            
            # Store alert
            self.alert_history.append(alert)
            
            # Update counters
            self._update_rate_counters(rule)
            
            # Trim history
            if len(self.alert_history) > self.max_alerts_history:
                self.alert_history = self.alert_history[-self.max_alerts_history//2:]
            
            self.logger.info(f"Triggered alert: {rule.name} ({rule.severity.value})")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert for rule {rule.id}: {e}")
    
    def _format_alert_message(self, rule: AlertRule, context: Dict[str, Any]) -> str:
        """Format alert message with context."""
        base_message = f"{rule.description}\n\n"
        
        # Add relevant context
        if 'portfolio_value' in context:
            base_message += f"Portfolio Value: ${context['portfolio_value']:,.2f}\n"
        
        if 'daily_pnl' in context:
            base_message += f"Daily P&L: ${context['daily_pnl']:,.2f}\n"
        
        if 'drawdown' in context:
            base_message += f"Current Drawdown: {context['drawdown']:.2%}\n"
        
        if 'active_positions' in context:
            base_message += f"Active Positions: {context['active_positions']}\n"
        
        base_message += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return base_message
    
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to specific channel."""
        try:
            if channel == AlertChannel.EMAIL:
                return await self._send_email_alert(alert)
            elif channel == AlertChannel.SLACK:
                return await self._send_slack_alert(alert)
            elif channel == AlertChannel.DISCORD:
                return await self._send_discord_alert(alert)
            elif channel == AlertChannel.WEBHOOK:
                return await self._send_webhook_alert(alert)
            elif channel == AlertChannel.LOG:
                return self._send_log_alert(alert)
            elif channel == AlertChannel.CONSOLE:
                return self._send_console_alert(alert)
            else:
                self.logger.warning(f"Unknown alert channel: {channel}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending alert to {channel.value}: {e}")
            return False
    
    async def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            if not self.email_config.get('enabled'):
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_address')
            msg['To'] = ', '.join(self.email_config.get('to_addresses', []))
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Format message body
            body = f"""
Trading System Alert

Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.message}

Alert ID: {alert.id}
Rule ID: {alert.rule_id}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                self.email_config.get('smtp_server', 'smtp.gmail.com'),
                self.email_config.get('smtp_port', 587)
            )
            server.starttls()
            server.login(
                self.email_config.get('username'),
                self.email_config.get('password')
            )
            
            text = msg.as_string()
            server.sendmail(
                self.email_config.get('from_address'),
                self.email_config.get('to_addresses', []),
                text
            )
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            return False
    
    async def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            webhook_url = self.slack_config.get('webhook_url')
            if not webhook_url:
                return False
            
            # Format Slack message
            color_map = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning', 
                AlertSeverity.ERROR: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            payload = {
                "username": "TradingBot",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [{
                    "color": color_map.get(alert.severity, 'warning'),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "ts": int(alert.timestamp.timestamp()),
                    "fields": [
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": True
                        },
                        {
                            "title": "Rule ID", 
                            "value": alert.rule_id,
                            "short": True
                        }
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=10) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
            return False
    
    async def _send_discord_alert(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        try:
            webhook_url = self.discord_config.get('webhook_url')
            if not webhook_url:
                return False
            
            # Format Discord message
            color_map = {
                AlertSeverity.INFO: 0x00ff00,      # Green
                AlertSeverity.WARNING: 0xffff00,   # Yellow
                AlertSeverity.ERROR: 0xff8000,     # Orange
                AlertSeverity.CRITICAL: 0xff0000   # Red
            }
            
            payload = {
                "embeds": [{
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "description": alert.message,
                    "color": color_map.get(alert.severity, 0xffff00),
                    "timestamp": alert.timestamp.isoformat(),
                    "fields": [
                        {
                            "name": "Alert ID",
                            "value": alert.id,
                            "inline": True
                        },
                        {
                            "name": "Rule ID",
                            "value": alert.rule_id,
                            "inline": True
                        }
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=10) as response:
                    return response.status == 204  # Discord returns 204
                    
        except Exception as e:
            self.logger.error(f"Error sending Discord alert: {e}")
            return False
    
    async def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send alert to custom webhook."""
        try:
            webhook_url = self.webhook_config.get('url')
            if not webhook_url:
                return False
            
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "context": alert.context
            }
            
            headers = self.webhook_config.get('headers', {})
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url, 
                    json=payload, 
                    headers=headers,
                    timeout=10
                ) as response:
                    return response.status in [200, 201, 202]
                    
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
            return False
    
    def _send_log_alert(self, alert: Alert) -> bool:
        """Send alert to log."""
        try:
            log_method = getattr(self.logger, alert.severity.value, self.logger.info)
            log_method(f"ALERT: {alert.title} - {alert.message}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending log alert: {e}")
            return False
    
    def _send_console_alert(self, alert: Alert) -> bool:
        """Send alert to console."""
        try:
            severity_colors = {
                AlertSeverity.INFO: '\033[92m',      # Green
                AlertSeverity.WARNING: '\033[93m',   # Yellow  
                AlertSeverity.ERROR: '\033[91m',     # Red
                AlertSeverity.CRITICAL: '\033[95m'   # Magenta
            }
            
            reset_color = '\033[0m'
            color = severity_colors.get(alert.severity, '')
            
            print(f"{color}[{alert.severity.value.upper()}] {alert.title}{reset_color}")
            print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Message: {alert.message}")
            print(f"Alert ID: {alert.id}")
            print("-" * 50)
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending console alert: {e}")
            return False
    
    def _update_rate_counters(self, rule: AlertRule):
        """Update rate limiting counters."""
        now = datetime.now()
        rule_id = rule.id
        
        # Update last sent time
        self.last_sent_times[rule_id] = now
        
        # Update hourly counter
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        hour_key = current_hour.isoformat()
        
        if rule_id not in self.alert_counters:
            self.alert_counters[rule_id] = {}
        
        self.alert_counters[rule_id][hour_key] = self.alert_counters[rule_id].get(hour_key, 0) + 1
        
        # Clean old counters (older than 24 hours)
        cutoff_time = now - timedelta(hours=24)
        for rule_id in list(self.alert_counters.keys()):
            for hour_key in list(self.alert_counters[rule_id].keys()):
                try:
                    hour_time = datetime.fromisoformat(hour_key)
                    if hour_time < cutoff_time:
                        del self.alert_counters[rule_id][hour_key]
                except:
                    pass
    
    def get_alert_history(self, hours: int = 24, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history within specified timeframe."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary."""
        recent_alerts = self.get_alert_history(24)
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        return {
            'alerting_enabled': self.alerting_enabled,
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'recent_alerts_24h': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'alert_channels_configured': len([
                ch for ch in AlertChannel 
                if self._is_channel_configured(ch)
            ]),
            'last_alert': recent_alerts[0].timestamp.isoformat() if recent_alerts else None
        }
    
    def _is_channel_configured(self, channel: AlertChannel) -> bool:
        """Check if alert channel is properly configured."""
        if channel == AlertChannel.EMAIL:
            return bool(self.email_config.get('enabled'))
        elif channel == AlertChannel.SLACK:
            return bool(self.slack_config.get('webhook_url'))
        elif channel == AlertChannel.DISCORD:
            return bool(self.discord_config.get('webhook_url'))
        elif channel == AlertChannel.WEBHOOK:
            return bool(self.webhook_config.get('url'))
        elif channel in [AlertChannel.LOG, AlertChannel.CONSOLE]:
            return True
        return False
    
    async def send_test_alert(self, channel: AlertChannel):
        """Send a test alert to verify channel configuration."""
        test_alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            title="Test Alert",
            message="This is a test alert to verify channel configuration.",
            severity=AlertSeverity.INFO,
            timestamp=datetime.now()
        )
        
        success = await self._send_to_channel(test_alert, channel)
        
        if success:
            self.logger.info(f"Test alert sent successfully to {channel.value}")
        else:
            self.logger.error(f"Failed to send test alert to {channel.value}")
        
        return success