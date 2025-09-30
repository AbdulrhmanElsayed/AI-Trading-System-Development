"""
Monitoring System Demo

Demonstrates the complete monitoring system setup and integration.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.monitoring import (
    MonitoringCoordinator,
    MonitoringConfig,
    AlertTemplates,
    AlertSeverity
)


class MonitoringDemo:
    """Demonstration of the monitoring system."""
    
    def __init__(self):
        # Setup configuration
        self.config_manager = ConfigManager()
        
        # Load monitoring configuration
        monitoring_config = MonitoringConfig.get_complete_monitoring_config()
        self.config_manager.config.update(monitoring_config)
        
        # Setup logger
        self.logger = TradingLogger("MonitoringDemo")
        
        # Initialize monitoring coordinator
        self.monitoring_coordinator = MonitoringCoordinator(self.config_manager)
        
    async def setup_monitoring_system(self):
        """Setup and demonstrate the complete monitoring system."""
        try:
            self.logger.info("Setting up comprehensive monitoring system...")
            
            # Initialize monitoring coordinator
            await self.monitoring_coordinator.initialize()
            
            # Configure alerting system with templates
            await self._configure_alerting_system()
            
            # Setup dashboard
            await self._setup_dashboard()
            
            # Simulate system integrations
            await self._simulate_system_integrations()
            
            self.logger.info("Monitoring system setup complete!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring system: {e}")
            return False
    
    async def _configure_alerting_system(self):
        """Configure the alerting system with predefined templates."""
        self.logger.info("Configuring alerting system...")
        
        # Add alert templates
        alert_templates = AlertTemplates.get_all_alert_templates()
        
        for alert_name, alert_config in alert_templates.items():
            await self.monitoring_coordinator.alerting_system.add_alert_rule(
                alert_name, alert_config
            )
            
        self.logger.info(f"Added {len(alert_templates)} alert rules")
    
    async def _setup_dashboard(self):
        """Setup the monitoring dashboard."""
        self.logger.info("Setting up monitoring dashboard...")
        
        # Start dashboard server
        dashboard_url = self.monitoring_coordinator.get_dashboard_url()
        
        self.logger.info(f"Dashboard available at: {dashboard_url}")
    
    async def _simulate_system_integrations(self):
        """Simulate integrations with other system components."""
        self.logger.info("Simulating system integrations...")
        
        # Mock portfolio manager
        class MockPortfolioManager:
            async def get_current_portfolio(self):
                return MockPortfolio()
        
        class MockPortfolio:
            def __init__(self):
                self.total_value = 10000.0
                self.cash = 2000.0
                self.total_pnl = 500.0
                self.positions = [1, 2, 3]  # Mock positions
                self.timestamp = datetime.now()
        
        # Register mock components
        mock_portfolio = MockPortfolioManager()
        self.monitoring_coordinator.register_portfolio_manager(mock_portfolio)
        
        self.logger.info("System integrations simulated")
    
    async def demonstrate_monitoring_features(self):
        """Demonstrate key monitoring features."""
        self.logger.info("Demonstrating monitoring features...")
        
        # 1. Show monitoring status
        status = self.monitoring_coordinator.get_monitoring_status()
        self.logger.info(f"Monitoring Status: {status}")
        
        # 2. Send test alert
        await self.monitoring_coordinator.send_test_alert(AlertSeverity.INFO)
        
        # 3. Simulate performance data
        await self._simulate_performance_data()
        
        # 4. Trigger system health check
        health_summary = self.monitoring_coordinator.system_health_monitor.get_system_summary()
        self.logger.info(f"System Health: {health_summary}")
        
        # 5. Show dashboard data
        self.logger.info(f"Dashboard URL: {self.monitoring_coordinator.get_dashboard_url()}")
        
        self.logger.info("Feature demonstration complete!")
    
    async def _simulate_performance_data(self):
        """Simulate portfolio performance data."""
        try:
            # Create mock portfolio and performance data
            mock_portfolio = {
                'total_value': 9800.0,  # Slightly down to trigger alerts
                'cash': 1800.0,
                'positions': [
                    {'symbol': 'AAPL', 'quantity': 10, 'current_price': 150.0},
                    {'symbol': 'MSFT', 'quantity': 5, 'current_price': 300.0}
                ]
            }
            
            mock_performance = {
                'daily_return': -0.02,  # -2% to demonstrate alerting
                'total_return': 0.05,
                'sharpe_ratio': 0.8,
                'max_drawdown': -0.08,
                'volatility': 0.15
            }
            
            # Update performance monitor
            await self.monitoring_coordinator.update_portfolio_performance(
                mock_portfolio, mock_performance
            )
            
            self.logger.info("Performance data simulated and updated")
            
        except Exception as e:
            self.logger.error(f"Error simulating performance data: {e}")
    
    async def run_monitoring_loop(self, duration_minutes: int = 5):
        """Run monitoring system for specified duration."""
        self.logger.info(f"Running monitoring system for {duration_minutes} minutes...")
        
        end_time = datetime.now().timestamp() + (duration_minutes * 60)
        
        while datetime.now().timestamp() < end_time:
            try:
                # Update system data periodically
                await self._simulate_performance_data()
                
                # Log current status
                status = self.monitoring_coordinator.get_monitoring_status()
                self.logger.info(f"System active - Components: {len(status)} - Time remaining: {int((end_time - datetime.now().timestamp()) / 60)} minutes")
                
                # Wait before next update
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
        
        self.logger.info("Monitoring loop completed")
    
    async def shutdown(self):
        """Shutdown the monitoring system."""
        self.logger.info("Shutting down monitoring system...")
        
        await self.monitoring_coordinator.shutdown()
        
        self.logger.info("Monitoring system shutdown complete")
    
    async def generate_monitoring_report(self) -> str:
        """Generate a comprehensive monitoring report."""
        try:
            # Get monitoring status
            status = self.monitoring_coordinator.get_monitoring_status()
            
            # Get recent alerts
            recent_alerts = self.monitoring_coordinator.alerting_system.get_alert_history(24)
            
            # Get system health
            health_summary = self.monitoring_coordinator.system_health_monitor.get_system_summary()
            
            # Generate report
            report = f"""
# AI Trading System Monitoring Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Status
- Monitoring Active: {status['coordinator']['active']}
- Components Initialized: {status['coordinator']['initialized']}
- Dashboard URL: {self.monitoring_coordinator.get_dashboard_url()}

## Registered Systems
- Portfolio Manager: {status['coordinator']['registered_systems']['portfolio_manager']}
- Risk Manager: {status['coordinator']['registered_systems']['risk_manager']}
- Execution Engine: {status['coordinator']['registered_systems']['execution_engine']}

## System Health Summary
- Overall Status: {health_summary.get('overall_status', 'Unknown')}
- Active Components: {len(health_summary.get('component_health', {}))}
- System Load: {health_summary.get('system_load', 'N/A')}

## Recent Alerts ({len(recent_alerts)})
"""
            
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                report += f"- {alert.timestamp.strftime('%H:%M:%S')} [{alert.severity.value}] {alert.title}\n"
            
            report += f"""

## Performance Monitoring
{status.get('performance_monitor', 'Not available')}

## Dashboard Status
{status.get('dashboard', 'Not available')}

---
Report generated by AI Trading System Monitoring
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating monitoring report: {e}")
            return f"Error generating report: {e}"


async def main():
    """Main demonstration function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demo = MonitoringDemo()
    
    try:
        print("🔧 AI Trading System - Monitoring Demonstration")
        print("=" * 60)
        
        # Setup monitoring system
        print("\n📊 Setting up monitoring system...")
        success = await demo.setup_monitoring_system()
        
        if not success:
            print("❌ Failed to setup monitoring system")
            return
        
        print("✅ Monitoring system setup complete!")
        
        # Demonstrate features
        print("\n🎯 Demonstrating monitoring features...")
        await demo.demonstrate_monitoring_features()
        
        # Generate report
        print("\n📋 Generating monitoring report...")
        report = await demo.generate_monitoring_report()
        print(report)
        
        # Optional: Run monitoring loop
        print("\n⚡ Monitoring system is active!")
        print(f"Dashboard available at: {demo.monitoring_coordinator.get_dashboard_url()}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            await demo.run_monitoring_loop(2)  # Run for 2 minutes
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        
    except Exception as e:
        print(f"❌ Error in monitoring demo: {e}")
    
    finally:
        # Cleanup
        print("\n🧹 Shutting down monitoring system...")
        await demo.shutdown()
        print("✅ Monitoring demo complete!")


if __name__ == "__main__":
    asyncio.run(main())