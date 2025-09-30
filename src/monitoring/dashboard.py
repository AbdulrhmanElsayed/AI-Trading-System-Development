"""
Web Dashboard

Real-time web dashboard for monitoring trading system performance.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import websockets
import weakref

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class DashboardServer:
    """Web dashboard server for real-time monitoring."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("DashboardServer")
        
        # Server configuration
        self.host = config.get('monitoring.dashboard.host', 'localhost')
        self.port = config.get('monitoring.dashboard.port', 8080)
        self.websocket_port = config.get('monitoring.dashboard.websocket_port', 8081)
        
        # Dashboard state
        self.dashboard_data = {
            'portfolio': {},
            'performance': {},
            'positions': [],
            'orders': [],
            'alerts': [],
            'system_health': {},
            'market_data': {},
            'last_update': None
        }
        
        # WebSocket connections for real-time updates
        self.websocket_clients = weakref.WeakSet()
        
        # Data update callbacks
        self.data_callbacks = {}
        
        # Server tasks
        self.server_task = None
        self.websocket_server = None
        
    async def initialize(self):
        """Initialize dashboard server.""" 
        self.logger.info(f"Initializing Dashboard Server on {self.host}:{self.port}")
        
        # Start WebSocket server for real-time updates
        await self._start_websocket_server()
        
        # Start HTTP server (simplified implementation)
        await self._start_http_server()
        
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time data."""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                self.host,
                self.websocket_port
            )
            self.logger.info(f"WebSocket server started on {self.host}:{self.websocket_port}")
            
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket client connections."""
        try:
            self.websocket_clients.add(websocket)
            self.logger.info(f"WebSocket client connected from {websocket.remote_address}")
            
            # Send initial data
            await websocket.send(json.dumps({
                'type': 'initial_data',
                'data': self.dashboard_data
            }))
            
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON message'
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket client disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
    
    async def _handle_websocket_message(self, websocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
                
            elif msg_type == 'subscribe':
                # Subscribe to specific data updates
                subscription = message.get('subscription', [])
                await websocket.send(json.dumps({
                    'type': 'subscribed',
                    'subscription': subscription
                }))
                
            elif msg_type == 'get_data':
                # Request specific data
                data_type = message.get('data_type')
                if data_type in self.dashboard_data:
                    await websocket.send(json.dumps({
                        'type': 'data_response',
                        'data_type': data_type,
                        'data': self.dashboard_data[data_type]
                    }))
                    
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def _start_http_server(self):
        """Start HTTP server (simplified implementation)."""
        # In a full implementation, you would use a proper web framework like FastAPI or Flask
        # This is a simplified version for demonstration
        self.logger.info(f"HTTP server would start on {self.host}:{self.port}")
        self.logger.info("Use a proper web framework like FastAPI for production")
    
    def register_data_callback(self, data_type: str, callback):
        """Register callback for data updates."""
        self.data_callbacks[data_type] = callback
        self.logger.info(f"Registered data callback for: {data_type}")
    
    async def update_portfolio_data(self, portfolio_data: Dict[str, Any]):
        """Update portfolio data."""
        self.dashboard_data['portfolio'] = portfolio_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('portfolio', portfolio_data)
    
    async def update_performance_data(self, performance_data: Dict[str, Any]):
        """Update performance data."""
        self.dashboard_data['performance'] = performance_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('performance', performance_data)
    
    async def update_positions_data(self, positions_data: List[Dict[str, Any]]):
        """Update positions data."""
        self.dashboard_data['positions'] = positions_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('positions', positions_data)
    
    async def update_orders_data(self, orders_data: List[Dict[str, Any]]):
        """Update orders data."""
        self.dashboard_data['orders'] = orders_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('orders', orders_data)
    
    async def update_alerts_data(self, alerts_data: List[Dict[str, Any]]):
        """Update alerts data."""
        self.dashboard_data['alerts'] = alerts_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('alerts', alerts_data)
    
    async def update_system_health_data(self, health_data: Dict[str, Any]):
        """Update system health data."""
        self.dashboard_data['system_health'] = health_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('system_health', health_data)
    
    async def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data."""
        self.dashboard_data['market_data'] = market_data
        self.dashboard_data['last_update'] = datetime.now().isoformat()
        
        await self._broadcast_update('market_data', market_data)
    
    async def _broadcast_update(self, data_type: str, data: Any):
        """Broadcast data update to all WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            'type': 'data_update',
            'data_type': data_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Send to all connected clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.discard(client)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard server status."""
        return {
            'server_running': self.websocket_server is not None,
            'host': self.host,
            'port': self.port,
            'websocket_port': self.websocket_port,
            'connected_clients': len(self.websocket_clients),
            'last_update': self.dashboard_data.get('last_update'),
            'data_types': list(self.dashboard_data.keys())
        }
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard (simplified version)."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading System Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        .dashboard-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .dashboard-header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #4CAF50;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-change {{
            font-size: 14px;
        }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-healthy {{ background-color: #4CAF50; }}
        .status-warning {{ background-color: #ff9800; }}
        .status-error {{ background-color: #f44336; }}
        .charts-container {{
            margin-top: 30px;
        }}
        .chart-placeholder {{
            background-color: #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>AI Trading System Dashboard</h1>
            <p>Real-time monitoring and analytics</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Portfolio Value</div>
                <div class="metric-value" id="portfolio-value">$0.00</div>
                <div class="metric-change" id="portfolio-change">0.00%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Daily P&L</div>
                <div class="metric-value" id="daily-pnl">$0.00</div>
                <div class="metric-change" id="daily-return">0.00%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active Positions</div>
                <div class="metric-value" id="active-positions">0</div>
                <div class="metric-change">positions</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">System Status</div>
                <div class="metric-value">
                    <span class="status-indicator status-healthy" id="system-status-indicator"></span>
                    <span id="system-status-text">Healthy</span>
                </div>
                <div class="metric-change" id="system-uptime">Uptime: 0h</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-placeholder">
                Portfolio Performance Chart
                <br><small>Chart integration would be implemented with libraries like Chart.js or D3.js</small>
            </div>
            
            <div class="chart-placeholder">
                System Resource Usage
                <br><small>Real-time CPU, Memory, and Disk usage charts</small>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:{self.websocket_port}');
        
        ws.onopen = function(event) {{
            console.log('Connected to dashboard WebSocket');
        }};
        
        ws.onmessage = function(event) {{
            const message = JSON.parse(event.data);
            
            if (message.type === 'data_update' || message.type === 'initial_data') {{
                updateDashboard(message.data);
            }}
        }};
        
        ws.onclose = function(event) {{
            console.log('Disconnected from dashboard WebSocket');
            // Implement reconnection logic here
        }};
        
        function updateDashboard(data) {{
            // Update portfolio metrics
            if (data.portfolio) {{
                const portfolio = data.portfolio;
                document.getElementById('portfolio-value').textContent = 
                    formatCurrency(portfolio.current_value || 0);
                document.getElementById('portfolio-change').textContent = 
                    formatPercent(portfolio.total_return || 0);
                document.getElementById('portfolio-change').className = 
                    'metric-change ' + (portfolio.total_return >= 0 ? 'positive' : 'negative');
            }}
            
            // Update performance metrics
            if (data.performance) {{
                const performance = data.performance;
                document.getElementById('daily-pnl').textContent = 
                    formatCurrency(performance.daily_return || 0);
                document.getElementById('daily-return').textContent = 
                    formatPercent(performance.daily_return || 0);
                document.getElementById('daily-return').className = 
                    'metric-change ' + (performance.daily_return >= 0 ? 'positive' : 'negative');
            }}
            
            // Update positions
            if (data.positions) {{
                document.getElementById('active-positions').textContent = data.positions.length;
            }}
            
            // Update system health
            if (data.system_health) {{
                const health = data.system_health;
                const statusElement = document.getElementById('system-status-text');
                const indicatorElement = document.getElementById('system-status-indicator');
                
                statusElement.textContent = health.overall_status || 'Unknown';
                
                // Update status indicator color
                indicatorElement.className = 'status-indicator';
                if (health.overall_status === 'healthy') {{
                    indicatorElement.classList.add('status-healthy');
                }} else if (health.overall_status === 'warning') {{
                    indicatorElement.classList.add('status-warning');
                }} else {{
                    indicatorElement.classList.add('status-error');
                }}
                
                if (health.uptime_hours) {{
                    document.getElementById('system-uptime').textContent = 
                        `Uptime: ${{Math.round(health.uptime_hours)}}h`;
                }}
            }}
        }}
        
        function formatCurrency(value) {{
            return new Intl.NumberFormat('en-US', {{
                style: 'currency',
                currency: 'USD'
            }}).format(value);
        }}
        
        function formatPercent(value) {{
            return new Intl.NumberFormat('en-US', {{
                style: 'percent',
                minimumFractionDigits: 2
            }}).format(value);
        }}
        
        // Send periodic ping to keep connection alive
        setInterval(() => {{
            if (ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: 'ping' }}));
            }}
        }}, 30000);
    </script>
</body>
</html>
        """
    
    async def shutdown(self):
        """Shutdown dashboard server."""
        self.logger.info("Shutting down Dashboard Server")
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Cancel server task
        if self.server_task:
            self.server_task.cancel()
        
        self.logger.info("Dashboard Server shutdown complete")


class DashboardManager:
    """Manages the dashboard and coordinates data updates."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("DashboardManager")
        
        # Dashboard server
        self.dashboard_server = DashboardServer(config)
        
        # Data sources
        self.data_sources = {}
        
        # Update tasks
        self.update_tasks = []
        
    async def initialize(self):
        """Initialize dashboard manager."""
        self.logger.info("Dashboard Manager initialized")
        
        # Initialize dashboard server
        await self.dashboard_server.initialize()
        
        # Start periodic update tasks
        self._start_update_tasks()
    
    def _start_update_tasks(self):
        """Start periodic data update tasks."""
        # Update portfolio data every 10 seconds
        portfolio_task = asyncio.create_task(self._update_portfolio_data_loop())
        self.update_tasks.append(portfolio_task)
        
        # Update system health every 30 seconds
        health_task = asyncio.create_task(self._update_health_data_loop())
        self.update_tasks.append(health_task)
    
    async def _update_portfolio_data_loop(self):
        """Update portfolio data periodically."""
        while True:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                if 'portfolio_manager' in self.data_sources:
                    portfolio_data = await self._get_portfolio_data()
                    await self.dashboard_server.update_portfolio_data(portfolio_data)
                
                if 'performance_monitor' in self.data_sources:
                    performance_data = await self._get_performance_data()
                    await self.dashboard_server.update_performance_data(performance_data)
                
            except Exception as e:
                self.logger.error(f"Error updating portfolio data: {e}")
    
    async def _update_health_data_loop(self):
        """Update system health data periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                if 'system_health' in self.data_sources:
                    health_data = await self._get_health_data()
                    await self.dashboard_server.update_system_health_data(health_data)
                
            except Exception as e:
                self.logger.error(f"Error updating health data: {e}")
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data from portfolio manager."""
        try:
            portfolio_manager = self.data_sources.get('portfolio_manager')
            if not portfolio_manager:
                return {}
            
            # This would integrate with the actual portfolio manager
            return {
                'current_value': 100000.0,  # Placeholder
                'total_return': 0.05,       # Placeholder
                'cash_balance': 10000.0     # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return {}
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data from performance monitor."""
        try:
            performance_monitor = self.data_sources.get('performance_monitor')
            if not performance_monitor:
                return {}
            
            # This would integrate with the actual performance monitor
            return {
                'daily_return': 0.02,     # Placeholder
                'sharpe_ratio': 1.5,      # Placeholder
                'max_drawdown': 0.03      # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance data: {e}")
            return {}
    
    async def _get_health_data(self) -> Dict[str, Any]:
        """Get system health data from health monitor."""
        try:
            health_monitor = self.data_sources.get('system_health')
            if not health_monitor:
                return {}
            
            # This would integrate with the actual health monitor
            return {
                'overall_status': 'healthy',   # Placeholder
                'uptime_hours': 24,           # Placeholder
                'cpu_usage': 45.0,            # Placeholder
                'memory_usage': 60.0          # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health data: {e}")
            return {}
    
    def register_data_source(self, name: str, source):
        """Register a data source for the dashboard."""
        self.data_sources[name] = source
        self.logger.info(f"Registered data source: {name}")
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.dashboard_server.host}:{self.dashboard_server.port}"
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status."""
        return {
            'dashboard_server': self.dashboard_server.get_dashboard_status(),
            'data_sources': list(self.data_sources.keys()),
            'update_tasks': len(self.update_tasks)
        }
    
    async def shutdown(self):
        """Shutdown dashboard manager."""
        self.logger.info("Shutting down Dashboard Manager")
        
        # Cancel update tasks
        for task in self.update_tasks:
            task.cancel()
        
        # Shutdown dashboard server
        await self.dashboard_server.shutdown()
        
        self.logger.info("Dashboard Manager shutdown complete")