"""
AI Trading System Main Entry Point

This is the main application entry point that orchestrates all trading system components.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.data.market_data_manager import MarketDataManager
from src.ml.model_manager import ModelManager
from src.risk.risk_manager import RiskManager
from src.execution.execution_engine import ExecutionEngine
from src.monitoring.system_monitor import SystemMonitor


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the trading system."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger("TradingSystem", self.config.get('system.log_level', 'INFO'))
        
        # Core components
        self.market_data_manager = None
        self.model_manager = None
        self.risk_manager = None
        self.execution_engine = None
        self.system_monitor = None
        
        # System state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing AI Trading System...")
            
            # Initialize market data manager
            self.logger.info("Initializing market data manager...")
            self.market_data_manager = MarketDataManager(self.config)
            await self.market_data_manager.initialize()
            
            # Initialize ML model manager
            self.logger.info("Initializing ML model manager...")
            self.model_manager = ModelManager(self.config)
            await self.model_manager.initialize()
            
            # Initialize risk manager
            self.logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(self.config)
            await self.risk_manager.initialize()
            
            # Initialize execution engine
            self.logger.info("Initializing execution engine...")
            self.execution_engine = ExecutionEngine(self.config)
            await self.execution_engine.initialize()
            
            # Initialize system monitor
            self.logger.info("Initializing system monitor...")
            self.system_monitor = SystemMonitor(self.config)
            await self.system_monitor.initialize()
            
            self.logger.info("All components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def start(self):
        """Start the trading system."""
        try:
            self.logger.info("Starting AI Trading System...")
            
            # Start all components
            tasks = []
            
            if self.market_data_manager:
                tasks.append(asyncio.create_task(self.market_data_manager.start()))
            
            if self.model_manager:
                tasks.append(asyncio.create_task(self.model_manager.start()))
            
            if self.risk_manager:
                tasks.append(asyncio.create_task(self.risk_manager.start()))
            
            if self.execution_engine:
                tasks.append(asyncio.create_task(self.execution_engine.start()))
            
            if self.system_monitor:
                tasks.append(asyncio.create_task(self.system_monitor.start()))
            
            # Start main trading loop
            tasks.append(asyncio.create_task(self._main_trading_loop()))
            
            self.is_running = True
            self.logger.info("Trading system started successfully!")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in trading system: {e}")
            raise
    
    async def _main_trading_loop(self):
        """Main trading loop that coordinates all components."""
        self.logger.info("Starting main trading loop...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get latest market data
                market_data = await self.market_data_manager.get_latest_data()
                
                if market_data:
                    # Generate trading signals using ML models
                    signals = await self.model_manager.generate_signals(market_data)
                    
                    # Apply risk management filters
                    filtered_signals = await self.risk_manager.filter_signals(signals)
                    
                    # Execute approved trades
                    if filtered_signals:
                        await self.execution_engine.execute_signals(filtered_signals)
                    
                    # Update system metrics
                    await self.system_monitor.update_metrics()
                
                # Sleep before next iteration
                await asyncio.sleep(self.config.get('trading.trading_frequency_seconds', 60))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                # Continue running but log the error
                await asyncio.sleep(10)  # Brief pause before retry
    
    async def shutdown(self):
        """Graceful shutdown of the trading system."""
        if self.is_running:
            self.logger.info("Shutting down trading system...")
            self.is_running = False
            
            # Stop all components in reverse order
            if self.execution_engine:
                await self.execution_engine.stop()
            
            if self.risk_manager:
                await self.risk_manager.stop()
            
            if self.model_manager:
                await self.model_manager.stop()
            
            if self.market_data_manager:
                await self.market_data_manager.stop()
            
            if self.system_monitor:
                await self.system_monitor.stop()
            
            self.shutdown_event.set()
            self.logger.info("Trading system shutdown complete.")


async def main():
    """Main application entry point."""
    system = None
    try:
        # Initialize and start the trading system
        system = TradingSystem()
        await system.initialize()
        await system.start()
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1
    finally:
        if system:
            await system.shutdown()
    
    return 0


if __name__ == "__main__":
    # Run the main application
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)