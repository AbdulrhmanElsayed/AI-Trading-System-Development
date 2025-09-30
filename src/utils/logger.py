"""
Logging Configuration Module

Provides centralized logging setup and utilities for the trading system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = setup_logger(name, level, f"{name.lower()}.log")
    
    def trade_executed(self, symbol: str, action: str, quantity: float, price: float):
        """Log trade execution."""
        self.logger.info(f"TRADE: {action} {quantity} {symbol} @ {price}")
    
    def signal_generated(self, symbol: str, signal_type: str, confidence: float):
        """Log trading signal generation."""
        self.logger.info(f"SIGNAL: {symbol} - {signal_type} (confidence: {confidence:.2f})")
    
    def risk_alert(self, message: str):
        """Log risk management alert."""
        self.logger.warning(f"RISK ALERT: {message}")
    
    def performance_update(self, pnl: float, drawdown: float, sharpe: float):
        """Log performance metrics."""
        self.logger.info(f"PERFORMANCE: PnL={pnl:.2f}, DD={drawdown:.2%}, Sharpe={sharpe:.2f}")
    
    def error(self, message: str):
        """Log error."""
        self.logger.error(message)
    
    def info(self, message: str):
        """Log info."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug."""
        self.logger.debug(message)