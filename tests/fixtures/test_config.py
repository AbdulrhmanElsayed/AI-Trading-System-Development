"""
Test Configuration Manager

Provides test-specific configurations and settings.
"""

import os
from typing import Dict, Any
from pathlib import Path


class TestConfigManager:
    """Configuration manager for test environment."""
    
    def __init__(self):
        self.config = self._load_test_config()
    
    def _load_test_config(self) -> Dict[str, Any]:
        """Load test-specific configuration."""
        return {
            # Database configuration
            'database': {
                'url': 'sqlite:///:memory:',
                'echo': False,
                'pool_size': 1,
                'max_overflow': 0
            },
            
            # Redis configuration (mock)
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 15,  # Use test database
                'decode_responses': True,
                'socket_connect_timeout': 1,
                'socket_timeout': 1
            },
            
            # Data source configuration
            'data_sources': {
                'yahoo_finance': {
                    'enabled': False,  # Disabled in tests
                    'api_key': 'test_key'
                },
                'alpha_vantage': {
                    'enabled': False,
                    'api_key': 'test_key'
                },
                'binance': {
                    'enabled': False,
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            },
            
            # Broker configuration (mock)
            'brokers': {
                'alpaca': {
                    'base_url': 'https://paper-api.alpaca.markets',
                    'api_key': 'test_key',
                    'api_secret': 'test_secret',
                    'paper_trading': True
                },
                'interactive_brokers': {
                    'host': '127.0.0.1',
                    'port': 7497,  # Paper trading port
                    'client_id': 1,
                    'paper_trading': True
                }
            },
            
            # Machine Learning configuration
            'machine_learning': {
                'models_path': str(Path(__file__).parent.parent / 'test_models'),
                'feature_cache_size': 1000,
                'prediction_cache_ttl': 300,
                'model_update_frequency': 3600,
                'ensemble_weights': {
                    'lstm': 0.3,
                    'xgboost': 0.4,
                    'random_forest': 0.3
                }
            },
            
            # Risk management configuration
            'risk_management': {
                'max_position_size': 0.05,  # 5% of portfolio
                'max_daily_loss': 0.02,     # 2% daily loss limit
                'max_drawdown': 0.10,       # 10% max drawdown
                'stop_loss': 0.05,          # 5% stop loss
                'take_profit': 0.10,        # 10% take profit
                'correlation_threshold': 0.7,
                'var_confidence': 0.95,
                'lookback_period': 252
            },
            
            # Portfolio configuration
            'portfolio': {
                'initial_capital': 100000.0,
                'max_positions': 20,
                'rebalance_frequency': 'daily',
                'transaction_cost': 0.001,  # 0.1% transaction cost
                'slippage': 0.0005          # 0.05% slippage
            },
            
            # Execution configuration
            'execution': {
                'order_timeout': 30,        # 30 seconds
                'max_retries': 3,
                'retry_delay': 1,           # 1 second
                'latency_threshold': 1000,  # 1000ms
                'fill_probability': 0.95    # 95% fill probability in tests
            },
            
            # Monitoring configuration
            'monitoring': {
                'metrics_retention_days': 7,  # Shorter retention for tests
                'alert_cooldown': 60,         # 60 seconds
                'dashboard_port': 8081,       # Different port for tests
                'websocket_ping_interval': 10,
                'log_level': 'DEBUG'
            },
            
            # Backtesting configuration
            'backtesting': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'benchmark': 'SPY',
                'commission': 0.001,
                'slippage': 0.0005,
                'initial_cash': 100000.0,
                'lookback_days': 252
            },
            
            # Testing specific settings
            'testing': {
                'fast_mode': True,
                'mock_external_apis': True,
                'generate_test_data': True,
                'test_data_size': 1000,
                'max_test_duration': 300,  # 5 minutes max per test
                'parallel_tests': True,
                'cleanup_after_tests': True
            },
            
            # Logging configuration
            'logging': {
                'level': 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_handler': False,      # No file logging in tests
                'console_handler': True,
                'capture_warnings': True
            },
            
            # Feature engineering configuration
            'features': {
                'technical_indicators': [
                    'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
                    'stochastic', 'williams_r', 'cci', 'atr'
                ],
                'fundamental_indicators': [
                    'pe_ratio', 'pb_ratio', 'dividend_yield', 'market_cap'
                ],
                'sentiment_indicators': [
                    'news_sentiment', 'social_sentiment', 'analyst_ratings'
                ],
                'lookback_periods': [5, 10, 20, 50, 100, 200],
                'feature_selection_method': 'mutual_info',
                'max_features': 100
            }
        }
    
    def get_config(self, section: str = None) -> Any:
        """Get configuration section or entire config."""
        if section:
            return self.config.get(section, {})
        return self.config
    
    def get_database_url(self) -> str:
        """Get database URL for testing."""
        return self.config['database']['url']
    
    def get_test_symbols(self) -> list:
        """Get list of symbols for testing."""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    def get_test_date_range(self) -> tuple:
        """Get test date range."""
        return (
            self.config['backtesting']['start_date'],
            self.config['backtesting']['end_date']
        )
    
    def is_fast_mode(self) -> bool:
        """Check if running in fast test mode."""
        return self.config['testing']['fast_mode']
    
    def should_mock_apis(self) -> bool:
        """Check if external APIs should be mocked."""
        return self.config['testing']['mock_external_apis']
    
    def get_initial_capital(self) -> float:
        """Get initial capital for testing."""
        return self.config['portfolio']['initial_capital']
    
    def get_test_data_size(self) -> int:
        """Get size of test data to generate."""
        return self.config['testing']['test_data_size']
    
    def override_config(self, updates: Dict[str, Any]):
        """Override configuration values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def reset_config(self):
        """Reset configuration to defaults."""
        self.config = self._load_test_config()
    
    def set_environment(self, env: str):
        """Set test environment (unit, integration, performance)."""
        env_configs = {
            'unit': {
                'testing': {'fast_mode': True, 'test_data_size': 100},
                'database': {'url': 'sqlite:///:memory:'}
            },
            'integration': {
                'testing': {'fast_mode': False, 'test_data_size': 1000},
                'database': {'url': 'sqlite:///test_integration.db'}
            },
            'performance': {
                'testing': {'fast_mode': False, 'test_data_size': 10000},
                'database': {'url': 'sqlite:///test_performance.db'}
            }
        }
        
        if env in env_configs:
            self.override_config(env_configs[env])
    
    def create_temp_config_file(self, config_dict: Dict[str, Any] = None) -> Path:
        """Create temporary configuration file for testing."""
        import tempfile
        import json
        
        config_to_write = config_dict or self.config
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_to_write, f, indent=2, default=str)
            return Path(f.name)
    
    def get_mock_api_responses(self) -> Dict[str, Any]:
        """Get mock API response templates."""
        return {
            'yahoo_finance': {
                'quote': {
                    'symbol': 'AAPL',
                    'price': 150.0,
                    'change': 1.5,
                    'change_percent': 1.0,
                    'volume': 1000000,
                    'market_cap': 2500000000000
                },
                'history': {
                    'dates': ['2023-01-01', '2023-01-02', '2023-01-03'],
                    'prices': [148.5, 149.0, 150.0],
                    'volumes': [900000, 950000, 1000000]
                }
            },
            'alpha_vantage': {
                'daily': {
                    'Meta Data': {
                        '2. Symbol': 'AAPL',
                        '3. Last Refreshed': '2023-01-03'
                    },
                    'Time Series (Daily)': {
                        '2023-01-03': {'4. close': '150.0', '5. volume': '1000000'},
                        '2023-01-02': {'4. close': '149.0', '5. volume': '950000'},
                        '2023-01-01': {'4. close': '148.5', '5. volume': '900000'}
                    }
                }
            },
            'binance': {
                'ticker': {
                    'symbol': 'BTCUSDT',
                    'price': '45000.0',
                    'volume': '1000.0'
                },
                'klines': [
                    [1640995200000, '44500.0', '45500.0', '44000.0', '45000.0', '100.0']
                ]
            }
        }