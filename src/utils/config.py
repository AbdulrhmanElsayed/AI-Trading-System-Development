"""
Configuration Management Module

Handles loading and managing system configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self.config = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Replace environment variable placeholders
            self._replace_env_vars(self.config)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """Recursively replace environment variable placeholders."""
        if isinstance(obj, dict):
            return {key: self._replace_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # Extract environment variable name
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)  # Return original if env var not found
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing section configuration
        """
        return self.get(section, {})
    
    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that all required configuration keys are present.
        
        Args:
            required_keys: List of required keys in dot notation
            
        Returns:
            True if all keys present, False otherwise
            
        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = []
        
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return complete configuration as dictionary."""
        return self.config.copy()