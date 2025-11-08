"""
Configuration module for the Product Recommendation System.

This module handles configuration loading from environment variables and default values.
"""

import os
from typing import Optional

# Default configuration values
DEFAULT_CONFIG = {
    'csv_path': 'home appliance skus lowes.csv',
    'log_level': 'INFO',
    'default_top_k': 10,
    'similar_top_k': 8,
    'window_width': 1000,
    'window_height': 700,
}


def get_config(key: str, default: Optional[str] = None) -> str:
    """
    Get configuration value from environment variable or default.
    
    Args:
        key: Configuration key
        default: Default value if not found
        
    Returns:
        Configuration value as string
    """
    # Try environment variable first
    value = os.getenv(key.upper())
    if value:
        return value
    
    # Try default config
    if key.lower() in DEFAULT_CONFIG:
        return str(DEFAULT_CONFIG[key.lower()])
    
    # Return provided default
    return default or ""


def get_csv_path() -> str:
    """Get CSV file path from config or environment."""
    return get_config('csv_path', DEFAULT_CONFIG['csv_path'])


def get_log_level() -> str:
    """Get log level from config or environment."""
    return get_config('log_level', DEFAULT_CONFIG['log_level'])

