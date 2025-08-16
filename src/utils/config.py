"""
Configuration management utilities for RetHyperspaceID.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration manager for RetHyperspaceID."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_path(self, subpath: str = "") -> Path:
        """
        Get data directory path.
        
        Args:
            subpath: Subdirectory within data directory
            
        Returns:
            Full path to data subdirectory
        """
        base_path = Path(self.get('data.raw_dir', 'data/raw')).parent
        return base_path / subpath
    
    def get_artifacts_path(self, subpath: str = "") -> Path:
        """
        Get artifacts directory path.
        
        Args:
            subpath: Subdirectory within artifacts directory
            
        Returns:
            Full path to artifacts subdirectory
        """
        base_path = Path(self.get('data.artifacts_dir', 'artifacts'))
        return base_path / subpath
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Output file path (defaults to original path)
        """
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            keys = key.split('.')
            config = self.config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
