"""
Loki logging integration for centralized log aggregation.
Sends logs directly to Grafana Loki without requiring Promtail.
"""

import os
import logging
import socket
from typing import Optional, Dict, Any

import bittensor as bt

import logging_loki



class ExcludeLoggerFilter(logging.Filter):
    """Filter to exclude certain loggers from being sent to Loki to prevent recursion."""
    
    # Loggers that should be excluded
    EXCLUDED_LOGGERS = [
        'urllib3',
        'requests',
        'logging_loki',
        'httpcore',
        'httpx',
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to exclude log records from these loggers."""
        logger_name = record.name
        
        # Exclude if logger name starts with any excluded prefix
        for excluded in self.EXCLUDED_LOGGERS:
            if logger_name.startswith(excluded):
                return False
        
        return True


class LokiLoggingHandler:
    """Manages Loki logging handler setup and configuration."""
    
    def __init__(
        self,
        loki_url: str,
        validator_name: Optional[str] = None,
        wallet_name: Optional[str] = None,
        hotkey: Optional[str] = None,
        additional_tags: Optional[Dict[str, str]] = None,
    ):
        self.loki_url = loki_url
        self.validator_name = validator_name or socket.gethostname()
        self.wallet_name = wallet_name or "unknown"
        self.hotkey = hotkey or "unknown"
        self.additional_tags = additional_tags or {}
        self.handler: Optional[logging.Handler] = None
    
    def get_tags(self) -> Dict[str, str]:
        """Build tags dict for Loki labels."""
        tags = {
            "job": "cancer_ai_validator",
            "validator": self.validator_name,
            "wallet": self.wallet_name,
            "hotkey": self.hotkey[:16] if len(self.hotkey) > 16 else self.hotkey,
            "hostname": socket.gethostname(),
        }
        tags.update(self.additional_tags)
        return tags
    
    def create_handler(self) -> Optional[logging.Handler]:
        """Create and return a Loki logging handler."""
        
        if not self.loki_url:
            return None
        
        try:
            # push endpoint
            url = self.loki_url.rstrip('/')
            if not url.endswith('/loki/api/v1/push'):
                url = f"{url}/loki/api/v1/push"
            
            self.handler = logging_loki.LokiHandler(
                url=url,
                tags=self.get_tags(),
                version="1",
            )
            
            # Set formatter
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self.handler.setFormatter(formatter)
            
            # Add filter to prevent recursion
            self.handler.addFilter(ExcludeLoggerFilter())
            
            return self.handler
            
        except Exception as e:
            bt.logging.error(f"Failed to create Loki handler: {e}")
            return None


def setup_loki_logging(config: "bt.Config") -> Optional[logging.Handler]:
    """
    Set up Loki logging based on config.
    
    Args:
        config: Bittensor config object with loki settings
        
    Returns:
        Loki handler if successfully created, None otherwise
    """
    # Check if Loki is enabled
    if not config.loki_url:
        return None
    
    # Get validator identification
    validator_name = config.validator_name
    wallet_name = config.wallet.name
    hotkey = config.wallet.hotkey
    
    
    loki_handler = LokiLoggingHandler(
        loki_url=config.loki_url,
        validator_name=validator_name,
        wallet_name=wallet_name,
        hotkey=hotkey,
    )
    
    handler = loki_handler.create_handler()
    
    if handler:
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        bt.logging.info(
            f"Loki logging enabled: {config.loki_url} "
            f"(validator={validator_name or 'auto'}, wallet={wallet_name})"
        )
    
    return handler


