"""
Archive node wrapper with fallback support.
Centralizes all archive node operations and fallback logic.
"""

from functools import wraps
from typing import Optional
import argparse
import sys
import bittensor as bt
from retry import retry
from websockets.client import OPEN as WS_OPEN


def _create_subtensor(url: str) -> bt.subtensor:
    """Create a subtensor instance for a specific URL."""
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    
    original_argv = sys.argv
    sys.argv = ['', '--subtensor.chain_endpoint', url, '--subtensor.network', '']
    try:
        config = bt.config(parser=parser)
        config.subtensor.network = None
        return bt.subtensor(config=config)
    finally:
        sys.argv = original_argv


def _create_archive_subtensor_with_fallback(archive_node_url: str, archive_node_fallback_url: str | None = None) -> bt.subtensor:
    """
    Creates an archive subtensor with enhanced fallback support that can switch nodes dynamically.
    """
    
    archive_urls = [url for url in [archive_node_url, archive_node_fallback_url] if url is not None]
    
    # Create the primary subtensor
    primary_subtensor = None
    
    for i, archive_url in enumerate(archive_urls):
        try:
            bt.logging.trace(f"Connecting to archive node: {archive_url}")
            subtensor = _create_subtensor(archive_url)
            subtensor.substrate.connect()
            bt.logging.debug(f"Connected to archive node: {archive_url}")
            
            if i == 0:
                primary_subtensor = subtensor
                primary_subtensor._fallback_urls = archive_urls[1:] if len(archive_urls) > 1 else []
                primary_subtensor._current_url = archive_url
                break
                
        except Exception as e:
            bt.logging.warning(f"Failed to connect to archive node ({archive_url}): {e}")

    if primary_subtensor is None:
        bt.logging.error("Failed to connect any archive nodes")
        raise RuntimeError("Failed to connect to any archive nodes")
    
    # Override the connect method to support dynamic fallback
    original_connect = primary_subtensor.substrate.connect
    
    def enhanced_connect(*args, **kwargs):
        # Try current connection first
        try:
            current = getattr(primary_subtensor.substrate, "ws", None)
            if current is not None and hasattr(current, 'state') and current.state == WS_OPEN:
                return current
        except Exception:
            pass
        
        # Try to reconnect with current URL
        try:
            current_url = getattr(primary_subtensor, '_current_url', 'unknown')
            bt.logging.trace(f"Reconnecting to: {current_url}")
            return original_connect(*args, **kwargs)
        except Exception as e:
            current_url = getattr(primary_subtensor, '_current_url', 'unknown')
            bt.logging.warning(f"Reconnect to {current_url} failed: {e}")
            
            # Try fallback URLs
            for fallback_url in getattr(primary_subtensor, '_fallback_urls', []):
                try:
                    bt.logging.info(f"Fallback to: {fallback_url}")
                    fallback_subtensor = _create_subtensor(fallback_url)
                    fallback_subtensor.substrate.connect()
                    
                    primary_subtensor.substrate = fallback_subtensor.substrate
                    primary_subtensor._current_url = fallback_url
                    bt.logging.info(f"Switched to fallback: {fallback_url}")
                    return fallback_subtensor.substrate.ws
                    
                except Exception as fallback_error:
                    bt.logging.warning(f"Fallback {fallback_url} failed: {fallback_error}")
                    continue
            
            raise RuntimeError("All connection attempts failed")
    
    primary_subtensor.substrate.connect = enhanced_connect
    return primary_subtensor


class ArchiveNodeWrapper:
    """
    Wrapper for archive node operations with automatic fallback support.
    Uses primary and fallback archive nodes from config.
    """
    
    def __init__(self, archive_node_url: str, archive_node_fallback_url: str | None = None):
        self.archive_node_url = archive_node_url
        self.archive_node_fallback_url = archive_node_fallback_url
        self._primary_subtensor = None
        self._fallback_subtensor = None
    
    def _get_primary_subtensor(self) -> bt.subtensor:
        """Get or create primary subtensor connection."""
        if self._primary_subtensor is None:
            self._primary_subtensor = _create_archive_subtensor_with_fallback(self.archive_node_url)
        return self._primary_subtensor
    
    def _get_fallback_subtensor(self) -> bt.subtensor | None:
        """Get or create fallback subtensor connection."""
        if self.archive_node_fallback_url is None:
            return None
        if self._fallback_subtensor is None:
            self._fallback_subtensor = _create_archive_subtensor_with_fallback(self.archive_node_fallback_url)
        return self._fallback_subtensor
    
    @retry(tries=3, delay=1, backoff=2, max_delay=30)
    def get_block_hash(self, block_number: int) -> str:
        """Get block hash with automatic fallback."""
        # Try primary node first
        try:
            block_hash = self._get_primary_subtensor().get_block_hash(block_number)
            if block_hash is not None:
                return block_hash
        except Exception as e:
            bt.logging.warning(f"Primary archive node failed: {e}")
        
        # Try fallback node if available
        fallback_subtensor = self._get_fallback_subtensor()
        if fallback_subtensor is not None:
            try:
                bt.logging.trace(f"Trying fallback node: {self.archive_node_fallback_url}")
                block_hash = fallback_subtensor.get_block_hash(block_number)
                if block_hash is not None:
                    bt.logging.trace(f"Successfully used fallback: {self.archive_node_fallback_url}")
                    return block_hash
            except Exception as e:
                bt.logging.warning(f"Fallback node failed: {e}")
        
        raise ValueError(f"Block hash not found for block number {block_number}")
    
    @retry(tries=3, delay=1, backoff=2, max_delay=30)
    def query_storage(self, module: str, storage_function: str, block_hash: str):
        """Query storage with automatic fallback."""
        # Try primary node first
        try:
            result = self._get_primary_subtensor().substrate.query(
                module=module, storage_function=storage_function, block_hash=block_hash
            )
            if result is not None:
                return result
        except Exception as e:
            bt.logging.warning(f"Primary archive node storage query failed: {e}")
        
        # Try fallback node if available
        fallback_subtensor = self._get_fallback_subtensor()
        if fallback_subtensor is not None:
            try:
                bt.logging.trace(f"Trying storage query on fallback: {self.archive_node_fallback_url}")
                result = fallback_subtensor.substrate.query(
                    module=module, storage_function=storage_function, block_hash=block_hash
                )
                if result is not None:
                    bt.logging.trace(f"Successfully queried storage from fallback")
                    return result
            except Exception as e:
                bt.logging.error(f"Fallback storage query failed: {e}")
        
        raise ValueError("Storage query failed for all archive nodes")


def get_archive_subtensor(archive_node_url: str, archive_node_fallback_url: str | None = None, subtensor: Optional[bt.subtensor] = None) -> bt.subtensor:
    """
    Get an archive subtensor with fallback support.
    If subtensor is provided, it will be enhanced with fallback capabilities.
    If not provided, a new one will be created.
    """
    return _create_archive_subtensor_with_fallback(archive_node_url, archive_node_fallback_url)


class WebSocketManager:
    """
    Manages WebSocket connections with automatic reconnection and fallback support.
    Can be used by any component that needs robust WebSocket handling.
    """
    
    def __init__(self, subtensor: bt.subtensor):
        self.subtensor = subtensor
        self._orig_ws_connect = subtensor.substrate.connect
        subtensor.substrate.connect = self._ws_connect
        
        try:
            ws = subtensor.substrate.connect()
            bt.logging.info(f"[WebSocketManager] Initial WS state: {ws.state}")
        except Exception as e:
            bt.logging.error("Initial WS connect failed: %s", e, exc_info=True)
    
    def _ws_connect(self, *args, **kwargs):
        """
        Replacement for substrate.connect().
        Reuses existing WebSocketClientProtocol if State.OPEN;
        otherwise performs a fresh handshake via original connect().
        """
        # Check current socket
        current = getattr(self.subtensor.substrate, "ws", None)
        if current is not None and current.state == WS_OPEN:
            return current

        # If socket not open, reconnect
        bt.logging.warning("⚠️ Subtensor WebSocket not OPEN—reconnecting…")
        try:
            new_ws = self._connect_ws_with_retry(*args, **kwargs)
        except Exception as e:
            bt.logging.error("Failed to reconnect WebSocket: %s", e, exc_info=True)
            raise

        # Update the substrate.ws attribute so future calls reuse this socket
        setattr(self.subtensor.substrate, "ws", new_ws)
        return new_ws

    @retry(tries=5, delay=0.5, backoff=2, max_delay=5)
    def _connect_ws_with_retry(self, *args, **kwargs):
        return self._orig_ws_connect(*args, **kwargs)
    
    def close(self):
        """Close the WebSocket connection."""
        try:
            bt.logging.debug("Closing WebSocket connection.")
            self.subtensor.substrate.close_websocket()
        except Exception:
            pass
