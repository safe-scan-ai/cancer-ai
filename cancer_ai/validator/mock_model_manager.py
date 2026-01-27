"""Mock ModelManager for testing with local ONNX files."""

import os
import asyncio
from typing import Optional
from datetime import datetime, timezone

import bittensor as bt

from .models import ModelInfo
from .exceptions import ModelRunException


class MockModelManager:
    """
    Simple Mock ModelManager for testing with local ONNX files.
    
    Provides stub methods and a list of mock models for evaluation.
    """
    
    def __init__(self, config, db_controller, subtensor=None, parent=None):
        self.config = config
        self.db_controller = db_controller
        self.subtensor = subtensor
        self.parent = parent
        
        # Ensure model directory exists
        if not os.path.exists(self.config.models.model_dir):
            os.makedirs(self.config.models.model_dir)
        
        # Initialize stores
        self.hotkey_store: dict[str, ModelInfo] = {}
        self._setup_mock_models()
    
    def _setup_mock_models(self):
        """Setup mock models using local ONNX files with competition_id filtering."""
        mock_models = [
            {
                "hotkey": "5HNkT3stM0d3lTr1c0rd3r3",
                "competition_id": "tricorder-3", 
                "file_path": "sample_tricorder_3_model.onnx",
                "hf_repo_id": "mock/tricorder3",
                "hf_model_filename": "sample_tricorder_3_model.onnx",
                "hf_code_filename": "tricorder3_code.py",
                "hf_repo_type": "model"
            }
        ]
        
        for model_data in mock_models:
            full_path = os.path.join(os.getcwd(), model_data["file_path"])
            if os.path.exists(full_path):
                model_info = ModelInfo(
                    model_data["hf_repo_id"],
                    model_data["hf_model_filename"],
                    model_data["hf_code_filename"],
                    model_data["hf_repo_type"]
                )
                model_info.file_path = full_path
                model_info.competition_id = model_data["competition_id"]
                model_info.model_size_mb = os.path.getsize(full_path) / (1024 * 1024)
                
                # Store in internal mock database
                self.hotkey_store[model_data["hotkey"]] = model_info
                bt.logging.info(f"Mock model added for hotkey {model_data['hotkey']} "
                              f"(competition: {model_data['competition_id']}): {full_path}")
            else:
                bt.logging.warning(f"Mock model file not found: {full_path}")
    
    def get_latest_models(self, hotkeys: list[str], competition_id: str, _cutoff: int = None) -> dict[str, ModelInfo]:
        """Return mock models filtered by competition_id."""
        if _cutoff is not None:
            bt.logging.debug(f"Mock: cutoff parameter {_cutoff} ignored in mock mode")
        
        bt.logging.info(f"Mock: Getting latest models for competition '{competition_id}' "
                       f"from {len(hotkeys)} hotkeys")
        
        # Convert hotkeys to list if needed
        if not isinstance(hotkeys, list):
            hotkeys_list = list(hotkeys)
        else:
            hotkeys_list = hotkeys
        
        # Add our mock hotkeys to the list
        mock_hotkeys = list(self.hotkey_store.keys())
        all_hotkeys = list(set(hotkeys_list + mock_hotkeys))
        
        latest_models = {}
        for hotkey in all_hotkeys:
            if hotkey in self.hotkey_store:
                model_info = self.hotkey_store[hotkey]
                if model_info.competition_id == competition_id:
                    latest_models[hotkey] = model_info
                    bt.logging.info(f"Mock: Found model for hotkey {hotkey} "
                                  f"in competition {competition_id}")
        
        bt.logging.info(f"Mock: Returning {len(latest_models)} models for competition {competition_id}")
        return latest_models
    
    async def download_miner_model(self, hotkey, token: Optional[str] = None) -> bool:
        return hotkey in self.hotkey_store
    
    async def verify_model_hash(self, hotkey: str):
        return
    
    def delete_model(self, hotkey) -> None:
        if hotkey in self.hotkey_store:
            bt.logging.info(f"Mock deleting model: {hotkey}")
            del self.hotkey_store[hotkey]
    
    def get_pioneer_models(self, grouped_hotkeys: list[list[str]]) -> tuple[list[str], dict[str, str]]:
        return [], {}
    
    def verify_model_hashes(self, hotkeys: list[str]) -> None:
        bt.logging.info("Mock model hash verification - skipping")
        return
    
    async def model_license_valid(self, hotkey: str) -> tuple[bool, str]:
        bt.logging.info(f"Mock license validation for hotkey {hotkey} - always valid")
        return True, "Mock license valid"
