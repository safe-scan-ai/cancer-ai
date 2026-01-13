"""
Dataset encryption and decryption manager for Cancer-AI.
"""
import os
import zipfile
from pathlib import Path
from typing import Optional
import bittensor as bt
from huggingface_hub import hf_hub_download, login

from .hotkey_crypto import decrypt_file_with_hotkey, load_hotkey_from_file


class EncryptedDatasetManager:
    """Manages encrypted dataset download and decryption."""
    
    def __init__(self, 
                 wallet_name: str,
                 hotkey_name: str = "default",
                 hf_repo: str = None,
                 hf_token: Optional[str] = None,
                 cache_dir: str = "./encrypted_datasets"):
        
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.hf_repo = hf_repo or os.getenv("HF_ENCRYPTED_DATASETS_REPO")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get validator's hotkey
        self.hotkey_ss58 = self._get_hotkey_ss58()
        
    def _get_hotkey_ss58(self) -> str:
        """Get validator's SS58 address from hotkey file."""
        try:
            hotkey = load_hotkey_from_file(self.wallet_name, self.hotkey_name)
            return hotkey.ss58_address
        except Exception as e:
            bt.logging.error(f"Failed to load hotkey: {e}")
            return ""
    
    def download_and_decrypt_dataset(self, 
                                     competition_id: str,
                                     dataset_filename: str) -> Path:
        """Download encrypted dataset and decrypt it."""
        if not self.hf_token:
            raise ValueError("HF_TOKEN not set")
        
        if not self.hotkey_ss58:
            raise ValueError("Could not determine hotkey SS58 address")
        
        # Login to HuggingFace
        login(token=self.hf_token)
        
        # Download path
        hf_path = f"{competition_id}/{self.hotkey_ss58}/{dataset_filename}"
        
        bt.logging.info(f"Downloading encrypted dataset: {hf_path}")
        
        try:
            encrypted_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_path,
                repo_type="dataset",
                token=self.hf_token,
                cache_dir=str(self.cache_dir / "downloads")
            )
            bt.logging.info(f"Downloaded to: {encrypted_path}")
        except Exception as e:
            bt.logging.error(f"Failed to download: {e}")
            raise
        
        # Decrypt
        bt.logging.info("Decrypting dataset.")
        try:
            decrypted_path = decrypt_file_with_hotkey(
                encrypted_path,
                self.wallet_name,
                self.hotkey_name
            )
            bt.logging.info(f"Decrypted to: {decrypted_path}")
        except Exception as e:
            bt.logging.error(f"Failed to decrypt: {e}")
            raise
        
        # Extract ZIP
        extract_dir = self.cache_dir / f"{competition_id}_decrypted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        bt.logging.info(f"Extracting to: {extract_dir}")
        try:
            with zipfile.ZipFile(decrypted_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            bt.logging.info("Extraction complete")
        except Exception as e:
            bt.logging.error(f"Failed to extract: {e}")
            raise
        
        return extract_dir