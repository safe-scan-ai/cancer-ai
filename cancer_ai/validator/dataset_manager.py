import os
import shutil
from pathlib import Path
from typing import List, Tuple
import json

from huggingface_hub import HfApi
import bittensor as bt

import asyncio
from ..utils.hotkey_crypto import decrypt_file_with_hotkey, load_hotkey_from_file
from .manager import SerializableManager
from .exceptions import DatasetManagerException
from .utils import run_command, log_time
from .dataset_handlers.image_csv import DatasetImagesCSV
from ..utils.structured_logger import log

class DatasetManager(SerializableManager):
    def __init__(
        self,
        config,
        competition_id: str,
        hf_repo_id: str,
        hf_filename: str,
        hf_repo_type: str,
        use_auth: bool = True,
        local_fs_mode: bool = False,
    ) -> None:
        """
        Initializes a new instance of the DatasetManager class.

        Args:
            config: The configuration object.
            competition_id (str): The ID of the competition.
            dataset_hf_id (str): The Hugging Face ID of the dataset.
            file_hf_id (str): The Hugging Face ID of the file.

        Returns:
            None
        """
        self.config = config

        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.hf_repo_type = hf_repo_type
        self.competition_id = competition_id
        self.use_auth = use_auth
        self.local_compressed_path = ""
        self.local_extracted_dir = Path(self.config.models.dataset_dir, competition_id)
        self.data: Tuple[List, List] = ()
        self.handler = None
        self.local_fs_mode = local_fs_mode
        self.wallet_name = config.wallet.name
        self.hotkey_name = config.wallet.hotkey
        self.hotkey_ss58 = None  
        self.encrypted_cache_dir = Path(config.neuron.full_path) / "encrypted_datasets"
        self.encrypted_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        return {}
    
    def _get_hotkey_ss58(self) -> str:
        """Get validator's SS58 address from hotkey file."""
        if self.hotkey_ss58:
            return self.hotkey_ss58

        try:
            hotkey = load_hotkey_from_file(self.wallet_name, self.hotkey_name)
            self.hotkey_ss58 = hotkey.ss58_address
            bt.logging.info(f"Loaded validator hotkey: {self.hotkey_ss58[:16]}...")
            return self.hotkey_ss58
        except Exception as e:
            bt.logging.error(f"Failed to load hotkey: {e}")
            raise ValueError(f"Cannot load validator hotkey for decryption: {e}")

    @log_time
    async def download_dataset(self):
        if not os.path.exists(self.local_extracted_dir):
            os.makedirs(self.local_extracted_dir)
        
        # Get validator's hotkey SS58
        hotkey_ss58 = self._get_hotkey_ss58()
        
        # Path format: {competition_id}/{validator_hotkey}/{filename}
        hf_path = f"{self.competition_id}/{hotkey_ss58}/{self.hf_filename}"
        
        bt.logging.info(f"Downloading encrypted dataset.")
        bt.logging.info(f"Repo: {self.hf_repo_id}.")
        bt.logging.info(f"Path: {hf_path}.")
        bt.logging.info(f"Validator: {hotkey_ss58}.")

        try:    
            self.local_compressed_path = await asyncio.to_thread(
                HfApi(token=self.config.hf_token).hf_hub_download,
                self.hf_repo_id,
                hf_path,
                cache_dir=self.encrypted_cache_dir,
                repo_type=self.hf_repo_type,
            )
            bt.logging.info(f"Downloaded encrypted dataset to: {self.local_compressed_path}")
        except Exception as e:
            bt.logging.error(f"Failed to download encrypted dataset: {e}")
            raise
    
    @log_time
    async def decrypt_dataset(self) -> None:
        """Decrypt the downloaded encrypted dataset."""
        if not self.local_compressed_path:
            raise DatasetManagerException("No encrypted dataset downloaded")
        
        bt.logging.info(f"Encrypted file: {self.local_compressed_path}")
        bt.logging.info(f"Using hotkey: {self.hotkey_name}")
        
        try:
            # Decrypt using validator's private key
            decrypted_path = await asyncio.to_thread(
                decrypt_file_with_hotkey,
                self.local_compressed_path,
                self.wallet_name,
                self.hotkey_name
            )
            
            # Update local_compressed_path to point to decrypted file
            self.local_compressed_path = decrypted_path
            bt.logging.info(f"Decrypted to: {decrypted_path}")
            
        except ValueError as e:
            bt.logging.error("Dataset was not encrypted!")

    def delete_dataset(self) -> None:
        """Delete dataset from disk"""

        bt.logging.info("Deleting dataset: ")

        try:
            if not os.access(self.config.models.dataset_dir, os.W_OK):
                bt.logging.error(f"No write permissions for: {self.local_extracted_dir}")
                return

            # Optional: Check if any files are open or being used.
            shutil.rmtree(self.config.models.dataset_dir)
            bt.logging.info("Dataset deleted")
        except OSError as e:
            bt.logging.error(f"Failed to delete dataset from disk: {e}")

    @log_time
    async def unzip_dataset(self) -> None:
        """Unzip dataset"""

        self.local_extracted_dir = Path(
            self.config.models.dataset_dir, self.competition_id
        )
        # delete old unpacked dataset
        if os.path.exists(self.local_extracted_dir):
            os.system(f"chmod -R u+rw {self.local_extracted_dir} && rm -R {self.local_extracted_dir}")

        bt.logging.debug(f"Dataset extracted to: { self.local_compressed_path}")

        # Ensure the extraction directory exists
        os.makedirs(self.local_extracted_dir, exist_ok=True)
        
        # TODO add error handling
        zip_file_path = self.local_compressed_path
        extract_dir = self.local_extracted_dir
        command = f'unzip -o "{zip_file_path}" -d {extract_dir} && chmod -R u+rw {extract_dir}'
        _, err = await run_command(command)
        if err:
            bt.logging.error(f"Error unzipping dataset: {err}")
            raise DatasetManagerException(f"Error unzipping dataset: {err}")
        bt.logging.info("Dataset unzipped")

    def set_dataset_handler(self) -> None:
        """Detect dataset type and set handler"""
        if not self.local_compressed_path:
            raise DatasetManagerException(
                f"Dataset '{self.config.competition_id}' not downloaded"
            )
        
        # Look for CSV file in the extracted directory or its subdirectories
        # Try common names: labels.csv, test.csv, data.csv, etc.
        csv_names = ["labels.csv", "test.csv", "data.csv", "metadata.csv", "dataset.csv"]
        labels_csv_path = None
        dataset_root_dir = None
        
        # Check directly in extracted dir
        for csv_name in csv_names:
            direct_csv_path = Path(self.local_extracted_dir, csv_name)
            if direct_csv_path.exists():
                labels_csv_path = direct_csv_path
                dataset_root_dir = self.local_extracted_dir
                bt.logging.info(f"Found CSV file: {csv_name}")
                break
        
        # If not found, check in subdirectories
        if not labels_csv_path:
            for item in os.listdir(self.local_extracted_dir):
                subdir_path = Path(self.local_extracted_dir, item)
                if subdir_path.is_dir() and not item.startswith('__'):  # Skip __MACOSX etc
                    for csv_name in csv_names:
                        potential_csv = Path(subdir_path, csv_name)
                        if potential_csv.exists():
                            labels_csv_path = potential_csv
                            dataset_root_dir = subdir_path
                            bt.logging.info(f"Found CSV file in subdirectory {subdir_path}: {csv_name}")
                            break
                    if labels_csv_path:
                        break
        
        # If still not found, look for any .csv file
        if not labels_csv_path:
            bt.logging.info("Specific CSV names not found, looking for any .csv file...")
            # Check directly in extracted dir
            for item in os.listdir(self.local_extracted_dir):
                if item.endswith('.csv'):
                    labels_csv_path = Path(self.local_extracted_dir, item)
                    dataset_root_dir = self.local_extracted_dir
                    bt.logging.info(f"Found CSV file: {item}")
                    break
            
            # Check in subdirectories
            if not labels_csv_path:
                for item in os.listdir(self.local_extracted_dir):
                    subdir_path = Path(self.local_extracted_dir, item)
                    if subdir_path.is_dir() and not item.startswith('__'):
                        for subitem in os.listdir(subdir_path):
                            if subitem.endswith('.csv'):
                                labels_csv_path = Path(subdir_path, subitem)
                                dataset_root_dir = subdir_path
                                bt.logging.info(f"Found CSV file in subdirectory {subdir_path}: {subitem}")
                                break
                        if labels_csv_path:
                            break
        
        if labels_csv_path and dataset_root_dir:
            self.handler = DatasetImagesCSV(
                self.config,
                dataset_root_dir,
                labels_csv_path,
            )
        else:
            raise NotImplementedError(f"Dataset handler not implemented - no CSV file found in {self.local_extracted_dir}")

    async def prepare_dataset(self) -> None:
        """Download dataset, unzip and set dataset handler"""
        if self.local_fs_mode:
            self.local_compressed_path = self.hf_filename

            if self.hf_filename.endswith(".encrypted"):
                bt.logging.info(f"Detected encrypted file in local mode. Decrypting '{self.competition_id}'")
                try:
                    decrypted_path = await asyncio.to_thread(
                        decrypt_file_with_hotkey,
                        self.hf_filename,
                        self.wallet_name,
                        self.hotkey_name
                    )
                    # Update path to decrypted file
                    self.local_compressed_path = decrypted_path
                    bt.logging.info(f"Decrypted to: {decrypted_path}")
                except Exception as e:
                    bt.logging.error(f"Failed to decrypt encrypted dataset: {e}")
                    raise DatasetManagerException(f"Failed to decrypt encrypted dataset: {e}")
        else:
            bt.logging.info(f"Downloading dataset '{self.competition_id}'")
            await self.download_dataset()
            bt.logging.info(f"Decrypting dataset '{self.competition_id}'")
            await self.decrypt_dataset()

        bt.logging.info(f"Unzipping dataset '{self.competition_id}'")
        await self.unzip_dataset()
        bt.logging.info(f"Setting dataset handler '{self.competition_id}'")
        self.set_dataset_handler()
        bt.logging.info(f"Preprocessing dataset '{self.competition_id}'")
        self.data = await self.handler.get_training_data()
        
        # Log dataset information
        if self.data and len(self.data) >= 2:
            x_train, y_train = self.data[0], self.data[1]
            import numpy as np
            
            # Calculate class distribution
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            class_distribution = dict(zip(unique_classes.tolist(), class_counts.tolist()))
            
            dataset_info = {
                'hf_repo': self.hf_repo_id,
                'hf_filename': self.hf_filename,
                'total_samples': len(x_train),
                'class_distribution': class_distribution,
                'num_classes': len(unique_classes)
            }
            
            log.dataset.info(f"Dataset loaded: {json.dumps(dataset_info, indent=2)}")

    async def get_data(self) -> Tuple[List, List, List]:
        """Get data from dataset handler"""
        if not self.data:
            raise DatasetManagerException(
                f"Dataset '{self.competition_id}' not initalized "
            )
        # Handle backward compatibility - if data has 2 elements, add empty metadata
        if len(self.data) == 2:
            x_data, y_data = self.data
            metadata = [{'age': None, 'gender': None} for _ in x_data]
            return x_data, y_data, metadata
        return self.data
