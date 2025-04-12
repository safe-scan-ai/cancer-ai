import os
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timezone, timedelta

import bittensor as bt
from huggingface_hub import HfApi, HfFileSystem

from .models import ModelInfo
from .manager import SerializableManager
from .exceptions import ModelRunException




class ModelManager(SerializableManager):
    def __init__(self, config, db_controller) -> None:
        self.config = config
        self.db_controller = db_controller

        if not os.path.exists(self.config.models.model_dir):
            os.makedirs(self.config.models.model_dir)
        self.api = HfApi()
        self.hotkey_store: dict[str, ModelInfo] = {}

    def get_state(self):
        return {k: asdict(v) for k, v in self.hotkey_store.items() if is_dataclass(v)}

    def set_state(self, hotkey_models: dict):
        self.hotkey_store = {k: ModelInfo(**v) for k, v in hotkey_models.items()}

    async def download_miner_model(self, hotkey) -> bool:
        """Downloads the newest model from Hugging Face and saves it to disk.
        Returns:
            bool: True if the model was downloaded successfully, False otherwise.
        """
        model_info = self.hotkey_store[hotkey]
        
        if self.config.hf_token:
            fs = HfFileSystem(token=self.config.hf_token)
        else:
            fs = HfFileSystem()
        repo_path = os.path.join(model_info.hf_repo_id, model_info.hf_model_filename)
        
        # List files in the repository and get file date
        try:
            files = fs.ls(model_info.hf_repo_id)
        except Exception as e:
            bt.logging.error(f"Failed to list files in repository {model_info.hf_repo_id}: {e}")
            return False
            
        # Find the specific file and its upload date
        file_date = None
        for file in files:
            if file["name"].endswith(model_info.hf_model_filename):
                # Extract the upload date
                file_date = file["last_commit"]["date"]
                break
        
        if not file_date:
            bt.logging.error(f"File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id}")
            return False
            
        # Parse and check if the model is too recent to download
        is_too_recent, parsed_date = self.is_model_too_recent(file_date, model_info.hf_model_filename, hotkey)
        if is_too_recent:
            return False
        
        file_date = parsed_date
        
        # Download the file
        try:
            model_info.file_path = self.api.hf_hub_download(
                repo_id=model_info.hf_repo_id,
                repo_type=model_info.hf_repo_type,
                filename=model_info.hf_model_filename,
                cache_dir=self.config.models.model_dir,
                token=self.config.hf_token if hasattr(self.config, "hf_token") else None,
            )
        except Exception as e:
            bt.logging.error(f"Failed to download model file: {e}")
            return False

        # Verify the downloaded file exists
        if not os.path.exists(model_info.file_path):
            bt.logging.error(f"Downloaded file does not exist at {model_info.file_path}")
            return False
        
        bt.logging.info(f"Successfully downloaded model file to {model_info.file_path}")
        return True

    def is_model_too_recent(self, file_date, filename, hotkey):
        """Checks if a model file was uploaded too recently based on the cutoff time.
        
        Args:
            file_date: The date when the file was uploaded (string or datetime)
            filename: The name of the model file
            hotkey: The hotkey of the miner
            
        Returns:
            tuple: (is_too_recent, parsed_date) where is_too_recent is a boolean indicating if the model
                  is too recent to download, and parsed_date is the parsed datetime object with timezone
        """
        # Ensure file_date is a datetime with timezone
        try:
            if isinstance(file_date, str):
                file_date = datetime.fromisoformat(file_date)
            if file_date.tzinfo is None:
                file_date = file_date.replace(tzinfo=timezone.utc)
        except Exception as e:
            bt.logging.error(f"Failed to parse file date {file_date}: {e}")
            return True, None

        bt.logging.debug(f"File {filename} was uploaded on: {file_date}")
        
        # Check if file is newer than our cutoff date (uploaded within last X minutes)
        now = datetime.now(timezone.utc)  # Get current time in UTC
        
        # Calculate time difference in minutes
        time_diff = (now - file_date).total_seconds() / 60
        
        if time_diff < self.config.models_query_cutoff:
            bt.logging.warning(f"Skipping model for hotkey {hotkey} because it was uploaded {time_diff:.2f} minutes ago, which is within the cutoff of {self.config.models_query_cutoff} minutes")
            return True, file_date
            
        return False, file_date
        

    def add_model(
        self,
        hotkey,
        hf_repo_id,
        hf_model_filename,
        hf_code_filename=None,
        hf_repo_type=None,
    ) -> None:
        """Saves locally information about a new model."""
        self.hotkey_store[hotkey] = ModelInfo(
            hf_repo_id, hf_model_filename, hf_code_filename, hf_repo_type
        )

    def delete_model(self, hotkey) -> None:
        """Deletes locally information about a model and the corresponding file on disk."""

        bt.logging.info(f"Deleting model: {hotkey}")
        if hotkey in self.hotkey_store and self.hotkey_store[hotkey].file_path:
            os.remove(self.hotkey_store[hotkey].file_path)
        self.hotkey_store[hotkey] = None


    def get_pioneer_models(self, grouped_hotkeys: list[list[str]]) -> list[str]:
        """
        Does a check on whether chain submit date was later then HF commit date. If not slashes.
        Compares chain submit date duplicated models to elect a pioneer based on block of submission (date)
        """
        pioneers = []

        if self.config.hf_token:
            fs = HfFileSystem(token=self.config.hf_token)
        else:
            fs = HfFileSystem()

        for group in grouped_hotkeys:
            candidate_hotkeys = []

            for hotkey in group:
                model_info = self.hotkey_store.get(hotkey)
                if not model_info:
                    bt.logging.error(f"Model info for hotkey {hotkey} not found.")
                    continue

                try:
                    files = fs.ls(model_info.hf_repo_id)
                except Exception as e:
                    bt.logging.error(f"Failed to list files in {model_info.hf_repo_id}: {e}")
                    continue

                file_date_str = None
                for file in files:
                    if file["name"].endswith(model_info.hf_model_filename):
                        file_date_str = file["last_commit"]["date"]
                        break

                if not file_date_str:
                    bt.logging.error(
                        f"File {model_info.hf_model_filename} not found in {model_info.hf_repo_id} for {hotkey}"
                    )
                    continue

                try:
                    if isinstance(file_date_str, datetime):
                        hf_commit_date = file_date_str
                    else:
                        hf_commit_date = datetime.fromisoformat(str(file_date_str))

                    if hf_commit_date.tzinfo is None or hf_commit_date.tzinfo.utcoffset(hf_commit_date) is None:
                        hf_commit_date = hf_commit_date.replace(tzinfo=timezone.utc)
                    else:
                        hf_commit_date = hf_commit_date.astimezone(timezone.utc)

                except Exception as e:
                    bt.logging.error(f"Failed to parse HF commit date {file_date_str} for {hotkey}: {e}")
                    continue

                try:
                    block_timestamp = self.db_controller.get_block_timestamp(model_info.block)
                    block_timestamp = block_timestamp.replace(tzinfo=timezone.utc)
                except Exception as e:
                    bt.logging.error(f"Failed to get block timestamp for {model_info.block}: {e}")
                    continue
                
                if hf_commit_date <= block_timestamp:
                    candidate_hotkeys.append((hotkey, model_info.block))

            if candidate_hotkeys:
                pioneer_hotkey = min(candidate_hotkeys, key=lambda x: x[1])[0]
                pioneers.append(pioneer_hotkey)
        return pioneers
