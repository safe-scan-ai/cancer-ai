import os
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timezone

import bittensor as bt
from huggingface_hub import HfApi, hf_hub_url, HfFileSystem

from .manager import SerializableManager
from .exceptions import ModelRunException


@dataclass
class ModelInfo:
    hf_repo_id: str | None = None
    hf_model_filename: str | None = None
    hf_code_filename: str | None = None
    hf_repo_type: str | None = None

    competition_id: str | None = None
    file_path: str | None = None
    model_type: str | None = None


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
        chain_model_date = await self.get_newest_saved_model_date(hotkey)
        if chain_model_date and chain_model_date.tzinfo is None:
            chain_model_date = chain_model_date.replace(tzinfo=timezone.utc)
        if not chain_model_date:
            bt.logging.error(f"Failed to get the newest saved model's date for hotkey {hotkey} from the local DB. Model download skipped.")
            return False

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
            if file['name'] == repo_path:
                # Extract the upload date
                file_date = file["last_commit"]["date"]
                break
        
        if not file_date:
            bt.logging.error(f"File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id}")
            return False
            
        # Ensure file_date is a datetime with timezone
        try:
            if isinstance(file_date, str):
                file_date = datetime.fromisoformat(file_date)
            if file_date.tzinfo is None:
                file_date = file_date.replace(tzinfo=timezone.utc)
        except Exception as e:
            bt.logging.error(f"Failed to parse file date {file_date}: {e}")
            return False
            
        bt.logging.info(f"File {model_info.hf_model_filename} was uploaded on: {file_date}")
        
        # Check if file is newer than our cutoff date
        if file_date < chain_model_date:
            bt.logging.info(f"Downloading model for hotkey {hotkey} because file date {file_date} is older than chain model date {chain_model_date}")
        else:
            bt.logging.info(f"Skipping model for hotkey {hotkey} because file date {file_date} is newer than or equal to chain model date {chain_model_date}")
            return False
        
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

    async def get_newest_saved_model_date(self, hotkey):
        """Fetches the newest saved model's date for a given hotkey from the local database."""
        newest_saved_model = self.db_controller.get_latest_model(hotkey, self.config.models_query_cutoff)
        if not newest_saved_model:
            bt.logging.error(f"Failed to get latest model from local DB for hotkey {hotkey}")
            return None
        return self.db_controller.get_block_timestamp(newest_saved_model.block)


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
