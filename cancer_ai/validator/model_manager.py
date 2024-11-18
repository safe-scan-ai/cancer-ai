import os
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timezone

import bittensor as bt
from huggingface_hub import HfApi

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
        bt.logging.debug(f"Downloading miner model for hotkey {hotkey}: {model_info}")
        
        chain_model_date = await self.get_newest_saved_model_date(hotkey)
        if chain_model_date and chain_model_date.tzinfo is None:
            chain_model_date = chain_model_date.replace(tzinfo=timezone.utc)
        if not chain_model_date:
            bt.logging.error(f"Failed to get the newest saved model's date for hotkey {hotkey} from the local DB. Model download skipped.")
            return False

        try:
            commits = self.api.list_repo_commits(
                repo_id=model_info.hf_repo_id,
                token=self.config.hf_token if hasattr(self.config, "hf_token") else None
            )
            
            model_commit = self.get_commit_with_file_change(
                commits, model_info, chain_model_date
            )

            if model_commit:
                try:
                    self.download_model_at_commit(model_commit, model_info)
                    bt.logging.info(f"Downloaded an older model version for hotkey {hotkey} (date: {model_commit.created_at}).")
                    return True
                except Exception as e:
                    bt.logging.error(f"Failed to download model at commit {model_commit.commit_id}: {e}")
                    return False
            else:
                bt.logging.error(f"No matching or older model found for hotkey {hotkey} based on the saved date. Download skipped.")
                return False

        except Exception as e:
            bt.logging.error(f"Failed to download model: {e}")
            return False

    async def get_newest_saved_model_date(self, hotkey):
        """Fetches the newest saved model's date for a given hotkey from the local database."""
        newest_saved_model = self.db_controller.get_latest_model(hotkey, self.config.models_query_cutoff)
        bt.logging.warning(f"Newest saved model for hotkey {hotkey}: {newest_saved_model.hf_repo_type}")
        if not newest_saved_model:
            bt.logging.error(f"Failed to get latest model from local DB for hotkey {hotkey}")
            return None
        return self.db_controller.get_block_timestamp(newest_saved_model.block)

    def get_commit_with_file_change(self, commits, model_info, chain_model_date):
        """Finds the commit where the specific file exists and matches the date criteria."""
        older_commit = None

        for commit in commits:
            commit_id = commit.commit_id
            commit_date = commit.created_at
            if commit_date.tzinfo is None:
                commit_date = commit_date.replace(tzinfo=timezone.utc)

            if commit_date > chain_model_date:
                bt.logging.debug(f"Skipping commit {commit_id} because it is newer than chain_model_date")
                continue

            # Check if the file exists at this commit
            try:
                files = self.api.list_repo_files(
                    repo_id=model_info.hf_repo_id,
                    revision=commit_id,
                    token=self.config.hf_token if hasattr(self.config, "hf_token") else None,
                )

                if model_info.hf_model_filename in files and (older_commit is None or commit_date > older_commit.created_at):
                    older_commit = commit

            except Exception as e:
                bt.logging.error(f"Failed to list files at commit {commit_id}: {e}")
                continue

        if older_commit:
            bt.logging.info(f"Found an older model version of commit {older_commit.commit_id}")
        else:
            bt.logging.error("No suitable older commit with the required file was found.")
        return older_commit

    
    def download_model_at_commit(self, commit, model_info):
        try:
            model_info.file_path = self.api.hf_hub_download(
                repo_id=model_info.hf_repo_id,
                repo_type=model_info.hf_repo_type,
                filename=model_info.hf_model_filename,
                cache_dir=self.config.models.model_dir,
                revision=commit.commit_id,
                token=self.config.hf_token if hasattr(self.config, "hf_token") else None,
            )
            if not os.path.exists(model_info.file_path):
                bt.logging.error(f"Downloaded file does not exist at {model_info.file_path}")
                raise FileNotFoundError(f"File {model_info.file_path} was not found after download.")
            bt.logging.info(f"Successfully downloaded model file to {model_info.file_path}")
        except Exception as e:
            bt.logging.error(f"Failed to download model file at commit {commit.commit_id}: {e}")
            raise

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
