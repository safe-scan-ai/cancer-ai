import os
import shutil
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi
import bittensor as bt

from .manager import SerializableManager
from .utils import run_command, log_time
from .dataset_handlers.image_csv import DatasetImagesCSV
from .exceptions import DatasetManagerException

class DatasetManager(SerializableManager):
    def __init__(
        self,
        config,
        competition_id: str,
        hf_repo_id: str,
        hf_filename: str,
        hf_repo_type: str,
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
        self.local_compressed_path = ""
        self.local_extracted_dir = Path(self.config.models.dataset_dir, competition_id)
        self.data: Tuple[List, List] = ()
        self.handler = None

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        return {}

    @log_time
    async def download_dataset(self):
        if not os.path.exists(self.local_extracted_dir):
            os.makedirs(self.local_extracted_dir)

        self.local_compressed_path = HfApi().hf_hub_download(
            self.hf_repo_id,
            self.hf_filename,
            cache_dir=Path(self.config.models.dataset_dir),
            repo_type=self.hf_repo_type,
            token=self.config.hf_token if hasattr(self.config, "hf_token") else None,
        )

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

        bt.logging.debug(f"Dataset extracted to: { self.local_compressed_path}")
        os.system(f"rm -R {self.local_extracted_dir}")
        # TODO add error handling
        out, err = await run_command(
            f"unzip {self.local_compressed_path} -d {self.local_extracted_dir}"
        )
        if err:
            bt.logging.error(f"Error unzipping dataset: {err}")
            raise DatasetManagerException(f"Error unzipping dataset: {err}")
        bt.logging.info("Dataset unzipped")

    def set_dataset_handler(self) -> None:
        """Detect dataset type and set handler"""
        if not self.local_compressed_path:
            raise DatasetManagerException(
                f"Dataset '{self.config.competition.id}' not downloaded"
            )
        # is csv in directory
        if os.path.exists(Path(self.local_extracted_dir, "labels.csv")):
            self.handler = DatasetImagesCSV(
                self.config,
                self.local_extracted_dir,
                Path(self.local_extracted_dir, "labels.csv"),
            )
        else:
            raise NotImplementedError("Dataset handler not implemented")

    async def prepare_dataset(self) -> None:
        """Download dataset, unzip and set dataset handler"""
        bt.logging.info(f"Downloading dataset '{self.competition_id}'")
        await self.download_dataset()
        bt.logging.info(f"Unzipping dataset '{self.competition_id}'")
        await self.unzip_dataset()
        bt.logging.info(f"Setting dataset handler '{self.competition_id}'")
        self.set_dataset_handler()
        bt.logging.info(f"Preprocessing dataset '{self.competition_id}'")
        self.data = await self.handler.get_training_data()

    async def get_data(self) -> Tuple[List, List]:
        """Get data from dataset handler"""
        if not self.data:
            raise DatasetManagerException(
                f"Dataset '{self.competition_id}' not initalized "
            )
        return self.data
