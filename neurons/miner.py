import asyncio
import copy
import time
import os
from pathlib import Path


import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import HfApi, login as hf_login
import huggingface_hub
from huggingface_hub import hf_hub_download
import onnx
import argparse
import hashlib

from cancer_ai.validator.utils import run_command
from cancer_ai.validator.model_run_manager import ModelRunManager
from cancer_ai.validator.models import ModelInfo
from cancer_ai.validator.dataset_manager import DatasetManager
from cancer_ai.validator.competition_manager import COMPETITION_HANDLER_MAPPING

from cancer_ai.base.base_miner import BaseNeuron
from cancer_ai.chain_models_store import ChainMinerModel, ChainModelMetadata
from cancer_ai.utils.config import path_config, add_miner_args
from cancer_ai.validator.utils import (
    get_newest_competition_packages,
    chain_miner_to_model_info,
)
from cancer_ai.validator.model_manager import ModelManager
from cancer_ai.validator.model_db import ModelDBController


LICENSE_NOTICE = """
🔒 License Notice:
To share your model for Safe Scan competition, it must be released under the MIT license.

✅ By continuing, you confirm that your model is licensed under the MIT License,
which allows open use, modification, and distribution with attribution.

📤 Make sure your HuggingFace repository has license set to MIT.
"""


class MinerManagerCLI:
    def __init__(self, config=None):

        # setting basic Bittensor objects
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = path_config(self)
        self.config.merge(base_config)
        self.config.logging.debug = True
        BaseNeuron.check_config(self.config)
        bt.logging.set_config(config=self.config.logging)

        self.code_zip_path = None

        self.wallet = None
        self.subtensor = None
        self.metagraph = None
        self.hotkey = None
        self.metadata_store = None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """Method for injecting miner arguments to the parser."""
        add_miner_args(cls, parser)

    async def upload_to_hf(self) -> None:
        """Uploads model and code to Hugging Face."""
        bt.logging.info("Uploading model to Hugging Face.")
        hf_api = HfApi()
        hf_login(token=self.config.hf_token)

        hf_model_path = self.config.hf_model_name
        hf_code_path = self.code_zip_path
        bt.logging.info(f"Model path: {hf_model_path}")
        bt.logging.info(f"Code path: {hf_code_path}")

        path = hf_api.upload_file(
            path_or_fileobj=self.config.model_path,
            path_in_repo=hf_model_path,
            repo_id=self.config.hf_repo_id,
            token=self.config.hf_token,
        )
        bt.logging.info("Uploading code to Hugging Face.")
        path = hf_api.upload_file(
            path_or_fileobj=self.code_zip_path,
            path_in_repo=Path(hf_code_path).name,
            repo_id=self.config.hf_repo_id,
            token=self.config.hf_token,
        )
        bt.logging.info(f"Code uploaded to Hugging Face: {path}")
        bt.logging.info(f"Uploaded model to Hugging Face: {path}")

    @staticmethod
    def is_onnx_model(model_path: str) -> bool:
        """Checks if model is an ONNX model."""
        if not os.path.exists(model_path):
            bt.logging.error("Model file does not exist")
            return False
        try:
            onnx.checker.check_model(model_path)
        except onnx.checker.ValidationError as e:
            bt.logging.warning(e)
            return False
        return True

    async def evaluate_model(self, is_self_check: bool = False) -> None:
        """Unified evaluation function for both local and self-check modes.

        Args:
            is_self_check: If True, downloads model from chain first. If False, uses local model.
        """
        if is_self_check:
            bt.logging.info(
                "Self-check mode: downloading and evaluating model from chain"
            )

            if not self.config.hotkey:
                bt.logging.error("--hotkey argument is required for self-check mode")
                return

            if not self.config.competition_id:
                bt.logging.error(
                    "--competition_id argument is required for self-check mode"
                )
                return

            # Initialize chain connection
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)

            bt.logging.info(f"Checking model for hotkey: {self.config.hotkey}")
            bt.logging.info(f"Competition ID: {self.config.competition_id}")

            # Initialize metadata store and DB controller
            self.metadata_store = ChainModelMetadata(
                subtensor=self.subtensor, netuid=self.config.netuid
            )

            # Set default db_path for self-check mode
            if not hasattr(self.config, "db_path") or self.config.db_path is None:
                self.config.db_path = "/tmp/miner_self_check.db"

            db_controller = ModelDBController(
                subtensor=self.subtensor, config=self.config
            )

            # Get UID for the hotkey
            try:
                uid = self.metagraph.hotkeys.index(self.config.hotkey)
                bt.logging.info(f"✅ Found UID {uid} for hotkey {self.config.hotkey}")
            except ValueError:
                bt.logging.error(f"Hotkey {self.config.hotkey} not found in metagraph")
                return

            # Retrieve model metadata from chain
            try:
                chain_model = await self.metadata_store.retrieve_model_metadata(
                    self.config.hotkey, uid
                )
                bt.logging.info(
                    f"✅ Retrieved model metadata: {chain_model.to_compressed_str()}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to retrieve model metadata from chain: {e}")
                return

            # Check if model belongs to the specified competition
            if chain_model.competition_id != self.config.competition_id:
                bt.logging.error(
                    f"Model competition_id '{chain_model.competition_id}' does not match "
                    f"requested competition_id '{self.config.competition_id}'"
                )
                return

            # Convert chain model to ModelInfo
            model_info = chain_miner_to_model_info(chain_model)
            bt.logging.info(
                f"✅ Model info: {model_info.hf_repo_id}/{model_info.hf_model_filename}"
            )

            # Initialize model manager for downloading
            model_manager = ModelManager(
                config=self.config,
                db_controller=db_controller,
                subtensor=self.subtensor,
            )
            model_manager.hotkey_store[self.config.hotkey] = model_info

            # Download model
            bt.logging.info("Downloading model from HuggingFace...")
            success, error = await model_manager.download_miner_model(
                self.config.hotkey, token=self.config.hf_token
            )
            if not success:
                bt.logging.error(f"Failed to download model: {error}")
                return

            bt.logging.info(
                f"✅ Model downloaded successfully to: {model_info.file_path}"
            )
        else:
            bt.logging.info("Local evaluation mode")
            model_info = ModelInfo(file_path=self.config.model_path)

        # Common evaluation logic for both modes
        try:
            dataset_packages = await get_newest_competition_packages(self.config)
        except Exception as e:
            bt.logging.error(f"Error retrieving competition packages: {e}")
            return

        for package in dataset_packages:
            bt.logging.info(
                f"Evaluating with dataset: {package['dataset_hf_filename']}"
            )

            dataset_manager = DatasetManager(
                self.config,
                self.config.competition_id,
                package["dataset_hf_repo"],
                package["dataset_hf_filename"],
                package["dataset_hf_repo_type"],
                use_auth=False,
            )
            await dataset_manager.prepare_dataset()

            X_test, y_test, metadata = await dataset_manager.get_data()

            # Initialize competition handler
            competition_handler = COMPETITION_HANDLER_MAPPING[self.config.competition_id](
                X_test=X_test, y_test=y_test, metadata=metadata, config=self.config
            )

            # Set preprocessing directory and preprocess data once
            competition_handler.set_preprocessed_data_dir(
                self.config.models.dataset_dir
            )
            await competition_handler.preprocess_and_serialize_data(X_test)

            y_test = competition_handler.prepare_y_pred(y_test)

            start_time = time.time()
            run_manager = ModelRunManager(config=self.config, model=model_info)
            preprocessed_data_gen = (
                competition_handler.get_preprocessed_data_generator()
            )
            y_pred = await run_manager.run(preprocessed_data_gen)
            run_time_s = time.time() - start_time

            # Get model size for efficiency score
            model_size_mb = (
                model_info.model_size_mb
                if hasattr(model_info, "model_size_mb") and model_info.model_size_mb
                else None
            )
            if (
                model_size_mb is None
                and model_info.file_path
                and os.path.exists(model_info.file_path)
            ):
                model_size_bytes = os.path.getsize(model_info.file_path)
                model_size_mb = model_size_bytes / (1024 * 1024)

            model_result = competition_handler.get_model_result(
                y_test, y_pred, run_time_s, model_size_mb
            )

            if is_self_check:
                bt.logging.success("SELF-CHECK EVALUATION RESULTS")
                bt.logging.info(f"Hotkey: {self.config.hotkey}")
                bt.logging.info(f"Competition: {self.config.competition_id}")
                bt.logging.info(
                    f"Model: {model_info.hf_repo_id}/{model_info.hf_model_filename}"
                )
                if model_size_mb:
                    bt.logging.info(f"Model Size: {model_size_mb:.2f} MB")
                bt.logging.success(f"\n{model_result.model_dump_json(indent=4)}")
                bt.logging.success("=" * 60)
            else:
                bt.logging.info(
                    f"✅ Evaluation results:\n{model_result.model_dump_json(indent=4)}"
                )

            # Cleanup preprocessed data
            competition_handler.cleanup_preprocessed_data()

            if self.config.clean_after_run:
                dataset_manager.delete_dataset()

    async def compress_code(self) -> None:
        bt.logging.info("Compressing code")
        bt.logging.info(f"Code directory: {self.config.code_directory}")

        code_dir = Path(self.config.code_directory)
        self.code_zip_path = str(code_dir.parent / f"{code_dir.name}.zip")

        out, err = await run_command(
            f"zip -r {self.code_zip_path} {self.config.code_directory}/*"
        )
        if err:
            bt.logging.info("Error zipping code")
            bt.logging.error(err)
            return
        bt.logging.info(f"Code zip path: {self.code_zip_path}")

    def _compute_model_hash(self, repo_id, model_filename):
        """Compute a 64-character hexadecimal SHA-256 hash of the model file from Hugging Face."""
        try:
            model_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                repo_type="model",
            )
            sha256 = hashlib.sha256()
            with open(model_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            full_hash = sha256.hexdigest()  # SHA-256 gives 64 chars
            bt.logging.info(f"Computed 64-character hash: {full_hash}")
            return full_hash
        except Exception as e:
            bt.logging.error(f"Failed to compute model hash: {e}")
            return None

    async def submit_model(self) -> None:
        # Check if the required model and files are present in hugging face repo
        print(LICENSE_NOTICE)
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.hotkey = self.wallet.hotkey.ss58_address

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

        self.metadata_store = ChainModelMetadata(
            subtensor=self.subtensor, netuid=self.config.netuid, wallet=self.wallet
        )

        if len(self.config.hf_repo_id.encode("utf-8")) > 32:
            bt.logging.error("hf_repo_id must be 32 bytes or less")
            return

        if len(self.config.hf_model_name.encode("utf-8")) > 32:
            bt.logging.error("hf_model_filename must be 32 bytes or less")
            return

        if len(self.config.hf_code_filename.encode("utf-8")) > 31:
            bt.logging.error("hf_code_filename must be 31 bytes or less")
            return

        if not self._check_hf_file_exists(
            self.config.hf_repo_id, self.config.hf_model_name, self.config.hf_repo_type
        ):
            return

        if not self._check_hf_file_exists(
            self.config.hf_repo_id,
            self.config.hf_code_filename,
            self.config.hf_repo_type,
        ):
            return

        if (
            Path(self.config.hf_code_filename).stem
            != Path(self.config.hf_model_name).stem
        ):
            bt.logging.error(
                "hf_model_filename and hf_code_filename must have the same name"
            )
            return

        model_hash = self._compute_model_hash(
            self.config.hf_repo_id, self.config.hf_model_name
        )

        if not model_hash:
            bt.logging.error("Failed to compute model hash")
            return

        # Push model metadata to chain
        model_id = ChainMinerModel(
            competition_id=self.config.competition_id,
            hf_repo_id=self.config.hf_repo_id,
            hf_model_filename=self.config.hf_model_name,
            hf_repo_type="model",
            hf_code_filename=self.config.hf_code_filename,
            block=None,
            model_hash=model_hash,
        )
        await self.metadata_store.store_model_metadata(model_id)
        bt.logging.success(
            f"Successfully pushed model metadata on chain. Model ID: {model_id}"
        )

    def _check_hf_file_exists(self, repo_id, filename, repo_type):
        if not huggingface_hub.file_exists(
            repo_id=repo_id, filename=filename, repo_type=repo_type
        ):
            bt.logging.error(f"{filename} not found in Hugging Face repo")
            return False
        return True

    async def main(self) -> None:

        # bt.logging(config=self.config)

        # Validate action-specific requirements
        if self.config.action == "self-check":
            if not self.config.hotkey:
                bt.logging.error("--hotkey argument is required for self-check action")
                return
            if not self.config.competition_id:
                bt.logging.error(
                    "--competition_id argument is required for self-check action"
                )
                return
        elif self.config.action != "submit":
            if not self.config.model_path:
                bt.logging.error("Missing --model_path argument")
                return
            if not MinerManagerCLI.is_onnx_model(self.config.model_path):
                bt.logging.error("Provided model is not in ONNX format")
                return

        match self.config.action:
            case "submit":
                await self.submit_model()
            case "evaluate":
                await self.evaluate_model(is_self_check=False)
            case "upload":
                await self.compress_code()
                await self.upload_to_hf()
            case "self-check":
                await self.evaluate_model(is_self_check=True)
            case _:
                bt.logging.error(f"Unrecognized action: {self.config.action}")


if __name__ == "__main__":
    load_dotenv()
    cli_manager = MinerManagerCLI()
    asyncio.run(cli_manager.main())
