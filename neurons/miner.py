import asyncio
import copy
import time
import os
from pathlib import Path

import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import HfApi, login as hf_login
import huggingface_hub
import onnx
import argparse

from cancer_ai.validator.utils import run_command
from cancer_ai.validator.model_run_manager import ModelRunManager, ModelInfo
from cancer_ai.validator.dataset_manager import DatasetManager
from cancer_ai.validator.competition_manager import COMPETITION_HANDLER_MAPPING

from cancer_ai.base.base_miner import BaseNeuron
from cancer_ai.chain_models_store import ChainMinerModel, ChainModelMetadata
from cancer_ai.utils.config import path_config, add_miner_args
from cancer_ai.validator.utils import get_newest_competition_packages


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

    async def evaluate_model(self) -> None:
        bt.logging.info("Evaluate model mode")

        run_manager = ModelRunManager(
            config=self.config, model=ModelInfo(file_path=self.config.model_path)
        )

        try:
            dataset_packages = await get_newest_competition_packages(self.config)
        except Exception as e:
            bt.logging.error(f"Error retrieving competition packages: {e}")
            return
        
        for package in dataset_packages:
            dataset_manager = DatasetManager(
                self.config,
                self.config.competition_id,
                package["dataset_hf_repo"],
                package["dataset_hf_filename"],
                package["dataset_hf_repo_type"],
                use_auth=False
            )
            await dataset_manager.prepare_dataset()

            X_test, y_test = await dataset_manager.get_data()

            competition_handler = COMPETITION_HANDLER_MAPPING[self.config.competition_id](
                X_test=X_test, y_test=y_test
            )

            y_test = competition_handler.prepare_y_pred(y_test)

            start_time = time.time()
            y_pred = await run_manager.run(X_test)
            run_time_s = time.time() - start_time

            # print(y_pred)
            model_result = competition_handler.get_model_result(y_test, y_pred, run_time_s)
            bt.logging.info(
                f"Evalutaion results:\n{model_result.model_dump_json(indent=4)}"
            )
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

    async def submit_model(self) -> None:
        # Check if the required model and files are present in hugging face repo

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

        print(self.config)

        if not self._check_hf_file_exists(self.config.hf_repo_id, self.config.hf_model_name, self.config.hf_repo_type):
            return

        if not self._check_hf_file_exists(self.config.hf_repo_id, self.config.hf_code_filename, self.config.hf_repo_type):
            return

        # Push model metadata to chain
        model_id = ChainMinerModel(
            competition_id=self.config.competition_id,
            hf_repo_id=self.config.hf_repo_id,
            hf_model_filename=self.config.hf_model_name,
            hf_repo_type=self.config.hf_repo_type,
            hf_code_filename=self.config.hf_code_filename,
            block=None,
        )
        await self.metadata_store.store_model_metadata(model_id)
        bt.logging.success(
            f"Successfully pushed model metadata on chain. Model ID: {model_id}"
        )

    def _check_hf_file_exists(self, repo_id, filename, repo_type):
        if not huggingface_hub.file_exists(repo_id=repo_id, filename=filename, repo_type=repo_type):
            bt.logging.error(f"{filename} not found in Hugging Face repo")
            return False
        return True

    async def main(self) -> None:

        # bt.logging(config=self.config)
        if self.config.action != "submit" and not self.config.model_path:
            bt.logging.error("Missing --model-path argument")
            return
        if self.config.action != "submit" and not MinerManagerCLI.is_onnx_model(
            self.config.model_path
        ):
            bt.logging.error("Provided model with is not in ONNX format")
            return

        match self.config.action:
            case "submit":
                await self.submit_model()
            case "evaluate":
                await self.evaluate_model()
            case "upload":
                await self.compress_code()
                await self.upload_to_hf()
            case _:
                bt.logging.error(f"Unrecognized action: {self.config.action}")


if __name__ == "__main__":
    load_dotenv()
    cli_manager = MinerManagerCLI()
    asyncio.run(cli_manager.main())
