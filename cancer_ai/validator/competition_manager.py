import time
from typing import List, Tuple

import bittensor as bt
import wandb
import hashlib

from dotenv import load_dotenv

from .manager import SerializableManager
from .model_manager import ModelManager, ModelInfo
from .dataset_manager import DatasetManager
from .model_run_manager import ModelRunManager
from .exceptions import ModelRunException
from .model_db import ModelDBController
from .utils import chain_miner_to_model_info

from .competition_handlers.melanoma_handler import MelanomaCompetitionHandler
from .competition_handlers.base_handler import ModelEvaluationResult
from .tests.mock_data import get_mock_hotkeys_with_models
from cancer_ai.chain_models_store import (
    ChainModelMetadata,
    ChainMinerModel,
)

load_dotenv()

COMPETITION_HANDLER_MAPPING = {
    "melanoma-1": MelanomaCompetitionHandler,
    "melanoma-testnet": MelanomaCompetitionHandler,
    "melanoma-7": MelanomaCompetitionHandler,
    "melanoma-2": MelanomaCompetitionHandler,
    "melanoma-3": MelanomaCompetitionHandler,
}


class ImagePredictionCompetition:
    def score_model(
        self, model_info: ModelInfo, pred_y: List, model_pred_y: List
    ) -> float:
        pass


class CompetitionManager(SerializableManager):
    """
    CompetitionManager is responsible for managing a competition.

    It handles the scoring, model management and synchronization with the chain.
    """

    def __init__(
        self,
        config,
        subtensor: bt.subtensor,
        hotkeys: list[str],
        validator_hotkey: str,
        competition_id: str,
        dataset_hf_repo: str,
        dataset_hf_filename: str,
        dataset_hf_repo_type: str,
        db_controller: ModelDBController,
        test_mode: bool = False,
        local_fs_mode: bool = False,
    ) -> None:
        """
        Responsible for managing a competition.

        Args:
        config (dict): Config dictionary.
        competition_id (str): Unique identifier for the competition.
        """
        bt.logging.trace(f"Initializing Competition: {competition_id}")
        self.config = config
        self.subtensor = subtensor
        self.competition_id = competition_id
        self.results: list[tuple[str, ModelEvaluationResult]] = []
        self.error_results: list[tuple[str, str]] = []
        self.model_manager = ModelManager(self.config, db_controller)
        self.dataset_manager = DatasetManager(
            config=self.config,
            competition_id=competition_id,
            hf_repo_id=dataset_hf_repo,
            hf_filename=dataset_hf_filename,
            hf_repo_type=dataset_hf_repo_type,
            local_fs_mode=local_fs_mode,
        )
        self.chain_model_metadata_store = ChainModelMetadata(
            self.subtensor, self.config.netuid
        )

        self.hotkeys = hotkeys
        self.validator_hotkey = validator_hotkey
        self.db_controller = db_controller
        self.test_mode = test_mode
        self.local_fs_mode = local_fs_mode

    def __repr__(self) -> str:
        return f"CompetitionManager<{self.competition_id}>"

    

    def get_state(self):
        return {
            "competition_id": self.competition_id,
            "model_manager": self.model_manager.get_state(),
        }

    def set_state(self, state: dict):
        self.competition_id = state["competition_id"]
        self.model_manager.set_state(state["model_manager"])

    async def chain_miner_to_model_info(
        self, chain_miner_model: ChainMinerModel
    ) -> ModelInfo:
        if chain_miner_model.competition_id != self.competition_id:
            bt.logging.debug(
                f"Chain miner model {chain_miner_model.to_compressed_str()} does not belong to this competition"
            )
            raise ValueError("Chain miner model does not belong to this competition")
        model_info = ModelInfo(
            hf_repo_id=chain_miner_model.hf_repo_id,
            hf_model_filename=chain_miner_model.hf_model_filename,
            hf_code_filename=chain_miner_model.hf_code_filename,
            hf_repo_type=chain_miner_model.hf_repo_type,
            competition_id=chain_miner_model.competition_id,
            block=chain_miner_model.block,
            model_hash=chain_miner_model.model_hash,
        )
        return model_info

    async def get_mock_miner_models(self):
        """Get registered mineres from testnet subnet 163"""
        self.model_manager.hotkey_store = get_mock_hotkeys_with_models()

    async def update_miner_models(self):
        """
        Updates hotkeys and downloads information of models from the chain
        """
        bt.logging.info("Selecting models for competition")
        bt.logging.info(f"Amount of hotkeys: {len(self.hotkeys)}")

        latest_models = self.db_controller.get_latest_models(
            self.hotkeys, self.competition_id, self.config.models_query_cutoff
        )
        for hotkey, model in latest_models.items():
            model_info = chain_miner_to_model_info(model)
            if model_info.competition_id != self.competition_id:
                bt.logging.warning(
                    f"Miner {hotkey} with competition id {model.competition_id} does not belong to {self.competition_id} competition, skipping"
                )
                continue
            self.model_manager.hotkey_store[hotkey] = model_info
        bt.logging.info(
            f"Amount of hotkeys with valid models: {len(self.model_manager.hotkey_store)}"
        )

    async def evaluate(self) -> Tuple[str | None, ModelEvaluationResult | None]:
        """Returns hotkey and competition id of winning model miner"""
        bt.logging.info(f"Start of evaluation of {self.competition_id}")
        
        hotkeys_to_slash = []
        # TODO add mock models functionality

        await self.update_miner_models()
        if len(self.model_manager.hotkey_store) == 0:
            bt.logging.error("No models to evaluate")
            return None, None

        
        await self.dataset_manager.prepare_dataset()
        X_test, y_test = await self.dataset_manager.get_data()

        competition_handler = COMPETITION_HANDLER_MAPPING[self.competition_id](
            X_test=X_test, y_test=y_test
        )
        y_test = competition_handler.prepare_y_pred(y_test)
        evaluation_counter = 0 
        models_amount = len(self.model_manager.hotkey_store.items())
        bt.logging.info(f"Evaluating {models_amount} models")

        for miner_hotkey, model_info in self.model_manager.hotkey_store.items():
            evaluation_counter +=1
            bt.logging.info(f"Evaluating {evaluation_counter}/{models_amount} hotkey: {miner_hotkey}")
            model_downloaded = await self.model_manager.download_miner_model(miner_hotkey)
            if not model_downloaded:
                bt.logging.error(
                    f"Failed to download model for hotkey {miner_hotkey}  Skipping."
                )
                self.error_results.append(miner_hotkey, "Failed to download model")
                continue

            computed_hash = self._compute_model_hash(model_info.file_path)
            if not computed_hash:
                bt.logging.info("Could not determine model hash. Skipping.")
                self.error_results.append(miner_hotkey, "Could not determine model hash")
                continue
        
            if computed_hash != model_info.model_hash:
                bt.logging.info(f"The hash of model uploaded by {miner_hotkey} does not match hash of model submitted on-chain. Slashing.")
                self.error_results.append(miner_hotkey, "The hash of model uploaded does not match hash of model submitted on-chain")
                hotkeys_to_slash.append(miner_hotkey)

            model_manager = ModelRunManager(
                self.config, self.model_manager.hotkey_store[miner_hotkey]
            )
            start_time = time.time()

            try:
                y_pred = await model_manager.run(X_test)
            except ModelRunException as e:
                bt.logging.error(
                    f"Model hotkey: {miner_hotkey} failed to run. Skipping. error: {e}"
                )
                self.error_results.append(miner_hotkey, f"Failed to run model: {e}")
                continue

            try:
                model_result = competition_handler.get_model_result(
                    y_test, y_pred, time.time() - start_time
                )
                self.results.append((miner_hotkey, model_result))
            except Exception as e:
                bt.logging.error(
                    f"Error evaluating model for hotkey: {miner_hotkey}. Error: {str(e)}"
                )
                self.error_results.append(miner_hotkey, f"Error evaluating model: {e}")
                bt.logging.info(f"Skipping model {miner_hotkey} due to evaluation error. error: {e}")

        if len(self.results) == 0:
            bt.logging.error("No models were able to run")
            return None, None
        
        # see if there are any duplicate scores, slash the copied models owners
        grouped_duplicated_hotkeys = self.group_duplicate_scores(hotkeys_to_slash)
        bt.logging.info(f"duplicated models: {grouped_duplicated_hotkeys}")
        if len(grouped_duplicated_hotkeys) > 0:
            pioneer_models_hotkeys = self.model_manager.get_pioneer_models(grouped_duplicated_hotkeys)
            hotkeys_to_slash.extend([hotkey for group in grouped_duplicated_hotkeys for hotkey in group if hotkey not in pioneer_models_hotkeys])
            self.slash_model_copiers(hotkeys_to_slash)
        
        winning_hotkey, winning_model_result = sorted(
            self.results, key=lambda x: x[1].score, reverse=True
        )[0]

        for miner_hotkey, model_result in self.results:
            bt.logging.info(f"Model from {miner_hotkey} successfully evaluated")
            bt.logging.trace(
                f"Model result for {miner_hotkey}:\n {model_result.model_dump_json(indent=4)} \n"
            )

        bt.logging.info(
            f"Winning hotkey for competition {self.competition_id}: {winning_hotkey}"
        )
        self.dataset_manager.delete_dataset()
        return winning_hotkey, winning_model_result


    def group_duplicate_scores(self, hotkeys_to_slash: list[str]) -> list[list[str]]:
        """
        Groups hotkeys for models with identical scores.
        """
        score_to_hotkeys = {}
        for hotkey, result in self.results:
            if hotkey in hotkeys_to_slash:
                continue
            score = round(result.score, 6)
            if score in score_to_hotkeys and score > 0.0:
                score_to_hotkeys[score].append(hotkey)
            else:
                score_to_hotkeys[score] = [hotkey]
        
        grouped_duplicates = [hotkeys for hotkeys in score_to_hotkeys.values() if len(hotkeys) > 1]
        return grouped_duplicates
    
    def slash_model_copiers(self, hotkeys_to_slash: list[str]):
        for hotkey, result in self.results:
            if hotkey in hotkeys_to_slash:
                bt.logging.info(f"Slashing model copier for hotkey: {hotkey} (setting score to 0.0)")
                result.score = 0.0

    def _compute_model_hash(self, file_path) -> str:
        """Compute an 8-character hexadecimal SHA-1 hash of the model file."""
        sha1 = hashlib.sha1()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha1.update(chunk)
            full_hash = sha1.hexdigest()
            truncated_hash = full_hash[:8]
            bt.logging.info(f"Computed 8-character hash: {truncated_hash}")
            return truncated_hash
        except Exception as e:
            bt.logging.error(f"Error computing hash for {file_path}: {e}")
            return None