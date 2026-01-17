import time
from typing import List, Tuple, Optional
from datetime import datetime

import bittensor as bt
import hashlib

from dotenv import load_dotenv

from cancer_ai.utils.structured_logger import log, LogCategory

from .manager import SerializableManager
from .model_manager import ModelManager
from .mock_model_manager import MockModelManager
from .dataset_manager import DatasetManager
from .model_run_manager import ModelRunManager
from .exceptions import ModelRunException
from .model_db import ModelDBController
from .utils import chain_miner_to_model_info
from .models import ModelInfo

from .competition_handlers.base_handler import BaseCompetitionHandler, BaseModelEvaluationResult
from .competition_handlers.melanoma_handler import MelanomaCompetitionHandler
from .competition_handlers.tricorder_2_handler import Tricorder2CompetitionHandler
from .competition_handlers.tricorder_3_handler import Tricorder3CompetitionHandler
from .tests.mock_data import get_mock_hotkeys_with_models
from cancer_ai.chain_models_store import (
    ChainModelMetadata,
    ChainMinerModel,
)

load_dotenv()

COMPETITION_HANDLER_MAPPING = {
    "melanoma-1": MelanomaCompetitionHandler,
    "melanoma-testnet": MelanomaCompetitionHandler,
    "melanoma-testnet2": MelanomaCompetitionHandler,
    "melanoma-7": MelanomaCompetitionHandler,
    "melanoma-2": MelanomaCompetitionHandler,
    "melanoma-3": MelanomaCompetitionHandler,
    "tricorder-2": Tricorder2CompetitionHandler,
    "tricorder-3": Tricorder3CompetitionHandler,
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
        dataset_release_date: Optional[datetime] = None,
        test_mode: bool = False,
        local_fs_mode: bool = False,
    ) -> None:
        """
        Responsible for managing a competition.

        Args:
        config (dict): Config dictionary.
        competition_id (str): Unique identifier for the competition.
        """
        log.trace(f"Initializing Competition: {competition_id}")
        self.config = config
        self.subtensor = subtensor
        self.competition_id = competition_id
        self.results: list[tuple[str, BaseModelEvaluationResult]] = []
        self.error_results: list[tuple[str, str]] = [] 
        # Initialize model manager based on config
        if hasattr(config, 'mock_models') and config.mock_models:
            self.model_manager = MockModelManager(config, db_controller, subtensor, self)
        else:
            self.model_manager = ModelManager(config, db_controller, subtensor, self)
            self.model_manager.dataset_release_date = dataset_release_date
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
        self.dataset_release_date = dataset_release_date
        self.dataset_hf_repo = dataset_hf_repo
        self.dataset_hf_filename = dataset_hf_filename
        self.competition_handler: Optional[BaseCompetitionHandler] = None

    def __repr__(self) -> str:
        return f"CompetitionManager<{self.competition_id}>"

    def _create_error_result(self, error_message: str) -> BaseModelEvaluationResult:
        """Create an error result using the competition-specific result class."""
        if self.competition_id.startswith("melanoma"):
            from .competition_handlers.melanoma_handler import MelanomaEvaluationResult
            return MelanomaEvaluationResult(score=0.0, error=error_message)
        elif self.competition_id.startswith("tricorder"):
            from .competition_handlers.tricorder_common import TricorderEvaluationResult
            return TricorderEvaluationResult(score=0.0, error=error_message)
        else:
            # Fallback: create base class but ensure it has to_log_dict method
            result = BaseModelEvaluationResult(score=0.0, error=error_message)
            if not hasattr(result, 'to_log_dict'):
                # Add the method if it's missing (shouldn't happen with proper inheritance)
                result.to_log_dict = lambda: {
                    "score": result.score,
                    "error": result.error,
                    "run_time_s": result.run_time_s,
                    "tested_entries": result.tested_entries,
                    "predictions_raw": result.predictions_raw,
                }
            return result


    async def chain_miner_to_model_info(
        self, chain_miner_model: ChainMinerModel
    ) -> ModelInfo:
        if chain_miner_model.competition_id != self.competition_id:
            log.debug(
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
        log.info("Selecting models for competition")
        log.info(f"Amount of hotkeys: {len(self.hotkeys)}")

        latest_models = self.db_controller.get_latest_models(
            self.hotkeys, self.competition_id, self.config.models_query_cutoff
        )
        if hasattr(self.config, 'mock_models') and self.config.mock_models:
            for hotkey, model_info in latest_models.items():
                self.model_manager.hotkey_store[hotkey] = model_info
            log.info(
                f"Amount of hotkeys with valid mock models: {len(self.model_manager.hotkey_store)}"
            )
            return
        for hotkey, model in latest_models.items():
            model_info = chain_miner_to_model_info(model)
            if model_info.competition_id != self.competition_id:
                log.warning(
                    f"Miner {hotkey} with competition id {model.competition_id} does not belong to {self.competition_id} competition, skipping"
                )
                continue
            self.model_manager.hotkey_store[hotkey] = model_info
        log.info(
            f"Amount of hotkeys with valid models: {len(self.model_manager.hotkey_store)}"
        )

    async def _evaluate_single_model(
        self, 
        miner_hotkey: str, 
        model_info, 
        y_test, 
        evaluation_counter: int,
        models_amount: int
    ) -> tuple[BaseModelEvaluationResult | None, bool]:
        """
        Evaluate a single model and return the result and whether it should be slashed.
        
        Args:
            miner_hotkey: The hotkey of the miner
            model_info: Model information object
            y_test: Test labels
            evaluation_counter: Current evaluation number for logging
            models_amount: Total number of models for logging
            
        Returns:
            tuple of (model_result or None, should_slash)
        """
        log.set_competition(self.competition_id)
        log.set_competition_action("evaluate")
        log.set_dataset(self.dataset_hf_repo, self.dataset_hf_filename)
        log.set_miner_hotkey(miner_hotkey)

        log.info("======== START EVALUATION ========")
        log.info(f"Evaluating {evaluation_counter}/{models_amount} hotkey: {miner_hotkey}")
        log.info("======== MODEL DOWNLOAD ========")
        
        # Download model
        model_downloaded, download_error = await self.model_manager.download_miner_model(miner_hotkey, token=self.config.hf_token)
        if not model_downloaded:
            error_msg = download_error or "Failed to download model"
            log.error(f"Failed to download model for hotkey {miner_hotkey}: {error_msg}. Skipping.")
            return self._create_error_result(error_msg), False

        # Verify hash
        computed_hash = self._compute_model_hash(model_info.file_path)
        if not computed_hash:
            log.info("Could not determine model hash. Skipping.")
            return self._create_error_result("Could not determine model hash"), False
    
        if computed_hash != model_info.model_hash:
            hf_info = f"{model_info.hf_repo_id}/{model_info.hf_model_filename}" if model_info.hf_repo_id and model_info.hf_model_filename else "unknown"
            log.info(f"The hash of model uploaded by {miner_hotkey} ({hf_info}) does not match hash of model submitted on-chain. Slashing.")
            return self._create_error_result("The hash of model uploaded does not match hash of model submitted on-chain"), True

        # Run inference
        inference_start_time = time.time()
        model_manager = None

        try:
            model_manager = ModelRunManager(
                self.config, self.model_manager.hotkey_store[miner_hotkey]
            )
            # Get fresh preprocessed data generator for each model
            preprocessed_data_gen = self.competition_handler.get_preprocessed_data_generator()
            log.info("======== MODEL INFERENCE ========")
            y_pred = await model_manager.run(preprocessed_data_gen)
        except ModelRunException as e:
            log.competition.error(f"Model hotkey: {miner_hotkey} failed to run. Skipping. error: {e}")
            return self._create_error_result(f"Failed to run model: {e}"), False
        finally:
            log.info("======== CLEANUP ========")
            # Clean up
            if model_manager and hasattr(model_manager, 'handler') and hasattr(model_manager.handler, 'cleanup'):
                try:
                    model_manager.handler.cleanup()
                except Exception:
                    pass
            if model_manager:
                del model_manager

        # Evaluate results
        total_time = time.time() - inference_start_time
        try:
            model_result = self.competition_handler.get_model_result(
                y_test, y_pred, total_time, model_info.model_size_mb
            )
            return model_result, False
        except Exception as e:
            log.competition.error(f"Error evaluating model for hotkey: {miner_hotkey}. Error: {str(e)}", exc_info=True)
            log.competition.info(f"Skipping model {miner_hotkey} due to evaluation error. error: {e}")
            return self._create_error_result(f"Error evaluating model: {e}"), False

    async def evaluate(self) -> Tuple[str | None, BaseModelEvaluationResult | None]:
        """Returns hotkey and competition id of winning model miner"""
        log.set_competition(self.competition_id)
        log.set_competition_action("evaluate")
        log.info(f"Start of evaluation of {self.competition_id}")
        
        hotkeys_to_slash = []
        # TODO add mock models functionality

        await self.update_miner_models()
        if len(self.model_manager.hotkey_store) == 0:
            log.error("No models to evaluate")
            return None, None

        
        await self.dataset_manager.prepare_dataset()
        X_test, y_test, metadata = await self.dataset_manager.get_data()

        # Pass metadata to tricorder handlers, otherwise use default parameters
        if self.competition_id in ["tricorder-3", "tricorder-2"]:
            self.competition_handler: BaseCompetitionHandler = COMPETITION_HANDLER_MAPPING[self.competition_id](
                X_test=X_test, y_test=y_test, metadata=metadata, config=self.config
            )
        else:
            self.competition_handler: BaseCompetitionHandler = COMPETITION_HANDLER_MAPPING[self.competition_id](
                X_test=X_test, y_test=y_test, config=self.config
            )
        
        # Set preprocessing directory and preprocess data once
        self.competition_handler.set_preprocessed_data_dir(self.config.models.dataset_dir)
        await self.competition_handler.preprocess_and_serialize_data(X_test)
        
        y_test = self.competition_handler.prepare_y_pred(y_test)
        evaluation_counter = 0 
        models_amount = len(self.model_manager.hotkey_store.items())
        log.info(f"Evaluating {models_amount} models")

        # Local testing whitelist
        WHITELIST_HOTKEYS = {
        }

        for miner_hotkey, model_info in self.model_manager.hotkey_store.items():
            # Only evaluate whitelisted hotkeys or all if whitelist is empty
            if WHITELIST_HOTKEYS and miner_hotkey not in WHITELIST_HOTKEYS:
                continue
            
            evaluation_counter += 1
            model_result, should_slash = await self._evaluate_single_model(
                miner_hotkey, model_info, y_test, evaluation_counter, models_amount
            )
            
            if model_result is not None:
                self.results.append((miner_hotkey, model_result))
            
            if should_slash:
                hotkeys_to_slash.append(miner_hotkey)

        if len(self.results) == 0:
            log.error("No models were able to run")
            return None, None
        
        # Validate final scores - check for suspicious zero scores

        log.info("======== VALIDATING FINAL SCORES ========")
        
        if self.competition_id in ["tricorder-3", "tricorder-2"]:
            zero_score_models = []
            for model_id, result in self.results:
                if result.score == 0.0 and not result.error:
                    # Zero score but no error - this is suspicious
                    zero_score_models.append({
                        "model_id": model_id,
                        "accuracy": getattr(result, 'accuracy', None),
                        "weighted_f1": getattr(result, 'weighted_f1', None),
                        "efficiency_score": getattr(result, 'efficiency_score', None),
                        "error": result.error
                    })
            
            if zero_score_models:
                log.warning(f"Found {len(zero_score_models)} models with zero score but no error!")
                for model_info in zero_score_models:
                    log.warning(f"Zero score model: {model_info}")
                    log.warning(f"This may indicate a scoring calculation bug!")
        
        log.info("======== PROCESSING DUPLICATE DETECTION ========")
        # Process duplicate detection separately from other slashing reasons
        duplicate_hotkeys_to_slash = self._process_duplicate_detection(hotkeys_to_slash)
        
        log.info(f"======== DUPLICATE HOTKEYS TO SLASH: {duplicate_hotkeys_to_slash} ========")
        
        # Combine all hotkeys to slash (from hash mismatches and duplicate detection)
        all_hotkeys_to_slash = hotkeys_to_slash + duplicate_hotkeys_to_slash
        
        # Final deduplication to prevent multiple slashing entries
        all_hotkeys_to_slash = list(set(all_hotkeys_to_slash))
        log.info(f"======== ALL HOTKEYS TO SLASH: {all_hotkeys_to_slash} ========")
        for hotkey in all_hotkeys_to_slash:
            log.set_miner_hotkey(hotkey)
            log.info(f"Slashing model copier: {hotkey}")
            log.clear_miner_hotkey()
        self.slash_model_copiers(all_hotkeys_to_slash)
        
        winning_hotkey, winning_model_result = sorted(
            self.results, key=lambda x: x[1].score, reverse=True
        )[0]
        
        log.info(f"======== WINNING MODEL: {winning_hotkey} with score {winning_model_result.score} ========")

        log.info(
            f"Winning hotkey for competition {self.competition_id}: {winning_hotkey}"
        )
        
        # Cleanup preprocessed data
        self.competition_handler.cleanup_preprocessed_data()
        self.dataset_manager.delete_dataset()
        return winning_hotkey, winning_model_result




    def group_duplicate_scores(self, hotkeys_to_slash: list[str]) -> list[list[str]]:
        """
        Groups hotkeys for models whose full evaluation‐metric tuple is identical.
        """
        metrics_to_hotkeys: dict[tuple, list[str]] = {}

        for hotkey, result in self.results:
            if hotkey in hotkeys_to_slash:
                continue
            
            # Skip models with score 0.0 from duplicate detection
            if result.score == 0.0:
                continue
            
            comparable_result = self.competition_handler.get_comparable_result(result)
            metrics_to_hotkeys.setdefault(comparable_result, []).append(hotkey)

        return [group for group in metrics_to_hotkeys.values() if len(group) > 1]

    
    def slash_model_copiers(self, hotkeys_to_slash: list[str]):
        for hotkey, result in self.results:
            if hotkey in hotkeys_to_slash:
                # Get HF repo info for logging
                model_info = self.model_manager.hotkey_store.get(hotkey)
                hf_info = f"{model_info.hf_repo_id}/{model_info.hf_model_filename}" if model_info and model_info.hf_repo_id and model_info.hf_model_filename else "unknown"
                
                log.set_miner_hotkey(hotkey)
                log.info(f"Slashing model copier for hotkey: {hotkey} ({hf_info}) - setting score to 0.0")
                log.clear_miner_hotkey()
                result.score = 0.0
                if not result.error:
                    result.error = "Slashing model copier - setting score to 0.0"

    def _compute_model_hash(self, file_path) -> str:
        """Compute a 64-character hexadecimal SHA-256 hash of the model file."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
                    sha256.hexdigest()
            full_hash = sha256.hexdigest()
            truncated_hash = full_hash[:8]
            log.info(f"Computed 8-character hash: {truncated_hash}")
            return truncated_hash
        except Exception as e:
            log.competition.error(f"Error computing hash for {file_path}: {e}", exc_info=True)
            return None
    
    def _process_duplicate_detection(self, hotkeys_to_slash: list) -> list:
        """Process duplicate detection and return list of hotkeys to slash for copying."""
        duplicate_hotkeys_to_slash = []
        
        # see if there are any duplicate scores, slash the copied models owners
        grouped_duplicated_hotkeys = self.group_duplicate_scores(hotkeys_to_slash)
        log.info(f"duplicated models: {grouped_duplicated_hotkeys}")
        if len(grouped_duplicated_hotkeys) > 0:
            pioneer_models_hotkeys, validation_errors = self.model_manager.get_pioneer_models(grouped_duplicated_hotkeys)
            
            # Apply validation errors to results
            if validation_errors:
                for hotkey, error_msg in validation_errors.items():
                    for h, result in self.results:
                        if h == hotkey:
                            result.error = error_msg
                            result.score = 0.0
                            log.info(f"Setting validation error for {hotkey}: {error_msg}")
            
            # Log pioneer vs copies for better traceability while keeping slashing logs unchanged
            for group in grouped_duplicated_hotkeys:
                pioneer_hotkey = None
                for hotkey in group:
                    if hotkey in pioneer_models_hotkeys:
                        pioneer_hotkey = hotkey
                        break
                if pioneer_hotkey is not None:
                    copies = [hotkey for hotkey in group if hotkey != pioneer_hotkey]
                    # Get HF repo info for pioneer and copies
                    pioneer_info = self.model_manager.hotkey_store.get(pioneer_hotkey)
                    pioneer_hf = f"{pioneer_info.hf_repo_id}/{pioneer_info.hf_model_filename}" if pioneer_info and pioneer_info.hf_repo_id and pioneer_info.hf_model_filename else "unknown"
                    
                    copy_hf_info = []
                    for copy_hotkey in copies:
                        copy_info = self.model_manager.hotkey_store.get(copy_hotkey)
                        copy_hf = f"{copy_info.hf_repo_id}/{copy_info.hf_model_filename}" if copy_info and copy_info.hf_repo_id and copy_info.hf_model_filename else "unknown"
                        copy_hf_info.append(f"{copy_hotkey} ({copy_hf})")
                    
                    for copy_hotkey in copies:
                        log.set_miner_hotkey(copy_hotkey)
                        log.info(
                            f"Duplicate prediction-signature group - pioneer: {pioneer_hotkey} ({pioneer_hf}), copies: {copy_hf_info}"
                        )
                        log.clear_miner_hotkey()
            duplicate_hotkeys_to_slash = [hotkey for group in grouped_duplicated_hotkeys for hotkey in group if hotkey not in pioneer_models_hotkeys]
        
        return duplicate_hotkeys_to_slash
