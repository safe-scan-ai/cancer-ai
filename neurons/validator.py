# The MIT License (MIT)
# Copyright 2023 Yuma Rao
# Copyright 2024 Safe-Scan

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import asyncio
import os
import traceback
import json
import threading
import datetime
import csv
import zipfile
from uuid import uuid4

import bittensor as bt
import numpy as np
import wandb

from cancer_ai.chain_models_store import ChainModelMetadata
from cancer_ai.validator.rewarder import CompetitionResultsStore
from cancer_ai.base.base_validator import BaseValidatorNeuron
from cancer_ai.validator.cancer_ai_logo import cancer_ai_logo
from cancer_ai.validator.utils import (
    fetch_organization_data_references,
    sync_organizations_data_references,
    check_for_new_dataset_files,
    get_local_dataset,
)
from cancer_ai.validator.model_db import ModelDBController
from cancer_ai.validator.competition_manager import CompetitionManager
from cancer_ai.validator.models import OrganizationDataReferenceFactory, NewDatasetFile
from cancer_ai.validator.models import WanDBLogCompetitionWinners, WanDBLogBase, WanDBLogModelErrorEntry
from huggingface_hub import HfApi

BLACKLIST_FILE_PATH = "config/hotkey_blacklist.json"
BLACKLIST_FILE_PATH_TESTNET = "config/hotkey_blacklist_testnet.json"

class Validator(BaseValidatorNeuron):
    
    def __init__(self, config=None, exit_event=None):
        print(cancer_ai_logo)
        super(Validator, self).__init__(config=config)
        self.hotkey = self.wallet.hotkey.ss58_address
        self.db_controller = ModelDBController(db_path=self.config.db_path, subtensor=self.subtensor)

        self.chain_models = ChainModelMetadata(
            self.subtensor, self.config.netuid, self.wallet
        )
        self.last_miners_refresh: float = None
        self.last_monitor_datasets: float = None

        self.hf_api = HfApi()

        self.exit_event = exit_event

    async def concurrent_forward(self):

        coroutines = [
            self.refresh_miners(),
        ]
        if self.config.filesystem_evaluation:
            coroutines.append(self.filesystem_test_evaluation())
        else:
            coroutines.append(self.monitor_datasets())
    
        await asyncio.gather(*coroutines)



    async def refresh_miners(self):
        """
        Downloads miner's models from the chain and stores them in the DB
        """

        if self.last_miners_refresh is not None and (
            time.time() - self.last_miners_refresh
            < self.config.miners_refresh_interval * 60
        ):
            bt.logging.trace("Skipping model refresh, not enough time passed")
            return

        bt.logging.info("Synchronizing miners from the chain")
        bt.logging.info(f"Amount of hotkeys: {len(self.hotkeys)}")

        blacklist_file = (
            BLACKLIST_FILE_PATH_TESTNET
            if self.config.test_mode
            else BLACKLIST_FILE_PATH
        )

        with open(blacklist_file, "r", encoding="utf-8") as f:
            BLACKLISTED_HOTKEYS = json.load(f)
        
        for i, hotkey in enumerate(self.hotkeys):
            if hotkey in BLACKLISTED_HOTKEYS:
                bt.logging.debug(f"Skipping blacklisted hotkey {hotkey}")
                continue

            hotkey = str(hotkey)
            bt.logging.debug(f"Downloading model {i+1}/{len(self.hotkeys)} from hotkey {hotkey}")
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
                chain_model_metadata = await self.chain_models.retrieve_model_metadata(hotkey, uid)
            except Exception as e:
                bt.logging.warning(f"Cannot get miner model for hotkey {hotkey} from the chain: {e}. Skipping.")
                continue

            try:
                self.db_controller.add_model(chain_model_metadata, hotkey)
            except Exception as e:
                # Check if it's a model_hash length constraint error
                if "CHECK constraint failed: LENGTH(model_hash) <= 8" in str(e):
                    bt.logging.error(
                        f"Invalid model hash for hotkey {hotkey}: "
                        f"Hash '{chain_model_metadata.model_hash}' exceeds 8-character limit. "
                        f"Model info will not be persisted to database."
                    )
                else:
                    bt.logging.error(f"An error occured while trying to persist the model info: {e}", exc_info=True)

        self.db_controller.clean_old_records(self.hotkeys)
        self.last_miners_refresh = time.time()
        self.save_state()

    async def filesystem_test_evaluation(self):
        time.sleep(5)
        data_package = get_local_dataset(self.config.local_dataset_dir)
        if not data_package:
            bt.logging.info("No new data packages found.")
            return
        competition_manager = CompetitionManager(
                config=self.config,
                subtensor=self.subtensor,
                hotkeys=self.hotkeys,
                validator_hotkey=self.hotkey,
                competition_id=data_package.competition_id,
                dataset_hf_repo="",
                dataset_hf_filename = data_package.dataset_hf_filename,
                dataset_hf_repo_type="dataset",
                db_controller = self.db_controller,
                test_mode = self.config.test_mode,
                local_fs_mode=True,
            )
        try:
            winning_hotkey, _ = await competition_manager.evaluate()
            if not winning_hotkey:
                bt.logging.error("NO WINNING HOTKEY")
        except Exception as e:
            bt.logging.error(f"Error evaluating {data_package.dataset_hf_filename}: {e}", exc_info=True)
            return

        models_results = competition_manager.results
        
      
        try:
            top_hotkey = self.competition_results_store.get_top_hotkey(data_package.competition_id)
        except ValueError:
            bt.logging.warning(f"No top hotkey available for competition {data_package.competition_id}")
            top_hotkey = None
        
        
        results_file_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{data_package.competition_id}.csv"
        await self.log_results_to_csv(results_file_name, data_package, top_hotkey, models_results)
        if winning_hotkey:
            bt.logging.info(f"Competition result for {data_package.competition_id}: {winning_hotkey}")


        # bt.logging.warning("Competition results store before update")
        # bt.logging.warning(self.competition_results_store.model_dump_json())
        competition_weights = await self.competition_results_store.update_competition_results(data_package.competition_id, models_results, self.config, self.metagraph.hotkeys, self.hf_api, self.db_controller)
        # bt.logging.warning("Competition results store after update")
        # bt.logging.warning(self.competition_results_store.model_dump_json())
        self.update_scores(competition_weights, 0.000001, 0.000002)


    async def monitor_datasets(self):
        """Main validation logic, triggered by new datastes on huggingface"""
        
        if self.last_monitor_datasets is not None and (
                time.time() - self.last_monitor_datasets
                < self.config.monitor_datasets_interval
            ):
            return
        self.last_monitor_datasets = time.time()
        bt.logging.info("Starting monitor_datasets")

        try:
            yaml_data = await fetch_organization_data_references(
                self.config.datasets_config_hf_repo_id,
                self.hf_api,
            )
            await sync_organizations_data_references(yaml_data)
        except Exception as e:
            bt.logging.error(f"Error in monitor_datasets initial setup: {e}\n Stack trace: {traceback.format_exc()}")
            return
        
        self.organizations_data_references = OrganizationDataReferenceFactory.get_instance()
        bt.logging.info("Fetched and synced organization data references")
        
        try:
            data_packages: list[NewDatasetFile] = await check_for_new_dataset_files(self.hf_api, self.org_latest_updates)
        except Exception as e:
            stack_trace = traceback.format_exc()
            bt.logging.error(f"Error checking for new dataset files: {e}\n Stack trace: {stack_trace}")
            return
        
        if not data_packages:
            bt.logging.info("No new data packages found.")
            return
        
        bt.logging.info(f"Found {len(data_packages)} new data packages")
        
        for data_package in data_packages:
            competition_id = data_package.competition_id
            competition_uuid = uuid4().hex
            competition_start_time = datetime.datetime.now()
            bt.logging.info(f"Starting competition for {competition_id}")    
            competition_manager = CompetitionManager(
                config=self.config,
                subtensor=self.subtensor,
                hotkeys=self.hotkeys,
                validator_hotkey=self.hotkey,
                competition_id=competition_id,
                dataset_hf_repo=data_package.dataset_hf_repo,
                dataset_hf_filename=data_package.dataset_hf_filename,
                dataset_hf_repo_type="dataset",
                db_controller = self.db_controller,
                test_mode = self.config.test_mode,
            )

            winning_hotkey = None
            try:
                winning_hotkey, _ = await competition_manager.evaluate()

            except Exception:
                stack_trace = traceback.format_exc()
                bt.logging.error(f"Cannot run {competition_id}: {stack_trace}")
                try:
                    wandb.init(project=competition_id, group="competition_evaluation")
                    error_log = WanDBLogBase(
                        uuid=competition_uuid,
                        log_type="competition_error",
                        competition_id=competition_id,
                        run_time_s=(datetime.datetime.now() - competition_start_time).seconds,
                        validator_hotkey=self.wallet.hotkey.ss58_address,
                        errors=str(stack_trace),
                        dataset_filename=data_package.dataset_hf_filename
                    )
                    wandb.log(error_log.model_dump())
                    wandb.finish()
                except Exception as wandb_error:
                    bt.logging.warning(f"Failed to log to wandb: {wandb_error}")
                continue

            if not winning_hotkey:
                bt.logging.warning("Could not determine the winner of competition")
                continue
            winning_model_link = self.db_controller.get_latest_model(hotkey=winning_hotkey, cutoff_time=self.config.models_query_cutoff).hf_link

            
            # Update competition results
            bt.logging.info(f"Competition result for {competition_id}: {winning_hotkey}")
            competition_weights = await self.competition_results_store.update_competition_results(competition_id, competition_manager.results, self.config, self.metagraph.hotkeys, self.hf_api, self.db_controller)
            self.update_scores(competition_weights, 0.0001, 0.0002)

            average_winning_hotkey = self.competition_results_store.get_top_hotkey(competition_id)
            try:
                winner_log = WanDBLogCompetitionWinners(
                    uuid=competition_uuid,
                    competition_id=competition_id,

                    competition_winning_hotkey=winning_hotkey,
                    competition_winning_uid=self.metagraph.hotkeys.index(winning_hotkey),

                    average_winning_hotkey=average_winning_hotkey,
                    average_winning_uid=self.metagraph.hotkeys.index(average_winning_hotkey),

                    validator_hotkey=self.wallet.hotkey.ss58_address,
                    model_link=winning_model_link,
                    dataset_filename=data_package.dataset_hf_filename,
                    run_time_s=(datetime.datetime.now() - competition_start_time).seconds
                )
                wandb.init(project=competition_id, group="competition_evaluation")
                wandb.log(winner_log.model_dump())
                wandb.finish()
            except Exception as wandb_error:
                bt.logging.warning(f"Failed to log competition winners to wandb: {wandb_error}")

            # log results to CSV
            csv_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{competition_id}.csv"
            await self.log_results_to_csv(csv_filename, data_package, winning_hotkey, competition_manager.results)

            # Logging results
            try:
                wandb.init(project=competition_id, group="model_evaluation")
                wandb_initialized = True
            except Exception as wandb_error:
                bt.logging.warning(f"Failed to initialize wandb for model evaluation: {wandb_error}")
                wandb_initialized = False
                
            for miner_hotkey, evaluation_result in competition_manager.results:
                if miner_hotkey in competition_manager.error_results:
                    continue

                try:
                    model = self.db_controller.get_latest_model(
                        hotkey=miner_hotkey,
                        cutoff_time=self.config.models_query_cutoff,
                    )
                    avg_score = 0.0
                    if (
                        data_package.competition_id in self.competition_results_store.average_scores and 
                        miner_hotkey in self.competition_results_store.average_scores[competition_id]
                    ):
                        avg_score = self.competition_results_store.average_scores[competition_id][miner_hotkey]
                    
                    if wandb_initialized:
                        try:
                            ActualWanDBLogModelEntryClass = competition_manager.competition_handler.WanDBLogModelClass
                            model_log = ActualWanDBLogModelEntryClass(
                                uuid=competition_uuid,
                                competition_id=competition_id,
                                miner_hotkey=miner_hotkey,
                                uid=self.metagraph.hotkeys.index(miner_hotkey),
                                validator_hotkey=self.wallet.hotkey.ss58_address,
                                model_url=model.hf_link,
                                average_score=avg_score,
                                run_time_s=evaluation_result.run_time_s,
                                dataset_filename=data_package.dataset_hf_filename,
                                **evaluation_result.to_log_dict(),
                            )
                            wandb.log(model_log.model_dump())
                        except Exception as wandb_error:
                            bt.logging.warning(f"Failed to log model results to wandb for hotkey {miner_hotkey}: {wandb_error}")
                except Exception as e:
                    bt.logging.error(f"Error processing model results for hotkey {miner_hotkey}: {e}")
                    continue

            # Logging errors
            if wandb_initialized:
                try:
                    for miner_hotkey, error_message in competition_manager.error_results:
                        model_log = WanDBLogModelErrorEntry(
                            uuid=competition_uuid,
                            competition_id=competition_id,
                            miner_hotkey=miner_hotkey,
                            uid=self.metagraph.hotkeys.index(miner_hotkey),
                            validator_hotkey=self.wallet.hotkey.ss58_address,
                            dataset_filename=data_package.dataset_hf_filename,
                            errors=error_message,
                        )
                        wandb.log(model_log.model_dump())
                except Exception as wandb_error:
                    bt.logging.warning(f"Failed to log error results to wandb: {wandb_error}")
            
            # Finish wandb run if it was initialized
            if wandb_initialized:
                try:
                    wandb.finish()
                except Exception as wandb_error:
                    bt.logging.warning(f"Failed to finish wandb run: {wandb_error}")
            
            # Save state only after successful competition evaluation
            # This ensures that org_latest_updates are persisted only for successfully processed packages
            self.save_state()
    

    def update_scores(
        self,
        competition_weights: dict[str, float],
        min_min_score: float,
        max_min_score: float
    ):
        """
        For each competition:
        1) Award the winner its full `weight`.
        2) Linearly spread concrete minimal values in [min_min_score … max_min_score]
            across the other non‐winner hotkeys (highest raw → max_min_score, lowest → min_min_score).
        3) Do NOT multiply those minimal values by the weight—just add them directly.
        """
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        for comp_id, weight in competition_weights.items():
            try:
                winner_hotkey = self.competition_results_store.get_top_hotkey(comp_id)
            except ValueError as e:
                bt.logging.warning(f"[{comp_id}] cannot determine winner: {e}")
                continue

            if winner_hotkey in self.metagraph.hotkeys:
                winner_idx = self.metagraph.hotkeys.index(winner_hotkey)
                self.scores[winner_idx] += weight
                bt.logging.info(
                    f"[{comp_id}] +{weight:.6f} to winner {winner_hotkey}"
                )
            else:
                bt.logging.warning(
                    f"[{comp_id}] winner {winner_hotkey!r} not in metagraph"
                )

            try:
                all_hotkeys = self.competition_results_store.get_hotkeys_with_non_zero_scores(comp_id)
            except ValueError as e:
                bt.logging.warning(f"[{comp_id}] {e}")
                continue

            # remove the winner from the list
            non_winners = [hk for hk in all_hotkeys if hk != winner_hotkey]
            k = len(non_winners)
            if k == 0:
                continue

            # compute the minimal-value sequence:
            # index 0 (highest score) → max_min_score,
            # index k-1 (lowest) → min_min_score
            if k > 1:
                span = max_min_score - min_min_score
                step = span / (k - 1)
                minimal_values = [
                    max_min_score - i * step
                    for i in range(k)
                ]
            else:
                # single runner-up gets the top of the band
                minimal_values = [max_min_score]

            # apply those concrete minimal values (not scaled by weight)
            for minimal, hk in zip(minimal_values, non_winners):
                if hk in self.metagraph.hotkeys:
                    idx = self.metagraph.hotkeys.index(hk)
                    self.scores[idx] += minimal
                    bt.logging.info(
                        f"[{comp_id}] +{minimal:.6f} to non-winner {hk}"
                    )
                else:
                    bt.logging.warning(
                        f"[{comp_id}] non-winner {hk!r} not in metagraph"
                    )

        bt.logging.debug(
            "Scores from update_scores:\n"
            f"{np.array2string(self.scores, precision=7, floatmode='fixed', separator=', ', suppress_small=True)}"
        )

        self.save_state()


    async def log_results_to_csv(self, file_name: str, data_package: NewDatasetFile, top_hotkey: str, models_results: list):
        """Debug method for dumping rewards for testing """
        
        csv_file_path = os.path.join("evaluation-results", file_name)
        bt.logging.info(f"Logging results to CSV for {data_package.competition_id} to file {csv_file_path}")
        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if os.stat(csv_file_path).st_size == 0:
                writer.writerow(["Package name", "Date", "Hotkey", "Score", "Average","Winner"])
            competition_id = data_package.competition_id
            for hotkey, model_result in models_results:
                avg_score = 0.0
                if (competition_id in self.competition_results_store.average_scores and 
                    hotkey in self.competition_results_store.average_scores[competition_id]):
                    avg_score = self.competition_results_store.average_scores[competition_id][hotkey]
                
                if hotkey == top_hotkey:
                    writer.writerow([os.path.basename(data_package.dataset_hf_filename), 
                                    datetime.datetime.now(datetime.timezone.utc), 
                                    hotkey, 
                                    round(model_result.score, 6), 
                                    round(avg_score, 6), 
                                    "X"])
                else:
                    writer.writerow([os.path.basename(data_package.dataset_hf_filename), 
                                    datetime.datetime.now(), 
                                    hotkey, 
                                    round(model_result.score, 6), 
                                    round(avg_score, 6), 
                                    " "])


    # Custom JSON encoder to handle datetime objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            return super().default(obj)
    
    def save_state(self):
        """Saves the state of the validator to a file."""
        if not getattr(self, "organizations_data_references", None):
            self.organizations_data_references = OrganizationDataReferenceFactory.get_instance()
        
        scores_list = self.scores.tolist() if hasattr(self.scores, 'tolist') else []
        hotkeys_list = self.hotkeys.tolist() if hasattr(self.hotkeys, 'tolist') else self.hotkeys
        
        state_dict = {
            'scores': scores_list,
            'hotkeys': hotkeys_list,
            'organizations_data_references': self.organizations_data_references.model_dump(),
            'org_latest_updates': self.org_latest_updates,
            'competition_results_store': self.competition_results_store.model_dump()
        }
        
        state_path = self.config.neuron.full_path + "/state.json"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        try:
            with open(state_path, 'w') as f:
                json.dump(state_dict, f, indent=2, cls=self.DateTimeEncoder)
                f.flush()
                f.close()
        except TypeError as e:
            bt.logging.error(f"Error serializing state to JSON: {e}", exc_info=True)
            for key, value in state_dict.items():
                try:
                    json.dumps(value, cls=self.DateTimeEncoder)
                except TypeError as e:
                    bt.logging.error(f"Problem serializing field '{key}': {e}")
        except Exception as e:
            bt.logging.error(f"Error saving validator state: {e}", exc_info=True)
            if 'f' in locals() and f:
                f.flush()
                f.close()
    
    def create_empty_state(self):
        """Creates an empty state file."""
        empty_state = {
            'scores': [],
            'hotkeys': [],
            'organizations_data_references': self.organizations_data_references.model_dump(),
            'org_latest_updates': {},
            'competition_results_store': self.competition_results_store.model_dump()
        }
        
        state_path = self.config.neuron.full_path + "/state.json"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        with open(state_path, 'w') as f:
            json.dump(empty_state, f, indent=2, cls=self.DateTimeEncoder)
    
    def load_state(self):
        """Loads the state of the validator from a file."""
        json_path = self.config.neuron.full_path + "/state.json"
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    state = json.load(f)
                self._convert_datetime_strings(state)
                self.scores = np.array(state['scores'], dtype=np.float32)
                self.hotkeys = np.array(state['hotkeys'])
                factory = OrganizationDataReferenceFactory.get_instance()
                factory.update_from_dict(state['organizations_data_references'])
                self.organizations_data_references = factory
                self.org_latest_updates = state['org_latest_updates']
                self.competition_results_store = CompetitionResultsStore.model_validate(
                    state['competition_results_store']
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                bt.logging.error(f"Error loading JSON state: {e}")
                if 'f' in locals() and f:
                    f.close()
                    bt.logging.info("Validator state file closed after loading.")
        else:
            bt.logging.warning("No state file found. Creating an empty one.")
            self.create_empty_state()
            return

        if 'f' in locals() and f:
            f.close()

    def _convert_datetime_strings(self, state_dict):
        """Helper method to convert ISO format datetime strings back to datetime objects."""
        if 'org_latest_updates' in state_dict and state_dict['org_latest_updates']:
            for org_id, timestamp in state_dict['org_latest_updates'].items():
                if isinstance(timestamp, str):
                    state_dict['org_latest_updates'][org_id] = datetime.datetime.fromisoformat(timestamp)



    


if __name__ == "__main__":
    bt.logging.info("Setting up main thread interrupt handle.")
    exit_event = threading.Event()
    with Validator(exit_event=exit_event) as validator:
        while True:
            time.sleep(5)
            if exit_event.is_set():
                bt.logging.info("Exit event received. Shutting down...")
                break
