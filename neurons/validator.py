# -*- coding: utf-8 -*-
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
from cancer_ai.validator.rewarder import CompetitionResultsStore, SCORE_DISTRIBUTION
from cancer_ai.base.base_validator import BaseValidatorNeuron
from cancer_ai.validator.cancer_ai_logo import cancer_ai_logo
from cancer_ai.utils.structured_logger import log
from cancer_ai.validator.utils import (
    check_for_new_dataset_files,
    get_local_dataset,
)
from cancer_ai.validator.model_db import ModelDBController
from cancer_ai.validator.competition_manager import CompetitionManager
from cancer_ai.validator.models import OrganizationDataReferenceFactory, NewDatasetFile
from cancer_ai.validator.models import WanDBLogBase
from cancer_ai.validator.validator_helpers import setup_organization_data_references
from huggingface_hub import HfApi

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
            # self.refresh_miners(),
        ]
        if self.config.filesystem_evaluation:
            coroutines.append(self.filesystem_test_evaluation())
        else:
            coroutines.append(self.monitor_datasets())
    
        await asyncio.gather(*coroutines)



    async def refresh_miners(self):
        """Downloads miner's models from the chain and stores them in the DB."""
        from cancer_ai.validator.validator_helpers import (
            should_refresh_miners, load_blacklisted_hotkeys, process_miner_models
        )
        
        if not should_refresh_miners(self.last_miners_refresh, self.config.miners_refresh_interval):
            return

        log.validation.info(f"Synchronizing {len(self.hotkeys)} miners from the chain")
        #bt.logging.info(f"Synchronizing {len(self.hotkeys)} miners from the chain")
        blacklisted_hotkeys = load_blacklisted_hotkeys(self.config.test_mode)
        
        await process_miner_models(self, blacklisted_hotkeys)
        
        self.db_controller.clean_old_records(self.hotkeys)
        self.last_miners_refresh = time.time()
        self.save_state()

    async def filesystem_test_evaluation(self):
        time.sleep(5)
        data_package = get_local_dataset(self.config.local_dataset_dir)
        if not data_package:
            log.competition.info("No new data packages found.")
            #bt.logging.info("No new data packages found.")
            return
        
        await self.run_competition_for_data_package(data_package, local_mode=True)

    
    async def monitor_datasets(self) -> None:
        """Main validation logic, triggered by new datasets being pushed to huggingface repositories for evaluation"""
        
        if self.last_monitor_datasets is not None and (
                time.time() - self.last_monitor_datasets
                < self.config.monitor_datasets_interval
            ):
            return
        self.last_monitor_datasets = time.time()
        log.competition.info("Starting monitor_datasets")
        #bt.logging.info("Starting monitor_datasets")

        # Setup organization data references (cached) - always refresh to ensure latest data
        try:
            self.organizations_data_references = await setup_organization_data_references(self)
        except Exception as e:
            log.dataset.error(f"Error in monitor_datasets initial setup: {e}\n Stack trace: {traceback.format_exc()}")
            #bt.logging.error(f"Error in monitor_datasets initial setup: {e}\n Stack trace: {traceback.format_exc()}")
            return
        
        # Initialize org_latest_updates with all organization IDs if not present or empty
        if not hasattr(self, 'org_latest_updates') or self.org_latest_updates is None:
            self.org_latest_updates = {}
        
        # Ensure org_latest_updates contains all current organizations
        for org in self.organizations_data_references.organizations:
            if org.organization_id not in self.org_latest_updates:
                self.org_latest_updates[org.organization_id] = None
        
        log.dataset.info(f"org_latest_updates contains {len(self.org_latest_updates)} organizations")
        #bt.logging.info(f"org_latest_updates contains {len(self.org_latest_updates)} organizations")
        
        # Check for new dataset files
        try:
            log.dataset.info(f"Checking for new datasets with org_latest_updates: {self.org_latest_updates}")
            #bt.logging.info(f"Checking for new datasets with org_latest_updates: {self.org_latest_updates}")
            data_packages = await check_for_new_dataset_files(self.hf_api, self.org_latest_updates)
        except Exception as e:
            log.dataset.error(f"Error checking for new dataset files: {e}\n Stack trace: {traceback.format_exc()}")
            #bt.logging.error(f"Error checking for new dataset files: {e}\n Stack trace: {traceback.format_exc()}")
            return
        
        if not data_packages:
            log.dataset.info("No new data packages found")
            #bt.logging.info("No new data packages found")
            return
        
        log.competition.info(f"Processing {len(data_packages)} new data packages")
        #bt.logging.info(f"Processing {len(data_packages)} new data packages")
        
        # Refresh miners to get latest models before competition evaluation
        log.validation.info("Refreshing miners for new competition evaluation")
        #bt.logging.info("Refreshing miners for new competition evaluation")
        await self.refresh_miners()
        for data_package in data_packages:
            await self.run_competition_for_data_package(data_package)

        
    async def run_competition_for_data_package(self, data_package: NewDatasetFile, local_mode: bool = False):
        competition_id = data_package.competition_id
        log.set_competition(competition_id)
        log.competition.info(f"====== STARTING COMPETITION: {data_package.dataset_hf_filename} {'[LOCAL]' if local_mode else '[REMOTE]'} ======")
        #bt.logging.info(f"====== STARTING COMPETITION: {data_package.dataset_hf_filename} ({competition_id}) {'[LOCAL]' if local_mode else '[REMOTE]'} ======")
        competition_uuid = uuid4().hex
        competition_start_time = datetime.datetime.now()
        
        # Configure CompetitionManager based on mode
        competition_manager = CompetitionManager(
            config=self.config,
            subtensor=self.subtensor,
            hotkeys=self.hotkeys,
            validator_hotkey=self.hotkey,
            competition_id=competition_id,
            dataset_hf_repo="" if local_mode else data_package.dataset_hf_repo,
            dataset_hf_filename=data_package.dataset_hf_filename,
            dataset_hf_repo_type="dataset",
            db_controller = self.db_controller,
            test_mode = self.config.test_mode,
            local_fs_mode=local_mode,
        )

        winning_hotkey = None
        try:
            winning_hotkey, _ = await competition_manager.evaluate()
        except Exception:
            stack_trace = traceback.format_exc()
            log.competition.error(f"Cannot run competition: {stack_trace}")
            #bt.logging.error(f"Cannot run {competition_id}: {stack_trace}")
            try:
                if not self.config.wandb.off:
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
                log.wandb.warn(f"Failed to log to wandb: {wandb_error}")
                #bt.logging.warning(f"Failed to log to wandb: {wandb_error}")
            log.clear_competition()
            return

        if not winning_hotkey:
            log.competition.warn("Could not determine the winner of competition")
            log.clear_competition()
            #bt.logging.warning("Could not determine the winner of competition")
            return
        
        # Process competition results
        await self.process_competition_results(
            competition_id, competition_uuid, data_package, competition_manager,
            winning_hotkey, competition_start_time
        )

        log.competition.info("Successfully completed competition")
        log.competition.info("====== COMPETITION FINISHED ======")
        log.competition.info("==================================")
        log.clear_competition()
        #bt.logging.info(f"Successfully completed competition for {competition_id}")
        #bt.logging.info("====== COMPETITION FINISHED ======")
        #bt.logging.info("==================================")

    async def process_competition_results(
        self,
        competition_id: str,
        competition_uuid: str,
        data_package: NewDatasetFile,
        competition_manager: CompetitionManager,
        winning_hotkey: str,
        competition_start_time: datetime.datetime
    ):
        """Process and log competition results."""
        log.set_competition(competition_id)
        log.statistics.info(f"Competition result: {winning_hotkey}")
        log.clear_competition()
        #bt.logging.info(f"Competition result for {competition_id}: {winning_hotkey}")
        
        # Get winning model link
        winning_model_link = self.db_controller.get_latest_model(hotkey=winning_hotkey, cutoff_time=self.config.models_query_cutoff).hf_link
        
        # Update competition results
        competition_weights = await self.competition_results_store.update_competition_results(
            competition_id, competition_manager.results, self.config, self.metagraph.hotkeys, self.hf_api, self.db_controller
        )
        self.update_scores(competition_weights, 0.0001, 0.0002)

        average_winning_hotkey = self.competition_results_store.get_top_hotkey(competition_id)
        
        # log results to CSV
        csv_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{competition_id}.csv"
        await self.log_results_to_csv(csv_filename, data_package, winning_hotkey, competition_manager.results)

        # Log evaluation results to wandb/local files
        from cancer_ai.utils.wandb_local import log_evaluation_results
        await log_evaluation_results(
            self, competition_id, competition_uuid, data_package, competition_manager,
            winning_hotkey, winning_model_link, average_winning_hotkey, competition_start_time
        )
        
        # Save state only after successful competition evaluation
        self.save_state()

    
    def update_scores(
        self,
        competition_weights: dict[str, float],
        min_min_score: float,
        max_min_score: float
    ):
        """
        Distribute rewards to top 10 miners according to SCORE_DISTRIBUTION.
        All other miners get minimal scores.
        """
        
        
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        for comp_id, weight in competition_weights.items():
            log.set_competition(comp_id)
            try:
                all_hotkeys = self.competition_results_store.get_hotkeys_with_non_zero_scores(comp_id)
            except ValueError as e:
                log.statistics.warn(f"{e}")
                log.clear_competition()
                #bt.logging.warning(f"[{comp_id}] {e}")
                continue

            if not all_hotkeys:
                log.statistics.warn("No hotkeys with non-zero scores")
                log.clear_competition()
                #bt.logging.warning(f"[{comp_id}] No hotkeys with non-zero scores")
                continue

            # Sort hotkeys by average score (descending)
            sorted_hotkeys = all_hotkeys
            
            # Distribute rewards to top 10 miners
            for rank, hotkey in enumerate(sorted_hotkeys[:10], 1):
                if hotkey in self.metagraph.hotkeys:
                    idx = self.metagraph.hotkeys.index(hotkey)
                    reward_fraction = SCORE_DISTRIBUTION.get(rank, 0.0)
                    reward_amount = weight * reward_fraction
                    self.scores[idx] += reward_amount
                    log.set_hotkey(hotkey)
                    log.statistics.info(
                        f"Rank {rank}: +{reward_amount:.6f} ({reward_fraction*100:.1f}%)"
                    )
                    log.clear_hotkey()
                    #bt.logging.info(
                    #    f"[{comp_id}] Rank {rank}: +{reward_amount:.6f} ({reward_fraction*100:.1f}%) to {hotkey}"
                    #)
                else:
                    log.set_hotkey(hotkey)
                    log.statistics.warn(f"Rank {rank} hotkey not in metagraph")
                    log.clear_hotkey()
                    #bt.logging.warning(
                    #    f"[{comp_id}] Rank {rank} hotkey {hotkey!r} not in metagraph"
                    #)

            # Award minimal scores to remaining miners (beyond top 10)
            remaining_hotkeys = sorted_hotkeys[10:]
            k = len(remaining_hotkeys)
            if k > 0:
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
                for minimal, hotkey in zip(minimal_values, remaining_hotkeys):
                    if hotkey in self.metagraph.hotkeys:
                        idx = self.metagraph.hotkeys.index(hotkey)
                        self.scores[idx] += minimal
                        log.set_hotkey(hotkey)
                        log.statistics.info(f"Beyond top 10: +{minimal:.6f}")
                        log.clear_hotkey()
                        #bt.logging.info(
                        #    f"[{comp_id}] Beyond top 10: +{minimal:.6f} to {hotkey}"
                        #)
                    else:
                        log.set_hotkey(hotkey)
                        log.statistics.warn("Non-top-10 hotkey not in metagraph")
                        log.clear_hotkey()
                        #bt.logging.warning(
                        #    f"[{comp_id}] Non-top-10 hotkey {hotkey!r} not in metagraph"
                        #)
            
            log.clear_competition()

        log.statistics.debug(
            "Scores from update_scores:\n"
            f"{np.array2string(self.scores, precision=7, floatmode='fixed', separator=', ', suppress_small=True)}"
        )
        #bt.logging.debug(
        #    "Scores from update_scores:\n"
        #    f"{np.array2string(self.scores, precision=7, floatmode='fixed', separator=', ', suppress_small=True)}"
        #)

        self.save_state()


    async def log_results_to_csv(self, file_name: str, data_package: NewDatasetFile, top_hotkey: str, models_results: list):
        """Debug method for dumping rewards for testing """
        
        csv_file_path = os.path.join("evaluation-results", file_name)
        log.set_competition(data_package.competition_id)
        log.statistics.info(f"Logging results to CSV to file {csv_file_path}")
        log.clear_competition()
        #bt.logging.info(f"Logging results to CSV for {data_package.competition_id} to file {csv_file_path}")
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
        temp_state_path = state_path + ".tmp"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        try:
            with open(temp_state_path, 'w') as f:
                json.dump(state_dict, f, indent=2, cls=self.DateTimeEncoder)
            # Atomically move the temporary file to the final destination
            os.rename(temp_state_path, state_path)
            log.validation.debug(f"Validator state saved to {state_path}")
            #bt.logging.debug(f"Validator state saved to {state_path}")
        except Exception as e:
            log.validation.error(f"Error saving validator state: {e}", exc_info=True)
            #bt.logging.error(f"Error saving validator state: {e}", exc_info=True)
            # Clean up the temporary file if it exists
            if os.path.exists(temp_state_path):
                os.remove(temp_state_path)
    
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
                log.validation.error(f"Error loading or parsing state file: {e}. Resetting to a clean state.")
                #bt.logging.error(f"Error loading or parsing state file: {e}. Resetting to a clean state.")
                self.create_empty_state()
                return
        else:
            log.validation.warn("No state file found. Creating an empty one.")
            #bt.logging.warning("No state file found. Creating an empty one.")
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
    log.validation.info("Setting up main thread interrupt handle.")
    #bt.logging.info("Setting up main thread interrupt handle.")
  
    try:
        bt.logging.disable_third_party_loggers()
    except:
        pass
    
    exit_event = threading.Event()
    with Validator(exit_event=exit_event) as validator:
        while True:
            time.sleep(5)
            if exit_event.is_set():
                log.validation.info("Exit event received. Shutting down...")
                #bt.logging.info("Exit event received. Shutting down...")
                break
