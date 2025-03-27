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
from cancer_ai.validator.models import WandBLogModelEntry, WanDBLogCompetitionWinner
from huggingface_hub import HfApi

BLACKLIST_FILE_PATH = "config/hotkey_blacklist.json"
BLACKLIST_FILE_PATH_TESTNET = "config/hotkey_blacklist_testnet.json"

class Validator(BaseValidatorNeuron):
    
    def __init__(self, config=None, exit_event=None):
        print(cancer_ai_logo)
        super(Validator, self).__init__(config=config)
        self.hotkey = self.wallet.hotkey.ss58_address
        self.db_controller = ModelDBController(self.config.db_path)

        self.chain_models = ChainModelMetadata(
            self.subtensor, self.config.netuid, self.wallet
        )
        self.last_miners_refresh: float = None
        self.last_monitor_datasets: float = None

        self.hf_api = HfApi()

        self.exit_event = exit_event

    async def concurrent_forward(self):
        try:
            bt.logging.info("Starting concurrent_forward execution")
            coroutines = [
                self.refresh_miners(),
            ]
            
            if self.config.filesystem_evaluation:
                bt.logging.info("Adding filesystem_test_evaluation to coroutines")
                coroutines.append(self.filesystem_test_evaluation())
            else:
                bt.logging.info("Adding monitor_datasets to coroutines")
                coroutines.append(self.monitor_datasets())
        
            bt.logging.info(f"Executing {len(coroutines)} coroutines with asyncio.gather")
            try:
                await asyncio.gather(*coroutines)
                bt.logging.info("Successfully completed all coroutines in concurrent_forward")
            except Exception as e:
                import traceback
                stack_trace = traceback.format_exc()
                bt.logging.error(f"Error during asyncio.gather in concurrent_forward: {e}")
                bt.logging.error(f"Stack trace from asyncio.gather:\n{stack_trace}")
                raise  # Re-raise to be caught by the outer try-except
        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            bt.logging.error(f"CRITICAL ERROR in concurrent_forward: {e}")
            bt.logging.error(f"Error type: {type(e).__name__}")
            bt.logging.error(f"Stack trace from concurrent_forward:\n{stack_trace}")
            
            # Log additional context
            bt.logging.error(f"Validator state when error occurred:")
            bt.logging.error(f"- Last miners refresh: {self.last_miners_refresh}")
            bt.logging.error(f"- Last monitor datasets: {self.last_monitor_datasets}")
            bt.logging.error(f"- Current time: {time.time()}")
            
            # Re-raise to be caught by the base validator's run method
            raise



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
            chain_model_metadata = await self.chain_models.retrieve_model_metadata(hotkey)
            if not chain_model_metadata:
                bt.logging.warning(
                    f"Cannot get miner model for hotkey {hotkey} from the chain, skipping"
                    )
                continue
            try:
                self.db_controller.add_model(chain_model_metadata, hotkey)
            except Exception as e:
                bt.logging.error(f"An error occured while trying to persist the model info: {e}")

        self.db_controller.clean_old_records(self.hotkeys)
        self.last_miners_refresh = time.time()
        self.save_state()

    async def filesystem_test_evaluation(self):
        time.sleep(1)
        data_package = get_local_dataset(self.config.local_dataset_dir)
        if not data_package:
            bt.logging.error("NO NEW DATA PACKAGES")
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
            bt.logging.error(f"Error evaluating {data_package.dataset_hf_filename}: {e}")

        models_results = competition_manager.results
        
      
        try:
            top_hotkey = self.competition_results_store.get_top_hotkey(data_package.competition_id)
        except ValueError:
            bt.logging.warning(f"No top hotkey available for competition {data_package.competition_id}")
            top_hotkey = None
        
        # Enable if you want to have results in CSV for debugging purposes
        # await self.log_results_to_csv(data_package, top_hotkey, models_results)

        bt.logging.info(f"Competition result for {data_package.competition_id}: {winning_hotkey}")

        bt.logging.warning("Competition results store before update")
        bt.logging.warning(self.competition_results_store.model_dump_json())
        competition_weights = await self.competition_results_store.update_competition_results(data_package.competition_id, models_results, self.config, self.metagraph.hotkeys, self.hf_api)
        bt.logging.warning("Competition results store after update")
        bt.logging.warning(self.competition_results_store.model_dump_json())
        self.update_scores(competition_weights)


    async def monitor_datasets(self):
        """Main validation logic, triggered by new datastes on huggingface"""
        try:
            if self.last_monitor_datasets is not None and (
                time.time() - self.last_monitor_datasets
                < self.config.monitor_datasets_interval
            ):
                return
            self.last_monitor_datasets = time.time()

            bt.logging.info("Starting monitor_datasets")
            yaml_data = await fetch_organization_data_references(
                self.config.datasets_config_hf_repo_id,
                self.hf_api,
                )        
                
            await sync_organizations_data_references(yaml_data)
            self.organizations_data_references = OrganizationDataReferenceFactory.get_instance()
            self.save_state()
            bt.logging.info("Fetched and synced organization data references")
        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            bt.logging.error(f"Error in monitor_datasets initial setup: {e}")
            bt.logging.error(f"Stack trace: {stack_trace}")
            return

        try:
            list_of_new_data_packages: list[NewDatasetFile] = await check_for_new_dataset_files(self.hf_api, self.org_latest_updates)
            self.save_state()

            if not list_of_new_data_packages:
                bt.logging.info("No new data packages found.")
                return
            
            bt.logging.info(f"Found {len(list_of_new_data_packages)} new data packages")
        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            bt.logging.error(f"Error checking for new dataset files: {e}")
            bt.logging.error(f"Stack trace: {stack_trace}")
            return
        
        for data_package in list_of_new_data_packages:
            try:
                bt.logging.info(f"Starting competition for {data_package.competition_id}")
                competition_manager = CompetitionManager(
                    config=self.config,
                    subtensor=self.subtensor,
                    hotkeys=self.hotkeys,
                    validator_hotkey=self.hotkey,
                    competition_id=data_package.competition_id,
                    dataset_hf_repo=data_package.dataset_hf_repo,
                    dataset_hf_filename=data_package.dataset_hf_filename,
                    dataset_hf_repo_type="dataset",
                    db_controller = self.db_controller,
                    test_mode = self.config.test_mode,
                )
            except Exception as e:
                import traceback
                stack_trace = traceback.format_exc()
                bt.logging.error(f"Error creating competition manager for {data_package.competition_id}: {e}")
                bt.logging.error(f"Stack trace: {stack_trace}")
                continue
            winning_hotkey = None
            winning_model_link = None
            try:
                winning_hotkey, winning_model_result = (
                    await competition_manager.evaluate()
                )
                if not winning_hotkey:
                    continue

                winning_model_link = self.db_controller.get_latest_model(hotkey=winning_hotkey).hf_link
            except Exception:
                formatted_traceback = traceback.format_exc()
                bt.logging.error(f"Error running competition: {formatted_traceback}")
                wandb.init(
                    reinit=True, project="competition_id", group="competition_evaluation"
                )
                
                error_log = WanDBLogCompetitionWinner(
                    competition_id=data_package.competition_id,
                    winning_evaluation_hotkey="",
                    run_time="",
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                    model_link=winning_model_link,
                    errors=str(formatted_traceback)
                )
                wandb.log(error_log.model_dump())
                wandb.finish()
                continue

            wandb.init(project=data_package.competition_id, group="competition_evaluation")
            
            
            winner_log = WanDBLogCompetitionWinner(
                competition_id=data_package.competition_id,
                winning_hotkey=winning_hotkey,
                validator_hotkey=self.wallet.hotkey.ss58_address,
                model_link=winning_model_link,
                errors=""
            )
            wandb.log(winner_log.model_dump())
            wandb.finish()

            # Update competition results
            bt.logging.info(f"Competition result for {data_package.competition_id}: {winning_hotkey}")
            competition_weights = await self.competition_results_store.update_competition_results(data_package.competition_id, competition_manager.results, self.config, self.metagraph.hotkeys, self.hf_api)
            self.update_scores(competition_weights)
            
            # Logging results
            for miner_hotkey, evaluation_result in competition_manager.results:
                try:
                    model = self.db_controller.get_latest_model(
                        hotkey=miner_hotkey
                    )
                    model_link = model.hf_link if model is not None else None
                    
                    avg_score = 0.0
                    if (data_package.competition_id in self.competition_results_store.average_scores and 
                        miner_hotkey in self.competition_results_store.average_scores[data_package.competition_id]):
                        avg_score = self.competition_results_store.average_scores[data_package.competition_id][miner_hotkey]
                    
                    wandb.init(project=data_package.competition_id, group="model_evaluation", reinit=True)
                    model_log = WandBLogModelEntry(
                        competition_id=data_package.competition_id,
                        miner_hotkey=miner_hotkey,
                        validator_hotkey=self.wallet.hotkey.ss58_address,
                        tested_entries=evaluation_result.tested_entries,
                        accuracy=evaluation_result.accuracy,
                        precision=evaluation_result.precision,
                        fbeta=evaluation_result.fbeta,
                        recall=evaluation_result.recall,
                        confusion_matrix=evaluation_result.confusion_matrix,
                        roc_curve={
                            "fpr": evaluation_result.fpr,
                            "tpr": evaluation_result.tpr,
                        },
                        model_link=model_link,
                        roc_auc=evaluation_result.roc_auc,
                        score=evaluation_result.score,
                        average_score=avg_score
                    )
                    wandb.log(model_log.model_dump())
                    wandb.finish()
                except Exception as e:
                    bt.logging.error(f"Error logging model results for hotkey {miner_hotkey}: {e}")
                    continue


    async def log_results_to_csv(self, data_package: NewDatasetFile, top_hotkey: str, models_results: list):
        """Debug method for dumping rewards for testing """

        csv_file = "filesystem_test_evaluation_results.csv"
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if os.stat(csv_file).st_size == 0:
                writer.writerow(["Package name", "Date", "Hotkey", "Score", "Average","Winner"])
            competition_id = data_package.competition_id
            for hotkey, model_result in models_results:
                avg_score = 0.0
                if (competition_id in self.competition_results_store.average_scores and 
                    hotkey in self.competition_results_store.average_scores[competition_id]):
                    avg_score = self.competition_results_store.average_scores[competition_id][hotkey]
                
                if hotkey == top_hotkey:
                    writer.writerow([os.path.basename(data_package.dataset_hf_filename), 
                                    datetime.datetime.now(), 
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


    def save_state(self):
        """Saves the state of the validator to a file."""
        if not getattr(self, "organizations_data_references", None):
            self.organizations_data_references = OrganizationDataReferenceFactory.get_instance()
        
        # Convert numpy arrays to lists for JSON serialization
        scores_list = self.scores.tolist() if hasattr(self.scores, 'tolist') else []
        hotkeys_list = self.hotkeys.tolist() if hasattr(self.hotkeys, 'tolist') else self.hotkeys
        
        # Create a complete state dictionary
        state_dict = {
            'scores': scores_list,
            'hotkeys': hotkeys_list,
            'organizations_data_references': self.organizations_data_references.model_dump(),
            'org_latest_updates': self.org_latest_updates,
            'competition_results_store': self.competition_results_store.model_dump()
        }
        
        # Ensure directory exists
        state_path = self.config.neuron.full_path + "/state.json"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        # Save as JSON file
        with open(state_path, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def create_empty_state(self):
        """Creates an empty state file."""
        # Create an empty state dictionary
        empty_state = {
            'scores': [],
            'hotkeys': [],
            'organizations_data_references': self.organizations_data_references.model_dump(),
            'org_latest_updates': {},
            'competition_results_store': self.competition_results_store.model_dump()
        }
        
        # Ensure directory exists
        state_path = self.config.neuron.full_path + "/state.json"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        # Save as JSON file
        with open(state_path, 'w') as f:
            json.dump(empty_state, f, indent=2)

    def load_state(self):
        """Loads the state of the validator from a file."""
        json_path = self.config.neuron.full_path + "/state.json"
        npz_path = self.config.neuron.full_path + "/state.npz"
        
        # Check if JSON state file exists
        if os.path.exists(json_path):
            try:
                # Load the state from JSON file
                with open(json_path, 'r') as f:
                    state = json.load(f)
                
                # Convert lists back to numpy arrays
                self.scores = np.array(state['scores'], dtype=np.float32)
                self.hotkeys = np.array(state['hotkeys'])
                
                # Load organization data references
                factory = OrganizationDataReferenceFactory.get_instance()
                factory.update_from_dict(state['organizations_data_references'])
                self.organizations_data_references = factory
                
                # Load org latest updates
                self.org_latest_updates = state['org_latest_updates']
                
                # Load competition results
                self.competition_results_store = CompetitionResultsStore.model_validate(
                    state['competition_results_store']
                )
                return
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                bt.logging.error(f"Error loading JSON state: {e}")
        
        # Fall back to npz if it exists (for backward compatibility)
        if os.path.exists(npz_path):
            try:
                # Load the state from npz file
                state = np.load(npz_path, allow_pickle=True)
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                
                try:
                    # Try to load JSON-serialized data from npz
                    org_data_str = state["organizations_data_references"].item()
                    org_data = json.loads(org_data_str)
                    factory = OrganizationDataReferenceFactory.get_instance()
                    factory.update_from_dict(org_data)
                    self.organizations_data_references = factory
                    
                    org_updates_str = state["org_latest_updates"].item()
                    self.org_latest_updates = json.loads(org_updates_str)
                    
                    comp_data_str = state["competition_results_store"].item()
                    competition_data = json.loads(comp_data_str)
                    self.competition_results_store = CompetitionResultsStore.model_validate(competition_data)
                except (json.JSONDecodeError, TypeError, KeyError):
                    # Fall back to standard approach for backward compatibility
                    factory = OrganizationDataReferenceFactory.get_instance()
                    saved_data = state["organizations_data_references"].item()
                    factory.update_from_dict(saved_data)
                    self.organizations_data_references = factory
                    
                    self.org_latest_updates = state["org_latest_updates"].item()
                    
                    competition_data = state["competition_results_store"].item()
                    self.competition_results_store = CompetitionResultsStore.model_validate(competition_data)
                
                # Save in the new JSON format for future use
                self.save_state()
                return
            except Exception as e:
                bt.logging.error(f"Error loading npz state: {e}")
        
        # If we get here, create an empty state
        self.create_empty_state()



    def update_scores(self, competition_weights: dict[str, float]):
        """Update scores based on competition weights."""
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        
        for competition_id, weight in competition_weights.items():
            try:
                winner_hotkey = self.competition_results_store.get_top_hotkey(competition_id)
                if winner_hotkey is not None:
                    if winner_hotkey in self.metagraph.hotkeys:
                        winner_idx = self.metagraph.hotkeys.index(winner_hotkey)
                        self.scores[winner_idx] += weight
                        bt.logging.info(f"Applied weight {weight} for competition {competition_id} winner {winner_hotkey}")
                    else:
                        bt.logging.warning(f"Winning hotkey {winner_hotkey} not found for competition {competition_id}")
            except ValueError as e:
                bt.logging.warning(f"Error getting top hotkey for competition {competition_id}: {e}")
                continue
        
        bt.logging.debug("Scores from UPDATE_SCORES:")
        bt.logging.debug(f"{self.scores}")
        self.save_state()


if __name__ == "__main__":
    bt.logging.info("Setting up main thread interrupt handle.")
    exit_event = threading.Event()
    with Validator(exit_event=exit_event) as validator:
        while True:
            time.sleep(5)
            if exit_event.is_set():
                bt.logging.info("Exit event received. Shutting down...")
                break
