from typing import Dict, Any, TYPE_CHECKING, Optional
import os
import json
import datetime

import wandb
import bittensor as bt



from neurons.validator import Validator
from cancer_ai.validator.utils import NewDatasetFile
from cancer_ai.validator.competition_manager import CompetitionManager
from cancer_ai.validator.models import WanDBLogModelErrorEntry, WanDBLogModelBase

class LocalWandbSaver:
    """Local wandb logging utilities for saving wandb data to JSON files."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize local wandb saver with timestamped directory."""
        if log_dir is None:
            log_dir = "logs"
        
        # Create timestamped directory for this run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize file paths
        self.competition_file = os.path.join(self.run_dir, "competition_evaluation.json")
        self.model_file = os.path.join(self.run_dir, "model_evaluation.json")
        
        # Initialize empty files if they don't exist
        self._init_file(self.competition_file, [])
        self._init_file(self.model_file, [])
        
        bt.logging.info(f"LocalWandbSaver initialized. Saving to: {self.run_dir}")
    
    def _init_file(self, filepath: str, initial_data: Any):
        """Initialize a JSON file with initial data."""
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def _append_to_file(self, filepath: str, data: Dict[str, Any]):
        """Append data to a JSON file (array format)."""
        try:
            # Read existing data
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            
            # Ensure it's a list
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            
            # Add timestamp to entry
            data['timestamp'] = datetime.datetime.now().isoformat()
            
            # Append new data
            existing_data.append(data)
            
            # Write back
            with open(filepath, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            bt.logging.error(f"Failed to append to {filepath}: {e}")
    
    def log_competition_winner(self, data: Dict[str, Any]):
        """Log competition winner data."""
        self._append_to_file(self.competition_file, data)
        bt.logging.info(f"Saved competition winner to {self.competition_file}")
    
    def log_model_evaluation(self, data: Dict[str, Any]):
        """Log model evaluation data."""
        self._append_to_file(self.model_file, data)
    
    def init_session(self, project: str, group: str):
        """Mock wandb.init() for local saving."""
        bt.logging.info(f"Local wandb session started: project={project}, group={group}")
        return self
    
    def log(self, data: Dict[str, Any]):
        """Mock wandb.log() for local saving."""
        # This method is called but we handle logging in the specific methods above
        pass
    
    def finish(self):
        """Mock wandb.finish() for local saving."""
        bt.logging.info("Local wandb session finished")


# Global instance for the current competition run
_local_wandb_saver: Optional[LocalWandbSaver] = None


def init_local_wandb(project: str, group: str, log_dir: Optional[str] = None) -> LocalWandbSaver:
    """Initialize local wandb session. Creates one instance per competition run."""
    global _local_wandb_saver
    if _local_wandb_saver is None:
        _local_wandb_saver = LocalWandbSaver(log_dir)
    return _local_wandb_saver.init_session(project, group)


def reset_local_wandb():
    """Reset the global local wandb saver. Call this when starting a new competition."""
    global _local_wandb_saver
    _local_wandb_saver = None


async def log_evaluation_results(
    validator: 'Validator',
    competition_id: str, 
    competition_uuid: str, 
    data_package: 'NewDatasetFile', 
    competition_manager: 'CompetitionManager',
    winning_hotkey: str,
    winning_model_link: str,
    average_winning_hotkey: str,
    competition_start_time: 'datetime.datetime'
) -> None:
    """Log evaluation results to wandb and/or local files."""
    # Log competition winners
    try:
        from cancer_ai.validator.models import WanDBLogCompetitionWinners
        winner_log: WanDBLogCompetitionWinners = WanDBLogCompetitionWinners(
            uuid=competition_uuid,
            competition_id=competition_id,
            competition_winning_hotkey=winning_hotkey,
            competition_winning_uid=validator.metagraph.hotkeys.index(winning_hotkey),
            average_winning_hotkey=average_winning_hotkey,
            average_winning_uid=validator.metagraph.hotkeys.index(average_winning_hotkey),
            validator_hotkey=validator.wallet.hotkey.ss58_address,
            model_link=winning_model_link,
            dataset_filename=data_package.dataset_hf_filename,
            run_time_s=(datetime.datetime.now() - competition_start_time).seconds
        )
        
        if validator.config.wandb.local_save:
            local_wandb = init_local_wandb(competition_id, "competition_evaluation")
            local_wandb.log_competition_winner(winner_log.model_dump())
            local_wandb.finish()
        
        if not validator.config.wandb.off:
            project_name = validator.config.wandb.project_name if validator.config.wandb.project_name else competition_id
            wandb.init(project=project_name, group="competition_evaluation")
            wandb.log(winner_log.model_dump())
            wandb.finish()
    except Exception as wandb_error:
        import bittensor as bt
        bt.logging.warning(f"Failed to log competition winners: {wandb_error}")
    
    # Log model evaluations
    local_wandb = None
    
    try:
        if validator.config.wandb.local_save:
            local_wandb = init_local_wandb(competition_id, "model_evaluation")
        
        if not validator.config.wandb.off:
            project_name = validator.config.wandb.project_name if validator.config.wandb.project_name else competition_id
            wandb.init(project=project_name, group="model_evaluation")
    except Exception as wandb_error:
        import bittensor as bt
        bt.logging.warning(f"Failed to initialize wandb for model evaluation: {wandb_error}")
        
    # Log all model results (both successful and error cases)
    for miner_hotkey, model_result in competition_manager.results:
        if model_result.error:
            # Error case: log as WanDBLogModelErrorEntry
            model_log: WanDBLogModelErrorEntry = WanDBLogModelErrorEntry(
                uuid=competition_uuid,
                competition_id=competition_id,
                miner_hotkey=miner_hotkey,
                uid=validator.metagraph.hotkeys.index(miner_hotkey),
                validator_hotkey=validator.wallet.hotkey.ss58_address,
                dataset_filename=data_package.dataset_hf_filename,
                errors=model_result.error,
            )
        else:
            # Success case: log using competition-specific WanDBLogModelClass
            try:
                model = validator.db_controller.get_latest_model(
                    hotkey=miner_hotkey,
                    cutoff_time=validator.config.models_query_cutoff,
                )
                avg_score: float = 0.0
                if (
                    data_package.competition_id in validator.competition_results_store.average_scores and 
                    miner_hotkey in validator.competition_results_store.average_scores[competition_id]
                ):
                    avg_score = validator.competition_results_store.average_scores[competition_id][miner_hotkey]
                
                # Get competition-specific metrics from model_result
                ActualWanDBLogModelEntryClass: type = competition_manager.competition_handler.WanDBLogModelClass
                model_log: WanDBLogModelBase = ActualWanDBLogModelEntryClass(
                    uuid=competition_uuid,
                    competition_id=competition_id,
                    miner_hotkey=miner_hotkey,
                    uid=validator.metagraph.hotkeys.index(miner_hotkey),
                    validator_hotkey=validator.wallet.hotkey.ss58_address,
                    model_url=model.hf_link,
                    code_url=model.hf_code_link,
                    average_score=avg_score,
                    run_time_s=model_result.run_time_s,
                    dataset_filename=data_package.dataset_hf_filename,
                    errors="",  # No errors for successful models
                    **model_result.to_log_dict(),
                )
            except Exception as wandb_error:
                import bittensor as bt
                bt.logging.warning(f"Failed to log model results for hotkey {miner_hotkey}: {wandb_error}")
                continue
        
        if validator.config.wandb.local_save and local_wandb is not None:
            local_wandb.log_model_evaluation(model_log.model_dump())
        
        if not validator.config.wandb.off:
            wandb.log(model_log.model_dump())
    
    # Finish wandb run
    try:
        if validator.config.wandb.local_save and local_wandb is not None:
            local_wandb.finish()
        
        if not validator.config.wandb.off:
            wandb.finish()
    except Exception as wandb_error:
        import bittensor as bt
        bt.logging.warning(f"Failed to finish wandb run: {wandb_error}")
