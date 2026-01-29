import asyncio
import json
from types import SimpleNamespace
from typing import List, Dict
import pytest

import bittensor as bt


from cancer_ai.validator.competition_manager import CompetitionManager
from cancer_ai.validator.rewarder import CompetitionWinnersStore, Rewarder
from cancer_ai.base.base_miner import BaseNeuron
from cancer_ai.utils.config import path_config
from cancer_ai.mock import MockSubtensor
from cancer_ai.validator.models import CompetitionsListModel, CompetitionModel
from cancer_ai.validator.model_db import ModelDBController

# TODO integrate with bt config
test_config = SimpleNamespace(
    **{
        "wandb_entity": "testnet",
        "wandb_project_name": "tricorder-3",
        "competition_id": "tricorder-3",
        "hotkeys": [],
        "subtensor": SimpleNamespace(**{"network": "test"}),
        "netuid": 163,
        "models": SimpleNamespace(
            **{
                "model_dir": "/tmp/models",
                "dataset_dir": "/tmp/datasets",
            }
        ),
        "hf_token": "HF_TOKEN",
        "db_path": "models.db",
    }
)

competitions_cfg = CompetitionsListModel(
    competitions=[
        CompetitionModel(
            competition_id="tricorder-3",
            category="skin",
            evaluation_times=["10:00"],
            dataset_hf_repo="safescanai/tricorder-3-competition-testnet",
            dataset_hf_filename="tricorder-3-dataset-testnet.zip",
            dataset_hf_repo_type="dataset",
        )
    ]
)


async def run_competitions(
    config: str,
    subtensor: bt.subtensor,
    hotkeys: List[str],
) -> Dict[str, str]:
    """Run all competitions, return the winning hotkey for each competition"""
    results = {}
    for competition_cfg in competitions_cfg.competitions:
        bt.logging.info("Starting competition: ", competition_cfg)

        competition_manager = CompetitionManager(
            config=config,
            subtensor=subtensor,
            hotkeys=hotkeys,
            validator_hotkey="Validator",
            competition_id=competition_cfg.competition_id,
            dataset_hf_repo=competition_cfg.dataset_hf_repo,
            dataset_hf_filename=competition_cfg.dataset_hf_filename,
            dataset_hf_repo_type=competition_cfg.dataset_hf_repo_type,
            test_mode=True,
            db_controller=ModelDBController(db_path=test_config.db_path, subtensor=subtensor)
        )
        results[competition_cfg.competition_id] = await competition_manager.evaluate()

        bt.logging.info(await competition_manager.evaluate())

    return results


def config_for_scheduler(subtensor: bt.subtensor) -> Dict[str, CompetitionManager]:
    """Returns CompetitionManager instances arranged by competition time"""
    time_arranged_competitions = {}
    for competition_cfg in competitions_cfg.competitions:
        for competition_time in competition_cfg["evaluation_time"]:
            time_arranged_competitions[competition_time] = CompetitionManager(
                config={},
                subtensor=subtensor,
                hotkeys=[],
                validator_hotkey="Validator",
                competition_id=competition_cfg.competition_id,
                dataset_hf_repo=competition_cfg.dataset_hf_repo,
                dataset_hf_filename=competition_cfg.dataset_hf_filename,
                dataset_hf_repo_type=competition_cfg.dataset_hf_repo_type,
                test_mode=True,
                db_controller=ModelDBController(db_path=test_config.db_path, subtensor=subtensor)
            )
    return time_arranged_competitions

if __name__ == "__main__":
    config = BaseNeuron.config()
    bt.logging.set_config(config=config)
    # if True:  # run them right away
    path_config = path_config(None)
    # config = config.merge(path_config)
    # BaseNeuron.check_config(config)
    bt.logging.set_config(config=config.logging)
    bt.logging.info(config)
    asyncio.run(run_competitions(test_config, MockSubtensor("123"), []))
