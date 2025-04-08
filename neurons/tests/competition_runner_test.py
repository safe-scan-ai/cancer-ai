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
from cancer_ai.validator.utils import get_competition_config
from cancer_ai.mock import MockSubtensor
from cancer_ai.validator.models import CompetitionsListModel, CompetitionModel
from cancer_ai.validator.model_db import ModelDBController


COMPETITION_FILEPATH = "config/competition_config_testnet.json"

# TODO integrate with bt config
test_config = SimpleNamespace(
    **{
        "wandb_entity": "testnet",
        "wandb_project_name": "melanoma-1",
        "competition_id": "melaonoma-1",
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

competitions_cfg = get_competition_config("config/competition_config_testnet.json")


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
            validator_hotkey="Walidator",
            competition_id=competition_cfg.competition_id,
            dataset_hf_repo=competition_cfg.dataset_hf_repo,
            dataset_hf_id=competition_cfg.dataset_hf_filename,
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
    for competition_cfg in competitions_cfg:
        for competition_time in competition_cfg["evaluation_time"]:
            time_arranged_competitions[competition_time] = CompetitionManager(
                config={},
                subtensor=subtensor,
                hotkeys=[],
                validator_hotkey="Walidator",
                competition_id=competition_cfg.competition_id,
                dataset_hf_repo=competition_cfg.dataset_hf_repo,
                dataset_hf_id=competition_cfg.dataset_hf_filename,
                dataset_hf_repo_type=competition_cfg.dataset_hf_repo_type,
                test_mode=True,
                db_controller=ModelDBController(db_path=test_config.db_path, subtensor=subtensor)
            )
    return time_arranged_competitions


@pytest.fixture
def competition_config():
    with open(COMPETITION_FILEPATH, "r") as f:
        return json.load(f)


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
