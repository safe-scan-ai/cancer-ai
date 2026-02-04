# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import os
import subprocess
import argparse
import bittensor as bt
from .logging import setup_events_logger

BLACKLIST_FILE_PATH = "config/hotkey_blacklist.json"    
BLACKLIST_FILE_PATH_TESTNET = "config/hotkey_blacklist_testnet.json"

def is_cuda_available():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT)
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("Log path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=76)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=180,
    )

    parser.add_argument(
        "--ignore_registered",
        action="store_true",
        help="Ignore hotkey registration check.",
        default=False,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.local_save",
        action="store_true",
        help="Save wandb data to local JSON files instead of sending to wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.state_off",
        action="store_true", 
        help="Disable logging validator state to wandb when wandb is enabled.",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run. By default - the one in dataset ",
        default=None,
    )


    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

    parser.add_argument(
        "--models_query_cutoff",
        type=int,
        help="The cutoff for the models query in minutes.",
        default=120,  
    )

    parser.add_argument(
        "--datasets_config_hf_repo_id",
        type=str,
        help="The reference to Hugging Face datasets config.",
        default="safescanai/competition-configuration",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path for for loading the starting model related to a training run.",
        default="./models",
    )

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="Hugging Face model repository ID",
        default="",
    )

    parser.add_argument(
        "--hf_model_name",
        type=str,
        help="Filename of the model to push to hugging face.",
    )
    parser.add_argument(
        "--hf_code_filename",
        type=str,
        help="Filename of the code zip  to push to hugging face.",
    )

    parser.add_argument(
        "--action",
        choices=["submit", "evaluate", "upload", "self-check"],
    )

    parser.add_argument(
        "--hotkey",
        type=str,
        help="Hotkey to check model for (used with --action self-check)",
        default=None,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to ONNX model, used for evaluation",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path for storing datasets.",
        default="./datasets",
    )

    parser.add_argument(
        "--clean_after_run",
        action="store_true",
        help="Whether to clean up (dataset, temporary files) after running",
        default=False,
    )

    parser.add_argument(
        "--code_directory",
        type=str,
        help="Path to code directory",
        default=".",
    )


def add_common_args(cls, parser):
    """Add validator and miner specific arguments to the parser."""
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        help="Path for storing competition participants models .",
    )

    parser.add_argument(
        "--models.model_dir",
        type=str,
        help="Path for storing competition participants models .",
        default="/tmp/models",
    )

    parser.add_argument(
        "--models.cache_cleanup_age_days",
        type=int,
        help="Age threshold in days for automatic cleanup of cached model files .",
        default=None,
    )

    parser.add_argument(
        "--models.dataset_dir",
        type=str,
        help="Path for storing datasets.",
        default="/tmp/datasets-extracted",
    )

    parser.add_argument(
        "--competition.config_path",
        type=str,
        help="Path with competition configuration .",
        default="./config/competition_config.json",
    )

    parser.add_argument(
        "--archive_node_url",
        type=str,
        help="The URL of the archive node to use",
        default="wss://archive.chain.opentensor.ai:443",
    )

    parser.add_argument(
        "--archive_node_fallback_url",
        type=str,
        help="The fallback URL of the archive node to use",
        default=None,
    )

    parser.add_argument(
        "--test_dataset_hf_repo",
        type=str,
        default=None,
        help="Test HF repo for dataset discovery (bypasses normal org config). Format: username/repo-name"
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=10,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=50,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--db_path",
        type=str,
        help="Path to the sqlite DB for storing the miners models reference",
        default="models.db"
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="melanoma-testnet",
    )

    parser.add_argument(
        "--wandb_project_name_state",
        type=str,
        help="The suffix to add to the project name for state logging.",
        default="validator-state",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="safe-scan-ai",
    )

    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test(net) mode",
        default=False,
    )

    parser.add_argument(
        "--mock_models",
        action="store_true",
        help="Use mock models for testing with local ONNX files",
        default=False,
    )


    parser.add_argument(
        "--miners_refresh_interval", # not used 
        type=int,
        help="The interval at which to refresh the miners in minutes",
        default=60,
    )

    parser.add_argument(
        "--monitor_datasets_interval",
        type=int,
        help="The interval at which to monitor the datasets in seconds",
        default=600, # 10 min
    )

    parser.add_argument(
        "--local_dataset_dir",
        type=str,
        help="Path to the local dataset directory",
        default="local_datasets/",
    )

    parser.add_argument(
        "--filesystem_evaluation",
        type=bool,
        help="Use local datasets instead of HF? Use together with --local_dataset_dir",
        default=False
    )

    parser.add_argument(
        "--validator_name",
        type=str,
        help="Unique name for this validator instance. Defaults to hostname if not set.",
        default=None,
    )

    parser.add_argument(
        "--logs_axiom_enabled",
        action="store_true",
        help="Enable Axiom logging for centralized log aggregation.",
        default=False,
    )

    parser.add_argument(
        "--axiom_dataset",
        type=str,
        help="Axiom dataset name for logs.",
        default="76_prod",
    )

    parser.add_argument(
        "--axiom_url",
        type=str,
        help="Axiom API base URL (deployment/region). You can also set AXIOM_URL env var.",
        default="https://eu-central-1.aws.edge.axiom.co",
    )

    ## Debugging variables
    parser.add_argument(
        "--debug.dont_refresh_miners",
        action="store_true",
        help="Don't refresh miners",
        default=False,
    )

def path_config(cls=None):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """

    # config from huggingface
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    add_common_args(cls, parser)
    if cls:
        cls.add_args(parser)
    return bt.config(parser)
