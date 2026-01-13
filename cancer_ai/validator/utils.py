from enum import Enum
import os
import json
import subprocess
from datetime import datetime
import asyncio
import time
from functools import wraps
import shutil
import yaml
import binascii
import bittensor as bt
from retry import retry
from huggingface_hub import HfApi, hf_hub_download
from typing import Union
from pathlib import Path
import sys
import platform


from cancer_ai.chain_models_store import ChainMinerModel
from .models import ModelInfo
from cancer_ai.validator.models import (
    NewDatasetFile,
    OrganizationDataReferenceFactory,
)


class ModelType(Enum):
    ONNX = "ONNX"
    TENSORFLOW_SAVEDMODEL = "TensorFlow SavedModel"
    KERAS_H5 = "Keras H5"
    PYTORCH = "PyTorch"
    SCIKIT_LEARN = "Scikit-learn"
    XGBOOST = "XGBoost"
    UNKNOWN = "Unknown format"


def log_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        module_name = func.__module__
        bt.logging.trace(
            f"'{module_name}.{func.__name__}'  took {end_time - start_time:.4f}s"
        )
        return result

    return wrapper


def log_system_info(repo_root: Path = None) -> None:
    """Log system and environment information for debugging and instance identification."""
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent
    
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=repo_root,
        )
        commit_hash = result.stdout.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit_hash = "unknown"

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            cwd=repo_root,
        )
        status_output = result.stdout.decode().strip()
        is_clean = status_output == ""
        cleanliness = "clean" if is_clean else "dirty"
        
        if not is_clean:
            uncommitted_files = ", ".join([line.split()[1] for line in status_output.split("\n") if line.strip()])
            cleanliness = f"dirty ({uncommitted_files})"
    except (FileNotFoundError, subprocess.CalledProcessError):
        cleanliness = "unknown"

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    platform_info = platform.platform()
    hostname = platform.node()
    processor = platform.processor()
    
    try:
        result = subprocess.run(["pip", "freeze"], check=True, capture_output=True, text=True)
        packages = " ".join(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        packages = "unknown"

    bt.logging.info(f"Git commit: {commit_hash}")
    bt.logging.info(f"Repo clean: {cleanliness}")
    bt.logging.info(f"Python version: {python_version}")
    bt.logging.info(f"Platform: {platform_info}")
    bt.logging.info(f"Hostname: {hostname}")
    bt.logging.info(f"Processor: {processor}")
    bt.logging.info(f"Installed packages: {packages}")


def detect_model_format(file_path) -> ModelType:
    _, ext = os.path.splitext(file_path)

    if ext == ".onnx":
        return ModelType.ONNX
    elif ext == ".h5":
        return ModelType.KERAS_H5
    elif ext in [".pt", ".pth"]:
        return ModelType.PYTORCH
    elif ext in [".pkl", ".joblib", ""]:
        return ModelType.SCIKIT_LEARN
    elif ext in [".model", ".json", ".txt"]:
        return ModelType.XGBOOST

    try:
        with open(file_path, "rb") as f:
            # TODO check if it works
            header = f.read(4)
            if (
                header == b"PK\x03\x04"
            ):  # Magic number for ZIP files (common in TensorFlow SavedModel)
                return ModelType.TENSORFLOW_SAVEDMODEL
            elif header[:2] == b"\x89H":  # Magic number for HDF5 files (used by Keras)
                return ModelType.KERAS_H5

    except Exception as e:
        bt.logging.error(f"Failed to detect model format: {e}")
        return ModelType.UNKNOWN

    return ModelType.UNKNOWN


async def run_command(cmd):
    # Start the subprocess
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    bt.logging.debug(f"Running command: {cmd}")
    # Wait for the subprocess to finish and capture the output
    stdout, stderr = await process.communicate()

    # Return the output and error if any
    return stdout.decode(), stderr.decode()



async def fetch_organization_data_references(
    hf_repo_id: str, hf_api: HfApi
) -> list[dict]:
    bt.logging.trace(
        f"Fetching organization data references from Hugging Face repo {hf_repo_id}"
    )
    yaml_data = []

    # prevent stale connections
    custom_headers = {"Connection": "close"}

    try:
        # blocks event loop while sleeping between retries
        files = _list_repo_tree_with_retry_sync(hf_api, hf_repo_id)
    except Exception as e:
        bt.logging.error("Failed to list repo tree after 10 attempts: %s", e)
        return yaml_data

    for file_info in files:
        if file_info.__class__.__name__ == "RepoFile":
            file_path = file_info.path

            if file_path.startswith("datasets/") and file_path.endswith(".yaml"):
                local_file_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    repo_type="space",
                    token=None,
                    filename=file_path,
                    headers=custom_headers,
                )

                last_commit_info = file_info.last_commit
                commit_date = last_commit_info.date if last_commit_info else None

                if commit_date is not None:
                    date_uploaded = commit_date
                else:
                    bt.logging.warning(
                        f"Could not get the last commit date for {file_path}"
                    )
                    date_uploaded = None

                with open(local_file_path, "r", encoding="utf-8") as f:
                    try:
                        data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        bt.logging.error(
                            f"Error parsing YAML file {file_path}: {str(e)}"
                        )
                        continue  # Skip this file due to parsing error

                yaml_data.append(
                    {
                        "file_name": file_path,
                        "yaml_data": data,
                        "date_uploaded": date_uploaded,
                    }
                )
        else:
            continue
    return yaml_data


async def fetch_yaml_data_from_local_repo(local_repo_path: str) -> list[dict]:
    """
    Fetches YAML data from all YAML files in the specified local directory.
    Returns a list of dictionaries containing file name, YAML data, and the last modified date.
    """
    yaml_data = []

    # Traverse through the local directory to find YAML files
    for root, _, files in os.walk(local_repo_path):
        for file_name in files:
            if file_name.endswith(".yaml"):
                file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(file_path, local_repo_path)
                commit_date = datetime.fromtimestamp(os.path.getmtime(file_path))

                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                yaml_data.append(
                    {
                        "file_name": relative_path,
                        "yaml_data": data,
                        "date_uploaded": commit_date,
                    }
                )

    return yaml_data


async def sync_organizations_data_references(fetched_yaml_files: list[dict]):
    """
    Synchronizes the OrganizationDataReferenceFactory state with the full content
    from the fetched YAML files.

    Each fetched YAML file is expected to contain a list of organization entries.
    The 'org_id' key from the YAML is remapped to 'organization_id' to match the model.
    """
    all_orgs = []
    for file in fetched_yaml_files:
        yaml_data = file["yaml_data"]
        # TODO IMPORTANT BUG use different organization_id for each entry 
        for entry in yaml_data:
            # Remap 'org_id' to 'organization_id' if needed.
            if "org_id" in entry:
                entry["organization_id"] = entry.pop("org_id")
            all_orgs.append(entry)

    update_data = {"organizations": all_orgs}

    factory = OrganizationDataReferenceFactory.get_instance()
    factory.update_from_dict(update_data)


async def get_newest_competition_packages(config: bt.Config, packages_count: int = 30) -> list[dict]:
    """
    Gets the link to the newest package for a specific competition.
    """
    newest_competition_packages: list[dict] = []

    
    hf_api = HfApi(token=config.hf_token)
    
    datasets_references = await fetch_organization_data_references(config.datasets_config_hf_repo_id, hf_api)
    await sync_organizations_data_references(datasets_references)
    org_reference = OrganizationDataReferenceFactory.get_instance()
    org = org_reference.find_organization_by_competition_id(config.competition_id)
    
    if not org:
        bt.logging.info(f"No organization found for competition ID: {config.competition_id}")
        return newest_competition_packages

    try:
        files = list_repo_tree_with_retry(
            hf_api=hf_api,
            repo_id=org.dataset_hf_repo,
            repo_type="dataset",
            recursive=True,
            expand=True,
        )
    except Exception as e:
        bt.logging.error(f"Failed to list repository tree for {org.dataset_hf_repo}: {e}")
        raise 
    
    relevant_files = [
        f for f in files
        if f.__class__.__name__ == "RepoFile"
        and f.path.startswith(org.dataset_hf_dir) and f.path.endswith(".zip")
    ]
    
    if not relevant_files:
        bt.logging.warning(f"No relevant files found in {org.dataset_hf_repo}/{org.dataset_hf_dir}")
        return newest_competition_packages
    
    sorted_files = sorted(
        relevant_files,
        key=lambda f: f.last_commit.date if f.last_commit else datetime.min,
        reverse=True
    )
    
    top_files = sorted_files[:packages_count]
    
    if not top_files:
        return newest_competition_packages
    newest_competition_packages = [
        {
            "dataset_hf_repo": org.dataset_hf_repo,
            "dataset_hf_filename": file.path,
            "dataset_hf_repo_type": "dataset",
        }
        for file in top_files
    ]

    return newest_competition_packages



async def check_for_new_dataset_files(
    hf_api: HfApi, org_latest_updates: dict
) -> list[NewDatasetFile]|None:
    """
    For each OrganizationDataReference stored in the singleton, this function:
      - Connects to the organization's public Hugging Face repo.
      - Lists files under the directory specified by dataset_hf_dir.
      - Determines the maximum commit date among those files.

    For a blank state, it returns the file with the latest commit date.
    On subsequent checks, it returns any file whose commit date is newer than the previously stored update.
    """
    results: list[NewDatasetFile] = []
    factory = OrganizationDataReferenceFactory.get_instance()
    updated_latest_updates: dict = dict(org_latest_updates)

    for org in factory.organizations:
        files = hf_api.list_repo_tree(
            repo_id=org.dataset_hf_repo,
            repo_type="dataset",
            recursive=True,
            expand=True,
        )
        relevant_files = [
            f
            for f in files
            if f.__class__.__name__ == "RepoFile"
            and f.path.startswith(org.dataset_hf_dir) and f.path.endswith(".zip")
        ]
        max_commit_date = None
        for f in relevant_files:
            commit_date = f.last_commit.date if f.last_commit else None
            if commit_date and (
                max_commit_date is None or commit_date > max_commit_date
            ):
                max_commit_date = commit_date

        new_files = []
        stored_update = org_latest_updates.get(org.organization_id)
        # if there is no stored_update and max_commit_date is present (any commit date is present)
        if stored_update is None and max_commit_date is not None:
            for f in relevant_files:
                commit_date = f.last_commit.date if f.last_commit else None
                if commit_date == max_commit_date:
                    new_files.append(f.path)
                    break
        # if there is any stored update then we implicitly expect that any commit date on the repo is present as well
        elif stored_update is not None:
            for f in relevant_files:
                commit_date = f.last_commit.date if f.last_commit else None
                if commit_date and commit_date > stored_update:
                    new_files.append(f.path)

        # update the stored latest update for this organization.
        if max_commit_date is not None:
            updated_latest_updates[org.organization_id] = max_commit_date

        for file_name in new_files:
            file_release_date = None
            for f in relevant_files:
                if f.path == file_name:
                    file_release_date = f.last_commit.date if f.last_commit else None
                    break
            results.append(
                NewDatasetFile(
                    competition_id=org.competition_id,
                    dataset_hf_repo=org.dataset_hf_repo,
                    dataset_hf_filename=file_name,
                    dataset_release_date=file_release_date,
                )
            )

    org_latest_updates.clear()
    org_latest_updates.update(updated_latest_updates)
    return results


async def get_competition_weights(config: bt.Config, hf_api: HfApi) -> dict[str, float]:
    """Get competition weights from the competition_weights.yml file."""
    local_file_path = hf_hub_download(
        repo_id=config.datasets_config_hf_repo_id,
        repo_type="space",
        filename="competition_weights.yml"
    )
    
    with open(local_file_path, 'r', encoding='utf-8') as file:
        weights_data = yaml.safe_load(file)
    
    weights_dict = {}
    if weights_data is not None:  # Handle empty file case
        for item in weights_data:
            weights_dict[item['competition_id']] = item['weight']
    
    return weights_dict

@retry(tries=10, delay=5, logger=bt.logging)
def list_repo_tree_with_retry(hf_api, repo_id, repo_type, recursive, expand):
    return hf_api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        recursive=recursive,
        expand=expand,
    )

def get_local_dataset(local_dataset_dir: str) -> NewDatasetFile|None:
    """Gets dataset package from local directory

    Directory needs to have speficic structure:
    Dir
        - to_be_released <- datasets to test
        - already_released <- function moves exhaused datasets to this directory

    """
    import random
    list_of_new_data_packages: list[NewDatasetFile] = []
    to_be_released_dir = os.path.join(local_dataset_dir, "to_be_released")
    already_released_dir = os.path.join(local_dataset_dir, "already_released")

    if not os.path.exists(to_be_released_dir):
        bt.logging.warning(f"Directory {to_be_released_dir} does not exist.")
        return None

    if not os.path.exists(already_released_dir):
        os.makedirs(already_released_dir, exist_ok=True)

    for filename in os.listdir(to_be_released_dir):
        if filename.endswith(".zip") or filename.endswith(".encrypted"):
            filepath = os.path.join(to_be_released_dir, filename)
            try:
                # Get file modification time before moving
                file_release_date = datetime.fromtimestamp(os.path.getmtime(filepath))
                # Move the file to the already_released directory.
                final_path = os.path.join(already_released_dir, filename)
                shutil.move(filepath, final_path)
                bt.logging.info(f"Successfully processed and moved {filename} to {already_released_dir}")
                return NewDatasetFile(
                    competition_id=random.choice(["tricorder-3"]), 
                    dataset_hf_repo="local",
                    dataset_hf_filename=final_path,
                    dataset_release_date=file_release_date,
                )
            except Exception as e:
                bt.logging.error(f"Error processing {filename}: {e}")

    return None


def chain_miner_to_model_info(chain_miner_model: ChainMinerModel) -> ModelInfo:
    return ModelInfo(
        hf_repo_id=chain_miner_model.hf_repo_id,
        hf_model_filename=chain_miner_model.hf_model_filename,
        hf_code_filename=chain_miner_model.hf_code_filename,
        hf_repo_type=chain_miner_model.hf_repo_type,
        competition_id=chain_miner_model.competition_id,
        block=chain_miner_model.block,
        model_hash=chain_miner_model.model_hash,
    )

@retry(
    Exception,
    tries=5,
    delay=3,
    backoff=3,
    max_delay=81,
    logger=bt.logging
)
def _list_repo_tree_with_retry_sync(hf_api: HfApi, hf_repo_id: str) -> list:
    return list_repo_tree_with_retry(
        hf_api=hf_api,
        repo_id=hf_repo_id,
        repo_type="space",
    
        recursive=True,
        expand=True,
    )

def decode_raw(raw_hex: str) -> str:
    """
    Decode a hex string (0x-prefixed or not) to UTF-8 if possible,
    otherwise return the original string.
    """
    try:
        # strip optional “0x”
        hex_str = raw_hex[2:] if raw_hex.startswith("0x") else raw_hex
        data = binascii.unhexlify(hex_str)
        return data.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return raw_hex
    
def decode_params(obj):
    """
    Recursively walk a dict/list and decode any 0x-prefixed strings.
    """
    if isinstance(obj, dict):
        return {k: decode_params(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_params(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("0x"):
        return decode_raw(obj)
    return obj