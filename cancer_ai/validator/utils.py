from enum import Enum
import os
import asyncio
import bittensor as bt
import json
import yaml
from cancer_ai.validator.models import CompetitionsListModel, CompetitionModel
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from cancer_ai.validator.models import (
    NewDatasetFile,
    OrganizationDataReferenceFactory,
)
from datetime import datetime
from typing import Any


class ModelType(Enum):
    ONNX = "ONNX"
    TENSORFLOW_SAVEDMODEL = "TensorFlow SavedModel"
    KERAS_H5 = "Keras H5"
    PYTORCH = "PyTorch"
    SCIKIT_LEARN = "Scikit-learn"
    XGBOOST = "XGBoost"
    UNKNOWN = "Unknown format"


import time
from functools import wraps


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

    except Exception:
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


def get_competition_config(path: str) -> CompetitionsListModel:
    with open(path, "r") as f:
        competitions_json = json.load(f)
    competitions = [CompetitionModel(**item) for item in competitions_json]
    return CompetitionsListModel(competitions=competitions)


async def fetch_organization_data_references(
    hf_repo_id: str, hf_token: str, hf_api: HfApi
) -> list[dict]:
    bt.logging.trace(
        f"Fetching organization data references from Hugging Face repo {hf_repo_id}"
    )

    # prevent stale connections
    custom_headers = {"Connection": "close"}

    files = hf_api.list_repo_tree(
        repo_id=hf_repo_id,
        repo_type="space",
        token=hf_token,
        recursive=True,
        expand=True,
    )

    yaml_data = []

    for file_info in files:
        if isinstance(file_info, RepoFile):
            file_path = file_info.path

            if file_path.startswith("datasets/") and file_path.endswith(".yaml"):
                local_file_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    repo_type="space",
                    token=hf_token,
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

                with open(local_file_path, "r") as f:
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

                with open(file_path, "r") as f:
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
        yaml_data = file["yaml_data"]  # each yaml_data is a list of dicts
        for entry in yaml_data:
            # Remap 'org_id' to 'organization_id' if needed.
            if "org_id" in entry:
                entry["organization_id"] = entry.pop("org_id")
            all_orgs.append(entry)

    # Prepare the dictionary to update the factory.
    update_data = {"organizations": all_orgs}

    # Update the singleton instance.
    factory = OrganizationDataReferenceFactory.get_instance()
    factory.update_from_dict(update_data)


async def get_new_dataset_files(
    hf_api: HfApi, org_latest_updates: dict
) -> list[NewDatasetFile]:
    """
    For each OrganizationDataReference, this function:
      - Lists .zip files in the specified dataset directory (non-recursive)
      - Returns .zip files newer than the last known update

    First check: Returns the newest .zip file
    Subsequent checks: Returns all .zip files newer than last known update
    """
    factory = OrganizationDataReferenceFactory.get_instance()
    if not factory.organizations:
        return []

    results = []

    for org in factory.organizations:
        # Get files directly in the specified directory
        files = hf_api.list_repo_tree(
            repo_id=org.dataset_hf_repo,
            repo_type="dataset",
            token=None,
            recursive=False,
            expand=True,
        )

        # Filter .zip files and get their dates
        file_dates = [
            (f.path, f.last_commit.date)
            for f in files
            if isinstance(f, RepoFile)
            and f.path.startswith(org.dataset_hf_dir)
            and f.path.endswith('.zip')
            and f.last_commit
            and f.last_commit.date
        ]

        if not file_dates:
            continue

        # Find newest file date
        newest_date = max(date for _, date in file_dates)
        last_known_update = org_latest_updates.get(org.organization_id)

        # Get new files based on check type
        if last_known_update is None:
            # First check - take newest file
            new_files = [path for path, date in file_dates if date == newest_date][:1]
        else:
            # Subsequent checks - take all files newer than last known update
            new_files = [path for path, date in file_dates if date > last_known_update]

        # Update last known update and record results
        org_latest_updates[org.organization_id] = newest_date
        results.extend(
            NewDatasetFile(
                competition_id=org.competition_id,
                dataset_hf_repo=org.dataset_hf_repo,
                dataset_hf_filename=file_path,
            )
            for file_path in new_files
        )

    return results
