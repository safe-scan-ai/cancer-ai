from enum import Enum
import os
import asyncio
import bittensor as bt
import json
import yaml
from cancer_ai.validator.models import CompetitionsListModel, CompetitionModel
from huggingface_hub import HfApi, hf_hub_download
from cancer_ai.validator.models import (
    DatasetReference,
    OrganizationDataReference,
    OrganizationDataReferenceFactory,
)
from datetime import datetime


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

async def fetch_organization_data_references(hf_repo_id: str, hf_token: str, hf_api: HfApi) -> list[dict]:
    bt.logging.trace(f"Fetching organization data references from Hugging Face repo {hf_repo_id}")

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
        if file_info.__class__.__name__ == 'RepoFile':
            file_path = file_info.path

            if file_path.startswith('datasets/') and file_path.endswith('.yaml'):
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
                    bt.logging.warning(f"Could not get the last commit date for {file_path}")
                    date_uploaded = None

                with open(local_file_path, 'r') as f:
                    data = yaml.safe_load(f)

                yaml_data.append({
                    'file_name': file_path,
                    'yaml_data': data,
                    'date_uploaded': date_uploaded
                })
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
            if file_name.endswith('.yaml'):
                file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(file_path, local_repo_path)
                commit_date = datetime.fromtimestamp(os.path.getmtime(file_path))

                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)

                yaml_data.append({
                    'file_name': relative_path,
                    'yaml_data': data,
                    'date_uploaded': commit_date
                })

    return yaml_data

async def get_new_organization_data_updates(fetched_yaml_files: list[dict]) -> list[DatasetReference]:
    factory = OrganizationDataReferenceFactory.get_instance()
    data_references: list[DatasetReference] = []

    for file in fetched_yaml_files:
        yaml_data = file['yaml_data']
        date_uploaded = file['date_uploaded']

        org_id = yaml_data[0]['organization_id']
        existing_org = next((org for org in factory.organizations if org.organization_id == org_id), None)

        if not existing_org:
            for entry in yaml_data:
                data_references.append(DatasetReference(
                    competition_id=entry['competition_id'],
                    dataset_hf_repo=entry['dataset_hf_repo'],
                    dataset_hf_filename=entry['dataset_hf_filename'],
                    dataset_hf_repo_type=entry['dataset_hf_repo_type'],
                    dataset_size=entry['dataset_size']
                ))
        
        if existing_org and date_uploaded != existing_org.date_uploaded:
            last_entry = yaml_data[-1]
            data_references.append(DatasetReference(
                competition_id=last_entry['competition_id'],
                dataset_hf_repo=last_entry['dataset_hf_repo'],
                dataset_hf_filename=last_entry['dataset_hf_filename'],
                dataset_hf_repo_type=last_entry['dataset_hf_repo_type'],
                dataset_size=last_entry['dataset_size']
            ))

    return data_references

async def update_organizations_data_references(fetched_yaml_files: list[dict]):
    bt.logging.trace("Updating organizations data references")
    factory = OrganizationDataReferenceFactory.get_instance()
    factory.organizations.clear()

    for file in fetched_yaml_files:
        yaml_data = file['yaml_data']
        new_org = OrganizationDataReference(
            organization_id=yaml_data[0]['organization_id'],
            contact_email=yaml_data[0]['contact_email'],
            bittensor_hotkey=yaml_data[0]['bittensor_hotkey'],
            data_packages=[
                DatasetReference(
                    competition_id=dp['competition_id'],
                    dataset_hf_repo=dp['dataset_hf_repo'],
                    dataset_hf_filename=dp['dataset_hf_filename'],
                    dataset_hf_repo_type=dp['dataset_hf_repo_type'],
                    dataset_size=dp['dataset_size']
                )
                for dp in yaml_data
            ],
            date_uploaded=file['date_uploaded']
        )
        factory.add_organizations([new_org])
