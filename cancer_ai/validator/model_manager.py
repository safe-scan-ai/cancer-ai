import os
import asyncio
import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional
from datetime import datetime, timezone, timedelta

import bittensor as bt
from huggingface_hub import HfApi, HfFileSystem

from .models import ModelInfo
from ..chain_models_store import _create_archive_subtensor
from .exceptions import ModelRunException
from .utils import decode_params
from websockets.client import OPEN as WS_OPEN


class ModelManager():
    def __init__(self, config, db_controller, subtensor: bt.subtensor, parent: Optional["CompetitionManager"] = None) -> None:
        self.config = config
        self.db_controller = db_controller

        if not os.path.exists(self.config.models.model_dir):
            os.makedirs(self.config.models.model_dir)
        self.api = HfApi(token=self.config.hf_token)
        self.hotkey_store: dict[str, ModelInfo] = {}
        self.parent = parent
        self.dataset_release_date: Optional[datetime] = None 
        
        # Retry configuration for HF API calls
        self.hf_max_retries = 4
        self.hf_initial_delay = 2
        self.hf_max_delay = 30

        if subtensor is not None and "test" not in subtensor.chain_endpoint.lower():
            archive_node_url = config.archive_node_url
            archive_node_fallback_url = config.archive_node_fallback_url

            if archive_node_url and archive_node_fallback_url:
                subtensor = _create_archive_subtensor(archive_node_url, archive_node_fallback_url)
            else:
                subtensor = bt.subtensor(network="archive")
        self.subtensor = subtensor

        # Capture the original connect() and override with _ws_connect wrapper
        # Substrate-interface calls connect() on every RPC under the hood,
        # so we wrap it to reuse the same socket unless it's truly closed.
        self._orig_ws_connect = self.subtensor.substrate.connect
        self.subtensor.substrate.connect = self._ws_connect

        ws = self.subtensor.substrate.connect()
        bt.logging.info(f"Initial WebSocket state: {ws.state}")

    def _ws_connect(self, *args, **kwargs):
        """
        Replacement for substrate.connect().
        Reuses existing WebSocketClientProtocol if State.OPEN;
        otherwise performs a fresh handshake via original connect().
        """
        # Check current socket
        current = getattr(self.subtensor.substrate, "ws", None)
        if current is not None and current.state == WS_OPEN:
            return current

        # If socket not open, reconnect
        bt.logging.warning("⚠️ Subtensor WebSocket not OPEN—reconnecting…")
        try:
            new_ws = self._orig_ws_connect(*args, **kwargs)
        except Exception as e:
            bt.logging.error("Failed to reconnect WebSocket: %s", e, exc_info=True)
            raise

        # Update the substrate.ws attribute so future calls reuse this socket
        setattr(self.subtensor.substrate, "ws", new_ws)
        return new_ws

    async def _hf_api_call_with_retry(self, api_call, *args, **kwargs):
        """Wrapper for HF API calls with exponential backoff retry logic."""
        for attempt in range(self.hf_max_retries):
            try:
                return api_call(*args, **kwargs)
            except Exception as e:
                if attempt == self.hf_max_retries - 1:
                    bt.logging.error(f"HF API call failed after {self.hf_max_retries} attempts. Error: {e}")
                    raise
                
                delay = min(self.hf_initial_delay * (2 ** attempt), self.hf_max_delay)
                bt.logging.warning(f"HF API call attempt {attempt + 1}/{self.hf_max_retries} failed. Retrying in {delay}s. Error: {e}")
                await asyncio.sleep(delay)

    async def model_license_valid(self, hotkey) -> tuple[bool, Optional[str]]:
        hf_id = self.hotkey_store[hotkey].hf_repo_id
        
        try:
            model_info = await self._hf_api_call_with_retry(self.api.model_info, hf_id, timeout=30)
        except Exception as e:
            return False, f"HF API ERROR: {e}"

        meta_license = None
        if model_info.card_data:
            meta_license = model_info.card_data.get("license")

        if meta_license and "mit" in meta_license.lower():
            return True, None

        return False, "NOT_MIT"

    async def download_miner_model(self, hotkey, token: Optional[str] = None) -> bool:
        """Downloads the newest model from Hugging Face and saves it to disk.
        Returns:
            bool: True if the model was downloaded successfully, False otherwise.
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 2  # seconds
        
        model_info = self.hotkey_store[hotkey]
                
        fs = HfFileSystem(token=token)

        

        is_valid, reason = await self.model_license_valid(hotkey)
        if not is_valid:
            hf_id = self.hotkey_store[hotkey].hf_repo_id

            if reason.startswith("HF API ERROR"):
                bt.logging.error(f"Could not verify license for {hf_id}: {reason.split(':', 1)[1]}")
                # self.parent.error_results.append((hotkey, "Couldn't verify license due to HF API error"))
            else:
                bt.logging.error(f"License for {hf_id} not found or invalid")
                # self.parent.error_results.append((hotkey, "MIT license not found or invalid"))
            return False


        bt.logging.debug(f"License found for {model_info.hf_repo_id}")
        # List files in the repository and get file date with retry
        
        file_found = False
        for retry_counter in range(MAX_RETRIES):
            try:
                files = fs.ls(model_info.hf_repo_id)
                
                # Find the specific file and its upload date
                target_filename = model_info.hf_model_filename.lower()
                for file in files:
                    filename = file["name"].split("/")[-1].lower()
                    if filename == target_filename:
                       
                        file_found = True
                        break
                        
                if file_found:  # If we found the file, break out of the retry loop
                    break
                else:
                    # File not found but repository exists, so we'll try again
                    if retry_counter < MAX_RETRIES - 1:
                        bt.logging.warning(f"Retry {retry_counter+1}/{MAX_RETRIES}: File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id}, retrying...")
                        await asyncio.sleep(RETRY_DELAY * (retry_counter + 1))
                    else:
                        bt.logging.error(f"File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id} after {MAX_RETRIES} attempts")
                        # self.parent.error_results.append((hotkey, f"File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id}"))
                        return False
                        
            except Exception as e:
                if retry_counter < MAX_RETRIES - 1:
                    bt.logging.warning(f"Retry {retry_counter+1}/{MAX_RETRIES}: Failed to list files in repository {model_info.hf_repo_id}: {e}")
                    await asyncio.sleep(RETRY_DELAY * (retry_counter + 1))  # Exponential backoff
                else:
                    bt.logging.error(f"Failed to list files in repository {model_info.hf_repo_id} after {MAX_RETRIES} attempts: {e}")
                    # self.parent.error_results.append((hotkey, f"Cannot list files in repo {model_info.hf_repo_id}"))
                    return False
        
            
        # Check if model is too recent using on-chain block timestamp
        if model_info.block is None:
            bt.logging.error(f"Model for hotkey {hotkey} has no block number. Cannot validate submission date.")
            return False        

        is_too_recent, submission_date = self.is_model_too_recent(
            model_info.block,  # Use on-chain block instead of HF file date
            model_info.hf_model_filename, 
            hotkey
        )
        if is_too_recent:
            bt.logging.warning(f"Model for hotkey {hotkey} was submitted too recently (on-chain date: {submission_date})")
            return False
        
        # Download the file with retry
        try:
            model_info.file_path = await self._hf_api_call_with_retry(
                self.api.hf_hub_download,
                repo_id=model_info.hf_repo_id,
                repo_type="model",
                filename=model_info.hf_model_filename,
                cache_dir=self.config.models.model_dir,
                token=self.config.hf_token if hasattr(self.config, "hf_token") else None,
            )
        except Exception as e:
            bt.logging.error(f"Failed to download model file: {e}")
            return False

        # Verify the downloaded file exists
        if not os.path.exists(model_info.file_path):
            bt.logging.error(f"Downloaded file does not exist at {model_info.file_path}")
            # self.parent.error_results.append((hotkey, f"Downloaded file does not exist at {model_info.file_path}"))
            return False
        
        # Check model size for efficiency scoring
        model_size_bytes = os.path.getsize(model_info.file_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Store model size for efficiency scoring
        model_info.model_size_mb = model_size_mb
        
        # Log model size with efficiency implications
        if model_size_mb <= 50:
            bt.logging.info(
                f"Model size: {model_size_mb:.1f}MB - Full efficiency score"
            )
        elif model_size_mb <= 150:
            efficiency_percent = ((150 - model_size_mb) / 100) * 100
            bt.logging.info(
                f"Model size: {model_size_mb:.1f}MB - {efficiency_percent:.0f}% efficiency score"
            )
        else:
            bt.logging.warning(
                f"Model size: {model_size_mb:.1f}MB - 0% efficiency score (exceeds 150MB)"
            )
        
        bt.logging.info(f"Successfully downloaded model file to {model_info.file_path}")
        return True

    def is_model_too_recent(self, block, filename, hotkey):
        """Checks if a model was submitted too recently based on on-chain block timestamp.
    
            Uses the immutable blockchain block timestamp instead of file upload date
            to prevent manipulation and ensure security.

            If dataset_release_date is set, compares model submission date against dataset release date.
            Otherwise, falls back to comparing against current time.

            Args:
                block: The block number when the model was submitted on-chain
                filename: The name of the model file (for logging)
                hotkey: The hotkey of the miner

            Returns:
                tuple: (is_too_recent, submission_date) where is_too_recent is a boolean indicating 
                       if the model is too recent to download, and submission_date is the datetime 
                       object with timezone
        """
        # Get on-chain submission timestamp from block number
        try:
            submission_date = self.db_controller.get_block_timestamp(block)
        except Exception as e:
            bt.logging.error(f"Failed to get block timestamp: {e}")
            return True, None  # Reject if we can't get timestamp
    
        bt.logging.debug(f"Model {filename} was submitted on-chain on: {submission_date}")
        
        # If dataset_release_date is set, compare against it instead of current time
        if self.dataset_release_date is not None:
            # Ensure dataset_release_date has timezone
            dataset_release = self.dataset_release_date
            if dataset_release.tzinfo is None:
                dataset_release = dataset_release.replace(tzinfo=timezone.utc)
            
            # Check if model was uploaded AFTER dataset release
            if submission_date > dataset_release:
                bt.logging.warning(
                    f"Skipping model for hotkey {hotkey} because it was submitted {submission_date} "
                    f"AFTER dataset release {dataset_release}. This indicates potential data leakage."
                )
                return True, submission_date
            
            # Also check the 120-minute cutoff from dataset release time
            time_diff = (dataset_release - submission_date).total_seconds() / 60
            if time_diff < self.config.models_query_cutoff:
                bt.logging.warning(
                    f"Skipping model for hotkey {hotkey} because it was submitted "
                    f"{time_diff:.2f} minutes before dataset release, which is within the cutoff of "
                    f"{self.config.models_query_cutoff} minutes"
                )
                return True, submission_date
            
            return False, submission_date
        else:
            # Fallback to old behavior if dataset_release_date is not set
            now = datetime.now(timezone.utc)  # Get current time in UTC
            
            # Calculate time difference in minutes
            time_diff = (now - submission_date).total_seconds() / 60
        
        if time_diff < self.config.models_query_cutoff:
            bt.logging.warning(f"Skipping model for hotkey {hotkey} because it was uploaded {time_diff:.2f} minutes ago, which is within the cutoff of {self.config.models_query_cutoff} minutes")
            return True, submission_date
            
        return False, submission_date
        

    def add_model(
        self,
        hotkey,
        hf_repo_id,
        hf_model_filename,
        hf_code_filename=None,
        hf_repo_type=None,
    ) -> None:
        """Saves locally information about a new model."""
        self.hotkey_store[hotkey] = ModelInfo(
            hf_repo_id, hf_model_filename, hf_code_filename, hf_repo_type
        )

    def delete_model(self, hotkey) -> None:
        """Deletes locally information about a model and the corresponding file on disk."""

        bt.logging.info(f"Deleting model: {hotkey}")
        if hotkey in self.hotkey_store and self.hotkey_store[hotkey].file_path:
            os.remove(self.hotkey_store[hotkey].file_path)
        self.hotkey_store[hotkey] = None

    def _extract_raw_value(self, fields: list[dict]) -> str:
        """Return the first Raw<n> value from the `fields` list."""
        for field in fields:
            for k, v in field.items():
                if k.startswith("Raw"):
                    return v
        raise KeyError("No Raw<n> entry found in `info.fields`")


    def get_pioneer_models(self, grouped_hotkeys: list[list[str]]) -> tuple[list[str], dict[str, str]]:
        """
        Does a check on whether chain submit date was later then HF commit date. If not slashes.
        Compares chain submit date duplicated models to elect a pioneer based on block of submission (date)
        Every hotkey that is not included in the candidate list is a subject to slashing.
        
        Returns:
            pioneers: list of hotkeys that are pioneers
            errors: dict of hotkey -> error message for models that failed validation
        """
        pioneers = []
        errors = {}

        if self.config.hf_token:
            fs = HfFileSystem(token=self.config.hf_token)
        else:
            fs = HfFileSystem()

        for group in grouped_hotkeys:
            candidate_hotkeys = []

            for hotkey in group:
                model_info = self.hotkey_store.get(hotkey)
                if not model_info:
                    bt.logging.error(f"Model info for hotkey {hotkey} not found.")
                    errors[hotkey] = "Model info not found"
                    continue

                try:
                    matches = fs.glob(f"{model_info.hf_repo_id}/extrinsic_record.json", refresh=True)
                    if not matches:
                        raise FileNotFoundError("extrinsic_record.json not found in repo")

                    record_path = matches[0]
                    with fs.open(record_path, "r", encoding="utf-8") as rf:
                        record_data = json.load(rf)

                    file_hotkey = record_data.get("hotkey")
                    extrinsic_id = record_data.get("extrinsic")
                    if file_hotkey != hotkey or not extrinsic_id:
                        raise ValueError(f"Invalid record contents: {record_data}")

                except Exception as e:
                    bt.logging.error(f"Failed to load HF repo extrinsic record for {hotkey}: {e}", exc_info=True)
                    errors[hotkey] = f"Invalid or missing extrinsic record in HF repo: {e}"
                    continue

                try:
                    blk_str, idx_str = extrinsic_id.split("-", 1)
                    block_num = int(blk_str)
                    ext_idx = int(idx_str, 16 if idx_str.lower().startswith("0x") else 10)

                    block_data = self.subtensor.substrate.get_block(block_number=block_num)
                    extrinsics = block_data.get("extrinsics", [])

                    if ext_idx < 0 or ext_idx >= len(extrinsics):
                        raise IndexError(f"Extrinsic index {ext_idx} out of bounds")

                    ext = extrinsics[ext_idx]
                    signer = ext.value.get("address")
                    if signer != hotkey:
                        raise ValueError(f"Extrinsic signer {signer} != expected hotkey {hotkey}")

                    call = ext.value.get("call", {})
                    module   = call.get("call_module")
                    function = call.get("call_function")

                    raw_params = {p["name"]: p["value"] for p in call.get("call_args", [])}
                    decoded_params = decode_params(raw_params)

                except ValueError as e:
                    # Validation errors (signer mismatch, hash mismatch)
                    bt.logging.error(f"Extrinsic validation failed for {hotkey}: {e}")
                    errors[hotkey] = f"Extrinsic validation failed: {e}"
                    continue
                except Exception as e:
                    bt.logging.exception(f"Failed to decode extrinsic {extrinsic_id} for {hotkey}: {e}", exc_info=True)
                    errors[hotkey] = f"Extrinsic decoding failed: {e}"
                    continue

                bt.logging.info(f"Found Extrinsic {extrinsic_id} → {module}.{function} {decoded_params} for hotkey {hotkey}")
                try:
                    info    = decoded_params.get("info", {})
                    fields  = info.get("fields", [])
                    raw_val = self._extract_raw_value(fields)
                    chain_model_hash       = raw_val.split(":")[-1]
                    participant_model_hash = self.hotkey_store[hotkey].model_hash

                    if chain_model_hash != participant_model_hash:
                        raise ValueError(
                            f"chain {chain_model_hash} != participant {participant_model_hash}"
                        )

                except Exception as e:
                    bt.logging.error(f"Model hash comparison failed for {hotkey}: {e}", exc_info=True)
                    errors[hotkey] = f"Model hash mismatch or extraction error: {e}"
                    continue
                
                candidate_hotkeys.append((hotkey, block_num))

            if candidate_hotkeys:
                pioneer_hotkey = min(candidate_hotkeys, key=lambda x: x[1])[0]
                pioneers.append(pioneer_hotkey)
        return pioneers, errors
