import os
import asyncio
import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional
from datetime import datetime, timezone, timedelta

import bittensor as bt
from huggingface_hub import HfApi, HfFileSystem

from .models import ModelInfo
from .exceptions import ModelRunException
from .utils import decode_params



class ModelManager():
    def __init__(self, config, db_controller, subtensor: bt.subtensor, parent: Optional["CompetitionManager"] = None) -> None:
        self.config = config
        self.db_controller = db_controller

        if not os.path.exists(self.config.models.model_dir):
            os.makedirs(self.config.models.model_dir)
        self.api = HfApi(token=self.config.hf_token)
        self.hotkey_store: dict[str, ModelInfo] = {}
        self.parent = parent
        self.subtensor = subtensor

        # Default subtensor is not archive, but we need it for fetching the historical extrinsics data
        if subtensor is not None and "test" not in self.subtensor.chain_endpoint.lower():
            self.subtensor = bt.subtensor(network="archive")

    async def model_license_valid(self, hotkey) -> tuple[bool, Optional[str]]:
        hf_id = self.hotkey_store[hotkey].hf_repo_id
        try:
            model_info = self.api.model_info(hf_id, timeout=30)
        except Exception as e:
            bt.logging.error(f"Cannot get information about repository {hf_id}. Error: {e}")
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

        repo_path = os.path.join(model_info.hf_repo_id, model_info.hf_model_filename)

        is_valid, reason = await self.model_license_valid(hotkey)
        if not is_valid:
            hf_id = self.hotkey_store[hotkey].hf_repo_id

            if reason.startswith("HF API ERROR"):
                bt.logging.error(f"Could not verify license for {hf_id}: {reason.split(':', 1)[1]}")
                self.parent.error_results.append((hotkey, "Couldn't verify license due to HF API error"))
            else:
                bt.logging.error(f"License for {hf_id} not found or invalid")
                self.parent.error_results.append((hotkey, "MIT license not found or invalid"))
            return False


        bt.logging.debug(f"License found for {model_info.hf_repo_id}")
        # List files in the repository and get file date with retry
        files = None
        file_date = None
        for retry_counter in range(MAX_RETRIES):
            try:
                files = fs.ls(model_info.hf_repo_id)
                
                # Find the specific file and its upload date
                for file in files:
                    if model_info.hf_model_filename.lower() in file["name"].lower():
                        # Extract the upload date
                        file_date = file["last_commit"]["date"]
                        break
                        
                if file_date:  # If we found the file, break out of the retry loop
                    break
                else:
                    # File not found but repository exists, so we'll try again
                    if retry_counter < MAX_RETRIES - 1:
                        bt.logging.warning(f"Retry {retry_counter+1}/{MAX_RETRIES}: File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id}, retrying...")
                        await asyncio.sleep(RETRY_DELAY * (retry_counter + 1))
                    else:
                        bt.logging.error(f"File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id} after {MAX_RETRIES} attempts")
                        self.parent.error_results.append((hotkey, f"File {model_info.hf_model_filename} not found in repository {model_info.hf_repo_id}"))
                        return False
                        
            except Exception as e:
                if retry_counter < MAX_RETRIES - 1:
                    bt.logging.warning(f"Retry {retry_counter+1}/{MAX_RETRIES}: Failed to list files in repository {model_info.hf_repo_id}: {e}")
                    await asyncio.sleep(RETRY_DELAY * (retry_counter + 1))  # Exponential backoff
                else:
                    bt.logging.error(f"Failed to list files in repository {model_info.hf_repo_id} after {MAX_RETRIES} attempts: {e}")
                    self.parent.error_results.append((hotkey, f"Cannot list files in repo {model_info.hf_repo_id}"))
                    return False
        
        # We don't need this check anymore since we handle it in the retry loop
            
        # Parse and check if the model is too recent to download
        is_too_recent, parsed_date = self.is_model_too_recent(file_date, model_info.hf_model_filename, hotkey)
        if is_too_recent:
            self.parent.error_results.append((hotkey, f"Model is too recent"))
            return False
        
        file_date = parsed_date
        
        # Download the file with retry
        for retry_counter in range(MAX_RETRIES):
            try:
                model_info.file_path = self.api.hf_hub_download(
                    repo_id=model_info.hf_repo_id,
                    repo_type="model",
                    filename=model_info.hf_model_filename,
                    cache_dir=self.config.models.model_dir,
                    token=self.config.hf_token if hasattr(self.config, "hf_token") else None,
                )
                break
            except Exception as e:
                if retry_counter < MAX_RETRIES - 1:
                    bt.logging.warning(f"Retry {retry_counter+1}/{MAX_RETRIES}: Failed to download model file: {e}")
                    await asyncio.sleep(RETRY_DELAY * (retry_counter + 1))  # Exponential backoff
                else:
                    bt.logging.error(f"Failed to download model file after {MAX_RETRIES} attempts: {e}")
                    self.parent.error_results.append((hotkey, f"Failed to download model file: {e}"))
                    return False

        # Verify the downloaded file exists
        if not os.path.exists(model_info.file_path):
            bt.logging.error(f"Downloaded file does not exist at {model_info.file_path}")
            self.parent.error_results.append((hotkey, f"Downloaded file does not exist at {model_info.file_path}"))
            return False
        
        bt.logging.info(f"Successfully downloaded model file to {model_info.file_path}")
        return True

    def is_model_too_recent(self, file_date, filename, hotkey):
        """Checks if a model file was uploaded too recently based on the cutoff time.
        
        Args:
            file_date: The date when the file was uploaded (string or datetime)
            filename: The name of the model file
            hotkey: The hotkey of the miner
            
        Returns:
            tuple: (is_too_recent, parsed_date) where is_too_recent is a boolean indicating if the model
                  is too recent to download, and parsed_date is the parsed datetime object with timezone
        """
        # Ensure file_date is a datetime with timezone
        try:
            if isinstance(file_date, str):
                file_date = datetime.fromisoformat(file_date)
            if file_date.tzinfo is None:
                file_date = file_date.replace(tzinfo=timezone.utc)
        except Exception as e:
            bt.logging.error(f"Failed to parse file date {file_date}: {e}")
            return True, None

        bt.logging.debug(f"File {filename} was uploaded on: {file_date}")
        
        # Check if file is newer than our cutoff date (uploaded within last X minutes)
        now = datetime.now(timezone.utc)  # Get current time in UTC
        
        # Calculate time difference in minutes
        time_diff = (now - file_date).total_seconds() / 60
        
        if time_diff < self.config.models_query_cutoff:
            bt.logging.warning(f"Skipping model for hotkey {hotkey} because it was uploaded {time_diff:.2f} minutes ago, which is within the cutoff of {self.config.models_query_cutoff} minutes")
            return True, file_date
            
        return False, file_date
        

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


    async def get_pioneer_models(self, grouped_hotkeys: list[list[str]]) -> list[str]:
        """
        Async version: For each group of hotkeys, find the earliest chain submission
        whose HF commit predates the on-chain timestamp.
        """
        # Batch-prep all HF + extrinsic entries
        entries = await self._prepare_entries(grouped_hotkeys)
        if not entries:
            return []

        # Batch-fetch all needed blocks at once
        block_nums = [e['block_num'] for e in entries]
        block_map = self._batch_fetch_blocks(block_nums)

        # Collect valid candidates per group index
        candidates: dict[int, list[tuple[str, int]]] = {}
        for e in entries:
            hk = e['hotkey']
            gi = e['group_index']
            blk = e['block_num']
            ext_idx = e['ext_idx']

            block = block_map.get(blk)
            if not block:
                bt.logging.error(f"Failed to fetch block {blk} for {hk}")
                self.parent.error_results.append((hk, f"Block {blk} not available"))
                continue

            # Validate extrinsic and extract details
            res = self._validate_extrinsic_in_block(hk, block, ext_idx)
            if not res['valid']:
                bt.logging.error(f"Extrinsic validation failed for {hk}: {res['error']}")
                self.parent.error_results.append((hk, res['error']))
                continue

            # Update hotkey_store with decoded model_hash and log the extrinsic
            self.hotkey_store[hk].model_hash = res['decoded_params'].get('model_hash')
            bt.logging.info(
                f"Found Extrinsic {blk}-{ext_idx} → {res['module']}.{res['function']} "
                f"{res['decoded_params']} for hotkey {hk}"
            )

            # Compare HF commit date to chain timestamp
            ts = self.db_controller.get_block_timestamp(blk).replace(tzinfo=timezone.utc)
            if e['hf_commit_date'] <= ts:
                candidates.setdefault(gi, []).append((hk, blk))

        # Elect pioneers: lowest block per group
        pioneers: list[str] = []
        for gi, lst in candidates.items():
            pioneer = min(lst, key=lambda x: x[1])[0]
            pioneers.append(pioneer)
        return pioneers

    async def _prepare_entries(self, grouped_hotkeys: list[list[str]]) -> list[dict]:
        fs = HfFileSystem(token=self.config.hf_token) if self.config.hf_token else HfFileSystem()
        entries: list[dict] = []
        for gi, group in enumerate(grouped_hotkeys):
            for hk in group:
                mi = self.hotkey_store.get(hk)
                if not mi:
                    bt.logging.error(f"Model info for hotkey {hk} not found.")
                    self.parent.error_results.append((hk, "Model info not found."))
                    continue
                try:
                    files = fs.ls(mi.hf_repo_id)
                    hf_date = self._extract_commit_date(files, mi.hf_model_filename)

                    rec = self._load_extrinsic_record(fs, mi.hf_repo_id)
                    blk, idx = self._parse_extrinsic_id(rec['extrinsic'], hk)

                    entries.append({
                        'hotkey': hk,
                        'group_index': gi,
                        'block_num': blk,
                        'ext_idx': idx,
                        'hf_commit_date': hf_date
                    })
                except Exception as e:
                    bt.logging.error(f"HF entry fail for {hk}: {e}")
                    self.parent.error_results.append((hk, str(e)))
        return entries

    def _extract_commit_date(self, files: list[dict], filename: str) -> datetime:
        for f in files:
            if f['name'].endswith(filename):
                dt = datetime.fromisoformat(str(f['last_commit']['date']))
                if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
        raise FileNotFoundError(f"{filename} not found in repo")

    def _load_extrinsic_record(self, fs: HfFileSystem, repo_id: str) -> dict:
        matches = fs.glob(f"{repo_id}/extrinsic_record.json", refresh=True)
        if not matches:
            raise FileNotFoundError("extrinsic_record.json missing")
        with fs.open(matches[0], 'r', encoding='utf-8') as rf:
            rec = json.load(rf)
        if not rec.get('hotkey') or not rec.get('extrinsic'):
            raise ValueError(f"Invalid record contents: {rec}")
        return rec

    def _parse_extrinsic_id(self, extrinsic_id: str, hotkey: str) -> tuple[int, int]:
        try:
            blk_str, idx_str = extrinsic_id.split('-', 1)
            block_num = int(blk_str)
            ext_idx = int(idx_str, 16 if idx_str.lower().startswith('0x') else 10)
            return block_num, ext_idx
        except Exception as e:
            raise ValueError(f"Cannot parse extrinsic_id {extrinsic_id}: {e}")

    def _batch_fetch_blocks(self, block_numbers: list[int]) -> dict[int, dict]:
        substrate = self.subtensor.substrate
        block_map: dict[int, dict] = {}
        with substrate.rpc_batch() as batch:
            for blk in block_numbers:
                batch.request('chain_getBlock', [blk])
            results = batch.send()
        for blk, res in zip(block_numbers, results):
            if res.get('result', {}).get('block'):
                block_map[blk] = res['result']['block']
            else:
                bt.logging.error(f"Failed to fetch block {blk}")
        return block_map

    def _validate_extrinsic_in_block(self, hotkey: str, block: dict, ext_idx: int) -> dict:
        extr = block.get('extrinsics', [])
        if ext_idx < 0 or ext_idx >= len(extr):
            return {'valid': False, 'error': f"Idx {ext_idx} out of bounds"}

        e = extr[ext_idx]
        signer = e.value.get('address')
        if signer != hotkey:
            return {'valid': False, 'error': 'Signer mismatch'}

        call = e.value.get('call', {})
        raw = {p['name']: p['value'] for p in call.get('call_args', [])}
        dec = decode_params(raw)

        # Check model-hash
        chain_hash = dec['fields'][0]['Raw92'].split(':')[-1]
        part_hash = self.hotkey_store[hotkey].model_hash
        if chain_hash != part_hash:
            return {'valid': False, 'error': 'Hash mismatch'}

        # Return full details for logging/assignment
        return {
            'valid': True,
            'module': call.get('call_module'),
            'function': call.get('call_function'),
            'decoded_params': dec
        }