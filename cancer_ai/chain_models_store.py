from typing import Optional, Type
import argparse
import sys
from pathlib import Path
import asyncio

import bittensor as bt
from pydantic import BaseModel, Field, ConfigDict
from retry import retry
from websockets.client import OPEN as WS_OPEN

from .utils.archive_node import get_archive_subtensor, WebSocketManager
from .utils.structured_logger import log


class ChainMinerModel(BaseModel):
    """Uniquely identifies a trained model"""

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    competition_id: Optional[str] = Field(description="The competition id")
    hf_repo_id: Optional[str] = Field(description="Hugging Face repository id.")
    hf_model_filename: Optional[str] = Field(description="Hugging Face model filename.")
    hf_repo_type: Optional[str] = Field(
        description="Hugging Face repository type.", default="model"
    )
    hf_code_filename: Optional[str] = Field(
        description="Hugging Face code zip filename."
    )
    block: Optional[int] = Field(
        description="Block on which this model was claimed on the chain."
    )

    model_hash: Optional[str] = Field(
        description="64-character SHA-256 of the model file from Hugging Face."
    )

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        model_filename_no_ext = Path(self.hf_model_filename).stem if self.hf_model_filename else ""
        return f"{self.hf_repo_id}:{model_filename_no_ext}:{self.competition_id}:{self.model_hash}"

    @property
    def hf_link(self) -> str:
        """Returns the Hugging Face link for the model."""
        return f"https://huggingface.co/{self.hf_repo_id}/blob/main/{self.hf_model_filename}"

    @property
    def hf_code_link(self) -> str:
        """Returns the Hugging Face link for the code."""
        if self.hf_code_filename is None:
            return ""
        return f"https://huggingface.co/{self.hf_repo_id}/blob/main/{self.hf_code_filename}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ChainMinerModel"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        if len(tokens) != 4:
            return None
        base_filename = tokens[1]
        return cls(
            hf_repo_id=tokens[0],
            hf_model_filename=f"{base_filename}.onnx",
            hf_code_filename=f"{base_filename}.zip",
            competition_id=tokens[2],
            hf_repo_type="model",
            model_hash=tokens[3],
            block=None,
        )


class ChainModelMetadata:
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        netuid: int,
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.wallet = wallet  # Wallet is only needed to write to the chain, not to read.
        self.netuid = netuid
        
        # Use WebSocketManager for connection management
        self.ws_manager = WebSocketManager(subtensor)
        self.subnet_metadata = self.subtensor.metagraph(self.netuid)

    async def store_model_metadata(self, model_id: ChainMinerModel):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        self.subtensor.commit(
            self.wallet,
            self.netuid,
            model_id.to_compressed_str(),
        )

    async def retrieve_model_metadata(self, hotkey: str, uid: int) -> ChainMinerModel:
        """Retrieves model metadata on this subnet for specific hotkey"""
        await asyncio.sleep(2)  # temp fix for 429
        
        metadata = get_metadata(self.subtensor, self.netuid, hotkey)

        if metadata is None:
            raise ValueError(f"No metadata found for hotkey {hotkey}")

        chain_str = get_commitment(self.subtensor, self.netuid, uid)
        
        if chain_str is None:
            raise ValueError(
                f"No chain string found for hotkey '{hotkey}' and uid {uid}"
            )

        model = ChainMinerModel.from_compressed_str(chain_str)
        bt.logging.trace(f"Model: {model}")
        if model is None:
            raise ValueError(
                f"Metadata might be in old format or invalid for hotkey '{hotkey}'. Raw value: {chain_str}"
            )
        
        # The block id at which the metadata is stored
        model.block = metadata["block"]
        
        # Set structured logger context for sync operations
        log.set_competition_action("sync")
        log.set_miner_hotkey(hotkey)
        if model.competition_id:
            log.set_competition(model.competition_id)
        
        bt.logging.info(f"Retrieved model metadata for hotkey {hotkey}, competition {model.competition_id}")
        
        # Clear miner-specific context after sync to avoid leaking
        log.set_miner_hotkey("")
        log.set_competition_action("")
        
        return model
    
    def close(self):
        """Close the WebSocket connection."""
        self.ws_manager.close()

@retry(tries=12, delay=1, backoff=2, max_delay=30)
def get_metadata(subtensor, netuid, hotkey):
    """Synchronous metadata fetch with retry logic."""
    try:
        return bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)
    except Exception as e:
        raise RuntimeError(
            f"Failed to get metadata from chain for hotkey '{hotkey}': {e}"
        ) from e
    
@retry(tries=12, delay=0.5, backoff=2, max_delay=20)
def get_commitment(subtensor, netuid, uid):
    """Synchronous commitment fetch with exponential-backoff and contextual errors."""
    try:
        return subtensor.get_commitment(netuid, uid)
    except Exception as e:
        raise RuntimeError(
            f"Failed to get commitment from chain for uid={uid}: {e}"
        ) from e