import functools
from typing import Optional, Type
import asyncio
from functools import wraps

import bittensor as bt
from pydantic import BaseModel, Field
from retry import retry



class ChainMinerModel(BaseModel):
    """Uniquely identifies a trained model"""

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

    class Config:
        arbitrary_types_allowed = True

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.hf_repo_id}:{self.hf_model_filename}:{self.hf_code_filename}:{self.competition_id}:{self.hf_repo_type}"

    @property
    def hf_link(self) -> str:
        """Returns the Hugging Face link for the model."""
        return f"https://huggingface.co/{self.hf_repo_id}/blob/main/{self.hf_model_filename}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ChainMinerModel"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        if len(tokens) != 5:
            return None
        return cls(
            hf_repo_id=tokens[0],
            hf_model_filename=tokens[1],
            hf_code_filename=tokens[2],
            competition_id=tokens[3],
            hf_repo_type=tokens[4],
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
        self.wallet = (
            wallet  # Wallet is only needed to write to the chain, not to read.
        )
        self.netuid = netuid
        self.subnet_metadata = self.subtensor.metagraph(self.netuid)

    async def store_model_metadata(self, model_id: ChainMinerModel):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        self.subtensor.commit(
            self.wallet,
            self.netuid,
            model_id.to_compressed_str(),
        )

    async def retrieve_model_metadata(self, hotkey: str) -> Optional[ChainMinerModel]:
        """Retrieves model metadata on this subnet for specific hotkey"""
        metadata = await get_metadata_with_timeout(self.subtensor, self.netuid, hotkey)
        if metadata is None:
            self.subnet_metadata = self.subtensor.metagraph(self.netuid)
            metadata = await get_metadata_with_timeout(self.subtensor, self.netuid, hotkey)
            if metadata is None:
                return None

        uids = self.subnet_metadata.uids
        hotkeys = self.subnet_metadata.hotkeys
        uid = next((uid for uid, hk in zip(uids, hotkeys) if hk == hotkey), None)

        try:
            chain_str = self.subtensor.get_commitment(self.netuid, uid)
        except Exception as e:
            bt.logging.debug(f"Failed to retrieve commitment for hotkey {hotkey}: {e}")
            return None

        if not metadata:
            return None

        model = None
        try:
            model = ChainMinerModel.from_compressed_str(chain_str)
            bt.logging.debug(f"Model: {model}")
            if model is None:
                bt.logging.error(
                    f"Metadata might be in old format on the chain for hotkey {hotkey}. Raw value: {chain_str}"
                )
                return None
        except Exception:
            # If the metadata format is not correct on the chain then we return None.
            bt.logging.error(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}. Raw value: {chain_str}"
            )
            return None
        # The block id at which the metadata is stored
        model.block = metadata["block"]
        return model

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                bt.logging.debug("Metadata retrieval timed out, refreshing subnet metadata")
                return None
        return wrapper
    return decorator

@retry(tries=10, delay=5)
def get_metadata_with_retry(subtensor, netuid, hotkey):
    return bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)

@timeout(10)  # 10 second timeout
async def get_metadata_with_timeout(subtensor, netuid, hotkey):
    return get_metadata_with_retry(subtensor, netuid, hotkey)
