import functools
from typing import Optional, Type
import asyncio
from functools import wraps

import bittensor as bt
from pydantic import BaseModel, Field
from retry import retry
from websockets.client import OPEN as WS_OPEN



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

    model_hash: Optional[str] = Field(
        description="8-byte SHA-1 hash of the model file from Hugging Face."
    )

    class Config:
        arbitrary_types_allowed = True

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.hf_repo_id}:{self.hf_model_filename}:{self.hf_code_filename}:{self.competition_id}:{self.hf_repo_type}:{self.model_hash}"

    @property
    def hf_link(self) -> str:
        """Returns the Hugging Face link for the model."""
        return f"https://huggingface.co/{self.hf_repo_id}/blob/main/{self.hf_model_filename}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ChainMinerModel"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        if len(tokens) != 6:
            return None
        return cls(
            hf_repo_id=tokens[0],
            hf_model_filename=tokens[1],
            hf_code_filename=tokens[2],
            competition_id=tokens[3],
            hf_repo_type=tokens[4],
            model_hash=tokens[5],
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

        self._orig_ws_connect = self.subtensor.substrate.connect
        self.subtensor.substrate.connect = self._ws_connect

        try:
            ws = self.subtensor.substrate.connect()
            bt.logging.info(f"[ChainModelMetadata] Initial WS state: {ws.state}")
        except Exception as e:
            bt.logging.error("Initial WS connect failed: %s", e, exc_info=True)

        self.subnet_metadata = self.subtensor.metagraph(self.netuid)


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

    async def retrieve_model_metadata(self, hotkey: str, uid: int) -> ChainMinerModel:
        """Retrieves model metadata on this subnet for specific hotkey"""
        await asyncio.sleep(2) # temp fix for 429
        
        metadata = get_metadata(self.subtensor, self.netuid, hotkey)

        if metadata is None:
            raise ValueError(f"No metadata found for hotkey {hotkey}")

        chain_str = get_commitment(self.subtensor, self.netuid, uid)
        
        if chain_str is None:
            raise ValueError(
                f"No chain string found for hotkey '{hotkey}' and uid {uid}"
            )

        model = ChainMinerModel.from_compressed_str(chain_str)
        bt.logging.debug(f"Model: {model}")
        if model is None:
            raise ValueError(
                f"Metadata might be in old format or invalid for hotkey '{hotkey}'. Raw value: {chain_str}"
            )
        
        # The block id at which the metadata is stored
        model.block = metadata["block"]
        return model
    
    def close(self):
        try:
            bt.logging.debug("Closing ModelDBController and websocket connection.")
            self.subtensor.substrate.close_websocket()
        except Exception:
            pass

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