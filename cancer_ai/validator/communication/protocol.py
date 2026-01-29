
import typing
import bittensor as bt

import pydantic
from typing import Dict, List, Optional


class CompetitionResultsSynapse(bt.Synapse):
    """
    Protocol for validators to share their competition evaluation results.
    Used for P2P validator communication to reach consensus on miner scores.
    
    Attributes:
    - validator_uid: UID of the validator sending/requesting results
    - competition_id: ID of the competition (e.g., 'tricorder-3')
    - cycle_id: Unique identifier for this evaluation cycle
    - timestamp: Unix timestamp when evaluation was completed
    - evaluation_results: Dictionary mapping miner hotkey to their score
    - dataset_hash: Hash of the dataset used (for verification)
    - model_count: Number of miner models evaluated
    - status: Status of evaluation ('complete', 'in_progress', 'request', 'not_ready')
    """
    
    model_config = pydantic.ConfigDict(protected_namespaces=())
    
    validator_uid: int = pydantic.Field(
        ...,
        title="Validator UID",
        description="UID of the validator sending results"
    )
    
    competition_id: str = pydantic.Field(
        ...,
        title="Competition ID",
        description="ID of the competition being evaluated"
    )
    
    cycle_id: str = pydantic.Field(
        ...,
        title="Cycle ID",
        description="Unique identifier for this evaluation cycle"
    )
    
    timestamp: float = pydantic.Field(
        ...,
        title="Timestamp",
        description="Unix timestamp when evaluation was completed"
    )
    
    # Evaluation results: dict mapping miner hotkey to score
    evaluation_results: Dict[str, float] = pydantic.Field(
        default_factory=dict,
        title="Evaluation Results",
        description="Dictionary mapping miner hotkey (SS58) to their score"
    )
    
    dataset_hash: str = pydantic.Field(
        default="",
        title="Dataset Hash",
        description="SHA256 hash of the dataset used for evaluation"
    )
    
    model_count: int = pydantic.Field(
        default=0,
        title="Model Count",
        description="Number of miner models evaluated"
    )
    
    status: str = pydantic.Field(
        default="complete",
        title="Status",
        description="Status: 'complete', 'in_progress', 'request', 'not_ready'"
    )
    
    def deserialize(self) -> "CompetitionResultsSynapse":
        """Return the complete synapse object."""
        return self

class Dummy(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by receiving axon.
    dummy_output: typing.Optional[int] = None

    def deserialize(self) -> int:
      
        return self.dummy_output
