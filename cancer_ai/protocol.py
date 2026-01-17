# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

import typing
import bittensor as bt

import pydantic
from typing import Dict, List, Optional
# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2

class CompetitionResultsSynapse(bt.Synapse):
    """
    Protocol for validators to share their competition evaluation results.
    Used for P2P validator communication to reach consensus on miner scores.
    
    Attributes:
    - validator_uid: UID of the validator sending/requesting results
    - competition_id: ID of the competition (e.g., 'tricorder-3', 'melanoma')
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
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by receiving axon.
    dummy_output: typing.Optional[int] = None

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.dummy_output
