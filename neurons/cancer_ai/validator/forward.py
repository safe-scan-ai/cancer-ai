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

import bittensor as bt

from cancer_ai.protocol import PredictionSynapse, ReasearcherTestingSynapse
from cancer_ai.validator.reward import get_rewards
from cancer_ai.utils.uids import get_all_uids


async def forward(self, base64_photo: str, challenge_type: str, model_name: str, input_metadata: dict):
    
    all_uids = get_all_uids(self)

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in all_uids],
        synapse=PredictionSynapse(base64_photo=base64_photo, challenge_type=challenge_type, model_name=model_name, input_metadata=input_metadata),
        deserialize=True,
        timeout=12,
    )

    # Log the results for monitoring purposes.
    print(f"Received responses: {responses}")

    rewards = get_rewards(self, responses=responses, max_time_penalty=self.config.max_time_penalty, factor=12)
    print(f"Scored rewards: {rewards}")

    self.update_scores(rewards, all_uids)

async def forward_to_researcher(self, researcher_uid: int, base64_photo: str, challenge_type: str, model_name: str, input_metadata: dict):
    response = await self.dendrite(
        axons=self.metagraph.axons[researcher_uid],
        synapse=ReasearcherTestingSynapse(base64_photo=base64_photo, challenge_type=challenge_type, model_name=model_name, input_metadata=input_metadata),
        deserialize=True,
        timeout=12,
    )

    print(f"Received response: {response}")
    if response.response_dict["identity_error"]:
        bt.logging.error(f"Miner with uid: {researcher_uid} was forwarded researcher synapse while not being a researcher.")
    
    #possibly returning score loss for a particular challenge, or returning the the whole set of predictions/losses
    
