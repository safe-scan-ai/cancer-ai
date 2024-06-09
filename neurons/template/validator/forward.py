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

from template.protocol import PredictionSynapse
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids


async def forward(self, base64_photo: str, challenge_type: str, model_name: str, input_metadata: dict):
    
    # TODO(developer): Define how validator selects only miner uids
    all_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in all_uids],
        synapse=PredictionSynapse(base64_photo=base64_photo, challenge_type=challenge_type, model_name=model_name, input_metadata=input_metadata),
        deserialize=True,
        timeout=12,
    )

    # Log the results for monitoring purposes.
    print(f"Received responses: {responses}")

    # TODO(probably): include uid in the responses and create new uids-responses to pass further to rewards

    rewards = get_rewards(self, responses=responses, max_time_penalty=0.4, factor=12)
    print(f"Scored rewards: {rewards}")

    self.update_scores(rewards, all_uids)
