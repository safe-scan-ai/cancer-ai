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
import time
import asyncio

from ..protocol import PredictionSynapse, ReasearcherTestingSynapse, MinerFeedbackSynapse
from ..validator.reward import get_rewards
from ..utils.uids import get_all_uids

# TODO
from enum import Enum

class MINER_MODE(Enum):
    Researcher = "researcher"
    Miner = "miner"


async def forward(self, image_url: str):
    all_uids = get_all_uids(self)

    #if the uids is the researcher which is in testing mode send him testing data
    for uid in all_uids:
        if uid.item() in self.all_uids_info and self.all_uids_info[uid.item()]["miner_mode"] == "researcher":
            asyncio.create_task(self.forward_researcher_test(uid.item()))

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in all_uids],
        synapse=PredictionSynapse(image_url=image_url),
        deserialize=True,
        timeout=12,
    )

    # Log the results for monitoring purposes.
    # bt.logging.debug(f"Received responses: {responses}")

    rewards = get_rewards(
        self,
        responses=responses,
        max_time_penalty=self.config.max_time_penalty,
        factor=12,
    )
    bt.logging.debug(f"Scored rewards: {rewards}")

    self.update_scores(rewards, all_uids)
    time.sleep(5)


async def forward_to_researcher(self, researcher_uid: int, test_data: list):
    images = {entry.id: entry.image_url for entry in test_data}

    response = await self.dendrite(
        axons=self.metagraph.axons[researcher_uid],
        synapse=ReasearcherTestingSynapse(images=images),
        deserialize=True,
        timeout=60 * 60 * 24,
    )

    if response is not None and response["identity_error"]:
        bt.logging.error(
            f"Miner with uid: {researcher_uid} was forwarded researcher synapse while not being a researcher with testing_session_id defined."
        )
    elif self.all_uids_info[researcher_uid] and response is not None and response["entries_num"]:
        # bt.logging.debug(f'Researcher with uid {researcher_uid} response {response}')
        
        researcher_score, current_model_score, num_entries, combined_predictions = await self.evaluate_model(response, test_data)
        if "testing_session_id" not in response["testing_session_id"]:
            bt.logging.error("Researcher with uid: {researcher_uid} did not send testing_session_id, not sending results do Stats API")
        else:
            asyncio.create_task(self.send_researchers_scores(researcher_score, current_model_score, num_entries,
                                                          combined_predictions, researcher_uid, response["testing_session_id"]))
        bt.logging.info(f'Researcher {researcher_uid}  researcher score: {researcher_score} \n current model score: {current_model_score}')

        # Send feedback to the miner
        feedback = [{"researcher_res": entry[0], "current_model_res": entry[1], "label": entry[2], "image_id": entry[3]} for entry in combined_predictions]
        asyncio.create_task(self.dendrite(
            axons=self.metagraph.axons[researcher_uid],
            synapse=MinerFeedbackSynapse(feedback=feedback),
            deserialize=False,
        ))
