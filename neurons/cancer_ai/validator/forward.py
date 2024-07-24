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
import asyncio

from cancer_ai.protocol import PredictionSynapse, ReasearcherTestingSynapse, MinerFeedbackSynapse
from cancer_ai.models import Feedback, FeedbackEntry
from cancer_ai.validator.reward import get_rewards
from cancer_ai.utils.uids import get_all_uids


async def forward(self, image_url: str):
    all_uids = get_all_uids(self)

    #if the uids is the researcher which is in testing mode send him testing data
    for uid in all_uids:
        if uid.item() in self.all_uids_info and self.all_uids_info[uid.item()]["is_tested"]:
            asyncio.create_task(self.forward_researcher_test(uid.item()))

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in all_uids],
        synapse=PredictionSynapse(image_url=image_url),
        deserialize=True,
        timeout=12,
    )

    # Log the results for monitoring purposes.
    print(f"Received responses: {responses}")

    rewards = get_rewards(
        self,
        responses=responses,
        max_time_penalty=self.config.max_time_penalty,
        factor=12,
    )
    print(f"Scored rewards: {rewards}")

    self.update_scores(rewards, all_uids)


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
            f"Miner with uid: {researcher_uid} was forwarded researcher synapse while not being a researcher."
        )

    # append tested entries num
    if self.all_uids_info[researcher_uid] and response["entries_num"]:
        self.all_uids_info[researcher_uid]["tested_entries_amount"] += response["entries_num"]
        print(f"Researcher with uid {researcher_uid} responded with {response['entries_num']} predictions.\n Total number of entries tested on researcher: {self.all_uids_info[researcher_uid]['tested_entries_amount']}")
        
        researcher_score, current_model_score, num_entries, combined_predictions = await self.evaluate_model(response, test_data)
        asyncio.create_task(self.send_researchers_scores(researcher_score, current_model_score, num_entries, combined_predictions, researcher_uid))
        print(
            f"Models comparison on {num_entries} entries:\n researcher score: {researcher_score} \n current model score: {current_model_score}"
        )

        # Send the scores to the miner as a feedback
        feedback_list = [FeedbackEntry(researcher_res=entry[0], current_model_res=entry[1], label=entry[2], image_id=entry[3]) for entry in combined_predictions]
        asyncio.create_task(self.dendrite(
            axons=self.metagraph.axons[researcher_uid],
            synapse=MinerFeedbackSynapse(feedback=Feedback(feedback=feedback_list)),
            deserialize=False,
        ))

    # switch off testing mode for the researcher when expected number of entries was tested
    if self.all_uids_info[researcher_uid]["tested_entries_amount"] >= self.config.researcher_testing_entries_amount:
        self.all_uids_info[researcher_uid]["is_tested"] = False
        self.all_uids_info[researcher_uid]["tested_entries_amount"] = 0
        print(f"Researcher with uid {researcher_uid} has finished testing.")
