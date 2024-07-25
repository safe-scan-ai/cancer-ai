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


import time
import bittensor as bt
import requests
import torch
import uuid
import json

from cancer_ai.validator import forward, forward_to_researcher
from cancer_ai.validator.models import DatasetEntries, ResearcherEntry, ResearcherScores
from cancer_ai.base.validator import BaseValidatorNeuron
from cancer_ai.protocol import MinerInfoSynapse
from pydantic import ValidationError


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

    async def forward(self):
        # How often you actually send synthetic challenges to the miner
        if self.step % self.config.forward_frequency > 0:
            return

        bt.logging.info("Updating available models & uids")
        await self.update_miners_identity()

        image_data = await self.get_image_data(1)
        if image_data is None or image_data.entries is None:
            return
        # TODO: use skin_type and model name fetched from the dataset api endpoint once its ready to serve this data
        challenge_data = {
            "image_url": image_data.entries[0].image_url,
        }

        return await forward(self, challenge_data["image_url"])

    async def forward_researcher_test(self, researcher_uid):
        bt.logging.info("Forwarding the test data to the researcher miner")
        test_data = []
        while not test_data:
            test_data = await self.get_image_data(
                self.config.researcher_testing_entries_package
            )
            if not test_data:
                bt.logging.error(
                    f"Error during fetching test data for researcher. Retrying in {self.config.fetching_interval} seconds"
                )
                time.sleep(self.config.fetching_interval)

        return await forward_to_researcher(self, researcher_uid, test_data.entries)

    async def get_image_data(self, amount):
        """This function fetches test data images from cancer-ai external api"""
        dataset_entries = None
        try:
            endpoint = (
                self.config.dataset_api + f"/dataset/skin/melanoma?amount={amount}"
            )
            headers = {"x-api-key": self.config.dataset_api_key}
            response = requests.get(url=endpoint, headers=headers)
            dataset_entries = DatasetEntries.parse_obj(response.json())

        except requests.RequestException as e:
            bt.logging.error(f"Failed to fetch test data: {e}")

        except ValidationError as e:
            bt.logging.error(f"Failed to fetch test data: {e}")

        finally:
            return dataset_entries

    async def evaluate_model(self, researcher_response, test_data):
        # researcher_response shape: {"models_response": {"sample_photo_id_1": 0.43,"sample_photo_id_2": 0.99, ...}}
        # test_data shape: [{"id": 1, "label": {"melanoma": false}, "image_url": "someurl"}, ...]

        combined_result = []
        for entry in test_data:
            if entry.id in researcher_response["models_response"]:
                combined_result.append(
                    (
                        researcher_response["models_response"][entry.id],
                        entry.current_model_response,
                        entry.label["melanoma"],
                        entry.id,
                    )
                )

        researcher_model_score = 0
        current_model_score = 0
        for entry in combined_result:
            researcher_pred, current_model_pred, label, id = entry
            if researcher_pred == current_model_pred:
                continue
            elif (label == True and researcher_pred > current_model_pred) or (
                label == False and researcher_pred < current_model_pred
            ):
                researcher_model_score += 1
            else:
                current_model_score += 1
        entries_num = len(combined_result)
        return researcher_model_score, current_model_score, entries_num, combined_result
    
    async def send_researchers_scores(self, researcher_score, current_model_score, num_entries,
                                       combined_predictions, researcher_uid, contact):
        
        entries = [ResearcherEntry(prediction=i[0], current_model_prediction=i[1], is_melanoma=i[2], image_id=i[3]) for i in combined_predictions]
        researcher_scores = ResearcherScores(entries=entries, researcher_score=researcher_score, current_model_score=current_model_score,
                                             num_entries=num_entries, testing_session_id=str(self.all_uids_info[researcher_uid]["testing_session_id"]), contact=contact)
        researcher_scores = researcher_scores.dict()

        try:
            headers = {"x-api-key": self.config.dataset_api_key}
            res = requests.post(
                self.config.stats_api + f"/researcher/testing/{researcher_uid}",
                json=researcher_scores,
                headers=headers,
            )
            if not res.ok:
                raise Exception(f"Received non-2xx status code: {res.status_code} - {res.text}")
        except Exception as e:
            bt.logging.error(f"Failed to send researchers scores to statistics api: {e}")

    async def update_miners_identity(self):
        valid_miners_info = await self.get_miners_info()
        if not valid_miners_info:
            bt.logging.warning("No active miner available")
        for uid, info in valid_miners_info.items():
            if info is None:
                print(f"Warning: No info available for UID {uid}")
                continue
            miner_state = self.all_uids_info.setdefault(
                uid,
                {
                    "miner_mode": "Unknown",
                    "min_stake": 100,
                    "device_info": {
                        "gpu_device_name": "Unknown",
                        "gpu_device_count": "Unknown",
                    },
                    "is_tested": False,
                    "tested_entries_amount": 0,
                },
            )

            miner_state["min_stake"] = info.get("min_stake", 100)
            miner_state["device_info"] = info.get("device_info", {})

            if (
                info["miner_mode"] == "researcher"
                and miner_state["miner_mode"] == "regular"
            ):
                miner_state["is_tested"] = True
                miner_state["testing_session_id"] = uuid.uuid4()
                bt.logging.success("New Researcher with uid {uid} started the testing challenge")
            miner_state["miner_mode"] = info["miner_mode"]

        bt.logging.success("Updated miner identity")
        self.save_state
        print("ALL_UIDS_INFO", self.all_uids_info)

        # TODO: enable once the cancer-ai API endpoint for storing miner info is ready
        # thread = Thread(target=self.store_miner_info, daemon=True)
        # thread.start()

    async def get_miners_info(self):
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        uid_to_axon = dict(zip(self.all_uids, self.metagraph.axons))
        query_axons = [uid_to_axon[int(uid)] for uid in self.all_uids]
        bt.logging.info("Requesting miner info")
        responses = self.dendrite.query(
            axons=query_axons,
            synapse=MinerInfoSynapse(),
            deserialize=False,
            timeout=10,
        )
        responses = {
            uid: response.response_dict
            for uid, response in zip(self.all_uids, responses)
        }
        return responses

    def store_miner_info(self):
        try:
            requests.post(
                self.config.storage_url + "/store_miner_info",
                json={
                    "uid": self.uid,
                    "info": self.all_uids_info,
                },
            )
        except Exception as e:
            bt.logging.error(f"Failed to store miner info: {e}")

    def update_and_get_top_researchers(self):
        """This function fetches top researchers with mapped rewards from cancer-ai external api"""
        try:
            response = requests.get(url=self.config.stats_api + "/emission-share")
            data = response.json()

            if not isinstance(data, dict):
                raise Exception("Invalid data")

            for key, value in data.items():
                if not (isinstance(key, int) and isinstance(value, float)):
                    raise Exception("Invalid data inside top researcher dict")

            self.top_researchers = dict(sorted(data.items()))
        except Exception as e:
            bt.logging.error(
                f"Failed to fetch top researchers: {e}. Proceeding with cached top researchers."
            )
        finally:
            return self.top_researchers

    def update_scores(self, rewards: torch.FloatTensor, uids: list[int]):
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            rewards = torch.nan_to_num(rewards, 0)

        # Check if `uids` is already a tensor and clone it to avoid the warning.
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, uids_tensor, rewards
        ).to(self.device)

        # Perform moving average with alpha parameter only on regular miners
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores.to(
            self.device
        )
        zero_threshold = 1e-4
        self.scores = torch.where(
            self.scores.abs() < zero_threshold,
            torch.zeros_like(self.scores),
            self.scores,
        )

        bt.logging.debug(f"Updated moving avg scores: {self.scores}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            print("Validator running...", time.time())
            time.sleep(5)
