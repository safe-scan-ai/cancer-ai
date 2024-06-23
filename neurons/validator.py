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

from threading import Thread
from cancer_ai.validator import forward
from cancer_ai.base.validator import BaseValidatorNeuron
from cancer_ai.protocol import MinerInfoSynapse



class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        self.all_uids_info = {
            uid: {"scores": [], "model_name": ""} for uid in self.all_uids
        }

    async def forward(self):
        # How often you actually send synthetic challenges to miner?
        if self.step % 1000 > 0:
            return
        
        # TODO: define if we want to update miners identity every forward(pretty often) or poll it e.g. once per minute?
        bt.logging.info("Updating available models & uids")
        await self.update_miners_identity()
        
        # TODO call the challenge generator url for photo and metadata
        photo_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
        challenge_type = "whitey"
        model_name = "skynet"
        input_metadata = {"dummy": "data"}
        
        return await forward(self, photo_b64, challenge_type, model_name, input_metadata)
    
    async def update_miners_identity(self):
        """
        1. Query model_name of available uids
        2. Update the available list
        """
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
                    "model_name": "Unknown",
                    "min_stake": 100,
                    "device_info": {
                        "gpu_device_name": "Unknown",
                        "gpu_device_count": "Unknown",
                     }
                },
            )

            miner_state["min_stake"] = info.get("min_stake", 100)
            miner_state["device_info"] = info.get("device_info", {})
            miner_state["model_name"] = info.get("model_name", "")

            if info["miner_mode"] == "researcher" and miner_state["miner_mode"] == "regular":
                #TODO: this should trigger calling the DB for test utilities to test the researcher
                ...
            miner_state["miner_mode"] = info["miner_mode"]
            #TODO: update miner's score (?) do we want to store score in miners info?

        bt.logging.success("Updated miner identity")
        print("ALL_UIDS_INFO", self.all_uids_info)

        # commented for not spamming on output
        # thread = Thread(target=self.store_miner_info, daemon=True)
        # thread.start()

    async def get_miners_info(self):
        """
        1. Query model_name of available uids
        """
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
        '''This function fetches top researchers with mapped rewards from cancer-ai external api'''
        try:
            response = requests.get(url = self.config.top_researchers_url)
            data = response.json()

            if not isinstance(data, dict):
                raise Exception("Invalid data")
            
            for key, value in data.items():
                if not (isinstance(key, int) and isinstance(value, float)):
                    raise Exception("Invalid data inside top researcher dict")
                
            self.top_researchers = dict(sorted(data.items()))
        except Exception as e:
            bt.logging.error(f"Failed to fetch top researchers: {e}. Proceeding with cached top researchers.")
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
        self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores.to(self.device)
        zero_threshold = 1e-4
        self.scores = torch.where(self.scores.abs() < zero_threshold, torch.zeros_like(self.scores), self.scores)

        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            print("Validator running...", time.time())
            time.sleep(5)
