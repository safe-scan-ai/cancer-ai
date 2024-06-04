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

from threading import Thread
from template.validator import forward
from template.base.validator import BaseValidatorNeuron
from template.protocol import MinerInfoSynapse



class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        print("load_state()")
        self.load_state()
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        self.all_uids_info = {
            uid: {"scores": [], "model_name": ""} for uid in self.all_uids
        }

    async def forward(self):
        bt.logging.info("Updating available models & uids")
        self.update_miners_identity()
        
        # TODO call the challenge generator url for photo and metadata
        photo_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
        challenge_type = "whitey"
        model_name = "skynet"
        input_metadata = {"dummy": "data"}
        
        return await forward(self, photo_b64, challenge_type, model_name, input_metadata)
    
    def update_miners_identity(self):
        """
        1. Query model_name of available uids
        2. Update the available list
        """
        valid_miners_info = self.get_miner_info()
        if not valid_miners_info:
            bt.logging.warning("No active miner available")
        for uid, info in valid_miners_info.items():
            if info is None:
                print(f"Warning: No info available for UID {uid}")
                continue
            miner_state = self.all_uids_info.setdefault(
                uid,
                {
                    "scores": [],
                    "model_name": "",
                },
            )

            miner_state["min_stake"] = info.get("min_stake", 100)
            miner_state["device_info"] = info.get("device_info", {})
            miner_state["model_name"] = info.get("model_name", "")

        bt.logging.success("Updated miner identity")
        print("ALL_UIDS_INFO", self.all_uids_info)

        thread = Thread(target=self.store_miner_info, daemon=True)
        thread.start()

    def get_miner_info(self):
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

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            print("Validator running...", time.time())
            time.sleep(5)
