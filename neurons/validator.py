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
import numpy as np
import tensorflow as tf
import asyncio
from datetime import datetime, timedelta

from cancer_ai.validator import forward, forward_to_researcher
from cancer_ai.validator.models import DatasetEntries, ResearcherEntry, ResearcherScores
from cancer_ai.base.validator import BaseValidatorNeuron
from cancer_ai.protocol import MinerInfoSynapse
from pydantic import ValidationError
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        model_path = hf_hub_download(
            "safe-scan-ai/skin-cancer-collection", "melanoma.keras"
        )
        self.base_model = tf.keras.models.load_model(model_path)
        bt.logging.info(f"Evaluation model built status: {self.base_model.built}")

        self.researchers_testing_timestamps = {}

        bt.logging.info("Starting miners identity update routine")
        asyncio.run_coroutine_threadsafe(self.run_miners_update_routine(), self.loop)

        bt.logging.info("Starting researcher testing watcher")
        asyncio.run_coroutine_threadsafe(self.run_researcher_testing_watcher(), self.loop)
        

    async def run_miners_update_routine(self):
        while True:
            bt.logging.info("Updating miners identity")
            await self.update_miners_identity()
            await self.update_researchers_testing_timestamps()
            await asyncio.sleep(self.config.update_miners_routine_interval)
    
    async def run_researcher_testing_watcher(self):
        while True:
            for uid, latest_update in self.researchers_testing_timestamps.items():
                if datetime.now() - latest_update > timedelta(minutes=self.config.researcher_testing_interval):
                    asyncio.create_task(self.forward_researcher_test(uid))
                    self.researchers_testing_timestamps[uid] = datetime.now()
            await asyncio.sleep(1)  # Prevent tight loop
    
    async def update_researchers_testing_timestamps(self):
        new_researchers_timestamps = {uid: datetime.now() for uid, data in self.all_uids_info.items() if data.get("miner_mode") == "researcher"}

        # Add new researchers or update existing ones' timestamps
        for uid, timestamp in new_researchers_timestamps.items():
            if uid not in self.researchers_testing_timestamps:
                self.researchers_testing_timestamps[uid] = timestamp
        
        # Find and remove researchers that are no longer researchers
        current_researchers_uids = set(new_researchers_timestamps.keys())
        existing_researchers_uids = set(self.researchers_testing_timestamps.keys())
        for uid in existing_researchers_uids - current_researchers_uids:
            del self.researchers_testing_timestamps[uid]

    async def forward(self):
        # How often you actually send synthetic challenges to the miner
        if self.step % self.config.forward_frequency > 0:
            return

        image_data = await self.get_image_data(1)
        if image_data is None or image_data.entries is None:
            # feed fallback image
            wikipedia_melaona_url = "https://upload.wikimedia.org/wikipedia/commons/6/6c/Melanoma.jpg"
            challenge_data = {
                "image_url": wikipedia_melaona_url,
            }
        else:   
            challenge_data = {
                "image_url": f"{self.config.dataset_api}{image_data.entries[0].image_url}",
            }
        return await forward(self, challenge_data["image_url"])

    async def forward_researcher_test(self, researcher_uid):
        bt.logging.info(f"Forwarding the test data to the researcher miner with uid {researcher_uid}")

        test_data = await self.get_image_data(
            self.config.researcher_testing_entries_package
        )
        if test_data is None:
            bt.logging.error(f"Failed to fetch test data for researcher with uid {researcher_uid}")
            return
        
        return await forward_to_researcher(self, researcher_uid, test_data.entries)

    async def get_image_data(self, amount):
        """This function fetches test data images from cancer-ai external api"""
        dataset_entries = None
        try:
            data = self.dataset_api.get_image_data(amount)
            # bt.logging.debug(f"Dataset API response: {data}")
            dataset_entries = DatasetEntries(**data)

        except requests.RequestException as e:
            bt.logging.error(f"Failed to fetch test data: {e}")

        except ValidationError as e:
            bt.logging.error(f"Failed to fetch test data: {e}")

        finally:
            return dataset_entries

    async def evaluate_researcher_model(self, researcher_response, test_data):
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
            researcher_pred, current_model_pred, label, _ = entry
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
                                       combined_predictions, researcher_uid, testing_session_id):
        entries = [ResearcherEntry(prediction=i[0], current_model_prediction=i[1], is_melanoma=i[2], image_id=i[3]) for i in combined_predictions]
        researcher_scores = ResearcherScores(entries=entries, researcher_score=researcher_score, current_model_score=current_model_score,
                                             num_entries=num_entries, testing_session_id=testing_session_id, hotkey=self.all_uids_info[researcher_uid]["hotkey"])
        researcher_scores = researcher_scores.dict()
        try:
            response_successful, message = self.stats_api.send_researcher_scores(researcher_uid, researcher_scores)
            if not response_successful:
                bt.logging.error(f"Failed to send researchers scores to statistics api: {message}")
            
        except Exception as e:
            bt.logging.error(f"Failed to send researchers scores to statistics api: {e}")

    async def update_miners_identity(self):
        not_available_uids = []
        valid_miners_info = await self.get_miners_info()
        if not valid_miners_info:
            bt.logging.warning("Updating miners identity: No active miner available")
        for uid, info in valid_miners_info.items():

            miner_state = self.all_uids_info.setdefault(
                uid,
                {
                    "miner_mode": "Unknown",
                    "min_stake": 100,
                    "device_info": {
                        "gpu_device_name": "Unknown",
                        "gpu_device_count": "Unknown",
                    },
                },
            )

            if info is None:
                not_available_uids.append(uid)
                miner_state["miner_mode"] = "Unknown",
                continue

            miner_state["min_stake"] = info.get("min_stake", 100)
            miner_state["device_info"] = info.get("device_info", {})
            miner_state["miner_mode"] = info["miner_mode"]
            miner_state["hotkey"] = self.metagraph.hotkeys[uid]

        self.save_state()
        api_response = self.stats_api.send_miner_info(self.all_uids_info)
        bt.logging.info(f"Updating miners identity: statistics API response: {api_response}")
        bt.logging.success("Updating miners identity: update successfull")
        bt.logging.info(f"Updating miners identity: No info available for UID {not_available_uids}")


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
    
    def is_miners_model_valid(self, image_url, miners_prediction):
        base_model_prediction = self.get_base_model_prediction(image_url)
        if base_model_prediction is None:
            # Indicate that the validation could not be performed
            return None
        if base_model_prediction != miners_prediction:
            return False
        return True

    def get_base_model_prediction(self, image_url):
        # Convert binary data to an image
        image = self.get_image(image_url)
        if image is None:
            return None
        image = load_img(BytesIO(image), target_size=(180, 180, 3))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        pred = self.base_model.predict(img_array)
        _, melanoma_probability = pred[0]
        return float(melanoma_probability)

    def get_image(self, image_url):
        jpg_image = None
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            if 'image/jpeg' in response.headers.get('Content-Type', ''):
                jpg_image = response.content
            else:
                bt.logging.error(f"URL does not point to a JPEG image: {image_url}")

        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error fetching image from {image_url}: {e}")
        except IOError as e:
            bt.logging.error(f"Error opening image from {image_url}: {e}")
        
        return jpg_image

    def store_miner_info(self):
        try:
            response_successful, message = self.stats_api.send_miner_info(self.all_uids_info)
            if not response_successful:
                bt.logging.error(f"Failed to store miner info: {message}")
        except Exception as e:
            bt.logging.error(f"Failed to store miner info: {e}")

    def update_top_researchers_from_api(self):
        """Fetches top researchers with mapped rewards from cancer-ai external API."""
        self.top_researchers = {}
        # bt.logging.debug(f"Stats API: Fetching top researchers for emission")
        try:
            response = requests.get(url=f"{self.config.stats_api}/emission-share")
            data = response.json()

            if not isinstance(data, dict):
                raise ValueError("Stats API: Invalid data type received, expected a dictionary")

            for key, value in data.items():
                if not (isinstance(key, int) and isinstance(value, float)):
                    raise ValueError(f"Stats API: Invalid data inside top researcher dict: {key}: {value}")

            self.top_researchers = dict(sorted(data.items()))
            # bt.logging.info("Refreshed list of top researchers")
            # bt.logging.debug(f"Top researchers from API: {self.top_researchers}")
        except Exception as e:
            bt.logging.error(
                f"Failed to fetch top researchers: {e}. Proceeding with cached top researchers."
            )

    def update_scores(self, rewards: np.ndarray, uids: list[int]):
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            rewards = np.nan_to_num(rewards, 0)

        # Convert uids to numpy array if not already
        uids_array = np.array(uids)

        # Initialize scores array if not already initialized
        if not hasattr(self, 'scores'):
            self.scores = np.zeros_like(rewards)

        # Create an array to hold scattered rewards
        scattered_rewards = np.zeros_like(self.scores)

        # Scatter the rewards to the appropriate indices
        scattered_rewards[uids_array] = rewards

        # Perform moving average with alpha parameter only on regular miners
        alpha = self.config.neuron.moving_average_alpha
        self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores

        zero_threshold = 1e-4
        self.scores = np.where(np.abs(self.scores) < zero_threshold, 0, self.scores)

        bt.logging.debug(f"Updated moving avg scores: {self.scores}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(5)
