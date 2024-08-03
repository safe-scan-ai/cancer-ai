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
import typing
from io import BytesIO

import bittensor as bt
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from huggingface_hub import hf_hub_download


# Bittensor Miner cancer_ai:
import cancer_ai

# import base miner class which takes care of most of the boilerplate
from cancer_ai.base.miner import BaseMinerNeuron
from cancer_ai.miner.forward import set_info, get_images, get_image, get_mode

from cancer_ai.miner.utils import is_valid_uuid
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO




class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.miner_info = set_info(self)
        bt.logging.info(f"Miner info: {self.miner_info}")

        model_path = hf_hub_download(
            "safe-scan-ai/skin-cancer-collection", "melanoma.keras"
        )
        self.regular_model = tf.keras.models.load_model(model_path)

        bt.logging.info(f"Regular model built status: {self.regular_model.built}")

    async def forward(
        self, synapse: cancer_ai.protocol.PredictionSynapse
    ) -> cancer_ai.protocol.PredictionSynapse:
        # Convert binary data to an image
        image = get_image(self, synapse.image_url)
        if image is None:
            return synapse
        
        image = load_img(BytesIO(image), target_size=(180, 180, 3))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        bt.logging.info("Regular miner mode: Running prediction")
        now = time.time()
        # Predict using the model
        pred = self.regular_model.predict(img_array)
        time_diff = time.time() - now
        bt.logging.info(f"Regular miner mode: Time taken to predict: {time_diff}")
        _, melanoma_probability = pred[0]
        synapse.response_dict = {
            "models_response": float(melanoma_probability),
            "miner_mode": get_mode(self),
            "miner_uid": self.uid,
            "image_url": synapse.image_url
        }
        return synapse

    async def forward_researcher(
        self, synapse: cancer_ai.protocol.ReasearcherTestingSynapse
    ) -> cancer_ai.protocol.ReasearcherTestingSynapse:
        if not self.config.researcher or not is_valid_uuid(
            self.config.testing_session_id
        ):
            synapse.response_dict = {"identity_error": True}
            return synapse

        images = get_images(self, synapse.images)

        # TODO(researcher owner): feed the ML model with the images
        # MOCK response for testing purposes
        bt.logging.info("Researcher miner mode: Got researcher task")
        import random

        mock_response = {
            "entries_num": len(images),
            "models_response": {},
            "identity_error": False,
            "testing_session_id": self.config.testing_session_id,
        }
        for image in images:
            mock_response["models_response"][image[0]] = random.randrange(1, 100) / 100
        synapse.response_dict = mock_response

        return synapse

    async def forward_info(
        self, synapse: cancer_ai.protocol.MinerInfoSynapse
    ) -> cancer_ai.protocol.MinerInfoSynapse:
        synapse.response_dict = self.miner_info

        return synapse

    async def forward_get_feedback(
        self, synapse: cancer_ai.protocol.MinerFeedbackSynapse
    ):

        feedback = synapse.feedback
        # TODO(researcher developer): write your logic to process feedback data
        # bt.logging.info("Researcher miner mode: feedback from model scores:", feedback)


    async def blacklist(
        self, synapse: cancer_ai.protocol.PredictionSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (cancer_ai.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: cancer_ai.protocol.PredictionSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (cancer_ai.protocol.PredictionSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
