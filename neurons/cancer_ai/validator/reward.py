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

import numpy as np
from typing import List
import bittensor as bt
from ..protocol import PredictionSynapse
from collections import Counter


def reward(
    response: PredictionSynapse,
    max_time_penalty: float,
    factor: float,
    most_common_prediction: float,
) -> float:
    if (
        response["response_dict"] is None
        or response["process_time"] > response["timeout"]
    ):
        return 0

    if response["response_dict"]["models_response"] != most_common_prediction:
        return 0

    time_penalty = min(
        max_time_penalty,
        max_time_penalty * pow(response["process_time"], 3) / pow(factor, 3),
    )
    # TODO Konrad what to log 
    # bt.logging.info(f"In rewards, query val: {query}, response val: {response}, rewards val: {1.0 if response == query * 2 else 0}")

    return 1 - time_penalty


def get_rewards(
    self,
    responses: List[PredictionSynapse],
    max_time_penalty: float,
    factor: float,
) -> np.ndarray:
    # getting the most common response to validate if miner used the right model
    predictions = []
    most_common_prediction = 1
    for res in responses:
        if (
            res is not None
            and isinstance(res, dict)
            and "response_dict" in res
            and isinstance(res["response_dict"], dict)
            and "models_response" in res["response_dict"]
        ):
            predictions.append(res["response_dict"]["models_response"])
    if predictions:
        counter = Counter(predictions)
        most_common_prediction, _ = counter.most_common(1)[0]
    
    return np.array(
        [ reward(response, max_time_penalty, factor, most_common_prediction) for response in responses ]
    )
