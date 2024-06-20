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

import torch
from typing import List
from template.protocol import PredictionSynapse

def reward(response: PredictionSynapse, max_time_penalty: float, factor: float) -> float:
    if response.response_dict is None or response.dendrite.process_time > response.timeout:
        return 0
    
    time_penalty = min(max_time_penalty, max_time_penalty * pow(response.dendrite.process_time, 3) / pow(factor, 3))
    return 1 - time_penalty


def get_rewards(
    self,
    responses: List[PredictionSynapse],
    max_time_penalty: float,
    factor: float,
) -> torch.FloatTensor:
    
    return torch.FloatTensor(
        [reward(response, max_time_penalty, factor) for response in responses]
    ).to(self.device)
