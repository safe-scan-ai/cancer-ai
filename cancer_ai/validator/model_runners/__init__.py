from abc import abstractmethod
from typing import AsyncGenerator
import numpy as np

class BaseRunnerHandler:
    def __init__(self, config, model_path: str) -> None:
        self.config = config
        self.model_path = model_path

    @abstractmethod
    async def run(self, preprocessed_data_generator: AsyncGenerator[np.ndarray, None]):
        """Execute the run process of the model with preprocessed data chunks."""
