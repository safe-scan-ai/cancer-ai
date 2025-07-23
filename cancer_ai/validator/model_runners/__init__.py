from abc import abstractmethod
from typing import AsyncGenerator, Union, Dict, Any, Tuple, List
import numpy as np

class BaseRunnerHandler:
    def __init__(self, config, model_path: str) -> None:
        self.config = config
        self.model_path = model_path

    @abstractmethod
    async def run(self, preprocessed_data_generator: AsyncGenerator[Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]], None]):
        """Execute the run process of the model with preprocessed data chunks."""
