import bittensor as bt
from typing import List

from .manager import SerializableManager
from .model_manager import ModelInfo
from .utils import detect_model_format, ModelType
from .model_runners.pytorch_runner import PytorchRunnerHandler
from .model_runners.tensorflow_runner import TensorflowRunnerHandler
from .model_runners.onnx_runner import OnnxRunnerHandler
from .exceptions import ModelRunException


MODEL_TYPE_HANDLERS = {
    ModelType.PYTORCH: PytorchRunnerHandler,
    ModelType.TENSORFLOW_SAVEDMODEL: TensorflowRunnerHandler,
    ModelType.ONNX: OnnxRunnerHandler,
}



class ModelRunManager(SerializableManager):
    def __init__(self, config, model: ModelInfo) -> None:
        self.config = config
        self.model = model
        self.set_runner_handler()

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        pass

    def set_runner_handler(self) -> None:
        """Sets the model runner handler based on the model type."""
        
        model_type = detect_model_format(self.model.file_path)
        # initializing ml model handler object
        
        model_handler = MODEL_TYPE_HANDLERS.get(model_type)
        if model_handler == None: 
            bt.logging.error (f"Unknown model format {self.model.hf_repo_id} {self.model.hf_repo_id}")
            raise ModelRunException("Unknown model format")

        
        self.handler = model_handler(self.config, self.model.file_path)

    async def run(self, pred_x: List) -> List:
        """
        Run the model with the given input.

        Returns:
            List: model predictions
        """

        model_predictions = await self.handler.run(pred_x)
        return model_predictions
