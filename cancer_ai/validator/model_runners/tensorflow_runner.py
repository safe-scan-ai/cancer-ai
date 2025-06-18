from . import BaseRunnerHandler
from typing import List, AsyncGenerator
import bittensor as bt
import numpy as np

class TensorflowRunnerHandler(BaseRunnerHandler):
    async def run(self, preprocessed_data_generator: AsyncGenerator[np.ndarray, None]) -> List:
        """
        Run TensorFlow model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays
            
        Returns:
            List of model predictions
        """
        import tensorflow as tf
        
        bt.logging.info("Running TensorFlow model inference on preprocessed data")
        
        model = tf.keras.models.load_model(self.model_path)
        results = []
        
        async for chunk in preprocessed_data_generator:
            try:
                # TensorFlow expects (N, H, W, C) format, so transpose from (N, C, H, W)
                chunk_transposed = np.transpose(chunk, (0, 2, 3, 1))
                chunk_results = model.predict(chunk_transposed, batch_size=10)
                results.extend(chunk_results)
            except Exception as e:
                bt.logging.error(f"TensorFlow inference error on chunk: {e}")
                continue
                
        return results
