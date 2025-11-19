from . import BaseRunnerHandler
from typing import List, AsyncGenerator
import bittensor as bt
import numpy as np
import asyncio

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
                bt.logging.debug(f"Running TensorFlow inference on chunk with shape {chunk.shape}")
                # TensorFlow expects (N, H, W, C) format, so transpose from (N, C, H, W)
                chunk_transposed = np.transpose(chunk, (0, 2, 3, 1))
                
                # Run inference with timeout protection
                chunk_results = await asyncio.wait_for(
                    asyncio.to_thread(model.predict, chunk_transposed, batch_size=10),
                    timeout=120.0  # 2 minutes max per chunk
                )
                results.extend(chunk_results)
                bt.logging.debug(f"TensorFlow inference completed, got {len(chunk_results)} results")
                
            except asyncio.TimeoutError:
                bt.logging.error("TensorFlow inference timeout after 120s on chunk")
                continue
            except Exception as e:
                bt.logging.error(f"TensorFlow inference error on chunk: {e}", exc_info=True)
                continue
                
        return results

    def cleanup(self):
        """Clean up resources for the Tensorflow runner."""
        pass