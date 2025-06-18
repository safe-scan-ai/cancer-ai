from typing import List, AsyncGenerator
import numpy as np
import bittensor as bt
from collections import defaultdict
from ..exceptions import ModelRunException

from . import BaseRunnerHandler


class OnnxRunnerHandler(BaseRunnerHandler):
    async def run(self, preprocessed_data_generator: AsyncGenerator[np.ndarray, None]) -> List:
        """
        Run ONNX model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays
            
        Returns:
            List of model predictions
        """
        import onnxruntime

        error_counter = defaultdict(int)

        try:
            session = onnxruntime.InferenceSession(self.model_path)
        except onnxruntime.OnnxRuntimeException:
            raise ModelRunException("Failed to create ONNX inference session")
        except Exception:
            raise ModelRunException("Failed to create ONNX inference session")

        results = []

        async for chunk in preprocessed_data_generator:
            try:
                input_name = session.get_inputs()[0].name
                input_data = {input_name: chunk}
                chunk_results = session.run(None, input_data)[0]
                results.extend(chunk_results)
            except Exception as e:
                bt.logging.debug(f"Inference error on chunk: {e}")
                error_counter['InferenceError'] += 1
                continue

        # Handle error summary
        if error_counter:
            error_summary = "; ".join([f"{count} {error_type.replace('_', ' ')}(s)" 
                                     for error_type, count in error_counter.items()])
            bt.logging.info(f"ONNX inference completed with issues: {error_summary}")
            
        if not results:
            raise ModelRunException("No results obtained from model inference")

        return results
