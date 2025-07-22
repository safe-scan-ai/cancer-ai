from . import BaseRunnerHandler
from typing import List, AsyncGenerator
import numpy as np
import bittensor as bt


class PytorchRunnerHandler(BaseRunnerHandler):
    async def run(self, preprocessed_data_generator: AsyncGenerator[np.ndarray, None]) -> List:
        """
        Run PyTorch model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays
            
        Returns:
            List of model predictions
        """
        import torch
        
        bt.logging.info("Running PyTorch model inference on preprocessed data")
        
        model = torch.load(self.model_path)
        model.eval()
        results = []
        
        async for chunk in preprocessed_data_generator:
            try:
                # Convert numpy array to torch tensor
                chunk_tensor = torch.from_numpy(chunk)
                with torch.no_grad():
                    chunk_results = model(chunk_tensor)
                    results.extend(chunk_results.cpu().numpy())
            except Exception as e:
                bt.logging.error(f"PyTorch inference error on chunk: {e}")
                continue
                
        return results