from . import BaseRunnerHandler
from typing import List, AsyncGenerator
import numpy as np
import bittensor as bt
import asyncio


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
                bt.logging.debug(f"Running PyTorch inference on chunk with shape {chunk.shape}")
                # Convert numpy array to torch tensor
                chunk_tensor = torch.from_numpy(chunk)
                
                # Run inference with timeout protection
                def _run_inference():
                    with torch.no_grad():
                        return model(chunk_tensor).cpu().numpy()
                
                chunk_results = await asyncio.wait_for(
                    asyncio.to_thread(_run_inference),
                    timeout=120.0  # 2 minutes max per chunk
                )
                results.extend(chunk_results)
                bt.logging.debug(f"PyTorch inference completed, got {len(chunk_results)} results")
                
            except asyncio.TimeoutError:
                bt.logging.error("PyTorch inference timeout after 120s on chunk")
                continue
            except Exception as e:
                bt.logging.error(f"PyTorch inference error on chunk: {e}", exc_info=True)
                continue
                
        return results

    def cleanup(self):
        """Clean up resources for the PyTorch runner."""
        pass