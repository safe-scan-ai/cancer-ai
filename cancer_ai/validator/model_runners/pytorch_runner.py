from . import BaseRunnerHandler
from typing import List, AsyncGenerator, Union, Dict, Any, Tuple
import numpy as np
import bittensor as bt


class PytorchRunnerHandler(BaseRunnerHandler):
    async def run(self, preprocessed_data_generator: AsyncGenerator[Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]], None]) -> List:
        """
        Run PyTorch model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays,
                                       or tuples of (numpy arrays, metadata) for tricorder
            
        Returns:
            List of model predictions
        """
        import torch
        
        bt.logging.info("Running PyTorch model inference on preprocessed data")
        
        model = torch.load(self.model_path)
        model.eval()
        results = []
        
        async for data in preprocessed_data_generator:
            try:
                # Handle both formats: plain numpy array or tuple with metadata
                if isinstance(data, tuple):
                    # Tricorder format: (image_data, metadata)
                    chunk, metadata = data
                    bt.logging.debug(f"Processing chunk with metadata: {len(metadata)} entries")
                    
                    # Convert numpy array to torch tensor
                    chunk_tensor = torch.from_numpy(chunk)
                    
                    # Prepare metadata tensor
                    metadata_tensor = self._prepare_metadata_tensor(metadata)
                    
                    with torch.no_grad():
                        # Pass both image and metadata to model
                        chunk_results = model(chunk_tensor, metadata_tensor)
                        results.extend(chunk_results.cpu().numpy())
                else:
                    # Melanoma format: plain numpy array (no metadata)
                    chunk = data
                    chunk_tensor = torch.from_numpy(chunk)
                    with torch.no_grad():
                        chunk_results = model(chunk_tensor)
                        results.extend(chunk_results.cpu().numpy())
            except Exception as e:
                bt.logging.error(f"PyTorch inference error on chunk: {e}")
                continue
                
        return results
    
    def _prepare_metadata_tensor(self, metadata: List[Dict[str, Any]]):
        """Convert metadata list to torch tensor for model input"""
        import torch
        
        # Convert metadata to numerical format
        metadata_array = []
        for entry in metadata:
            age = entry.get('age', 0) if entry.get('age') is not None else 0
            # Convert gender to numerical: male=1, female=0, unknown=-1
            gender_str = entry.get('gender', '').lower() if entry.get('gender') else ''
            if gender_str in ['male', 'm']:
                gender = 1
            elif gender_str in ['female', 'f']:
                gender = 0
            else:
                gender = -1  # Unknown/missing gender
            
            # Convert location to numerical: arm=1, feet=2, genitalia=3, hand=4, head=5, leg=6, torso=7, unknown=-1
            location_str = entry.get('location', '').lower() if entry.get('location') else ''
            location_mapping = {
                'arm': 1, 'feet': 2, 'genitalia': 3, 'hand': 4, 
                'head': 5, 'leg': 6, 'torso': 7
            }
            location = location_mapping.get(location_str, -1)  # Unknown/missing location
            
            metadata_array.append([age, gender, location])
        
        return torch.tensor(metadata_array, dtype=torch.float32)
