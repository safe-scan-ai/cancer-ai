from . import BaseRunnerHandler
from typing import List, AsyncGenerator, Union, Dict, Any, Tuple
import bittensor as bt
import numpy as np

class TensorflowRunnerHandler(BaseRunnerHandler):
    async def run(self, preprocessed_data_generator: AsyncGenerator[Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]], None]) -> List:
        """
        Run TensorFlow model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays,
                                       or tuples of (numpy arrays, metadata) for tricorder
            
        Returns:
            List of model predictions
        """
        import tensorflow as tf
        
        bt.logging.info("Running TensorFlow model inference on preprocessed data")
        
        model = tf.keras.models.load_model(self.model_path)
        results = []
        
        async for data in preprocessed_data_generator:
            try:
                # Handle both formats: plain numpy array or tuple with metadata
                if isinstance(data, tuple):
                    # Tricorder format: (image_data, metadata)
                    chunk, metadata = data
                    bt.logging.debug(f"Processing chunk with metadata: {len(metadata)} entries")
                    
                    # TensorFlow expects (N, H, W, C) format, so transpose from (N, C, H, W)
                    chunk_transposed = np.transpose(chunk, (0, 2, 3, 1))
                    
                    # Prepare metadata array
                    metadata_array = self._prepare_metadata_array(metadata)
                    
                    # Check if model expects multiple inputs
                    model_inputs = model.inputs
                    if len(model_inputs) >= 2:
                        # Model expects both image and metadata inputs
                        chunk_results = model.predict([chunk_transposed, metadata_array], batch_size=10)
                    else:
                        # Model only expects image input (fallback)
                        bt.logging.warning("Tricorder model only has one input - metadata will be ignored")
                        chunk_results = model.predict(chunk_transposed, batch_size=10)
                    
                    results.extend(chunk_results)
                else:
                    # Melanoma format: plain numpy array (no metadata)
                    chunk = data
                    # TensorFlow expects (N, H, W, C) format, so transpose from (N, C, H, W)
                    chunk_transposed = np.transpose(chunk, (0, 2, 3, 1))
                    chunk_results = model.predict(chunk_transposed, batch_size=10)
                    results.extend(chunk_results)
            except Exception as e:
                bt.logging.error(f"TensorFlow inference error on chunk: {e}")
                continue
                
        return results
    
    def _prepare_metadata_array(self, metadata: List[Dict[str, Any]]):
        """Convert metadata list to numpy array for TensorFlow model input"""
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
            
            metadata_array.append([age, gender])
        
        return np.array(metadata_array, dtype=np.float32)
