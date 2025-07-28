from typing import List, AsyncGenerator, Union, Dict, Any, Tuple
import numpy as np
import bittensor as bt
from collections import defaultdict
from ..exceptions import ModelRunException
from ..competition_handlers.tricorder_handler import LocationId

from . import BaseRunnerHandler


class OnnxRunnerHandler(BaseRunnerHandler):
    async def run(self, preprocessed_data_generator: AsyncGenerator[Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]], None]) -> List:
        """
        Run ONNX model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays,
                                       or tuples of (numpy arrays, metadata) for tricorder
            
        Returns:
            List of model predictions
        """
        import onnxruntime

        error_counter = defaultdict(int)

        try:
            session = onnxruntime.InferenceSession(self.model_path)
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred when loading ONNX model: {e}")
            raise ModelRunException(f"An unexpected error occurred when loading ONNX model: {e}") from e

        results = []

        async for data in preprocessed_data_generator:
            try:                
                # Handle both formats: plain numpy array or tuple with metadata
                if isinstance(data, tuple):
                    # Tricorder format: (image_data, metadata)
                    chunk, metadata = data
                    
                    # Prepare inputs for ONNX model
                    inputs = session.get_inputs()
                    input_data = {}
                    
                    if len(inputs) >= 2:
                        # Model expects both image and metadata inputs
                        image_input_name = inputs[0].name
                        metadata_input_name = inputs[1].name
                        metadata_array = self._prepare_metadata_array(metadata)
                        
                        input_data = {
                            image_input_name: chunk,
                            metadata_input_name: metadata_array
                        }
                    else:
                        # Model only expects image input (fallback)
                        input_data = {inputs[0].name: chunk}
                else:
                    # Melanoma format: plain numpy array (no metadata)
                    chunk = data
                    input_name = session.get_inputs()[0].name
                    input_data = {input_name: chunk}
                
                chunk_results = session.run(None, input_data)[0]
                results.extend(chunk_results)
                
            except Exception as e:
                bt.logging.warning(f"An error occurred during inference on chunk {data}: {e}")
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
    
    def _prepare_metadata_array(self, metadata: List[Dict[str, Any]]):
        """Convert metadata list to numpy array for ONNX model input"""
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
            
            # Convert location to numerical using LocationId enum
            location_str = entry.get('location', '').lower() if entry.get('location') else ''
            location = self._get_location_value(location_str)
            
            metadata_array.append([age, gender, location])
        
        return np.array(metadata_array, dtype=np.float32)
    
    def _get_location_value(self, location_str: str) -> int:
        """Convert location string to numerical value using LocationId enum."""
        if not location_str:
            return -1
        
        try:
            # Convert to uppercase to match enum names
            location_enum = LocationId[location_str.upper()]
            return location_enum.value
        except KeyError:
            # Unknown/invalid location
            return -1
