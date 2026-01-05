from typing import List, AsyncGenerator, Union, Tuple
import numpy as np
import bittensor as bt
import asyncio
from collections import defaultdict
from ..exceptions import ModelRunException

from . import BaseRunnerHandler


class OnnxRunnerHandler(BaseRunnerHandler):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)
        self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up ONNX session and release resources."""
        if self.session:
            bt.logging.debug("Cleaning up ONNX session.")
            self.session = None
            import gc
            gc.collect()
    
    def _get_model_input_size(self, session) -> Tuple[int, int]:
        """Extract expected image input size from ONNX model"""
        inputs = session.get_inputs()
        if inputs:
            shape = inputs[0].shape
            # Shape is typically [batch_size, channels, height, width]
            if len(shape) >= 4:
                h = shape[2] if isinstance(shape[2], int) else 512
                w = shape[3] if isinstance(shape[3], int) else 512
                return (h, w)
        return (512, 512)  # Default fallback
    
    def _resize_image_batch(self, chunk: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image batch to target size using PIL for quality"""
        from PIL import Image
        
        batch_size, channels, height, width = chunk.shape
        target_h, target_w = target_size
        
        if (height, width) == target_size:
            return chunk  # Already correct size
        
        resized_batch = []
        for i in range(batch_size):
            # Convert from (C, H, W) to (H, W, C) for PIL
            img_array = np.transpose(chunk[i], (1, 2, 0))
            # Scale back to 0-255 range for PIL
            img_array = (img_array * 255).astype(np.uint8)
            
            # Resize using PIL
            img = Image.fromarray(img_array)
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            # Convert back to (C, H, W) and normalize to 0-1
            resized_array = np.array(img, dtype=np.float32) / 255.0
            resized_array = np.transpose(resized_array, (2, 0, 1))
            resized_batch.append(resized_array)
        
        return np.array(resized_batch, dtype=np.float32)

    async def run(self, preprocessed_data_generator: AsyncGenerator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], None]) -> List:
        """
        Run ONNX model inference on preprocessed data chunks.
        
        Args:
            preprocessed_data_generator: Generator yielding preprocessed numpy arrays,
                                       or tuples of (numpy arrays, preprocessed_metadata) for tricorder
            
        Returns:
            List of model predictions
        """
        import onnxruntime

        error_counter = defaultdict(int)

        try:
            self.session = onnxruntime.InferenceSession(self.model_path)
        except onnxruntime.OnnxRuntimeException as e:
            bt.logging.error(f"ONNX runtime error when loading model: {e}")
            raise ModelRunException(f"ONNX runtime error when loading model: {e}") from e
        except OSError as e:
            bt.logging.error(f"File error when loading ONNX model: {e}")
            raise ModelRunException(f"File error when loading ONNX model: {e}") from e

        # Detect model's expected input size
        model_input_size = self._get_model_input_size(self.session)

        results = []

        async for data in preprocessed_data_generator:
            try:                
                # Handle both formats: plain numpy array or tuple with preprocessed metadata
                if isinstance(data, tuple):
                    # Tricorder format: (image_data, preprocessed_metadata)
                    chunk, metadata = data
                    
                    # Resize chunk to match model's expected input size
                    chunk = self._resize_image_batch(chunk, model_input_size)
                    
                    # Prepare inputs for ONNX model
                    inputs = self.session.get_inputs()

                    # Process each image individually
                    batch_size = chunk.shape[0]
                    
                    for i in range(batch_size):
                        # Extract single image: (3, 512, 512) -> (1, 3, 512, 512)
                        single_image = chunk[i:i+1]
                        # Extract single metadata: (, 3) -> (1, 3)
                        single_metadata = metadata[i:i+1]
                        
                        # Prepare input for single image
                        if len(inputs) >= 2:
                            # Model expects both image and metadata inputs
                            image_input_name = inputs[0].name
                            metadata_input_name = inputs[1].name
                            
                            input_data = {
                                image_input_name: single_image,
                                metadata_input_name: single_metadata
                            }
                        else:
                            # Model only expects image input (fallback)
                            input_data = {inputs[0].name: single_image}
                        
                        bt.logging.trace(f"Running ONNX inference on image {i+1}/{batch_size} with shape {single_image.shape}")
                        single_result = await asyncio.wait_for(
                            asyncio.to_thread(self.session.run, None, input_data),
                            timeout=120.0  # 2 minutes max per image
                        )
                        single_result = single_result[0]
                        if len(single_result.shape) > 1 and single_result.shape[0] == 1:
                            single_result = np.squeeze(single_result, axis=0)  # Remove batch dimension
                        results.append(single_result)
                    
                else:
                    # Melanoma format: plain numpy array (no metadata)
                    chunk = data
                    
                    # Resize chunk to match model's expected input size
                    chunk = self._resize_image_batch(chunk, model_input_size)
                    batch_size = chunk.shape[0]
                    for i in range(batch_size):
                        single_image = chunk[i:i+1]
                        input_name = self.session.get_inputs()[0].name
                        input_data = {input_name: single_image}
                        bt.logging.debug(f"Running ONNX inference on image {i+1}/{batch_size} with shape {single_image.shape}")
                        single_result = await asyncio.wait_for(
                            asyncio.to_thread(self.session.run, None, input_data),
                            timeout=120.0  # 2 minutes max per image
                        )
                        single_result = single_result[0]
                        if len(single_result.shape) > 1 and single_result.shape[0] == 1:
                            single_result = np.squeeze(single_result, axis=0)   # Remove batch dimension
                        results.append(single_result)
                
            except asyncio.TimeoutError:
                bt.logging.error(f"ONNX inference timeout after 120s on chunk")
                error_counter['TimeoutError'] += 1
                continue
            except Exception as e:
                bt.logging.error(f"ONNX inference error during chunk processing: {e}", exc_info=True)
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
