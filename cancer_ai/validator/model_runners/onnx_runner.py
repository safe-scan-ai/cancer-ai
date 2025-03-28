from typing import List, AsyncGenerator
import numpy as np
import bittensor as bt
from collections import defaultdict
from ..exceptions import ModelRunException
import os

from . import BaseRunnerHandler


class OnnxRunnerHandler(BaseRunnerHandler):
    async def get_chunk_of_data(
        self, X_test: List, chunk_size: int, error_counter: defaultdict
    ) -> AsyncGenerator[List, None]:
        """Opens images using PIL and yields a chunk of them while aggregating errors."""
        import PIL.Image as Image

        for i in range(0, len(X_test), chunk_size):
            bt.logging.debug(f"Processing chunk {i} to {i + chunk_size}")
            chunk = []
            for img_path in X_test[i: i + chunk_size]:
                try:
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(f"File does not exist: {img_path}")

                    with Image.open(img_path) as img:
                        img = img.convert('RGB')  # Ensure image is in RGB
                        chunk.append(img.copy())  # Copy to avoid issues after closing
                except FileNotFoundError:
                    error_counter['FileNotFoundError'] += 1
                    continue  # Skip this image
                except IOError:
                    error_counter['IOError'] += 1
                    continue  # Skip this image
                except Exception:
                    error_counter['UnexpectedError'] += 1
                    continue  # Skip this image

            if not chunk:
                error_counter['EmptyChunk'] += 1
                continue  # Skip this chunk if no images are loaded

            try:
                chunk = self.preprocess_data(chunk)
            except ModelRunException:
                error_counter['PreprocessingFailure'] += 1
                continue  # Skip this chunk if preprocessing fails
            except Exception:
                error_counter['UnexpectedPreprocessingError'] += 1
                continue  # Skip this chunk if preprocessing fails

            yield chunk

    def preprocess_data(self, X_test: List) -> List:
        new_X_test = []
        target_size = (224, 224)  # TODO: Change this to the correct size

        for idx, img in enumerate(X_test):
            try:
                img = img.resize(target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                if img_array.ndim == 2:  # Grayscale image
                    img_array = np.stack((img_array,) * 3, axis=-1)
                elif img_array.shape[-1] != 3:
                    raise ValueError(f"Unexpected number of channels: {img_array.shape[-1]}")

                img_array = np.transpose(
                    img_array, (2, 0, 1)
                )  # Transpose image to (C, H, W)
                new_X_test.append(img_array)
            except (AttributeError, ValueError):
                # These are non-critical issues; skip the image
                continue  # Optionally, you can count these if needed
            except Exception:
                # Log unexpected preprocessing errors
                continue  # Optionally, you can count these if needed

        if not new_X_test:
            raise ModelRunException("No images were successfully preprocessed")

        try:
            new_X_test = np.array(new_X_test, dtype=np.float32)
        except Exception:
            raise ModelRunException("Failed to convert preprocessed images to numpy array")

        return new_X_test

    async def run(self, X_test: List) -> List:
        import onnxruntime

        error_counter = defaultdict(int)  # Initialize error counters

        try:
            session = onnxruntime.InferenceSession(self.model_path)
        except onnxruntime.OnnxRuntimeException:
            raise ModelRunException("Failed to create ONNX inference session")
        except Exception:
            raise ModelRunException("Failed to create ONNX inference session")

        results = []

        async for chunk in self.get_chunk_of_data(X_test, chunk_size=200, error_counter=error_counter):
            try:
                input_batch = np.stack(chunk, axis=0)
            except ValueError:
                error_counter['StackingError'] += 1
                continue  # Skip this batch
            except Exception:
                error_counter['UnexpectedStackingError'] += 1
                continue  # Skip this batch

            try:
                input_name = session.get_inputs()[0].name
                input_data = {input_name: input_batch}
                chunk_results = session.run(None, input_data)[0]
                results.extend(chunk_results)
            except Exception:
                error_counter['UnexpectedInferenceError'] += 1
                continue  # Skip this batch

        # After processing all chunks, handle the error summary
        if error_counter:
            error_summary = []
            for error_type, count in error_counter.items():
                error_summary.append(f"{count} {error_type.replace('_', ' ')}(s)")

            summary_message = "; ".join(error_summary)
            bt.logging.info(f"Processing completed with the following issues: {summary_message}")
        if not results:
            raise ModelRunException("No results obtained from model inference")

        return results
