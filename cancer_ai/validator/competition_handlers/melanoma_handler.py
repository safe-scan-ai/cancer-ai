
from typing import List, AsyncGenerator
import numpy as np
import pickle
import os
from pathlib import Path
from collections import defaultdict
import bittensor as bt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    fbeta_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from cancer_ai.validator.models import WanDBLogModelBase
from .base_handler import BaseCompetitionHandler, BaseModelEvaluationResult
from cancer_ai.utils.structured_logger import log

class MelanomaWanDBLogModelEntry(WanDBLogModelBase):
    accuracy: float
    precision: float
    fbeta: float
    recall: float
    confusion_matrix: list
    roc_curve: dict
    roc_auc: float



class MelanomaEvaluationResult(BaseModelEvaluationResult):
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    fbeta: float = 0.0
    confusion_matrix: list = [[0, 0], [0, 0]]
    fpr: list = []
    tpr: list = []
    roc_auc: float = 0.0

    def to_log_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "fbeta": self.fbeta,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix,
            "roc_curve": {"fpr": self.fpr, "tpr": self.tpr} if self.fpr and self.tpr else {},
            "roc_auc": self.roc_auc,
        }

    class Config:
        arbitrary_types_allowed = True


# Weights for the competition, for calcualting model score
WEIGHT_FBETA = 0.6
WEIGHT_ACCURACY = 0.3
WEIGHT_AUC = 0.1

# Melanoma-specific preprocessing constants
MELANOMA_TARGET_SIZE = (512, 512)
MELANOMA_CHUNK_SIZE = 200


class MelanomaCompetitionHandler(BaseCompetitionHandler):
    WanDBLogModelClass = MelanomaWanDBLogModelEntry

    """Handler for melanoma competition - handles both data preprocessing and model evaluation"""

    def __init__(self, X_test, y_test, config=None) -> None:
        super().__init__(X_test, y_test)
        self.config = config
        self.preprocessed_data_dir = None
        self.preprocessed_chunks = []
        
    def set_preprocessed_data_dir(self, data_dir: str) -> None:
        """Set directory for storing preprocessed data"""
        self.preprocessed_data_dir = Path(data_dir) / "melanoma_preprocessed"
        self.preprocessed_data_dir.mkdir(exist_ok=True)

    async def preprocess_and_serialize_data(self, X_test: List[str]) -> List[str]:
        """
        Preprocess all images and serialize them to disk in chunks.
        Returns list of paths to serialized chunk files.
        """
        if not self.preprocessed_data_dir:
            raise ValueError("Preprocessed data directory not set")
            
        log.dataset.info(f"Preprocessing {len(X_test)} images for melanoma competition")
        #bt.logging.info(f"Preprocessing {len(X_test)} images for melanoma competition")
        error_counter = defaultdict(int)
        chunk_paths = []
        
        for i in range(0, len(X_test), MELANOMA_CHUNK_SIZE):
            log.dataset.debug(f"Processing chunk {i} to {i + MELANOMA_CHUNK_SIZE}")
            #bt.logging.debug(f"Processing chunk {i} to {i + MELANOMA_CHUNK_SIZE}")
            chunk_data = []
            
            for img_path in X_test[i: i + MELANOMA_CHUNK_SIZE]:
                try:
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(f"File does not exist: {img_path}")

                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        preprocessed_img = self._preprocess_single_image(img)
                        chunk_data.append(preprocessed_img)
                        
                except FileNotFoundError:
                    error_counter['FileNotFoundError'] += 1
                    continue
                except IOError:
                    error_counter['IOError'] += 1
                    continue
                except Exception as e:
                    log.dataset.debug(f"Unexpected error processing {img_path}: {e}")
                    #bt.logging.debug(f"Unexpected error processing {img_path}: {e}")
                    error_counter['UnexpectedError'] += 1
                    continue

            if chunk_data:
                try:
                    chunk_array = np.array(chunk_data, dtype=np.float32)
                    chunk_file = self.preprocessed_data_dir / f"chunk_{len(chunk_paths)}.pkl"
                    
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_array, f)
                    
                    chunk_paths.append(str(chunk_file))
                    log.dataset.debug(f"Saved chunk with {len(chunk_data)} images to {chunk_file}")
                    #bt.logging.debug(f"Saved chunk with {len(chunk_data)} images to {chunk_file}")
                    
                except Exception as e:
                    log.dataset.error(f"Failed to serialize chunk: {e}")
                    #bt.logging.error(f"Failed to serialize chunk: {e}")
                    error_counter['SerializationError'] += 1

        if error_counter:
            error_summary = "; ".join([f"{count} {error_type.replace('_', ' ')}(s)" 
                                     for error_type, count in error_counter.items()])
            log.dataset.info(f"Preprocessing completed with issues: {error_summary}")
            #bt.logging.info(f"Preprocessing completed with issues: {error_summary}")
            
        log.dataset.info(f"Preprocessed data saved in {len(chunk_paths)} chunks")
        #bt.logging.info(f"Preprocessed data saved in {len(chunk_paths)} chunks")
        self.preprocessed_chunks = chunk_paths
        return chunk_paths

    def _preprocess_single_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess a single PIL image for melanoma competition"""
        # Resize to target size
        img = img.resize(MELANOMA_TARGET_SIZE)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Handle grayscale images
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] != 3:
            raise ValueError(f"Unexpected number of channels: {img_array.shape[-1]}")

        # Transpose to (C, H, W) format
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    async def get_preprocessed_data_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generator that yields preprocessed data chunks"""
        for chunk_file in self.preprocessed_chunks:
            if os.path.exists(chunk_file):
                try:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                        yield chunk_data
                except Exception as e:
                    log.dataset.error(f"Error loading preprocessed chunk {chunk_file}: {e}")
                    #bt.logging.error(f"Error loading preprocessed chunk {chunk_file}: {e}")
                    continue
            else:
                log.dataset.warn(f"Preprocessed chunk file not found: {chunk_file}")
                #bt.logging.warning(f"Preprocessed chunk file not found: {chunk_file}")

    def preprocess_data(self):
        """Prepare the data for melanoma competition."""
        pass

    def cleanup_preprocessed_data(self) -> None:
        """Clean up preprocessed data files"""
        if self.preprocessed_data_dir and self.preprocessed_data_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.preprocessed_data_dir)
                log.dataset.debug("Cleaned up preprocessed data")
                #bt.logging.debug("Cleaned up preprocessed data")
            except Exception as e:
                log.dataset.error(f"Failed to cleanup preprocessed data: {e}")
                #bt.logging.error(f"Failed to cleanup preprocessed data: {e}")

    def prepare_y_pred(self, y_pred: np.ndarray) -> np.ndarray:
        return [1 if y == "True" else 0 for y in self.y_test]

    def calculate_score(self, fbeta: float, accuracy: float, roc_auc: float) -> float:
        return fbeta * WEIGHT_FBETA + accuracy * WEIGHT_ACCURACY + roc_auc * WEIGHT_AUC

    def get_model_result(
        self, y_test: List[float], y_pred, run_time_s: float
    ) -> MelanomaEvaluationResult:
        # Convert y_pred to numpy array if it's a list
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        
        # Handle the case where y_pred contains arrays instead of scalars
        try:
            # If y_pred is a 2D array, take the first column or flatten it if it's a single prediction per sample
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # If we have multiple predictions per sample, take the first column
                y_pred_flat = y_pred[:, 0]
            else:
                # Otherwise flatten the array to ensure it's 1D
                y_pred_flat = y_pred.flatten()
        except (AttributeError, TypeError):
            # If y_pred doesn't have shape attribute or other issues, use it directly
            y_pred_flat = y_pred
            
        y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred_flat]
        tested_entries = len(y_test)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        fbeta = fbeta_score(y_test, y_pred_binary, beta=2, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        fpr, tpr, _ = roc_curve(y_test, y_pred_flat)
        roc_auc = auc(fpr, tpr)

        score = self.calculate_score(fbeta, accuracy, roc_auc)

        return MelanomaEvaluationResult(
            tested_entries=tested_entries,
            run_time_s=run_time_s,
            accuracy=accuracy,
            precision=precision,
            fbeta=fbeta,
            recall=recall,
            confusion_matrix=conf_matrix.tolist(),
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            roc_auc=roc_auc,
            score=score,
            predictions_raw=y_pred_flat.tolist(),
        )

    def get_comparable_result_fields(self) -> tuple[str, ...]:
        """Field names for get_comparable_result, in order."""
        return (
            "accuracy",
            "precision",
            "recall",
            "fbeta",
            "predictions_raw",
        )

    def get_comparable_result(self, result: MelanomaEvaluationResult) -> tuple:
        """
        Create a comparable representation of the result for grouping duplicates.
        
        Args:
            result: The evaluation result object.
            
        Returns:
            A tuple of key metrics that can be used for comparison.
        """
        if not isinstance(result, MelanomaEvaluationResult):
            return tuple()

        return (
            round(result.accuracy, 6),
            round(result.precision, 6),
            round(result.recall, 6),
            round(result.fbeta, 6),
            tuple(result.predictions_raw),
        )
