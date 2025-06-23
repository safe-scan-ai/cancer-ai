from typing import List, Dict, Any, AsyncGenerator, Tuple, Optional, TypedDict, Literal
from dataclasses import dataclass
import numpy as np
import os
import pickle
from pathlib import Path
from collections import defaultdict
import bittensor as bt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from enum import Enum, IntEnum

from .base_handler import BaseCompetitionHandler, BaseModelEvaluationResult

# --- Constants ---
TARGET_SIZE = (224, 224)
CHUNK_SIZE = 200

# --- Data Structures ---
class RiskCategory(str, Enum):
    BENIGN = "benign"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"

class ClassInfo(TypedDict):
    """Metadata for each skin lesion class"""
    id: int                      # 1-based class ID
    name: str                    # Full class name
    short_name: str              # Short class identifier
    risk_category: RiskCategory  # Risk level
    weight: float                # Scoring weight

# 1-based class IDs for better readability
class ClassId(IntEnum):
    ACTINIC_KERATOSIS = 1
    BASAL_CELL_CARCINOMA = 2
    SEBORRHEIC_KERATOSIS = 3
    SQUAMOUS_CELL_CARCINOMA = 4
    VASCULAR_LESION = 5
    DERMATOFIBROMA = 6
    BENIGN_NEVUS = 7
    OTHER_NON_NEOPLASTIC = 8
    MELANOMA = 9
    OTHER_NEOPLASTIC = 10
    BENIGN = 11  # General benign class

# Class metadata mapping
CLASS_INFO: Dict[ClassId, ClassInfo] = {
    ClassId.ACTINIC_KERATOSIS: {
        "name": "Actinic Keratosis (AK)",
        "short_name": "AK",
        "risk_category": RiskCategory.BENIGN,
        "weight": 1.0
    },
    ClassId.BASAL_CELL_CARCINOMA: {
        "name": "Basal Cell Carcinoma (BCC)",
        "short_name": "BCC",
        "risk_category": RiskCategory.HIGH_RISK,
        "weight": 3.0
    },
    ClassId.SEBORRHEIC_KERATOSIS: {
        "name": "Seborrheic Keratosis (SK)",
        "short_name": "SK",
        "risk_category": RiskCategory.MEDIUM_RISK,
        "weight": 2.0
    },
    ClassId.SQUAMOUS_CELL_CARCINOMA: {
        "name": "Squamous Cell Carcinoma (SCC)",
        "short_name": "SCC",
        "risk_category": RiskCategory.HIGH_RISK,
        "weight": 3.0
    },
    ClassId.VASCULAR_LESION: {
        "name": "Vascular Lesion",
        "short_name": "VASC",
        "risk_category": RiskCategory.MEDIUM_RISK,
        "weight": 2.0
    },
    ClassId.DERMATOFIBROMA: {
        "name": "Dermatofibroma",
        "short_name": "DF",
        "risk_category": RiskCategory.BENIGN,
        "weight": 1.0
    },
    ClassId.BENIGN_NEVUS: {
        "name": "Benign Nevus",
        "short_name": "NEVUS",
        "risk_category": RiskCategory.BENIGN,
        "weight": 1.0
    },
    ClassId.OTHER_NON_NEOPLASTIC: {
        "name": "Other Non-Neoplastic",
        "short_name": "ONN",
        "risk_category": RiskCategory.BENIGN,
        "weight": 1.0
    },
    ClassId.MELANOMA: {
        "name": "Melanoma",
        "short_name": "MEL",
        "risk_category": RiskCategory.HIGH_RISK,
        "weight": 3.0
    },
    ClassId.OTHER_NEOPLASTIC: {
        "name": "Other Neoplastic",
        "short_name": "ON",
        "risk_category": RiskCategory.BENIGN,
        "weight": 1.0
    },
    ClassId.BENIGN: {
        "name": "Benign",
        "short_name": "BENIGN",
        "risk_category": RiskCategory.BENIGN,
        "weight": 0.5  # Lower weight since it's a catch-all benign class
    }
}

# Convert to 0-based indices for model output
CLASS_IDS_0_BASED = [cid - 1 for cid in ClassId]  # [0, 1, 2, ..., 9]
RISK_CATEGORIES = {
    RiskCategory.BENIGN: [cid - 1 for cid, info in CLASS_INFO.items() 
                         if info["risk_category"] == RiskCategory.BENIGN],
    RiskCategory.MEDIUM_RISK: [cid - 1 for cid, info in CLASS_INFO.items() 
                              if info["risk_category"] == RiskCategory.MEDIUM_RISK],
    RiskCategory.HIGH_RISK: [cid - 1 for cid, info in CLASS_INFO.items() 
                            if info["risk_category"] == RiskCategory.HIGH_RISK]
}

# For backward compatibility
BENIGN_CLASSES = RISK_CATEGORIES[RiskCategory.BENIGN]
MEDIUM_RISK_CLASSES = RISK_CATEGORIES[RiskCategory.MEDIUM_RISK]
HIGH_RISK_CLASSES = RISK_CATEGORIES[RiskCategory.HIGH_RISK]

# Weights for different risk categories
BENIGN_WEIGHT = 1.0
MEDIUM_RISK_WEIGHT = 2.0
HIGH_RISK_WEIGHT = 3.0

@dataclass
class TricorderEvaluationResult(BaseModelEvaluationResult):
    """Results from evaluating a model on the tricorder competition."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    fbeta: float = 0.0
    weighted_f1: float = 0.0
    f1_by_class: List[float] = None
    class_weights: List[float] = None
    confusion_matrix: List[List[int]] = None
    risk_category_scores: Dict[RiskCategory, float] = None
    
    def __post_init__(self):
        if self.f1_by_class is None:
            self.f1_by_class = [0.0] * len(ClassInfo)
        if self.class_weights is None:
            self.class_weights = [info["weight"] for info in CLASS_INFO.values()]
        if self.confusion_matrix is None:
            self.confusion_matrix = [[0] * len(ClassInfo) for _ in range(len(ClassInfo))]
        if self.risk_category_scores is None:
            self.risk_category_scores = {cat: 0.0 for cat in RiskCategory}


class TricorderCompetitionHandler(BaseCompetitionHandler):
    """Handler for skin lesion classification competition with 10 classes.
    
    This handler manages the entire competition pipeline including:
    - Data preprocessing and serialization
    - Model evaluation
    - Scoring based on competition rules
    """

    def __init__(self, X_test: List[str], y_test: List[int], config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(X_test, y_test)
        self.config = config or {}
        self.preprocessed_data_dir = None
        self.preprocessed_chunks = []
        
        # Convert string labels to 0-based indices
        self.y_test = []
        for y in y_test:
            if isinstance(y, str) and y in [info["short_name"] for info in CLASS_INFO.values()]:
                # Find class ID by short name
                class_id = next((cid for cid, info in CLASS_INFO.items() 
                              if info["short_name"] == y), None)
                if class_id is not None:
                    self.y_test.append(class_id - 1)  # Convert to 0-based
            elif isinstance(y, int) and y > 0:
                self.y_test.append(y - 1)  # Convert to 0-based if numeric
            else:
                raise ValueError(f"Invalid label: {y}")
        
        # Get class weights from CLASS_INFO
        self.class_weights = [info["weight"] for info in CLASS_INFO.values()]
        
        # Create mapping from short names to class indices for reference
        self.class_name_to_idx = {
            info["short_name"]: cid - 1 
            for cid, info in CLASS_INFO.items()
        }
        
        # Initialize metrics
        self.metrics = {
            'accuracy': 0.0,
            'weighted_f1': 0.0,
            'efficiency': 1.0  # Placeholder for efficiency score
        }

    def set_preprocessed_data_dir(self, data_dir: str) -> None:
        """Set directory for storing preprocessed data"""
        self.preprocessed_data_dir = Path(data_dir) / "tricorder_preprocessed"
        self.preprocessed_data_dir.mkdir(exist_ok=True)

    async def preprocess_and_serialize_data(self, X_test: List[str]) -> List[str]:
        """
        Preprocess all images and serialize them to disk in chunks.
        Returns list of paths to serialized chunk files.
        """
        if not self.preprocessed_data_dir:
            raise ValueError("Preprocessed data directory not set")
            
        bt.logging.info(f"Preprocessing {len(X_test)} images for tricorder competition")
        error_counter = defaultdict(int)
        chunk_paths = []
        
        for i in range(0, len(X_test), CHUNK_SIZE):
            bt.logging.debug(f"Processing chunk {i} to {i + CHUNK_SIZE}")
            chunk_data = []
            
            for img_path in X_test[i: i + CHUNK_SIZE]:
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
                    bt.logging.debug(f"Unexpected error processing {img_path}: {e}")
                    error_counter['UnexpectedError'] += 1
                    continue

            if chunk_data:
                try:
                    chunk_array = np.array(chunk_data, dtype=np.float32)
                    chunk_file = self.preprocessed_data_dir / f"chunk_{len(chunk_paths)}.pkl"
                    
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_array, f)
                    
                    chunk_paths.append(str(chunk_file))
                    bt.logging.debug(f"Saved chunk with {len(chunk_data)} images to {chunk_file}")
                    
                except Exception as e:
                    bt.logging.error(f"Failed to serialize chunk: {e}")
                    error_counter['SerializationError'] += 1

        if error_counter:
            error_summary = "; ".join([f"{count} {error_type.replace('_', ' ')}" 
                                     for error_type, count in error_counter.items()])
            bt.logging.info(f"Preprocessing completed with issues: {error_summary}")
            
        bt.logging.info(f"Preprocessed data saved in {len(chunk_paths)} chunks")
        self.preprocessed_chunks = chunk_paths
        return chunk_paths

    def _preprocess_single_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess a single PIL image for tricorder competition"""
        # Resize to target size
        img = img.resize(TARGET_SIZE)
        
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
                    bt.logging.error(f"Error loading preprocessed chunk {chunk_file}: {e}")
                    continue
            else:
                bt.logging.warning(f"Preprocessed chunk file not found: {chunk_file}")

    def cleanup_preprocessed_data(self) -> None:
        """Clean up preprocessed data files"""
        if self.preprocessed_data_dir and self.preprocessed_data_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.preprocessed_data_dir)
                bt.logging.debug("Cleaned up preprocessed data")
            except Exception as e:
                bt.logging.error(f"Failed to cleanup preprocessed data: {e}")

    def preprocess_data(self):
        """Legacy method - using preprocess_and_serialize_data instead"""
        pass

    def _calculate_risk_category_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[RiskCategory, float]:
        """Calculate F1 scores for each risk category."""
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        category_scores = {}
        
        for category, class_indices in RISK_CATEGORIES.items():
            if class_indices:  # Only calculate if there are classes in this category
                category_f1 = np.mean([f1_scores[i] for i in class_indices])
                category_scores[category] = float(category_f1)
            else:
                category_scores[category] = 0.0
                
        return category_scores

    def _calculate_weighted_f1(self, category_scores: Dict[RiskCategory, float]) -> float:
        """Calculate weighted F1 score based on risk categories."""
        total_weight = sum(info["weight"] for info in CLASS_INFO.values())
        weighted_sum = sum(
            score * CLASS_INFO[ClassId(HIGH_RISK_CLASSES[0] + 1)]["weight"] 
            if category == RiskCategory.HIGH_RISK else
            score * CLASS_INFO[ClassId(MEDIUM_RISK_CLASSES[0] + 1)]["weight"] 
            if category == RiskCategory.MEDIUM_RISK else
            score * CLASS_INFO[ClassId(BENIGN_CLASSES[0] + 1)]["weight"]
            for category, score in category_scores.items()
        )
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate final competition score (0-100)."""
        # 90% prediction quality (50% accuracy, 50% weighted F1)
        prediction_score = 0.5 * metrics['accuracy'] + 0.5 * metrics['weighted_f1']
        
        # 10% efficiency (placeholder - will be calculated based on model size and speed)
        efficiency_score = metrics.get('efficiency', 1.0)  # Default to max if not set
        
        # Final score (0-100 scale)
        final_score = 0.9 * prediction_score + 0.1 * efficiency_score
        return final_score * 100  # Convert to 0-100 scale

    def get_model_result(self, y_test: List[int], y_pred: List[float], run_time_s: float) -> TricorderEvaluationResult:
        """
        Evaluate model predictions and return detailed results.
        
        Args:
            y_test: List of true class indices (0-9)
            y_pred: List of predicted probabilities (shape [n_samples, 10])
            run_time_s: Inference time in seconds
            
        Returns:
            TricorderEvaluationResult with comprehensive evaluation metrics
        """
        try:
            # Convert to numpy arrays
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            
            # Convert string labels to indices if needed
            if isinstance(y_test[0], str):
                y_test = np.array([self.class_name_to_idx.get(y, -1) for y in y_test])
                if -1 in y_test:
                    raise ValueError("Invalid class name in y_test")
            # Ensure y_test is 0-based if numeric
            elif y_test.min() > 0:
                y_test = y_test - 1
            
            # Get predicted class indices
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate basic metrics
            accuracy = float(accuracy_score(y_test, y_pred_classes))
            precision = float(precision_score(y_test, y_pred_classes, average='weighted', zero_division=0))
            recall = float(recall_score(y_test, y_pred_classes, average='weighted', zero_division=0))
            fbeta = float(f1_score(y_test, y_pred_classes, average='weighted', zero_division=0))
            
            # Calculate F1 scores by class and risk category
            f1_scores = f1_score(y_test, y_pred_classes, average=None, zero_division=0)
            category_scores = self._calculate_risk_category_scores(y_test, y_pred_classes)
            weighted_f1 = self._calculate_weighted_f1(category_scores)
            
            # Update metrics
            self.metrics.update({
                'accuracy': accuracy,
                'weighted_f1': weighted_f1,
                'run_time_s': run_time_s
            })
            
            # Create result object
            result = TricorderEvaluationResult(
                tested_entries=len(y_test),
                run_time_s=run_time_s,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                fbeta=fbeta,
                weighted_f1=weighted_f1,
                f1_by_class=f1_scores.tolist(),
                class_weights=self.class_weights,
                confusion_matrix=confusion_matrix(y_test, y_pred_classes).tolist(),
                risk_category_scores=category_scores,
                score=self.calculate_score(self.metrics)
            )
            
            # Log important metrics
            bt.logging.info(f"Model evaluation results:")
            bt.logging.info(f"- Accuracy: {accuracy:.4f}")
            bt.logging.info(f"- Weighted F1: {weighted_f1:.4f}")
            for category, score in category_scores.items():
                bt.logging.info(f"- {category.value} F1: {score:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in get_model_result: {str(e)}"
            bt.logging.error(error_msg, exc_info=True)
            return TricorderEvaluationResult(
                tested_entries=len(y_test) if 'y_test' in locals() else 0,
                run_time_s=run_time_s,
                error=error_msg
            )