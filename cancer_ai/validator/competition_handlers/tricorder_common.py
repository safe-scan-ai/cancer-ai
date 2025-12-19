from enum import Enum, IntEnum
from typing import TypedDict, List, Dict, Any, AsyncGenerator, Tuple, Optional
from abc import ABC, abstractmethod

import os
import json
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

from pydantic import Field
import numpy as np
import bittensor as bt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

from cancer_ai.validator.models import WanDBLogModelBase
from .base_handler import BaseCompetitionHandler, BaseModelEvaluationResult

MAX_INVALID_ENTRIES = 2  # Maximum number of invalid entries allowed in the dataset

# --- Constants ---
TARGET_SIZE = (512, 512)
CHUNK_SIZE = 200

# Image preprocessing constants
NORMALIZATION_FACTOR = 255.0

# Risk category weights for scoring
CATEGORY_WEIGHTS = {"HIGH_RISK": 3.0, "MEDIUM_RISK": 2.0, "BENIGN": 1.0}

# Efficiency scoring constants
MIN_MODEL_SIZE_MB = 50
MAX_MODEL_SIZE_MB = 150
EFFICIENCY_RANGE_MB = 100  # MAX - MIN

# Final scoring weights
PREDICTION_WEIGHT = 0.9
EFFICIENCY_WEIGHT = 0.1
ACCURACY_WEIGHT = 0.5
WEIGHTED_F1_WEIGHT = 0.5

# Age validation
MAX_AGE = 120

# Tricorder-3 Model Architecture Constants
TRICORDER_3_NUM_CLASSES = 11
TRICORDER_3_NUM_DEMOGRAPHICS = 3
TRICORDER_3_IMAGE_SIZE = (512, 512)
TRICORDER_3_FEATURE_CHANNELS = [16, 32, 64]  # Conv layer output channels
TRICORDER_3_DEMOGRAPHICS_FEATURES = 16
TRICORDER_3_COMBINED_FEATURES = 64 + 16  # Image features + demographics features


# --- Data Structures ---
class RiskCategory(str, Enum):
    BENIGN = "benign"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"


class ClassInfo(TypedDict):
    """Metadata for each skin lesion class"""

    id: int  # 1-based class ID
    name: str  # Full class name
    short_name: str  # Short class identifier
    risk_category: RiskCategory  # Risk level
    weight: float  # Scoring weight


class LocationId(IntEnum):
    """Body location enumeration for skin lesions."""
    ARM = 1
    FEET = 2
    GENITALIA = 3
    HAND = 4
    HEAD = 5
    LEG = 6
    TORSO = 7


# Independent mapping of string names to enum values for easy lookup
LOCATION_NAME_TO_VALUE = {
    'ARM': 1,
    'FEET': 2,
    'GENITALIA': 3,
    'HAND': 4,
    'HEAD': 5,
    'LEG': 6,
    'TORSO': 7
}


def convert_metadata_to_array(metadata: List[Dict[str, Any]]) -> np.ndarray:
    """Convert metadata list to numpy array for model input.
    
    Args:
        metadata: List of dictionaries containing age, gender, and location
        
    Returns:
        Numpy array with shape (n_samples, 3) containing [age, gender, location]
    """
    metadata_array = []
    for entry in metadata:
        # Convert entry keys to lowercase for case-insensitive matching
        entry_lower = {k.lower(): v for k, v in entry.items()}
        
        age = entry_lower.get('age', 0) if entry_lower.get('age') is not None else 0
        # Convert gender to numerical: male=1, female=0, unknown=-1
        gender_str = entry_lower.get('gender', '').lower() if entry_lower.get('gender') else ''
        if gender_str in ['male', 'm']:
            gender = 1
        elif gender_str in ['female', 'f']:
            gender = 0
        else:
            gender = -1  # Unknown/missing gender
        
        # Convert location to numerical using LocationId enum
        location_str = entry_lower.get('location', '').lower() if entry_lower.get('location') else ''
        location = get_location_value(location_str)
        
        metadata_array.append([age, gender, location])
    
    return np.array(metadata_array, dtype=np.float32)


def get_location_value(location_str: str) -> int:
    """Convert location string to numerical value using LocationId enum.
    
    Args:
        location_str: Location string (e.g., 'arm', 'head', etc.)
        
    Returns:
        Integer location value, or -1 for unknown/invalid locations
    """
    if not location_str:
        return -1
    
    try:
        # Convert to uppercase to match mapping keys
        return LOCATION_NAME_TO_VALUE[location_str.upper()]
    except KeyError:
        # Unknown/invalid location
        return -1


# Weights for different risk categories
BENIGN_WEIGHT = 1.0
MEDIUM_RISK_WEIGHT = 2.0
HIGH_RISK_WEIGHT = 3.0


class TricorderWanDBLogModelEntry(WanDBLogModelBase):
    tested_entries: int
    accuracy: float
    precision: float
    fbeta: float
    recall: float
    confusion_matrix: list
    roc_curve: dict | None = None
    roc_auc: float | None = None
    weighted_f1: float | None = None
    f1_by_class: list | None = None
    class_weights: list | None = None
    risk_category_scores: dict | None = None
    predictions_raw: list | None = None
    error: str | None = None


class TricorderEvaluationResult(BaseModelEvaluationResult):
    """Results from evaluating a model on the tricorder competition."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    fbeta: float = 0.0
    weighted_f1: float = 0.0
    efficiency_score: float = 1.0
    f1_by_class: List[float] = Field(default_factory=list)
    class_weights: List[float] = Field(default_factory=list)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    risk_category_scores: Dict[RiskCategory, float] = Field(
        default_factory=lambda: {category: 0.0 for category in RiskCategory}
    )

    def to_log_dict(self) -> dict:
        return {
            "tested_entries": self.tested_entries,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "fbeta": self.fbeta,
            "recall": self.recall,
            "efficiency_score": self.efficiency_score,
            "confusion_matrix": self.confusion_matrix,
            "roc_curve": getattr(self, "roc_curve", None),
            "roc_auc": getattr(self, "roc_auc", None),
            "weighted_f1": getattr(self, "weighted_f1", None),
            "f1_by_class": getattr(self, "f1_by_class", None),
            "class_weights": getattr(self, "class_weights", None),
            "risk_category_scores": getattr(self, "risk_category_scores", None),
            "predictions_raw": getattr(self, "predictions_raw", None),
            "score": getattr(self, "score", None),
            "error": getattr(self, "error", None),
        }


class BaseTricorderCompetitionHandler(BaseCompetitionHandler, ABC):
    """Base class for Tricorder competition handlers with common functionality."""
    
    WanDBLogModelClass = TricorderWanDBLogModelEntry

    def __init__(
        self,
        X_test: List[str],
        y_test: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(X_test, y_test)
        self.config = config or {}
        self.metadata = metadata or [
            {"age": None, "gender": None, "location": None} for _ in X_test
        ]
        self.preprocessed_data_dir = None
        self.preprocessed_chunks = []

        # Get class-specific info from subclass
        self.CLASS_INFO = self.get_class_info()
        self.RISK_CATEGORIES = self._calculate_risk_categories()
        self.class_weights = [info["weight"] for info in self.CLASS_INFO.values()]
        self.class_name_to_idx = {
            info["short_name"]: cid - 1 for cid, info in self.CLASS_INFO.items()
        }

        self._validate_data()
        self._convert_labels()
        self._initialize_metrics()

    @abstractmethod
    def get_class_info(self) -> Dict[Any, Dict[str, Any]]:
        """Return the class information dictionary for this competition version."""
        pass

    def _calculate_risk_categories(self) -> Dict[RiskCategory, List[int]]:
        """Calculate risk categories based on CLASS_INFO."""
        return {
            RiskCategory.BENIGN: [
                cid - 1
                for cid, info in self.CLASS_INFO.items()
                if info["risk_category"] == RiskCategory.BENIGN
            ],
            RiskCategory.MEDIUM_RISK: [
                cid - 1
                for cid, info in self.CLASS_INFO.items()
                if info["risk_category"] == RiskCategory.MEDIUM_RISK
            ],
            RiskCategory.HIGH_RISK: [
                cid - 1
                for cid, info in self.CLASS_INFO.items()
                if info["risk_category"] == RiskCategory.HIGH_RISK
            ],
        }

    def _validate_data(self) -> None:
        """Validate metadata and labels."""
        validation_errors = []

        for i, meta_entry in enumerate(self.metadata):
            # Convert entry keys to lowercase for case-insensitive matching
            entry_lower = {k.lower(): v for k, v in meta_entry.items()}
            
            # Validate age
            age = entry_lower.get("age")
            if age is None:
                validation_errors.append(f"Missing age at index {i}")
            elif not isinstance(age, (int, float)) or age < 0 or age > MAX_AGE:
                validation_errors.append(
                    f"Invalid age at index {i}: {age} (must be 0-120)"
                )

            # Validate gender
            gender = entry_lower.get("gender")
            if gender is None:
                validation_errors.append(f"Missing gender at index {i}")
            else:
                gender_lower = str(gender).lower()
                if gender_lower not in ["m", "f", "male", "female"]:
                    validation_errors.append(
                        f"Invalid gender at index {i}: {gender} (must be 'm', 'f', 'male', 'female')"
                    )
                else:
                    meta_entry["gender"] = gender_lower

            # Validate location
            location = entry_lower.get("location")
            if location is None:
                validation_errors.append(f"Missing location at index {i}")
            else:
                location_lower = str(location).lower()
                valid_locations = [
                    "arm", "feet", "genitalia", "hand", "head", "leg", "torso",
                ]
                if location_lower not in valid_locations:
                    validation_errors.append(
                        f"Invalid location at index {i}: {location} (must be one of {valid_locations})"
                    )
                else:
                    meta_entry["location"] = location_lower

        # Validate labels
        valid_label_names = [info["short_name"] for info in self.CLASS_INFO.values()]
        for i, label in enumerate(self.y_test):
            if isinstance(label, str):
                if label not in valid_label_names:
                    validation_errors.append(
                        f"Invalid label at index {i}: {label} (must be one of {valid_label_names})"
                    )
            elif isinstance(label, int):
                if label < 1 or label > len(self.CLASS_INFO):
                    validation_errors.append(
                        f"Invalid label at index {i}: {label} (must be 1-{len(self.CLASS_INFO)})"
                    )
            else:
                validation_errors.append(
                    f"Invalid label type at index {i}: {type(label)} (must be string or int)"
                )

        # Handle validation errors
        if validation_errors:
            error_summary = "\n".join(validation_errors[:10])
            if len(validation_errors) > 10:
                error_summary += f"\n... and {len(validation_errors) - 10} more errors"

            bt.logging.warning(
                f"TRICORDER COMPETITION WARNING: Dataset validation has issues"
            )
            bt.logging.warning(f"Found {len(validation_errors)} validation errors:")
            bt.logging.warning(error_summary)

            if len(validation_errors) > MAX_INVALID_ENTRIES:
                bt.logging.error(
                    f"TRICORDER COMPETITION CANCELLED: Not enough valid data to evaluate"
                )
                bt.logging.error(
                    f" {len(validation_errors)} entries are invalid, maximum invalid: {MAX_INVALID_ENTRIES}"
                )
                raise ValueError(f"Not enough valid data to evaluate.")

    def _convert_labels(self) -> None:
        """Convert string labels to 0-based indices."""
        converted_labels = []
        for y in self.y_test:
            if isinstance(y, str) and y in [
                info["short_name"] for info in self.CLASS_INFO.values()
            ]:
                # Find class ID by short name
                class_id = next(
                    (
                        cid
                        for cid, info in self.CLASS_INFO.items()
                        if info["short_name"] == y
                    ),
                    None,
                )
                if class_id is not None:
                    converted_labels.append(class_id - 1)  # Convert to 0-based
            elif isinstance(y, int) and y > 0:
                converted_labels.append(y - 1)  # Convert to 0-based if numeric
            else:
                raise ValueError(f"Invalid label: {y}")
        self.y_test = converted_labels

    def _initialize_metrics(self) -> None:
        """Initialize metrics dictionary."""
        self.metrics = {
            "accuracy": 0.0,
            "weighted_f1": 0.0,
            "efficiency": 1.0,
        }

    def set_preprocessed_data_dir(self, data_dir: str) -> None:
        """Set directory for storing preprocessed data"""
        self.preprocessed_data_dir = Path(data_dir) / "tricorder_preprocessed"
        self.preprocessed_data_dir.mkdir(exist_ok=True)

    async def preprocess_and_serialize_data(self, X_test: List[str]) -> List[str]:
        """Preprocess all images with metadata and serialize them to disk in chunks."""
        if not self.preprocessed_data_dir:
            raise ValueError("Preprocessed data directory not set")
        
        if len(X_test) != len(self.y_test) or len(X_test) != len(self.metadata):
            raise ValueError(
                f"Mismatched lengths: X_test={len(X_test)}, "
                f"y_test={len(self.y_test)}, metadata={len(self.metadata)}"
            )

        bt.logging.trace(
            f"Preprocessing {len(X_test)} images for tricorder competition"
        )
        bt.logging.trace(f"Using chunk size: {CHUNK_SIZE}")
        bt.logging.trace(f"Available metadata entries: {len(self.metadata)}")
        error_counter = defaultdict(int)
        chunk_paths = []

        for i in range(0, len(X_test), CHUNK_SIZE):
            bt.logging.trace(
                f"Processing chunk {len(chunk_paths)} - images {i} to {min(i + CHUNK_SIZE, len(X_test))}"
            )
            chunk_data = []
            chunk_metadata = []

            for idx, img_path in enumerate(X_test[i : i + CHUNK_SIZE]):
                try:
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(f"File does not exist: {img_path}")

                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        preprocessed_img = self._preprocess_single_image(img)
                        chunk_data.append(preprocessed_img)

                        # Add corresponding metadata
                        global_idx = i + idx
                        if global_idx < len(self.metadata):
                            chunk_metadata.append(self.metadata[global_idx])
                        else:
                            chunk_metadata.append(
                                {"age": None, "gender": None, "location": None}
                            )

                except FileNotFoundError:
                    error_counter["FileNotFoundError"] += 1
                    continue
                except IOError:
                    error_counter["IOError"] += 1
                    continue
                except Exception as e:
                    bt.logging.debug(f"Unexpected error processing {img_path}: {e}")
                    error_counter["UnexpectedError"] += 1
                    continue

            if chunk_data:
                try:
                    chunk_array = np.array(chunk_data, dtype=np.float32)
                    chunk_file = (
                        self.preprocessed_data_dir / f"chunk_{len(chunk_paths)}.pkl"
                    )
                    metadata_file = (
                        self.preprocessed_data_dir / f"metadata_{len(chunk_paths)}.pkl"
                    )

                    with open(chunk_file, "wb") as f:
                        pickle.dump(chunk_array, f)

                    with open(metadata_file, "wb") as f:
                        pickle.dump(chunk_metadata, f)

                    chunk_paths.append(str(chunk_file))
                    bt.logging.trace(
                        f"Saved chunk with {len(chunk_data)} images and metadata to {chunk_file}"
                    )

                except Exception as e:
                    bt.logging.error(f"Failed to serialize chunk: {e}")
                    error_counter["SerializationError"] += 1

        if error_counter:
            error_summary = "; ".join(
                [
                    f"{count} {error_type.replace('_', ' ')}"
                    for error_type, count in error_counter.items()
                ]
            )
            bt.logging.trace(
                f"Preprocessing completed with issues: {error_summary}"
            )

        bt.logging.trace(
            f"Preprocessed data saved in {len(chunk_paths)} chunks"
        )
        bt.logging.trace(f"Chunk paths: {chunk_paths}")
        self.preprocessed_chunks = chunk_paths
        return chunk_paths

    def _preprocess_single_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess a single PIL image for tricorder competition"""
        # Resize to target size using a deterministic algorithm
        img = img.resize(TARGET_SIZE, Image.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / NORMALIZATION_FACTOR

        # Handle grayscale images
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] != 3:
            raise ValueError(f"Unexpected number of channels: {img_array.shape[-1]}")

        # Transpose to (C, H, W) format
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    async def get_preprocessed_data_generator(
        self,
    ) -> AsyncGenerator[Tuple[np.ndarray, np.ndarray], None]:
        """Generator that yields preprocessed data chunks with preprocessed metadata"""

        for i, chunk_file in enumerate(self.preprocessed_chunks):
            bt.logging.trace(f"Processing chunk {i}: {chunk_file}")
            if os.path.exists(chunk_file):
                try:
                    # Load image data
                    with open(chunk_file, "rb") as f:
                        chunk_data = pickle.load(f)

                    # Load corresponding metadata
                    metadata_file = str(Path(chunk_file).parent / f"metadata_{i}.pkl")
                    chunk_metadata = []
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "rb") as f:
                            chunk_metadata = pickle.load(f)
                    else:
                        # Default metadata if file doesn't exist
                        bt.logging.warning(
                            f"Metadata file not found, using defaults"
                        )
                        chunk_metadata = [
                            {"age": None, "gender": None, "location": None}
                            for _ in range(len(chunk_data))
                        ]

                    bt.logging.trace(
                        f"Yielding chunk {i} with {len(chunk_data)} samples and {len(chunk_metadata)} metadata"
                    )
                    preprocessed_metadata = convert_metadata_to_array(chunk_metadata)
                    yield chunk_data, preprocessed_metadata
                except Exception as e:
                    bt.logging.error(
                        f"Error loading preprocessed chunk {chunk_file}: {e}"
                    )
                    continue
            else:
                bt.logging.warning(f"Preprocessed chunk file not found: {chunk_file}")

    def cleanup_preprocessed_data(self) -> None:
        """Clean up preprocessed data files"""
        if self.preprocessed_data_dir and self.preprocessed_data_dir.exists():
            import shutil

            try:
                shutil.rmtree(self.preprocessed_data_dir)
                bt.logging.trace("Cleaned up preprocessed data")
            except Exception as e:
                bt.logging.error(f"Failed to cleanup preprocessed data: {e}")

    def preprocess_data(self):
        """Legacy method - using preprocess_and_serialize_data instead"""
        pass

    def prepare_y_pred(self, y_pred):
        """Convert string labels to 0-based indices for evaluation."""
        converted = []
        for y in y_pred:
            if isinstance(y, str):
                # Find class ID by short name
                class_id = next(
                    (
                        cid
                        for cid, info in self.CLASS_INFO.items()
                        if info["short_name"] == y
                    ),
                    None,
                )
                if class_id is not None:
                    converted.append(class_id - 1)  # Convert to 0-based
                else:
                    raise ValueError(f"Unknown class short name: {y}")
            elif isinstance(y, (int, float)):
                converted.append(int(y) - 1)  # Convert to 0-based if numeric
            else:
                raise ValueError(f"Invalid label type: {type(y).__name__}")
        return converted

    def _calculate_risk_category_scores(
        self, f1_scores: np.ndarray
    ) -> Dict[RiskCategory, float]:
        """Calculate F1 scores for each risk category based on pre-computed F1 scores per class."""
        category_scores = {}

        for category, class_indices in self.RISK_CATEGORIES.items():
            if class_indices:
                category_f1 = np.mean([f1_scores[i] for i in class_indices])
                category_scores[category] = float(category_f1)
            else:
                category_scores[category] = 0.0

        return category_scores

    def _calculate_weighted_f1(
        self, category_scores: Dict[RiskCategory, float]
    ) -> float:
        """Calculate weighted F1 score based on risk categories."""
        # Use category-level weights from constants
        category_weights = {
            RiskCategory.HIGH_RISK: CATEGORY_WEIGHTS["HIGH_RISK"],
            RiskCategory.MEDIUM_RISK: CATEGORY_WEIGHTS["MEDIUM_RISK"],
            RiskCategory.BENIGN: CATEGORY_WEIGHTS["BENIGN"],
        }

        total_weight = sum(category_weights.values())
        weighted_sum = sum(
            category_scores.get(category, 0.0) * weight
            for category, weight in category_weights.items()
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate final competition score (0-1)."""
        import bittensor as bt
        
        try:
            bt.logging.debug(f"calculate_score called with metrics={metrics}")
            bt.logging.debug(f"Scoring weights - ACCURACY_WEIGHT={ACCURACY_WEIGHT}, WEIGHTED_F1_WEIGHT={WEIGHTED_F1_WEIGHT}, PREDICTION_WEIGHT={PREDICTION_WEIGHT}, EFFICIENCY_WEIGHT={EFFICIENCY_WEIGHT}")
            
            # Prediction quality (accuracy + weighted F1)
            prediction_score = (
                ACCURACY_WEIGHT * metrics["accuracy"]
                + WEIGHTED_F1_WEIGHT * metrics["weighted_f1"]
            )
            bt.logging.debug(f"prediction_score = {ACCURACY_WEIGHT} * {metrics['accuracy']} + {WEIGHTED_F1_WEIGHT} * {metrics['weighted_f1']} = {prediction_score:.6f}")

            # Efficiency score
            efficiency_score = metrics.get("efficiency", 1.0)  # Default to max if not set
            bt.logging.debug(f"efficiency_score from metrics = {efficiency_score:.6f}")

            final_score = (
                PREDICTION_WEIGHT * prediction_score + EFFICIENCY_WEIGHT * efficiency_score
            )
            bt.logging.debug(f"final_score = {PREDICTION_WEIGHT} * {prediction_score:.6f} + {EFFICIENCY_WEIGHT} * {efficiency_score:.6f} = {final_score:.6f}")
            
            return final_score
        except Exception as e:
            bt.logging.error(f"ERROR in calculate_score: {e}, metrics: {metrics}", exc_info=True)
            return 0.0

    def get_model_result(
        self,
        y_test: List[int],
        y_pred: List[float],
        run_time_s: float,
        model_size_mb: float = None,
    ) -> TricorderEvaluationResult:
        """Evaluate model predictions and return detailed results."""
        import bittensor as bt
        
        try:
            bt.logging.debug(f"get_model_result called with y_test len={len(y_test)}, y_pred len={len(y_pred)}, model_size_mb={model_size_mb}")
            
            # Convert to numpy arrays
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            

            # Define all possible class labels
            labels = list(range(len(self.CLASS_INFO)))

            # Get predicted class indices - handle both 1D and 2D predictions
            if y_pred.ndim == 1:
                y_pred_classes = y_pred.astype(int)
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)

            # Calculate basic metrics
            accuracy = float(accuracy_score(y_test, y_pred_classes))
            precision = float(
                precision_score(
                    y_test,
                    y_pred_classes,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            )
            recall = float(
                recall_score(
                    y_test,
                    y_pred_classes,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            )
            fbeta = float(
                f1_score(
                    y_test,
                    y_pred_classes,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            )
            f1_scores = f1_score(
                y_test, y_pred_classes, labels=labels, average=None, zero_division=0
            )

            # Calculate risk category scores and weighted F1
            category_scores = self._calculate_risk_category_scores(f1_scores)
            
            weighted_f1 = self._calculate_weighted_f1(category_scores)

            # Log important metrics
            bt.logging.trace(f"Model evaluation results: Accuracy: {accuracy:.4f}, Weighted F1: {weighted_f1:.4f}")
            for category, score in category_scores.items():
                bt.logging.trace(f"- {category.value} F1: {score:.4f}")

        
            efficiency_score = 1.0  # Default to max if size not provided
            if model_size_mb is not None:
                if model_size_mb <= MIN_MODEL_SIZE_MB:
                    efficiency_score = 1.0  # Full efficiency score
                elif model_size_mb <= MAX_MODEL_SIZE_MB:
                    # Linear decay from 1.0 to 0.0 between MIN and MAX MB
                    efficiency_score = (
                        MAX_MODEL_SIZE_MB - model_size_mb
                    ) / EFFICIENCY_RANGE_MB
                else:
                    efficiency_score = 0.0  # No efficiency score above MAX MB

                bt.logging.trace(
                    f"- Model size: {model_size_mb:.1f}MB, Efficiency score: {efficiency_score:.2f}"
                )
            else:
                bt.logging.trace("No model size provided, using default efficiency=1.0")

            # Calculate final score using calculate_score method
            # Round metrics to ensure deterministic scoring across different hardware
            metrics = {
                "accuracy": round(accuracy, 6),
                "weighted_f1": round(weighted_f1, 6),
                "efficiency": round(efficiency_score, 6),
            }
            bt.logging.trace(f"Final metrics before calculate_score: {metrics}")
            
            score = self.calculate_score(metrics)
            bt.logging.trace(f"calculate_score returned: {score:.6f}")
            
            # Create result object
            bt.logging.debug("Creating TricorderEvaluationResult object...")
            result = TricorderEvaluationResult(
                tested_entries=len(y_test),
                run_time_s=run_time_s,
                predictions_raw=y_pred.tolist(),
                accuracy=metrics["accuracy"],
                precision=round(precision, 6),
                recall=round(recall, 6),
                fbeta=round(fbeta, 6),
                weighted_f1=metrics["weighted_f1"],
                efficiency_score=metrics["efficiency"],
                f1_by_class=f1_scores.tolist(),
                class_weights=self.class_weights,
                confusion_matrix=confusion_matrix(
                    y_test, y_pred_classes, labels=labels
                ).tolist(),
                risk_category_scores=category_scores,
                score=score,
            )

            bt.logging.trace(f"Result created successfully with score={result.score:.6f}")
            return result

        except Exception as e:
            error_msg = f"Error in get_model_result: {str(e)}"
            bt.logging.error("EXCEPTION in get_model_result!", exc_info=True)
            bt.logging.error(f"Exception details - {error_msg}")
            bt.logging.error(f"y_test available: {'y_test' in locals()}, y_pred available: {'y_pred' in locals()}")
            bt.logging.error(f"model_size_mb: {model_size_mb}, run_time_s: {run_time_s}")
            
            result = TricorderEvaluationResult(
                tested_entries=len(y_test) if "y_test" in locals() else 0,
                run_time_s=run_time_s,
                error=error_msg,
            )
            bt.logging.error(f"Returning error result with score={result.score}")
            return result

    def get_comparable_result_fields(self) -> tuple[str, ...]:
        """Field names for get_comparable_result, in order."""
        return (
            "accuracy",
            "weighted_f1",
            "risk_category_scores",
        )

    def get_comparable_result(self, result: TricorderEvaluationResult) -> tuple:
        """Create a comparable representation of the result for grouping duplicates."""
        if not isinstance(result, TricorderEvaluationResult):
            return tuple()

        
        predictions_array = np.asarray(result.predictions_raw, dtype=float)
        if predictions_array.ndim == 1:
            predictions_array = predictions_array.reshape(-1, 1)

        num_samples = int(predictions_array.shape[0])
        if num_samples == 0:
            return (
                round(result.accuracy, 6),
                round(result.weighted_f1, 6),
            )

        max_signature_samples = 200
        step = max(1, num_samples // max_signature_samples)
        indices = list(range(0, num_samples, step))

        top_classes = np.argmax(predictions_array, axis=1)
        top_probs = predictions_array[np.arange(num_samples), top_classes]

        signature: list[tuple[int, float]] = []
        for idx in indices:
            class_id = int(top_classes[idx])
            prob = round(float(top_probs[idx]), 6)
            signature.append((class_id, prob))

        encoded = json.dumps(signature, separators=(",", ":"), sort_keys=False)
        pred_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        return (pred_hash,)