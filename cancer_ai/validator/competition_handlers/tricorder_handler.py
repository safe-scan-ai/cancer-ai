from typing import List, Dict, Any, AsyncGenerator, Tuple, Optional, TypedDict, Literal
from pydantic import Field
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

MAX_INVALID_ENTRIES = 2  # Maximum number of invalid entries allowed in the dataset

# --- Constants ---
TARGET_SIZE = (512, 512)
CHUNK_SIZE = 200

# Image preprocessing constants
NORMALIZATION_FACTOR = 255.0

# Risk category weights for scoring
CATEGORY_WEIGHTS = {
    'HIGH_RISK': 3.0,
    'MEDIUM_RISK': 2.0,
    'BENIGN': 1.0
}

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

class LocationId(IntEnum):
    ARM = 1
    FEET = 2
    GENITALIA = 3
    HAND = 4
    HEAD = 5
    LEG = 6
    TORSO = 7

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
        "short_name": "NV",
        "risk_category": RiskCategory.BENIGN,
        "weight": 1.0
    },
    ClassId.OTHER_NON_NEOPLASTIC: {
        "name": "Other Non-Neoplastic",
        "short_name": "NON",
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

from cancer_ai.validator.models import WanDBLogModelBase

class TricorderWanDBLogModelEntry(WanDBLogModelBase):
    tested_entries: int
    model_url: str
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
    f1_by_class: List[float] = Field(default_factory=lambda: [0.0] * len(CLASS_INFO))
    class_weights: List[float] = Field(default_factory=lambda: [info["weight"] for info in CLASS_INFO.values()])
    confusion_matrix: List[List[int]] = Field(default_factory=lambda: [[0] * len(CLASS_INFO) for _ in range(len(CLASS_INFO))])
    risk_category_scores: Dict[RiskCategory, float] = Field(default_factory=lambda: {category: 0.0 for category in RiskCategory})

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

class TricorderCompetitionHandler(BaseCompetitionHandler):
    WanDBLogModelClass = TricorderWanDBLogModelEntry

    """Handler for skin lesion classification competition with 10 classes.
    
    This handler manages the entire competition pipeline including:
    - Data preprocessing and serialization
    - Model evaluation
    - Scoring based on competition rules
    """

    def __init__(self, X_test: List[str], y_test: List[int], metadata: Optional[List[Dict[str, Any]]] = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(X_test, y_test)
        self.config = config or {}
        self.metadata = metadata or [{'age': None, 'gender': None, 'location': None} for _ in X_test]
        self.preprocessed_data_dir = None
        self.preprocessed_chunks = []
        self.valid_indices = []  # Track indices of successfully processed entries
        self.y_test_filtered = []  # Filtered test labels matching valid entries
        
        validation_errors = []
        
        # Only validate metadata if it was actually provided (not default None values)
        metadata_provided = metadata is not None and len(metadata) > 0
        if metadata_provided:
            # Check if first entry has non-None values to determine if real metadata was provided
            first_meta = self.metadata[0] if self.metadata else {}
            has_real_metadata = any(v is not None for v in first_meta.values())
            
            if has_real_metadata:
                for i, meta_entry in enumerate(self.metadata):
                    # Validate age
                    age = meta_entry.get('age')
                    if age is None:
                        validation_errors.append(f"Missing age at index {i}")
                    elif not isinstance(age, (int, float)) or age < 0 or age > MAX_AGE:
                        validation_errors.append(f"Invalid age at index {i}: {age} (must be 0-120)")
                    
                    # Validate gender
                    gender = meta_entry.get('gender')
                    if gender is None:
                        validation_errors.append(f"Missing gender at index {i}")
                    else:
                        gender_lower = str(gender).lower()
                        if gender_lower not in ['m', 'f', 'male', 'female']:
                            validation_errors.append(f"Invalid gender at index {i}: {gender} (must be 'm', 'f', 'male', 'female')")
                        else:
                            meta_entry['gender'] = gender_lower
                    
                    # Validate location
                    location = meta_entry.get('location')
                    if location is None:
                        validation_errors.append(f"Missing location at index {i}")
                    else:
                        location_lower = str(location).lower()
                        valid_locations = ['arm', 'feet', 'genitalia', 'hand', 'head', 'leg', 'torso']
                        if location_lower not in valid_locations:
                            validation_errors.append(f"Invalid location at index {i}: {location} (must be one of {valid_locations})")
                        else:
                            meta_entry['location'] = location_lower
            else:
                bt.logging.info("No metadata provided for tricorder competition - using defaults")
        
        # Validate labels
        valid_label_names = [info["short_name"] for info in CLASS_INFO.values()]
        for i, label in enumerate(y_test):
            if isinstance(label, str):
                if label not in valid_label_names:
                    validation_errors.append(f"Invalid label at index {i}: {label} (must be one of {valid_label_names})")
            elif isinstance(label, int):
                if label < 1 or label > len(CLASS_INFO):
                    validation_errors.append(f"Invalid label at index {i}: {label} (must be 1-{len(CLASS_INFO)})")
            else:
                validation_errors.append(f"Invalid label type at index {i}: {type(label)} (must be string or int)")
        
        # If any validation errors, log them but continue with the competition
        # These are metadata/label validation errors, not image loading errors
        if validation_errors:
            error_summary = "\n".join(validation_errors[:10])
            if len(validation_errors) > 10:
                error_summary += f"\n... and {len(validation_errors) - 10} more errors"
            
            bt.logging.warning(f"TRICORDER COMPETITION WARNING: Dataset validation has metadata/label issues")
            bt.logging.warning(f"Found {len(validation_errors)} validation errors:")
            bt.logging.warning(error_summary)
            bt.logging.info("Competition will continue - these validation issues don't prevent evaluation")
        
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
        Preprocess all images with metadata and serialize them to disk in chunks.
        Returns list of paths to serialized chunk files.
        """
        if not self.preprocessed_data_dir:
            raise ValueError("Preprocessed data directory not set")
            
        bt.logging.debug(f"TRICORDER: Preprocessing {len(X_test)} images for tricorder competition")
        bt.logging.debug(f"TRICORDER: Using chunk size: {CHUNK_SIZE}")
        bt.logging.debug(f"TRICORDER: Available metadata entries: {len(self.metadata)}")
        error_counter = defaultdict(int)
        chunk_paths = []
        self.valid_indices = []  # Track indices of successfully processed entries
        
        for i in range(0, len(X_test), CHUNK_SIZE):
            bt.logging.debug(f"TRICORDER: Processing chunk {len(chunk_paths)} - images {i} to {min(i + CHUNK_SIZE, len(X_test))}")
            chunk_data = []
            chunk_metadata = []
            chunk_valid_indices = []
            
            for idx, img_path in enumerate(X_test[i: i + CHUNK_SIZE]):
                global_idx = i + idx
                try:
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(f"File does not exist: {img_path}")

                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        preprocessed_img = self._preprocess_single_image(img)
                        chunk_data.append(preprocessed_img)
                        
                        # Add corresponding metadata
                        if global_idx < len(self.metadata):
                            chunk_metadata.append(self.metadata[global_idx])
                        else:
                            chunk_metadata.append({'age': None, 'gender': None, 'location': None})
                        
                        # Track valid indices for later filtering
                        chunk_valid_indices.append(global_idx)
                        self.valid_indices.append(global_idx)
                        
                except FileNotFoundError:
                    error_counter['FileNotFoundError'] += 1
                    bt.logging.warning(f"File not found: {img_path} (index {global_idx})")
                    continue
                except IOError as e:
                    error_counter['IOError'] += 1
                    bt.logging.warning(f"IO error processing {img_path} (index {global_idx}): {e}")
                    continue
                except Exception as e:
                    bt.logging.warning(f"Unexpected error processing {img_path} (index {global_idx}): {e}")
                    error_counter['UnexpectedError'] += 1
                    continue

            if chunk_data:
                try:
                    chunk_array = np.array(chunk_data, dtype=np.float32)
                    chunk_file = self.preprocessed_data_dir / f"chunk_{len(chunk_paths)}.pkl"
                    metadata_file = self.preprocessed_data_dir / f"metadata_{len(chunk_paths)}.pkl"
                    
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_array, f)
                    
                    with open(metadata_file, 'wb') as f:
                        pickle.dump(chunk_metadata, f)
                    
                    chunk_paths.append(str(chunk_file))
                    bt.logging.debug(f"TRICORDER: Saved chunk with {len(chunk_data)} images and metadata to {chunk_file}")
                    
                except Exception as e:
                    bt.logging.error(f"TRICORDER: Failed to serialize chunk: {e}")
                    error_counter['SerializationError'] += 1

        # Check if we have too many invalid entries
        total_errors = sum(error_counter.values())
        valid_entries = len(self.valid_indices)
        total_entries = len(X_test)
        
        if total_errors > 0:
            error_summary = "; ".join([f"{count} {error_type.replace('_', ' ')}" 
                                     for error_type, count in error_counter.items()])
            bt.logging.warning(f"TRICORDER: Preprocessing completed with issues: {error_summary}")
            bt.logging.warning(f"TRICORDER: {total_errors}/{total_entries} entries failed to process")
            
            if total_errors > MAX_INVALID_ENTRIES:
                bt.logging.error(f"TRICORDER COMPETITION CANCELLED: Too many invalid entries")
                bt.logging.error(f"Found {total_errors} invalid entries, maximum allowed: {MAX_INVALID_ENTRIES}")
                raise ValueError(f"Too many invalid entries ({total_errors}), maximum allowed: {MAX_INVALID_ENTRIES}")
        
        # Filter y_test to match valid indices
        self.y_test_filtered = [self.y_test[i] for i in self.valid_indices]
        
        bt.logging.info(f"TRICORDER: Successfully processed {valid_entries}/{total_entries} entries")
        bt.logging.debug(f"TRICORDER: Preprocessed data saved in {len(chunk_paths)} chunks")
        bt.logging.debug(f"TRICORDER: Chunk paths: {chunk_paths}")
        self.preprocessed_chunks = chunk_paths
        return chunk_paths

    def _preprocess_single_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess a single PIL image for tricorder competition"""
        # Resize to target size
        img = img.resize(TARGET_SIZE)
        
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

    async def get_preprocessed_data_generator(self) -> AsyncGenerator[Tuple[np.ndarray, List[Dict[str, Any]]], None]:
        """Generator that yields preprocessed data chunks with metadata"""
        bt.logging.debug(f"TRICORDER: Starting data generator with {len(self.preprocessed_chunks)} chunks")
        
        for i, chunk_file in enumerate(self.preprocessed_chunks):
            bt.logging.debug(f"TRICORDER: Processing chunk {i}: {chunk_file}")
            if os.path.exists(chunk_file):
                try:
                    # Load image data
                    bt.logging.debug(f"TRICORDER: Loading image data from {chunk_file}")
                    with open(chunk_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    bt.logging.debug(f"TRICORDER: Loaded chunk data shape: {chunk_data.shape}")
                    
                    # Load corresponding metadata
                    metadata_file = str(Path(chunk_file).parent / f"metadata_{i}.pkl")
                    bt.logging.debug(f"TRICORDER: Loading metadata from {metadata_file}")
                    chunk_metadata = []
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'rb') as f:
                            chunk_metadata = pickle.load(f)
                        bt.logging.debug(f"TRICORDER: Loaded {len(chunk_metadata)} metadata entries")
                    else:
                        # Default metadata if file doesn't exist
                        bt.logging.warning(f"TRICORDER: Metadata file not found, using defaults")
                        chunk_metadata = [{'age': None, 'gender': None, 'location': None} for _ in range(len(chunk_data))]
                    
                    bt.logging.debug(f"TRICORDER: Yielding chunk {i} with {len(chunk_data)} samples and {len(chunk_metadata)} metadata")
                    yield chunk_data, chunk_metadata
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
        
    def prepare_y_pred(self, y_pred):
        """
        Convert string labels to 0-based indices for evaluation.
        
        Args:
            y_pred: List of prediction labels (either string short names or 1-based indices)
            
        Returns:
            List of 0-based class indices
        """
        converted = []
        for y in y_pred:
            if isinstance(y, str):
                # Find class ID by short name
                class_id = next((cid for cid, info in CLASS_INFO.items() 
                              if info["short_name"] == y), None)
                if class_id is not None:
                    converted.append(class_id - 1)  # Convert to 0-based
                else:
                    raise ValueError(f"Unknown class short name: {y}")
            elif isinstance(y, (int, float)):
                converted.append(int(y) - 1)  # Convert to 0-based if numeric
            else:
                raise ValueError(f"Invalid label type: {type(y).__name__}")
        return converted

    def _calculate_risk_category_scores(self, f1_scores: np.ndarray) -> Dict[RiskCategory, float]:
        """Calculate F1 scores for each risk category based on pre-computed F1 scores per class."""
        category_scores = {}
        
        for category, class_indices in RISK_CATEGORIES.items():
            if class_indices:
                category_f1 = np.mean([f1_scores[i] for i in class_indices])
                category_scores[category] = float(category_f1)
            else:
                category_scores[category] = 0.0
                
        return category_scores

    def _calculate_weighted_f1(self, category_scores: Dict[RiskCategory, float]) -> float:
        """Calculate weighted F1 score based on risk categories."""
        # Use category-level weights from constants
        category_weights = {
            RiskCategory.HIGH_RISK: CATEGORY_WEIGHTS['HIGH_RISK'],
            RiskCategory.MEDIUM_RISK: CATEGORY_WEIGHTS['MEDIUM_RISK'],
            RiskCategory.BENIGN: CATEGORY_WEIGHTS['BENIGN']
        }
        
        total_weight = sum(category_weights.values())
        weighted_sum = sum(
            category_scores.get(category, 0.0) * weight
            for category, weight in category_weights.items()
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate final competition score (0-1)."""
        # Prediction quality (accuracy + weighted F1)
        prediction_score = ACCURACY_WEIGHT * metrics['accuracy'] + WEIGHTED_F1_WEIGHT * metrics['weighted_f1']
        
        # Efficiency score
        efficiency_score = metrics.get('efficiency', 1.0)  # Default to max if not set
        
        final_score = PREDICTION_WEIGHT * prediction_score + EFFICIENCY_WEIGHT * efficiency_score
        return final_score

    def get_model_result(self, y_test: List[int], y_pred: List[float], run_time_s: float, model_size_mb: float = None) -> TricorderEvaluationResult:
        """
        Evaluate model predictions and return detailed results.
        
        Args:
            y_test: List of true class indices (0-9) - will be filtered to match valid entries
            y_pred: List of predicted probabilities (shape [n_samples, 10])
            run_time_s: Inference time in seconds
            model_size_mb: Model size in MB for efficiency calculation
            
        Returns:
            TricorderEvaluationResult with comprehensive evaluation metrics
        """
        try:
            # Use filtered test labels if available (after preprocessing), otherwise use provided y_test
            if hasattr(self, 'y_test_filtered') and self.y_test_filtered:
                y_test_to_use = self.y_test_filtered
                bt.logging.info(f"Using filtered test labels: {len(y_test_to_use)} entries")
            else:
                y_test_to_use = y_test
                bt.logging.warning("No filtered test labels available, using original y_test")
            
            # Convert to numpy arrays
            y_test = np.array(y_test_to_use)
            y_pred = np.array(y_pred)
            
            # Validate array shapes match
            if len(y_test) != len(y_pred):
                bt.logging.error(f"Array length mismatch: y_test={len(y_test)}, y_pred={len(y_pred)}")
                raise ValueError(f"Array length mismatch: y_test has {len(y_test)} samples, y_pred has {len(y_pred)} samples")

            # Define all possible class labels (0 to 9)
            labels = list(range(len(CLASS_INFO)))
            
            # Get predicted class indices
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate basic metrics
            accuracy = float(accuracy_score(y_test, y_pred_classes))
            precision = float(precision_score(y_test, y_pred_classes, labels=labels, average='weighted', zero_division=0))
            recall = float(recall_score(y_test, y_pred_classes, labels=labels, average='weighted', zero_division=0))
            fbeta = float(f1_score(y_test, y_pred_classes, labels=labels, average='weighted', zero_division=0))
            
            # Calculate F1 scores by class, ensuring all classes are included
            f1_scores = f1_score(y_test, y_pred_classes, labels=labels, average=None, zero_division=0)
            
            # Calculate risk category scores and weighted F1
            category_scores = self._calculate_risk_category_scores(f1_scores)
            weighted_f1 = self._calculate_weighted_f1(category_scores)

            # Log important metrics
            bt.logging.info(f"Model evaluation results:")
            bt.logging.info(f"- Accuracy: {accuracy:.4f}")
            bt.logging.info(f"- Weighted F1: {weighted_f1:.4f}")
            for category, score in category_scores.items():
                bt.logging.info(f"- {category.value} F1: {score:.4f}")

            # Calculate efficiency score based on model size
            efficiency_score = 1.0  # Default to max if size not provided
            if model_size_mb is not None:
                if model_size_mb <= MIN_MODEL_SIZE_MB:
                    efficiency_score = 1.0  # Full efficiency score
                elif model_size_mb <= MAX_MODEL_SIZE_MB:
                    # Linear decay from 1.0 to 0.0 between MIN and MAX MB
                    efficiency_score = (MAX_MODEL_SIZE_MB - model_size_mb) / EFFICIENCY_RANGE_MB
                else:
                    efficiency_score = 0.0  # No efficiency score above MAX MB
                
                bt.logging.info(f"- Model size: {model_size_mb:.1f}MB, Efficiency score: {efficiency_score:.2f}")
            
            # Calculate final score using calculate_score method
            metrics = {
                'accuracy': accuracy,
                'weighted_f1': weighted_f1,
                'efficiency': efficiency_score
            }
            score = self.calculate_score(metrics)
            # Create result object
            result = TricorderEvaluationResult(
                tested_entries=len(y_test),
                run_time_s=run_time_s,
                predictions_raw=y_pred.tolist(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                fbeta=fbeta,
                weighted_f1=weighted_f1,
                efficiency_score=efficiency_score,
                f1_by_class=f1_scores.tolist(),
                class_weights=self.class_weights,
                confusion_matrix=confusion_matrix(y_test, y_pred_classes, labels=labels).tolist(),
                risk_category_scores=category_scores,
                score=score
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error in get_model_result: {str(e)}"
            bt.logging.error(error_msg, exc_info=True)
            return TricorderEvaluationResult(
                tested_entries=len(y_test) if 'y_test' in locals() else 0,
                run_time_s=run_time_s,
                error=error_msg
            )

    def get_comparable_result_fields(self) -> tuple[str, ...]:
        """Field names for get_comparable_result, in order."""
        return (
            "accuracy",
            "weighted_f1",
            "risk_category_scores",
        )

    def get_comparable_result(self, result: TricorderEvaluationResult) -> tuple:
        """
        Create a comparable representation of the result for grouping duplicates.
        
        This method should be implemented by each competition handler to specify
        which metrics are used for comparing results.
        
        Args:
            result: The evaluation result object.
            
        Returns:
            A tuple of key metrics that can be used for comparison.
        """
        if not isinstance(result, TricorderEvaluationResult):
            return tuple()
            
        # Round floats to handle potential floating point inaccuracies
        return (
            round(result.accuracy, 6),
            round(result.weighted_f1, 6),
            # Sort risk category scores by key to ensure consistent order
            tuple(sorted((k.value, round(v, 6)) for k, v in result.risk_category_scores.items())),
        )