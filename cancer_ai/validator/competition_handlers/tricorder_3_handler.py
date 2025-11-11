from enum import IntEnum
from typing import Dict, Any, List

from .tricorder_common import (
    BaseTricorderCompetitionHandler, 
    RiskCategory, 
    MIN_MODEL_SIZE_MB, 
    MAX_MODEL_SIZE_MB, 
    EFFICIENCY_RANGE_MB,
    TricorderEvaluationResult
)

# Speed scoring constants (50% of efficiency score)
SPEED_WEIGHT = 0.5
SIZE_WEIGHT = 0.5


# 1-based class IDs for better readability
class ClassId(IntEnum):
    ACTINIC_KERATOSIS_INTRAEPITHELIAL_CARCINOMA = 1
    BASAL_CELL_CARCINOMA = 2
    OTHER_BENIGN_PROLIFERATIONS = 3
    BENIGN_KERATINOCYTIC_LESION = 4
    DERMATOFIBROMA = 5
    INFLAMMATORY_INFECTIOUS_CONDITIONS = 6
    OTHER_MALIGNANT_PROLIFERATIONS = 7
    MELANOMA = 8
    MELANOCYTIC_NEVUS = 9
    SQUAMOUS_CELL_CARCINOMA_KERATOACANTHOMA = 10
    VASCULAR_LESIONS_HEMORRHAGE = 11


class Tricorder3CompetitionHandler(BaseTricorderCompetitionHandler):
    """Handler for Tricorder-3 skin lesion classification competition with 11 classes."""

    def __init__(self, X_test: List[str], y_test: List[int], metadata: List[Dict[str, Any]] = None, config: Dict[str, Any] = None) -> None:
        super().__init__(X_test, y_test, metadata, config)
        
        # Speed tracking for efficiency scoring
        self.speed_results: List[Dict[str, float]] = []

    def get_class_info(self) -> Dict[ClassId, Dict[str, Any]]:
        """Return the class information dictionary for Tricorder-3."""
        return {
            ClassId.ACTINIC_KERATOSIS_INTRAEPITHELIAL_CARCINOMA: {
                "name": "Actinic keratosis/intraepidermal carcinoma",
                "short_name": "AKIEC",
                "risk_category": RiskCategory.MEDIUM_RISK,
                "weight": 2.0,
            },
            ClassId.BASAL_CELL_CARCINOMA: {
                "name": "Basal cell carcinoma",
                "short_name": "BCC",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.OTHER_BENIGN_PROLIFERATIONS: {
                "name": "Other benign proliferations including collisions",
                "short_name": "BEN_OTH",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.BENIGN_KERATINOCYTIC_LESION: {
                "name": "Benign keratinocytic lesion",
                "short_name": "BKL",
                "risk_category": RiskCategory.MEDIUM_RISK,
                "weight": 2.0,
            },
            ClassId.DERMATOFIBROMA: {
                "name": "Dermatofibroma",
                "short_name": "DF",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.INFLAMMATORY_INFECTIOUS_CONDITIONS: {
                "name": "Inflammatory and infectious",
                "short_name": "INF",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.OTHER_MALIGNANT_PROLIFERATIONS: {
                "name": "Other malignant proliferations including collisions",
                "short_name": "MAL_OTH",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.MELANOMA: {
                "name": "Melanoma",
                "short_name": "MEL",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.MELANOCYTIC_NEVUS: {
                "name": "Melanocytic Nevus, any type",
                "short_name": "NV",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.SQUAMOUS_CELL_CARCINOMA_KERATOACANTHOMA: {
                "name": "Squamous cell carcinoma/keratoacanthoma",
                "short_name": "SCCKA",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.VASCULAR_LESIONS_HEMORRHAGE: {
                "name": "Vascular lesions and hemorrhage",
                "short_name": "VASC",
                "risk_category": RiskCategory.MEDIUM_RISK,
                "weight": 2.0,
            },
        }

    def record_speed_result(self, model_id: str, inference_time_ms: float) -> None:
        """Record inference speed result for a model.
        
        Args:
            model_id: Unique identifier for the model
            inference_time_ms: Inference time in milliseconds for single image
        """
        self.speed_results.append({
            "model_id": model_id,
            "inference_time_ms": inference_time_ms
        })
        import bittensor as bt
        bt.logging.debug(f"Recorded speed for {model_id}: {inference_time_ms:.2f}ms")

    def calculate_speed_scores(self) -> Dict[str, float]:
        """Calculate speed scores for all recorded models based on relative performance.
        
        Returns:
            Dictionary mapping model_id to speed_score (0.0-1.0)
            Faster models get higher scores
        """
        if not self.speed_results:
            return {}
        
        # Extract inference times
        times = [result["inference_time_ms"] for result in self.speed_results]
        min_time = min(times)
        max_time = max(times)
        
        # Calculate speed scores (faster = higher score)
        speed_scores = {}
        for result in self.speed_results:
            model_id = result["model_id"]
            time_ms = result["inference_time_ms"]
            
            if max_time == min_time:
                # All models have same speed
                speed_score = 1.0
            else:
                # Linear scaling: fastest gets 1.0, slowest gets 0.0
                speed_score = (max_time - time_ms) / (max_time - min_time)
            
            speed_scores[model_id] = max(0.0, min(1.0, speed_score))
            import bittensor as bt
            bt.logging.debug(f"Speed score for {model_id}: {speed_score:.3f} (time: {time_ms:.2f}ms)")
        
        return speed_scores

    def calculate_efficiency_scores(self, model_sizes_mb: Dict[str, float]) -> Dict[str, float]:
        """Calculate combined efficiency scores (50% size + 50% speed).
        
        Args:
            model_sizes_mb: Dictionary mapping model_id to model size in MB
            
        Returns:
            Dictionary mapping model_id to efficiency_score (0.0-1.0)
        """
        speed_scores = self.calculate_speed_scores()
        efficiency_scores = {}
        
        for model_id in model_sizes_mb:
            # Calculate size score
            size_mb = model_sizes_mb[model_id]
            if size_mb <= MIN_MODEL_SIZE_MB:
                size_score = 1.0
            elif size_mb <= MAX_MODEL_SIZE_MB:
                size_score = (MAX_MODEL_SIZE_MB - size_mb) / EFFICIENCY_RANGE_MB
            else:
                size_score = 0.0
            
            # Get speed score (0.0 if not recorded)
            speed_score = speed_scores.get(model_id, 0.0)
            
            # Combined efficiency score (50% size + 50% speed)
            efficiency_score = SIZE_WEIGHT * size_score + SPEED_WEIGHT * speed_score
            efficiency_scores[model_id] = efficiency_score
            
            import bittensor as bt
            bt.logging.debug(
                f"Efficiency for {model_id}: size={size_score:.3f}, speed={speed_score:.3f}, "
                f"combined={efficiency_score:.3f}"
            )
        
        return efficiency_scores

    def update_results_with_efficiency(
        self, 
        results: List[TricorderEvaluationResult],
        model_ids: List[str],
        model_sizes_mb: Dict[str, float]
    ) -> List[TricorderEvaluationResult]:
        """Update evaluation results with efficiency scores and recalculate final scores.
        
        Args:
            results: List of evaluation results to update
            model_ids: List of model IDs corresponding to results
            model_sizes_mb: Dictionary mapping model_id to model size in MB
            
        Returns:
            Updated list of evaluation results
        """
        if len(results) != len(model_ids):
            import bittensor as bt
            bt.logging.error("Results and model_ids length mismatch")
            return results
        
        # Calculate efficiency scores
        efficiency_scores = self.calculate_efficiency_scores(model_sizes_mb)
        
        # Update results
        updated_results = []
        for result, model_id in zip(results, model_ids):
            # Create a copy to avoid modifying original
            updated_result = TricorderEvaluationResult(**result.dict())
            
            # Update efficiency score
            updated_result.efficiency_score = efficiency_scores.get(model_id, 0.0)
            
            # Recalculate final score
            metrics = {
                "accuracy": updated_result.accuracy,
                "weighted_f1": updated_result.weighted_f1,
                "efficiency": updated_result.efficiency_score,
            }
            updated_result.score = self.calculate_score(metrics)
            
            updated_results.append(updated_result)
            
            import bittensor as bt
            bt.logging.info(
                f"Updated {model_id}: efficiency={updated_result.efficiency_score:.3f}, "
                f"final_score={updated_result.score:.3f}"
            )
        
        return updated_results
