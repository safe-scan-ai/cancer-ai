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
from ...utils.structured_logger import log



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


    def calculate_efficiency_scores(self, model_sizes_mb: Dict[str, float]) -> Dict[str, float]:
        """Calculate deterministic efficiency scores based on model size.

        This score is based on a linear decay function. Models smaller than
        MIN_MODEL_SIZE_MB get a score of 1.0, and models larger than
        MAX_MODEL_SIZE_MB get a score of 0.0.

        Args:
            model_sizes_mb: Dictionary mapping model_id to model size in MB.

        Returns:
            A dictionary mapping model_id to its efficiency_score (0.0-1.0).
        """
        
        log.debug(f"TRICORDER-3: calculate_efficiency_scores called with {len(model_sizes_mb)} models")
        log.debug(f"TRICORDER-3: Efficiency thresholds - MIN={MIN_MODEL_SIZE_MB}MB, MAX={MAX_MODEL_SIZE_MB}MB, RANGE={EFFICIENCY_RANGE_MB}MB")
        
        efficiency_scores = {}
        
        for model_id in model_sizes_mb:
            # Calculate size score
            size_mb = model_sizes_mb[model_id]
            log.debug(f"TRICORDER-3: Processing {model_id} with size={size_mb:.2f}MB")
            
            if size_mb <= MIN_MODEL_SIZE_MB:
                size_score = 1.0
                log.debug(f"TRICORDER-3: {model_id} size <= MIN, size_score=1.0")
            elif size_mb <= MAX_MODEL_SIZE_MB:
                size_score = (MAX_MODEL_SIZE_MB - size_mb) / EFFICIENCY_RANGE_MB
                log.debug(f"TRICORDER-3: {model_id} size in range, size_score={size_score:.6f} = ({MAX_MODEL_SIZE_MB} - {size_mb}) / {EFFICIENCY_RANGE_MB}")
            else:
                size_score = 0.0
                log.debug(f"TRICORDER-3: {model_id} size > MAX, size_score=0.0")
            
            # Deterministic efficiency score based only on size
            efficiency_score = size_score
            efficiency_scores[model_id] = efficiency_score
            
            log.info(
                f"TRICORDER-3: Efficiency for {model_id}: size={size_mb:.2f}MB, score={efficiency_score:.6f}"
            )
        
        log.debug(f"TRICORDER-3: Calculated efficiency scores for {len(efficiency_scores)} models")
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
            log.competition.error("Results and model_ids length mismatch")
            return results
        
        # Calculate efficiency scores
        # Calculate deterministic efficiency scores based on size
        efficiency_scores = self.calculate_efficiency_scores(model_sizes_mb)
        
        # Update results
        updated_results = []
        
        for result, model_id in zip(results, model_ids):
            try:
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
                
                log.debug(f"Recalculating score for {model_id} with metrics: {metrics}")
                updated_result.score = self.calculate_score(metrics)
                
                updated_results.append(updated_result)
                
                log.info(
                    f"Updated {model_id}: efficiency={updated_result.efficiency_score:.3f}, "
                    f"final_score={updated_result.score:.3f}"
                )
            except Exception as e:
                log.competition.error(f"Error updating result for {model_id}: {e}", exc_info=True)
                # Keep original result if update fails
                updated_results.append(result)
        
        return updated_results
