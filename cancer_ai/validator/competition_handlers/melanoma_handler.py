from .base_handler import BaseCompetitionHandler
from .base_handler import ModelEvaluationResult

from typing import List
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    fbeta_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# Weights for the competition, for calcualting model score

WEIGHT_FBETA = 0.6
WEIGHT_ACCURACY = 0.3
WEIGHT_AUC = 0.1


class MelanomaCompetitionHandler(BaseCompetitionHandler):
    """Handler for melanoma competition"""

    def __init__(self, X_test, y_test) -> None:
        super().__init__(X_test, y_test)

    def prepare_y_pred(self, y_pred: np.ndarray) -> np.ndarray:
        return [1 if y == "True" else 0 for y in self.y_test]

    def calculate_score(self, fbeta: float, accuracy: float, roc_auc: float) -> float:
        return fbeta * WEIGHT_FBETA + accuracy * WEIGHT_ACCURACY + roc_auc * WEIGHT_AUC

    def get_model_result(
        self, y_test: List[float], y_pred, run_time_s: float
    ) -> ModelEvaluationResult:
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

        return ModelEvaluationResult(
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
        )
