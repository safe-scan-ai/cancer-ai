from enum import IntEnum
from typing import Dict, Any

from .tricorder_common import BaseTricorderCompetitionHandler, RiskCategory


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


class Tricorder2CompetitionHandler(BaseTricorderCompetitionHandler):
    """Handler for Tricorder-2 skin lesion classification competition with 10 classes."""

    def get_class_info(self) -> Dict[ClassId, Dict[str, Any]]:
        """Return the class information dictionary for Tricorder-2."""
        return {
            ClassId.ACTINIC_KERATOSIS: {
                "name": "Actinic Keratosis (AK)",
                "short_name": "AK",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.BASAL_CELL_CARCINOMA: {
                "name": "Basal Cell Carcinoma (BCC)",
                "short_name": "BCC",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.SEBORRHEIC_KERATOSIS: {
                "name": "Seborrheic Keratosis (SK)",
                "short_name": "SK",
                "risk_category": RiskCategory.MEDIUM_RISK,
                "weight": 2.0,
            },
            ClassId.SQUAMOUS_CELL_CARCINOMA: {
                "name": "Squamous Cell Carcinoma (SCC)",
                "short_name": "SCC",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.VASCULAR_LESION: {
                "name": "Vascular Lesion",
                "short_name": "VASC",
                "risk_category": RiskCategory.MEDIUM_RISK,
                "weight": 2.0,
            },
            ClassId.DERMATOFIBROMA: {
                "name": "Dermatofibroma",
                "short_name": "DF",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.BENIGN_NEVUS: {
                "name": "Benign Nevus",
                "short_name": "NV",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.OTHER_NON_NEOPLASTIC: {
                "name": "Other Non-Neoplastic",
                "short_name": "NON",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
            ClassId.MELANOMA: {
                "name": "Melanoma",
                "short_name": "MEL",
                "risk_category": RiskCategory.HIGH_RISK,
                "weight": 3.0,
            },
            ClassId.OTHER_NEOPLASTIC: {
                "name": "Other Neoplastic",
                "short_name": "ON",
                "risk_category": RiskCategory.BENIGN,
                "weight": 1.0,
            },
        }
