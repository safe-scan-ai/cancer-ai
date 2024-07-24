from pydantic import BaseModel, parse_obj_as
from typing import List, Dict, Any


class DatasetEntry(BaseModel):
    id: str
    label: Dict[str, Any]
    image_url: str
    current_model_response: float


class DatasetEntries(BaseModel):
    entries: List[DatasetEntry]


class ResearcherEntry(BaseModel):
    prediction: float
    current_model_prediction: float
    is_melanoma: bool
    image_id: str

class ResearcherScores(BaseModel):
    entries: list[ResearcherEntry]
    researcher_score: int
    current_model_score: int
    num_entries: int
    testing_session_id: str