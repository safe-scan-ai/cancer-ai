from pydantic import BaseModel, parse_obj_as
from typing import List, Dict


class FeedbackEntry(BaseModel):
    researcher_res: float
    current_model_res: float
    label: bool
    image_id: str

class Feedback(BaseModel):
    feedback: list[FeedbackEntry]