from pydantic import BaseModel

class FeedbackEntry(BaseModel):
    researcher_res: float
    current_model_res: float
    label: bool
    image_id: str

class Feedback(BaseModel):
    feedback: list[FeedbackEntry]