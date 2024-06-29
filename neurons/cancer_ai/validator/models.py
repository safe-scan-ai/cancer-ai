from pydantic import BaseModel, parse_obj_as
from typing import List, Dict

class DatasetEntry(BaseModel):
    id: str
    label: Dict[str, str]
    image_url: str
    
class DatasetEntries(BaseModel):
    entries: List[DatasetEntry]
