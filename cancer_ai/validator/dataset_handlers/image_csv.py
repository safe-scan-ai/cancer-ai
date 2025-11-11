from .base_handler import BaseDatasetHandler
from typing import List, Tuple
from dataclasses import dataclass
import csv
import aiofiles
from pathlib import Path

from ..utils import log_time


@dataclass
class ImageEntry:
    relative_path: str
    label: str | int | bool  # Generic label - could be is_melanoma (bool) or disease_type (str)
    age: int | None = None
    gender: str | None = None
    location: str | None = None


class DatasetImagesCSV(BaseDatasetHandler):
    """
    DatasetImagesCSV is responsible for handling the CSV dataset where directory structure looks as follows:

    ├── images
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── labels.csv
    """

    def __init__(self, config, dataset_path, label_path: str) -> None:
        super().__init__(config, dataset_path)
        self.label_path = label_path
        self.metadata_columns = ["filepath", "label", "age", "location", "gender"]

    @log_time
    async def sync_training_data(self):
        self.entries: List[ImageEntry] = []
        # go over csv file
        async with aiofiles.open(self.label_path, "r") as f:
            content = await f.read()
            
        # Parse CSV with DictReader for column-agnostic access
        import io
        reader = csv.DictReader(io.StringIO(content))
        
        for row in reader:
            # Convert all keys to lowercase for case-insensitive matching
            row_lower = {k.lower(): v for k, v in row.items()}
            
            # Get filepath - support different column names
            filepath = row_lower.get('newfilename') or row_lower.get('filepath') or row_lower.get('filename') or ''
            
            # Get label - support different column names
            label = row_lower.get('class') or row_lower.get('label') or ''
            
            # Parse age
            age = None
            age_str = row_lower.get('age') or ''
            if age_str:
                try:
                    age = int(age_str)
                except ValueError:
                    pass  # Keep as None if invalid
            
            # Parse location
            location = None
            location_str = row_lower.get('location') or ''
            if location_str:
                location = location_str.strip().lower()
                # Validate against expected location values
                valid_locations = ['arm', 'feet', 'genitalia', 'hand', 'head', 'leg', 'torso']
                if location not in valid_locations:
                    location = None  # Keep as None if invalid
            
            # Parse gender
            gender = None
            gender_str = row_lower.get('gender') or ''
            if gender_str:
                gender = gender_str.strip().lower()
                # Keep the raw value - validation will happen in tricorder handler
            
            self.entries.append(ImageEntry(
                relative_path=filepath,
                label=label,
                age=age,
                gender=gender,
                location=location
            ))

    @log_time
    async def get_training_data(self) -> Tuple[List, List, List]:
        """
        Get the training data.

        This method is responsible for loading the training data and returning a tuple containing three lists: 
        the first list contains paths to the images, the second list contains the labels, 
        and the third list contains patient metadata (age, gender).
        """
        await self.sync_training_data()
        pred_x = [
            Path(self.path, entry.relative_path).resolve()
            for entry in self.entries
        ]
        pred_y = [entry.label for entry in self.entries]
        pred_metadata = [
            {'age': entry.age, 'gender': entry.gender, 'location': entry.location} 
            for entry in self.entries
        ]
        await self.process_training_data()
        return pred_x, pred_y, pred_metadata
