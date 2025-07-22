from .base_handler import BaseDatasetHandler
from PIL import Image
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
        self.config = config
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.metadata_columns = ["filepath", "label", "age", "gender", "location"]

    @log_time
    async def sync_training_data(self):
        self.entries: List[ImageEntry] = []
        # go over csv file
        async with aiofiles.open(self.label_path, "r") as f:
            reader = csv.reader(await f.readlines())
            next(reader)  # skip first line
            for row in reader:
                # Parse basic fields
                filepath = row[0]
                label = row[1]  # Could be is_melanoma (0/1) or disease_type ("AK", "BCC", etc.)
                
                # Parse optional metadata fields
                age = None
                gender = None
                location = None
                
                if len(row) > 2 and row[2]:  # age column exists and not empty
                    try:
                        age = int(row[2])
                    except ValueError:
                        pass  # Keep as None if invalid
                        
                if len(row) > 3 and row[3]:  # gender column exists and not empty
                    gender = row[3].strip().lower()
                    if gender not in ['male', 'female', 'm', 'f']:
                        gender = None  # Keep as None if invalid
                        
                if len(row) > 4 and row[4]:  # location column exists and not empty
                    location = row[4].strip().lower()
                    # Validate against expected location values
                    valid_locations = ['arm', 'feet', 'genitalia', 'hand', 'head', 'leg', 'torso']
                    if location not in valid_locations:
                        location = None  # Keep as None if invalid
                
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
            Path(self.dataset_path, entry.relative_path).resolve()
            for entry in self.entries
        ]
        pred_y = [entry.label for entry in self.entries]
        pred_metadata = [
            {'age': entry.age, 'gender': entry.gender, 'location': entry.location} 
            for entry in self.entries
        ]
        await self.process_training_data()
        return pred_x, pred_y, pred_metadata
