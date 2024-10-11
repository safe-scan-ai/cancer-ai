from typing import List
from pydantic import BaseModel, EmailStr, Field, ValidationError
import yaml

class CompetitionModel(BaseModel):
    competition_id: str
    category: str | None = None
    evaluation_times: List[str]
    dataset_hf_repo: str
    dataset_hf_filename: str
    dataset_hf_repo_type: str


class CompetitionsListModel(BaseModel):
    competitions: List[CompetitionModel]

class DataPackageReference(BaseModel):
    competition_id: str = Field(..., min_length=1, description="Competition identifier")
    dataset_hf_repo: str = Field(..., min_length=1, description="Hugging Face repository path for the dataset")
    dataset_hf_filename: str = Field(..., min_length=1, description="Filename for the dataset in the repository")
    dataset_hf_repo_type: str = Field(..., min_length=1, description="Type of the Hugging Face repository (e.g., dataset)")
    dataset_size: int = Field(..., ge=1, description="Size of the dataset, must be a positive integer")

class OrganizationDataReference(BaseModel):
    organization_id: str = Field(..., min_length=1, description="Unique identifier for the organization")
    contact_email: EmailStr = Field(..., description="Contact email address for the organization")
    bittensor_hotkey: str = Field(..., min_length=1, description="Hotkey associated with the organization")
    data_packages: List[DataPackageReference] = Field(..., description="List of data packages for the organization")
