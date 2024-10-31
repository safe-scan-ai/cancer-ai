from typing import List, ClassVar, Optional, ClassVar, Optional
from pydantic import BaseModel, EmailStr, Field, ValidationError
from datetime import datetime, EmailStr, Field, ValidationError
from datetime import datetime

class CompetitionModel(BaseModel):
    competition_id: str
    category: str | None = None
    evaluation_times: List[str]
    dataset_hf_repo: str
    dataset_hf_filename: str
    dataset_hf_repo_type: str


class CompetitionsListModel(BaseModel):
    competitions: List[CompetitionModel]

class DatasetReference(BaseModel):
    competition_id: str = Field(..., min_length=1, description="Competition identifier")
    dataset_hf_repo: str = Field(..., min_length=1, description="Hugging Face repository path for the dataset")
    dataset_hf_filename: str = Field(..., min_length=1, description="Filename for the dataset in the repository")
    dataset_hf_repo_type: str = Field(..., min_length=1, description="Type of the Hugging Face repository (e.g., dataset)")
    dataset_size: int = Field(..., ge=1, description="Size of the dataset, must be a positive integer")

class OrganizationDataReference(BaseModel):
    organization_id: str = Field(..., min_length=1, description="Unique identifier for the organization")
    contact_email: EmailStr = Field(..., description="Contact email address for the organization")
    bittensor_hotkey: str = Field(..., min_length=1, description="Hotkey associated with the organization")
    data_packages: List[DatasetReference] = Field(..., description="List of data packages for the organization")
    date_uploaded: datetime = Field(..., description="Date the organization data was uploaded")

class OrganizationDataReferenceFactory(BaseModel):
    organizations: List[OrganizationDataReference] = Field(default_factory=list)
    _instance: ClassVar[Optional["OrganizationDataReferenceFactory"]] = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_organizations(self, organizations: List[OrganizationDataReference]):
        self.organizations.extend(organizations)
    
    def update_from_dict(self, data: dict):
        """Updates the singleton instance's state from a dictionary."""
        if "organizations" in data:
            # Convert each dict in 'organizations' to an OrganizationDataReference instance
            self.organizations = [OrganizationDataReference(**org) for org in data["organizations"]]
        for key, value in data.items():
            if key != "organizations":
                setattr(self, key, value)