from typing import List, ClassVar, Optional, ClassVar, Optional
from pydantic import BaseModel, EmailStr, Field, ValidationError
from datetime import datetime
from dataclasses import dataclass

class CompetitionModel(BaseModel):
    competition_id: str
    category: str | None = None
    evaluation_times: List[str]
    dataset_hf_repo: str
    dataset_hf_filename: str
    dataset_hf_repo_type: str


class CompetitionsListModel(BaseModel):
    competitions: List[CompetitionModel]

class OrganizationDataReference(BaseModel):
    competition_id: str = Field(..., min_length=1, description="Competition identifier")
    organization_id: str = Field(..., min_length=1, description="Unique identifier for the organization")
    dataset_hf_repo: str = Field(..., min_length=1, description="Hugging Face repository path for the dataset")
    dataset_hf_dir: str = Field("", min_length=0, description="Directory for the datasets in the repository")

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
            
    def find_organization_by_competition_id(self, competition_id: str) -> Optional[OrganizationDataReference]:
        """Find an organization by competition ID.
        Returns:
            The organization data reference for the given competition ID, or None if not found
        """
        return next((o for o in self.organizations if o.competition_id == competition_id), None)

class NewDatasetFile(BaseModel):
    competition_id: str = Field(..., min_length=1, description="Competition identifier")
    dataset_hf_repo: str = Field(..., min_length=1, description="Hugging Face repository path for the dataset")
    dataset_hf_filename: str = Field(..., min_length=1, description="Filename for the dataset in the repository")



class WanDBLogBase(BaseModel):
    """Base class for WandB log entries"""
    uuid: str # competition unique identifier
    log_type: str
    validator_hotkey: str
    dataset_filename: str
    
    competition_id: str
    
    errors: str = ""
    run_time_s: float = 0.0

class WanDBLogModelBase(WanDBLogBase):
    log_type: str = "model_results"
    uid: int
    miner_hotkey: str
    
    score: float = 0.0
    average_score: float = 0.0
    
class WanDBLogModelErrorEntry(WanDBLogModelBase):
    pass    
    

class WanDBLogCompetitionWinners(WanDBLogBase):
    """Summary of competition"""
    log_type: str = "competition_summary"
    
    competition_winning_hotkey: str
    competition_winning_uid: int

    average_winning_hotkey: str
    average_winning_uid: int


@dataclass
class ModelInfo:
    hf_repo_id: str | None = None
    hf_model_filename: str | None = None
    hf_code_filename: str | None = None
    hf_repo_type: str | None = None

    competition_id: str | None = None
    file_path: str | None = None
    model_type: str | None = None
    block: int | None = None
    model_hash: str | None = None
    model_size_mb: float | None = None
