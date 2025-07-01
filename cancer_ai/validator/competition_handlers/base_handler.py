from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel


class BaseModelEvaluationResult(BaseModel):
    score: float = 0.0
    predictions_raw: list = []
    error: str = ""

    run_time_s: float = 0.0
    tested_entries: int = 0

    class Config:
        arbitrary_types_allowed = True


class BaseCompetitionHandler(ABC):
    """
    Base class for handling different competition types.

    This class initializes the config and competition_id attributes.
    """

    def __init__(self, X_test: list, y_test: list) -> None:
        """
        Initializes the BaseCompetitionHandler object.
        """
        self.X_test = X_test
        self.y_test = y_test

    @abstractmethod
    def preprocess_and_serialize_data(self, X_test: list) -> list:
        """
        Abstract method to preprocess and serialize data.

        This method is responsible for preprocessing the data for the competition
        and serializing it for efficient reuse across multiple model evaluations.

        Args:
            X_test: List of input data (typically file paths for images)

        Returns:
            List of paths to serialized preprocessed data chunks
        """

    @abstractmethod
    def set_preprocessed_data_dir(self, data_dir: str) -> None:
        """
        Abstract method to set directory for storing preprocessed data.
        """

    @abstractmethod
    def get_preprocessed_data_generator(self):
        """
        Abstract method to get preprocessed data generator.

        Returns:
            Generator that yields preprocessed data chunks
        """

    @abstractmethod
    def cleanup_preprocessed_data(self) -> None:
        """
        Abstract method to cleanup preprocessed data files.
        """

    @abstractmethod
    def preprocess_data(self):
        """
        Abstract method to prepare the data.

        This method is responsible for preprocessing the data for the competition.
        """

    @abstractmethod
    def get_model_result(self, y_test: List[int], y_pred: List[float], run_time_s: float) -> tuple:
        """
        Abstract method to evaluate the competition.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_comparable_result(self, result: BaseModelEvaluationResult) -> tuple:
        """
        Create a comparable representation of the result for grouping duplicates.
        
        This method should be implemented by each competition handler to specify
        which metrics are used for comparing results.
        
        Args:
            result: The evaluation result object.
            
        Returns:
            A tuple of key metrics that can be used for comparison.
        """
        raise NotImplementedError
