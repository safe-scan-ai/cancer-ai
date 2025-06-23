from typing import Any
from abc import abstractmethod

from numpy import ndarray
import numpy as np
from pydantic import BaseModel, field_serializer


class BaseModelEvaluationResult(BaseModel):
    score: float = 0.0
    predictions_raw: list = []
    error: str = ""

    run_time_s: float = 0.0
    tested_entries: int = 0

    class Config:
        arbitrary_types_allowed = True


class TricorderEvaluationResult(BaseModelEvaluationResult):
    accuracy: float = 0.0
    precision: float = 0.0
    fbeta: float = 0.0


class BaseCompetitionHandler:
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
    def get_model_result(self) -> BaseModelEvaluationResult:
        """
        Abstract method to evaluate the competition.

        This method is responsible for evaluating the competition.
        """
