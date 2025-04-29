from typing import Dict, List, Tuple
import pandas as pd

from core.learning.learning_manager import LearningManager


class LearningEngine:
    """Handles machine learning aspects of staff planning."""
    
    def __init__(self):
        self.learner = LearningManager()
        self.clients_df = pd.DataFrame()
        self.mas_df = pd.DataFrame()

    def update_data(self, clients_df: pd.DataFrame, mas_df: pd.DataFrame) -> None:
        """Update the learning engine with new data."""
        self.clients_df = clients_df
        self.mas_df = mas_df

    def prepare_data(self, assignments: Dict) -> List[List]:
        """Prepare data for learning."""
        return self.learner.prepare_data(assignments, self.mas_df, self.clients_df)

    def predict_and_score(self, datapoint: List[List]) -> Tuple[int, float]:
        """Predict and score a datapoint."""
        return self.learner.predict_and_score(datapoint)

    def is_initialized(self) -> bool:
        """Check if the learning engine is properly initialized."""
        return not self.clients_df.empty and not self.mas_df.empty 