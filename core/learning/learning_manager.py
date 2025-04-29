"""Learning manager for the staff planning system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .data_preprocessor import DataPreprocessor
from .model import Model

class LearningManager:
    """Manages the learning process for staff planning."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the learning manager.
        
        Args:
            model_params: Parameters for the model
        """
        self.data_preprocessor = DataPreprocessor()
        self.model = Model(model_params)
        self.is_trained = False
    
    def train(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the model with the given data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            feature_columns: List of feature column names
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Preprocess data
        X_train, X_test, y_train, y_test = self.data_preprocessor.preprocess(
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state
        )
        
        # Train model
        self.model.train(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, data: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Make predictions for new data.
        
        Args:
            data: Input data
            feature_columns: List of feature column names
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Transform data
        X = self.data_preprocessor.transform(data[feature_columns])
        
        # Make predictions
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.get_feature_importance() 