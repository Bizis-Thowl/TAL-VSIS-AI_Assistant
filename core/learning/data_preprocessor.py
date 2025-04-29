"""Data preprocessing for the staff planning system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .feature_extractor import FeatureExtractor


class DataPreprocessor:
    """Preprocesses data for the staff planning system."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        self.feature_extractor = FeatureExtractor()
    
    def preprocess(self, data: pd.DataFrame, target_column: str, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data for training.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.target_column = target_column
        self.feature_columns = [col for col in data.columns if col != target_column]
        
        # Split features and target
        X = data[self.feature_columns].values
        y = data[target_column].values
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted scaler.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed numpy array
        """
        if not self.feature_columns:
            raise ValueError("Preprocessor must be fitted first")
            
        X = data[self.feature_columns].values
        return self.scaler.transform(X)
    
    def prepare_data(
        self,
        assignment: Dict,
        employees: pd.DataFrame,
        clients: pd.DataFrame
    ) -> Optional[List[List]]:
        """Prepare data for the model."""
        try:
            replacements = self._create_replacements_df(assignment["ma"], assignment["klient"])
            replacements = self._merge_data(replacements, employees, clients)
            result = self._extract_features(replacements)
            return self._format_result(result)
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None
    
    def _create_replacements_df(self, ma: str, klient: str) -> pd.DataFrame:
        """Create a DataFrame for replacements."""
        return pd.DataFrame({
            "mas": [ma],
            "clients": [klient]
        })
    
    def _merge_data(
        self,
        replacements: pd.DataFrame,
        employees: pd.DataFrame,
        clients: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge replacement data with employee and client data."""
        replacements = replacements.merge(
            employees,
            left_on="mas",
            right_on="id",
            how="inner"
        )
        return replacements.merge(
            clients,
            left_on="clients",
            right_on="id",
            how="inner",
            suffixes=("_mas", "_client")
        )
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from the data."""
        result = data.apply(self.feature_extractor.extract_features, axis=1)
        return pd.DataFrame(result.tolist())
    
    def _format_result(self, result_df: pd.DataFrame) -> List[List]:
        """Format the result for model input."""
        result_df = result_df.filter(items=FeatureExtractor.FEATURES)
        return [list(result_df.iloc[0])] 