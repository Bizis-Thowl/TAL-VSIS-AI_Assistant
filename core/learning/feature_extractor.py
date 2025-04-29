"""Feature extraction for the staff planning system."""

import pandas as pd

class FeatureExtractor:
    """Extracts features from raw data for the staff planning system."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_columns = []
        self.target_column = None
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the input data.
        
        Args:
            data: Input DataFrame containing raw data
            
        Returns:
            DataFrame with extracted features
        """
        # Create a copy to avoid modifying the original data
        features = data.copy()
        
        # Extract temporal features
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
            features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Extract categorical features
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != 'timestamp':  # Skip timestamp column
                features[col] = pd.Categorical(features[col]).codes
        
        # Update feature columns
        self.feature_columns = [col for col in features.columns if col != self.target_column]
        
        return features
    
    def set_target_column(self, column: str):
        """
        Set the target column for prediction.
        
        Args:
            column: Name of the target column
        """
        self.target_column = column 