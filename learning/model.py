"""Model management for the staff planning system."""

import numpy as np
from typing import Dict, Any, Optional
from pickle import dump, load
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

model_path = "models/isolation_forst.pkl"


class AbnormalityModel:
    """Manages the machine learning model for staff planning using Isolation Forest."""

    def __init__(
        self, model_params: Optional[Dict[str, Any]] = None, use_cache: bool = True
    ):
        """
        Initialize the model.

        Args:
            model_params: Parameters for the model. Defaults to:
                - n_estimators: 100
                - max_samples: 'auto'
                - contamination: 'auto'
                - random_state: 42
        """
        default_params = {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": "auto",
            "random_state": 42,
        }
        if model_params:
            default_params.update(model_params)

        if use_cache and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = load(f)
        else:
            self.model = IsolationForest(**default_params)

    def train(self, X: np.ndarray) -> None:
        """
        Train the model on the input data.

        Args:
            X: Input features (no labels needed for unsupervised learning)
        """
        self.model.fit(X)
        with open(model_path, "wb") as f:
            dump(self.model, f, protocol=5)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (returns -1 for outliers, 1 for inliers).

        Args:
            X: Input features

        Returns:
            Array of predictions (-1 for outliers, 1 for inliers)
        """
        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        The lower the score, the more abnormal the sample.

        Args:
            X: Input features

        Returns:
            Array of anomaly scores
        """
        return self.model.score_samples(X)

    def evaluate(
        self, X: np.ndarray, y_true: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        If y_true is provided, calculates ROC-AUC and PR-AUC.
        Otherwise, returns basic statistics about anomaly scores.

        Args:
            X: Input features
            y_true: Optional true labels (1 for normal, -1 for anomaly)

        Returns:
            Dictionary of evaluation metrics
        """
        scores = self.score_samples(X)

        metrics = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

        if y_true is not None:
            # Convert scores to probabilities (higher score = more normal)
            probabilities = 1 / (1 + np.exp(-scores))
            metrics.update(
                {
                    "roc_auc": float(roc_auc_score(y_true, probabilities)),
                    "pr_auc": float(average_precision_score(y_true, probabilities)),
                }
            )

        return metrics

    def get_decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get the decision function values.
        The lower the value, the more abnormal the sample.

        Args:
            X: Input features

        Returns:
            Array of decision function values
        """
        return self.model.decision_function(X)

    def visualize(self, X: np.ndarray) -> None:
        """
        Visualize the model's performance.

        Args:
            X: Input features
            y_true: Optional true labels (1 for normal, -1 for anomaly)
        """

        df = pd.DataFrame(
            X,
            columns=[
                "timeToSchool",
                "cl_experience",
                "school_experience",
                "priority",
                "ma_availability",
                "mobility",
                "geschlecht_relevant",
                "qualifications_met",
            ],
        )
        df["anomaly"] = self.predict(X)
        df["score"] = self.score_samples(X)

        g = sns.pairplot(
            df,
            hue="anomaly",
            height=2.5,
            palette={1: "blue", -1: "red"},
            diag_kind="kde",
            kind="hist",
        )

        # Bar plot of the number of normal vs anomaly points
        plt.figure(figsize=(8, 6))
        sns.countplot(
            x="anomaly",
            data=df,
            palette={1: "blue", -1: "red"},
            hue="anomaly",
            legend=False,
        )
        plt.title("Count of normal vs. anomaly points")
        plt.xlabel("Anomaly")
        plt.ylabel("Count")
        plt.xticks(ticks=[0, 1], labels=["Anomaly", "Normal"])
        plt.show()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        g.savefig(f"models/isolation_forest_visualization_{timestamp}.png")
