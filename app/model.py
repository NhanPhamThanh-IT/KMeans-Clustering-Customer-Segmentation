"""
model.py

Defines the CustomerSegmentModel class for prediction.
"""

import pickle


class CustomerSegmentModel:
    """Handles loading and predicting customer segment using K-Means model."""

    def __init__(self, model_path: str):
        """Initializes the model from a pickle file."""
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, features: list[list[float]]) -> int:
        """Predicts the cluster index for given input features.

        Args:
            features: A list of features [[income, score]].

        Returns:
            Cluster label (integer).
        """
        return self.model.predict(features)[0]
