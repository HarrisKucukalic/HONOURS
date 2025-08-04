import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    """A generic Random Forest model for NEM price prediction."""
    def __init__(self, region: str, **kwargs):
        """
        Initialises the Random Forest Regressor.
        region: The NEM region (e.g., "NSW") for which the model is being trained.
        kwargs: Hyperparameters for the RandomForestRegressor (e.g., n_estimators, max_depth).
        """
        self.region = region
        self.model = RandomForestRegressor(**kwargs)
        print(f"Initialized Random Forest model for {self.region}.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Random Forest model.
        X_train: Training feature data.
        y_train: Training target data.
        """
        print(f"Training Random Forest model on {self.region} data...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.
        X_test: Data to make predictions on.
        Returns: Predicted values.
        """
        print(f"Making predictions with Random Forest model for {self.region}...")
        return self.model.predict(X_test)