import numpy as np
import pandas as pd
import xgboost as xgb
class XGBoostModel:
    """A generic XGBoost model for NEM price prediction."""
    def __init__(self, region: str, **kwargs):
        """
        Initializes the XGBoost Regressor.
        region: The NEM region for which the model is being trained.
        kwargs: Hyperparameters for the xgb.XGBRegressor (e.g., n_estimators, learning_rate).
        """
        self.region = region
        self.model = xgb.XGBRegressor(**kwargs)
        print(f"Initialized XGBoost model for {self.region}.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the XGBoost model.
        X_train: Training feature data.
        y_train: Training target data.
        """
        print(f"Training XGBoost model on {self.region} data...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.
        X_test: Data to make predictions on.
        Returns: Predicted values.
        """
        print(f"Making predictions with XGBoost model for {self.region}...")
        return self.model.predict(X_test)