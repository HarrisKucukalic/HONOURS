import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import cupy as cp

class XGBoostModel:
    """A generic XGBoost model for NEM price prediction."""
    def __init__(self, region: str):
        """
        Initializes the XGBoost Regressor.
        region: The NEM region for which the model is being trained.
        kwargs: Hyperparameters for the xgb.XGBRegressor (e.g., n_estimators, learning_rate).
        """
        self.region = region
        self.model = xgb.XGBRegressor(booster='dart', eval_metric='rmse', device='cuda')
        print(f"Initialized XGBoost model for {self.region}.")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Trains the XGBoost model.
        X_train: Training feature data.
        y_train: Training target data.
        Dummy validation sets - they are not necessary
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on the test set.
        """
        print(f"Evaluating XGBoost model for {self.region} on test data...")

        # Convert the NumPy array from the CPU to a CuPy array on the GPU
        X_test_gpu = cp.asarray(X_test)

        # Now, predict using the GPU data. The warning will disappear.
        y_pred = self.model.predict(X_test_gpu)

        # The predictions will be on the GPU, so move them back to CPU for sklearn metrics
        y_pred_cpu = cp.asnumpy(y_pred)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_cpu)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_cpu))

        print(f"Evaluation complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {"MAE": mae, "RMSE": rmse}

    def save_results(self, results: dict, directory: str):
        """
        Saves the evaluation results to a file.

        Args:
            results (dict): A dictionary containing evaluation metrics.
            directory (str): The directory path to save the results file.
        """
        # Create the directory if it doesn't already exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Define the full path for the results file
        results_path = os.path.join(directory, 'evaluation_results.txt')

        # Write the results to the file
        with open(results_path, 'w') as f:
            f.write(f"Results for {self.model.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"Results saved to {results_path}")