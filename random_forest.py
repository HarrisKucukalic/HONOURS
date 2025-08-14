import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RandomForestModel:
    """A generic Random Forest model for NEM price prediction."""
    def __init__(self, region: str):
        """
        Initialises the Random Forest Regressor.
        region: The NEM region (e.g., "NSW") for which the model is being trained.
        kwargs: Hyperparameters for the RandomForestRegressor (e.g., n_estimators, max_depth).
        """
        self.region = region
        self.model = RandomForestRegressor(n_estimators=200, criterion='friedman_mse', random_state=42)
        print(f"Initialized Random Forest model for {self.region}.")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Trains the Random Forest model.
        X_train: Training feature data.
        y_train: Training target data.
        Dummy validation sets - they are not necessary
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on the test set and returns performance metrics.

        Args:
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): Test target data.

        Returns:
            dict: A dictionary containing evaluation metrics (MAE, RMSE).
        """
        print(f"Evaluating Random Forest model for {self.region} on test data...")

        # Get model predictions
        y_pred = self.predict(X_test)

        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Evaluation complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Return metrics in a dictionary
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