import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pso import PSO_Optimizer
from transformer import TransformerModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Transformer_PSO_Model:
    """
    A high-level wrapper to find optimal hyperparameters for a TransformerModel using PSO
    and then train the final model.
    """

    def __init__(self, region: str, input_shape: int):
        """
        Initializes the high-level Transformer-PSO model.
        """
        self.region = region
        self.input_shape = input_shape
        self.optimizer = PSO_Optimizer(
            model_class=TransformerModel,
            region=self.region,
            input_shape=self.input_shape
        )
        self.final_model = None
        print(f"Initialized Transformer_PSO_Model wrapper for {self.region}.")

    def train_model(self, X_train, y_train, X_val, y_val, search_space, pso_n_particles, pso_iters, final_epochs=50):
        """
        Finds optimal hyperparameters using PSO and then trains the final, optimized model.
        """
        print("\n--- Step 1: Finding Optimal Hyperparameters for Transformer using PSO ---")
        best_params = self.optimizer.find_optimal_hyperparameters(
            X_train, y_train, X_val, y_val, search_space, pso_n_particles, pso_iters
        )
        print(f"\nOptimal hyperparameters found: {best_params}")

        print("\n--- Step 2: Training Final Transformer Model with Optimal Hyperparameters ---")
        # Note: TransformerModel uses 'input_dim'
        self.final_model = TransformerModel(region=self.region, input_dim=self.input_shape, **best_params)

        # Combine training and validation data for the final training run
        X_full_train = np.concatenate((X_train, X_val))
        y_full_train = np.concatenate((y_train, y_val))

        self.final_model.train_model(
            X_full_train,
            y_full_train,
            epochs=final_epochs,
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate']
        )
        print("\nFinal model training complete.")

    def predict(self, X_test):
        """
        Makes predictions using the final, optimized model.
        """
        if self.final_model:
            print(f"Making predictions with final optimized Transformer model for {self.region}...")
            return self.final_model.predict(X_test)
        else:
            raise RuntimeError("Model has not been trained yet. Please call the train() method first.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on the test set and returns performance metrics.

        Args:
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): Test target data.

        Returns:
            dict: A dictionary containing evaluation metrics (MAE, RMSE).
        """
        print(f"Evaluating Transformer-PSO model for {self.region} on test data...")

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