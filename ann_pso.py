import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ann import ANNModel
from pso import PSO_Optimizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ANN_PSO_Model:
    """
    A high-level wrapper to find optimal hyperparameters for an ANNModel using PSO
    and then train the final model.
    """

    def __init__(self, region: str, input_shape: int):
        """
        Initializes the high-level ANN-PSO model.
        """
        self.region = region
        self.input_shape = input_shape
        self.optimizer = PSO_Optimizer(
            model_class=ANNModel,  # Pass the class itself
            region=self.region,
            input_shape=self.input_shape
        )
        self.final_model = None
        self.best_hyperparameters = None
        print(f"Initialized ANN_PSO_Model wrapper for {self.region}.")

    def run_training_loop(self, X_train, y_train, X_val, y_val, search_space, pso_n_particles, pso_iters):
        """
        Finds optimal hyperparameters using PSO and then trains the final, optimized model.
        """
        print("\n--- Step 1: Finding Optimal Hyperparameters using PSO ---")
        self.best_hyperparameters = self.optimizer.find_optimal_hyperparameters(
            search_space, X_train, y_train, X_val, y_val, pso_n_particles, pso_iters
        )
        print(f"\nOptimal hyperparameters found: {self.best_hyperparameters}")

        print("\n--- Step 2: Decoding and Separating Hyperparameters ---")

        # ▼▼▼ THIS IS THE NEW, CRUCIAL LOGIC ▼▼▼

        # 1. Decode the architecture from the raw PSO results
        num_layers = int(np.round(self.best_hyperparameters['num_layers']))
        architecture_list = [
            int(np.round(self.best_hyperparameters[f'neurons_layer_{j}'])) for j in range(num_layers)
        ]

        # 2. Create a dictionary specifically for the model's constructor
        model_constructor_params = {
            'layer_neurons': architecture_list
            # 'dropout_rate' could also be an optimizable parameter here
        }

        # 3. Create a dictionary for the training method's parameters
        training_params = {
            'epochs': int(np.round(self.best_hyperparameters['epochs'])),
            'batch_size': int(2 ** np.round(self.best_hyperparameters['batch_size_power'])),
            'learning_rate': self.best_hyperparameters['learning_rate']
        }

        # ▲▲▲ END OF NEW LOGIC ▲▲▲

        print(f"Decoded architecture: {architecture_list}")
        print(f"Training parameters: {training_params}")

        print("\n--- Step 3: Training Final Model with Optimal Hyperparameters ---")

        # Use the separated dictionaries to build and train the model
        self.final_model = ANNModel(
            region=self.region,
            input_shape=self.input_shape,
            **model_constructor_params  # Pass only the architecture params
        )

        # Train the final model using the separated training params
        self.final_model.train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            **training_params  # Pass the training params here
        )
        print("\nFinal model training complete.")

    def predict(self, X_test):
        """
        Makes predictions using the final, optimized model.
        """
        if self.final_model:
            print(f"Making predictions with final optimized model for {self.region}...")
            return self.final_model.predict(X_test)
        else:
            raise RuntimeError("Model has not been trained yet. Please call the run_training_loop() method first.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on the test set and returns performance metrics.
        """
        print(f"Evaluating ANN-PSO model for {self.region} on test data...")
        y_pred = self.predict(X_test)

        if y_test.ndim > 1:
            y_test = y_test.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Evaluation complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {"MAE": mae, "RMSE": rmse}

    def save_results(self, results: dict, directory: str):
        """
        Saves the evaluation results and the best hyperparameters to a file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        results_path = os.path.join(directory, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Results for {self.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write("\nBest Hyperparameters Found by PSO:\n")
            for key, value in self.best_hyperparameters.items():
                f.write(f"{key}: {value}\n")

        print(f"Results and hyperparameters saved to {results_path}")
