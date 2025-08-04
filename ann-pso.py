import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ann import ANNModel
from pso import PSO_Optimizer

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
            model_class=ANNModel,
            region=self.region,
            input_shape=self.input_shape
        )
        self.final_model = None
        print(f"Initialized ANN_PSO_Model wrapper for {self.region}.")

    def train(self, X_train, y_train, X_val, y_val, search_space, pso_n_particles, pso_iters, final_epochs=50):
        """
        Finds optimal hyperparameters using PSO and then trains the final, optimized model.

        Args:
            X_train, y_train: Data for training the temporary models during optimization.
            X_val, y_val: Data for evaluating the temporary models (calculating RMSE).
            search_space: A dictionary defining the min and max bounds for PSO.
            pso_n_particles: The number of particles in the swarm.
            pso_iters: The number of iterations for the PSO algorithm.
            final_epochs: The number of epochs to train the final model.
        """
        print("\n--- Step 1: Finding Optimal Hyperparameters using PSO ---")
        best_params = self.optimizer.find_optimal_hyperparameters(
            X_train, y_train, X_val, y_val, search_space, pso_n_particles, pso_iters
        )
        print(f"\nOptimal hyperparameters found: {best_params}")

        print("\n--- Step 2: Training Final Model with Optimal Hyperparameters ---")
        self.final_model = ANNModel(
            region=self.region,
            input_shape=self.input_shape,
            n_layer1=best_params['n_layer1'],
            n_layer2=best_params['n_layer2'],
            n_layer3=best_params['n_layer3']
        )

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
            print(f"Making predictions with final optimized model for {self.region}...")
            return self.final_model.predict(X_test)
        else:
            raise RuntimeError("Model has not been trained yet. Please call the train() method first.")