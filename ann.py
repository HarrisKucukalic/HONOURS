import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
class ANNModel(nn.Module):
    """A generic Artificial Neural Network (ANN) model for NEM price prediction using PyTorch."""

    def __init__(self, region: str, input_shape: int,
                 layer_neurons: list = None, dropout_rate: float = 0.3):

        super(ANNModel, self).__init__()
        self.region = region

        if layer_neurons is None:
            layer_neurons = [128, 64, 32]

        self.layers = nn.ModuleList()

        # This variable tracks the input size for each new layer. It starts with the model's input shape.
        in_features = input_shape

        # --- A single loop to create all hidden layers ---
        for out_features in layer_neurons:
            # Connect the previous layer's size (in_features) to the new size (out_features)
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.BatchNorm1d(out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

            # IMPORTANT: Update in_features to be the output size for the next loop iteration
            in_features = out_features

        # After the loop, 'in_features' holds the size of the last hidden layer (32)
        self.output_layer = nn.Linear(in_features, 1)

        self.to(device)
        print(f"Initialised DYNAMIC PyTorch ANN for {self.region} on {device} with {layer_neurons} hidden layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the dynamically built model.
        """
        # Loop through all the layers (Linear, BatchNorm, ReLU, Dropout) in the list
        for layer in self.layers:
            # BatchNorm can fail on a batch size of 1 during eval, so we add a check
            if isinstance(layer, nn.BatchNorm1d) and x.shape[0] <= 1:
                continue
            x = layer(x)

        # After the hidden layers, pass through the final output layer
        x = self.output_layer(x)

        return x
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 500, batch_size: int = 128, learning_rate: float = 0.001, patience: int = 10):
        """
        Trains the ANN model with early stopping.

        Args:
            X_train, y_train: Training data and labels.
            X_val, y_val: Validation data and labels for early stopping.
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per batch.
            learning_rate: Step size for the optimizer.
            patience: Number of epochs to wait for improvement before stopping.
        """
        print(f"Training ANN model on {self.region} data...")

        # --- Data Setup ---
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # --- Early Stopping Initialization ---
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # --- Training Phase ---
            self.train()
            epoch_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # --- Validation Phase ---
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

            # --- Early Stopping Logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model's state
                best_model_state = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.")
                break

        # Load the best model weights back into the model
        if best_model_state:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

        print("Training complete.")


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.
        """
        print(f"Making predictions with ANN model for {self.region}...")
        self.eval()  # Set model to evaluation mode
        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        with torch.no_grad():
            predictions = self(X_test_tensor)

        return predictions.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on the test set and returns performance metrics.

        Args:
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): Test target data.

        Returns:
            dict: A dictionary containing evaluation metrics (MAE, RMSE).
        """
        print(f"Evaluating ANN model for {self.region} on test data...")

        # Get model predictions using the existing predict method
        y_pred = self.predict(X_test)

        # Ensure y_test and y_pred are flat arrays for comparison
        if y_test.ndim > 1:
            y_test = y_test.flatten()

        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Evaluation complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Return metrics in a dictionary, as expected by the main script
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
            # Corrected line: Use self.__class__.__name__
            f.write(f"Results for {self.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"Results saved to {results_path}")