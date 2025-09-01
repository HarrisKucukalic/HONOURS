import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ANNModel(nn.Module):
    # A generic Artificial Neural Network (ANN) model for NEM price prediction using PyTorch.
    def __init__(self, region: str, input_shape: int, layer_neurons: list = None, dropout_rate: float = 0.3):
        super(ANNModel, self).__init__()
        self.region = region
        if layer_neurons is None:
            layer_neurons = [128, 64, 32]

        self.layers = nn.ModuleList()
        # This variable tracks the input size for each new layer. It starts with the model's input shape.
        in_features = input_shape
        # Creation of Hidden Layers
        for out_features in layer_neurons:
            # Connect the previous layer's size (in_features) to the new size (out_features)
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.BatchNorm1d(out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

            # Updates in_features to be the output size for the next iteration
            in_features = out_features

        # After the loop, 'in_features' holds the size of the last hidden layer.
        self.output_layer = nn.Linear(in_features, 1)
        self.to(device)
        print(f"Initialised DYNAMIC PyTorch ANN for {self.region} on {device} with {layer_neurons} hidden layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Loop through all the layers (Linear, BatchNorm, ReLU, Dropout) in the list
        for layer in self.layers:
            # BatchNorm can fail on a batch size of 1 during eval, so a check is conducted to avoid a crash
            if isinstance(layer, nn.BatchNorm1d) and x.shape[0] <= 1:
                continue
            x = layer(x)
        # After the hidden layers, pass through the final output layer
        x = self.output_layer(x)
        return x

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 500, batch_size: int = 128, learning_rate: float = 0.001, patience: int = 20):
        print(f"Training ANN model for {self.region}...")

        X_train_tensor = torch.from_numpy(X_train).float().to(device)
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1).to(device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=learning_rate)

        # Early Stopping set-up, avoids unnecessary training
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Create val_loader if validation data exists
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1).to(device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training Loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimiser.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)

            # Run validation only if val_loader exists
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    # Validate in batches
                    for inputs, targets in val_loader:
                        outputs = self(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)

                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                # Early Stopping Logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.")
                    break
            else:
                # If no validation, just print training progress
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}')

        # Load the best model weights back into the model if early stopping was used
        if best_model_state:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

        print("Training complete.")


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        print(f"Making predictions with ANN model for {self.region}...")
        # Set model to evaluation mode
        self.eval()
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            predictions = self(X_test_tensor)

        return predictions.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        print(f"Evaluating {self.__class__.__name__} for {self.region} on test data...")
        y_pred = self.predict(X_test)
        # Ensure arrays are flat for metric calculation and plotting
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        results = {"MAE": mae, "RMSE": rmse}
        print(f"Evaluation complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        # Return the metrics and the data needed for plotting
        return results, y_test_flat, y_pred_flat

    def save_results(self, results: dict, y_true: np.ndarray, y_pred: np.ndarray, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Save metrics (MAE, RMSE) in a text file.
        results_path = os.path.join(directory, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Results for {self.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Metrics saved to {results_path}")

        # Generate and Save Plots

        # Predicted vs. Actual Scatter Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.3, label='Model Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.title('Predicted vs. Actual RRP')
        plt.xlabel('Actual RRP')
        plt.ylabel('Predicted RRP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'predicted_vs_actual.png'))
        plt.close()

        # Residuals Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs. Predicted Values')
        plt.xlabel('Predicted RRP')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'residuals_plot.png'))
        plt.close()

        # Time Series Comparison (zoomed in on the first 1000 points)
        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:1000], label='Actual Values', color='blue')
        plt.plot(y_pred[:1000], label='Predicted Values', color='orange', alpha=0.8)
        plt.title('Time Series Comparison (First 1000 Test Points)')
        plt.xlabel('Time Step')
        plt.ylabel('RRP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'timeseries_comparison.png'))
        plt.close()

        print(f"Diagnostic plots saved to {directory}")