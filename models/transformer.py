import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerModel(nn.Module):
    def __init__(self, region: str, input_shape: int, model_dim: int = 128, n_heads: int = 8, n_encoder_layers: int = 6, ff_dim: int = 512, dropout: float = 0.3):
        super(TransformerModel, self).__init__()
        self.region = region
        self.encoder = nn.Linear(input_shape, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.Linear(model_dim, 1)
        self.to(device)
        print(f"Initialised PyTorch Transformer model for {self.region} on {device}.")

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global Average Pooling
        output = self.decoder(output)
        return output

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 500, batch_size: int = 128, learning_rate: float = 1e-4, patience: int = 10):
        print(f"Training Transformer model for {self.region}...")
        # Device and Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # Data Setup (on CPU) - avoid GPU memory overuse
        # Create datasets from numpy arrays (CPU memory)
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        # Training Initialisation
        criterion = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0
        print(f"Starting training on {device} for {epochs} epochs...")
        # Training Loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                # Move Batch to GPU
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * X_batch.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)
            # Validation Loop - if validation set is not empty
            if val_loader:
                # Set model to evaluation mode
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch_val, y_batch_val in val_loader:
                        # Move BATCH to GPU
                        X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                        outputs_val = self(X_batch_val)
                        loss = criterion(outputs_val, y_batch_val)
                        val_loss += loss.item() * X_batch_val.size(0)

                avg_val_loss = val_loss / len(val_loader.dataset)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # save Transformer Weights
                    torch.save(self.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.4f}")
                        break
            else:
                # If no validation set, just print training loss
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        print("Training complete.")

    def predict(self, X_test: np.ndarray, batch_size: int = 128) -> np.ndarray:
        print(f"Making predictions with Transformer model for {self.region}...")
        self.eval()
        if len(X_test.shape) == 2:
            X_test = X_test[:, np.newaxis, :]
        # Create a dataset and a loader to break the data into batches
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        all_predictions = []
        with torch.no_grad():
            # Process one small batch at a time, keeping memory low
            for (inputs,) in test_loader:
                outputs = self(inputs)
                all_predictions.append(outputs.cpu().numpy())

        # Combine the results from all the small batches
        return np.vstack(all_predictions)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        print(f"Evaluating final {self.__class__.__name__} for {self.region} on test data...")
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

        # Save Text Results
        results_path = os.path.join(directory, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Results for {self.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"Metrics and hyperparameters saved to {results_path}")

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