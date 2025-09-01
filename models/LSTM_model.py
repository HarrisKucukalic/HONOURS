import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import os
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Create a linear layer to learn attention weights
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_size)
        # Calculate attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # Apply weights to the LSTM output to get a context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context


class LSTMModel(nn.Module):
    def __init__(self, region, input_shape, hidden_size=256, num_layers=3, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.region = region
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_shape,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialise hidden and cell states for a unidirectional LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Apply dropout, layer normalisation, and attention
        out = self.dropout(out)
        out = self.layernorm(out)
        context = self.attention(out)
        # Pass through the final fully connected layer
        out = self.fc(context)
        return out

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 500, batch_size: int = 128, learning_rate: float = 0.001, patience: int = 50):
        print(f"Training LSTM model for {self.region} with early stopping (patience={patience})...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        # Early Stopping Initialisation
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = None
        print("Starting LSTM training...")
        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimiser.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                loss.backward()
                optimiser.step()
                epoch_train_loss += loss.item() * inputs.size(0)

            avg_train_loss = epoch_train_loss / len(train_loader.dataset)
            self.train_losses.append(avg_train_loss)

            # Validation
            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets.view(-1, 1))
                    epoch_val_loss += loss.item() * inputs.size(0)

            avg_val_loss = epoch_val_loss / len(val_loader.dataset)
            self.val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save the best model weights
                best_model_wts = copy.deepcopy(self.state_dict())
                print(f"Validation loss improved to {best_val_loss:.4f}. Saving model.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
        # Restore the best model weights
        if best_model_wts:
            print("Restoring best model weights.")
            self.load_state_dict(best_model_wts)

        print("Training complete.")

    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # Set the model to evaluation mode
        self.eval()
        # Create a DataLoader for the prediction data
        X_tensor = torch.from_numpy(X).float()
        pred_dataset = TensorDataset(X_tensor)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for inputs_batch in pred_loader:
                # inputs_batch is a list containing one tensor
                inputs = inputs_batch[0].to(device)
                outputs = self(inputs)
                all_preds.append(outputs.cpu().numpy())

        # Concatenate predictions from all batches
        y_pred = np.concatenate(all_preds, axis=0)
        return y_pred

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
        return results, y_test_flat, y_pred_flat

    def save_results(self, results: dict, y_true: np.ndarray, y_pred: np.ndarray, directory: str):
        """Saves evaluation metrics and diagnostic plots to the specified directory."""
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
