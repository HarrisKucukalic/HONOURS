import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
class ANNModel(nn.Module):
    """A generic Artificial Neural Network (ANN) model for NEM price prediction using PyTorch."""

    def __init__(self, region: str, input_shape: int):
        """
        Initializes the ANN model.
        region: The NEM region for which the model is being trained.
        input_shape: The number of input features.
        """
        super(ANNModel, self).__init__()
        self.region = region

        self.layer1 = nn.Linear(input_shape, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        self.to(device)  # Move the model to the configured device (GPU or CPU)
        print(f"Initialized PyTorch ANN model for {self.region} on {device}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return x

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32,
                    learning_rate: float = 0.001):
        """
        Trains the ANN model.
        """
        print(f"Training ANN model on {self.region} data...")
        self.train()  # Set model to training mode

        # Convert data to PyTorch Tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)

        # Create DataLoader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

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