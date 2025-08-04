import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerModel(nn.Module):
    """A generic Transformer-based model for NEM price prediction using PyTorch."""

    def __init__(self, region: str, input_dim: int, model_dim: int = 128, n_heads: int = 4, n_encoder_layers: int = 3,
                 ff_dim: int = 512, dropout: float = 0.1):
        """
        Initializes the Transformer model.
        """
        super(TransformerModel, self).__init__()
        self.region = region

        self.encoder = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.Linear(model_dim, 1)

        self.to(device)
        print(f"Initialised PyTorch Transformer model for {self.region} on {device}.")

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass."""
        src = self.encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global Average Pooling
        output = self.decoder(output)
        return output

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32,
                    learning_rate: float = 0.001):
        """Trains the Transformer model."""
        print(f"Training Transformer model on {self.region} data.")
        self.train()

        # Note: Transformer models expect input data in the shape (batch_size, seq_len, features)
        # Reshape if necessary.
        if len(X_train.shape) == 2:
            X_train = X_train[:, np.newaxis, :]

        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)

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
        """Makes predictions with the Transformer model."""
        print(f"Making predictions with Transformer model for {self.region}...")
        self.eval()

        if len(X_test.shape) == 2:
            X_test = X_test[:, np.newaxis, :]

        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        with torch.no_grad():
            predictions = self(X_test_tensor)

        return predictions.cpu().numpy()