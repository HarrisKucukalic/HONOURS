import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # Dropout between stacked layers
        )
        # Dropout layer after LSTM
        self.dropout = nn.Dropout(dropout)
        # Layer normalization after LSTM
        self.layernorm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        # Attention layer
        self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        # For bidirectional, hidden_size is doubled in the first dimension
        h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers,
                         x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers,
                         x.size(0),
                         self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_length, hidden_size * 2 if bidirectional)

        # Apply dropout and layer norm
        out = self.dropout(out)
        out = self.layernorm(out)

        # Apply attention
        context = self.attention(out)

        # Decode the hidden state (now with attention)
        out = self.fc(context)
        return out


def log_loss_curve(writer, train_losses, val_losses, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    writer.add_figure('Loss Curves', plt.gcf(), epoch)
    plt.close()

def train_lstm(model, train_data, val_data, sequence_length, input_size, batch_size, num_epochs, learning_rate=0.001,
               device='cuda'):
    # Move the model to the correct device (GPU/CPU)
    model = model.to(device)
    writer = SummaryWriter('runs/model_run')

    # Create DataLoader for training and validation data
    X_train, y_train = train_data
    X_val, y_val = val_data

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Loop over the training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels.argmax(dim=1))  # Convert one-hot labels to class indices

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels.argmax(dim=1)).sum().item()
            total_samples += inputs.size(0)

        # Calculate average training loss and accuracy
        train_loss /= total_samples
        train_losses.append(train_loss)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        train_accuracy = 100 * correct_predictions / total_samples
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # No need to calculate gradients during validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels.argmax(dim=1))

                # Track loss and accuracy
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels.argmax(dim=1)).sum().item()
                total_samples += inputs.size(0)

        # Calculate average validation loss and accuracy
        val_loss /= total_samples
        val_accuracy = 100 * correct_predictions / total_samples
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        val_losses.append(val_loss)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print("Training complete.")
    plot_loss_graph(train_losses, val_losses)
    log_loss_curve(writer, train_losses, val_losses, num_epochs)
    writer.close()


def plot_loss_graph(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
