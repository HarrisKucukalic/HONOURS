import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from models.transformer import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualise_model_with_tensorboard(model: nn.Module, input_tensor: torch.Tensor, log_dir: str):
    try:
        # Set the model to evaluation mode to ensure deterministic behavior
        model.eval()

        # Explicitly script the model to create a stable graph for the tracer
        scripted_model = torch.jit.script(model)

        # Create a SummaryWriter to write data to the log directory
        writer = SummaryWriter(log_dir)
        writer.add_graph(scripted_model, input_tensor)
        writer.close()

        print(f"Successfully created model graph in '{log_dir}'")
        print(f"To view, run: tensorboard --logdir={os.path.dirname(log_dir)}")

    except Exception as e:
        print(f"Error creating diagram for {log_dir}: {e}")


if __name__ == '__main__':
    TRANSFORMER_INPUT_FEATURES = 12
    TRANSFORMER_BATCH_SIZE = 16
    SEQUENCE_LENGTH = 30
    MODEL_DIM = 128
    N_HEADS = 4
    N_ENCODER_LAYERS = 3
    FF_DIM = 512

    # Instantiate Transformer model with dropout set to 0 for visualisation
    transformer_model = TransformerModel(
        region="NSW",
        input_dim=TRANSFORMER_INPUT_FEATURES,
        model_dim=MODEL_DIM,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        ff_dim=FF_DIM,
        dropout=0.0
    )

    # Create a dummy input tensor with the correct 3D shape for a Transformer
    # Shape: (batch_size, sequence_length, input_features)
    dummy_input_transformer = torch.randn(TRANSFORMER_BATCH_SIZE, SEQUENCE_LENGTH, TRANSFORMER_INPUT_FEATURES).to(
        device)

    # Generate the graph for the Transformer model in a different log directory
    visualise_model_with_tensorboard(transformer_model, dummy_input_transformer, log_dir="deep_learning_model_structures/transformer_model")




