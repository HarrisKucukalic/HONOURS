import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from LSTM import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchview import draw_graph
from torchinfo import summary
from torchviz import make_dot
from torch.utils.tensorboard import *

def load_data():
    return pd.read_csv(r'C:\projects\HONOURS\FinalData.csv')


def split_data(data, sequence_length=10, batch_size=32):
    X_batches = []
    y_batches = []
    # encode gait types
    data = data.copy()
    label_encoder = LabelEncoder()
    data['GaitType'] = label_encoder.fit_transform(data['GaitType'])
    print(label_encoder.classes_)
    # groups data with two keys to make primary key for each video
    grouped_data = data.groupby(['VideoID', 'GaitType'])

    for (video_id, gait_type), group in grouped_data:
        # ensures the order of the video remain the same
        group = group.sort_index()
        # gets all the features needed for the X data
        x_features = group.drop(columns=['VideoID', 'GaitType']).values
        # create sequence for the LSTM
        for i in range(len(group) - sequence_length):
            X_sequence = x_features[i:i + sequence_length]
            y_sequence = group['GaitType'].iloc[i + sequence_length - 1]
            # Gets labels as an array of boolean
            y_sequence_encoded = np.zeros(len(label_encoder.classes_))
            y_sequence_encoded[y_sequence] = 1

            X_batches.append(X_sequence)
            y_batches.append(y_sequence_encoded)

    # Convert batches into numpy arrays
    X_batches_np = np.array(X_batches)
    y_batches_np = np.array(y_batches)
    # Ensure each batch is evenly distributed, deals with alternative video lengths
    num_batches = len(X_batches_np) // batch_size
    X_batches_equalised = X_batches_np[:num_batches * batch_size]
    y_batches_equalised = y_batches_np[:num_batches * batch_size]
    # Reshape into correct batches
    X_batches_final = X_batches_equalised.reshape(len(X_batches_equalised), sequence_length, X_batches_equalised.shape[2])
    y_batches_final = y_batches_equalised.reshape(len(y_batches_equalised), len(label_encoder.classes_))

    return X_batches_final, y_batches_final


def custom_split_by_video_id(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Create unique group keys
    unique_keys = data.groupby(['VideoID', 'GaitType']).size().reset_index()[['VideoID', 'GaitType']]

    # First split: train vs temp (val + test)
    train_keys, temp_keys = train_test_split(
        unique_keys, test_size=(1 - train_ratio), random_state=random_state, shuffle=True
    )

    # Second split: val vs test from the temp set
    val_keys, test_keys = train_test_split(
        temp_keys, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state, shuffle=True
    )

    # Convert key sets to tuples for easy filtering
    train_keys = set(map(tuple, train_keys.values))
    val_keys = set(map(tuple, val_keys.values))
    test_keys = set(map(tuple, test_keys.values))

    # Filter the original data
    train_df = data[data.apply(lambda row: (row['VideoID'], row['GaitType']) in train_keys, axis=1)]
    val_df = data[data.apply(lambda row: (row['VideoID'], row['GaitType']) in val_keys, axis=1)]
    test_df = data[data.apply(lambda row: (row['VideoID'], row['GaitType']) in test_keys, axis=1)]

    return train_df, val_df, test_df

def plot_confusion_matrix(model, data, device='cuda', label_encoder=None):
    model.eval()

    X, y = data
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)

    true_labels = torch.argmax(y, dim=1).cpu().numpy()
    predicted_labels = predicted.cpu().numpy()

    # Get unique class labels
    classes = label_encoder.classes_ if label_encoder else np.unique(true_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

    # Get display labels
    tick_labels = (label_encoder.inverse_transform(classes)
                   if label_encoder else classes)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Gait Classification')
    plt.show()


def visualize_model(model, input_size):
    return draw_graph(
        model,
        input_size=input_size,  # (batch, seq_len, features)
        device='meta',  # No memory usage
        depth=2,  # Expand nested layers
        show_shapes=True
    )


if __name__ == '__main__':
    # Set device to local GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load and split data
    gait_data = load_data()
    print(gait_data.head())
    metrics = [
        'RightHipAngle', 'LeftHipAngle',
        'RightKneeAngle', 'LeftKneeAngle',
        'BodyLeanAngle', 'HeadTiltAngle'
    ]

    # Create frame index within each video
    gait_data = gait_data.reset_index(drop=True)
    gait_data['FrameIndex'] = gait_data.groupby('VideoID').cumcount()

    # Compute mean and median
    mean_curves = gait_data.groupby(['GaitType', 'FrameIndex'])[metrics].mean().reset_index()
    median_curves = gait_data.groupby(['GaitType', 'FrameIndex'])[metrics].median().reset_index()

    # Plot per GaitType
    for gait_type in mean_curves['GaitType'].unique():
        plt.figure(figsize=(12, 6))
        mean_subset = mean_curves[mean_curves['GaitType'] == gait_type]
        median_subset = median_curves[median_curves['GaitType'] == gait_type]

        for metric in metrics:
            # Plot mean as solid line
            plt.plot(mean_subset['FrameIndex'], mean_subset[metric], label=f'{metric} (mean)', linestyle='-')
            # Plot median as dashed line
            # plt.plot(median_subset['FrameIndex'], median_subset[metric], label=f'{metric} (median)', linestyle='--')

        plt.title(f'Mean Angles for {gait_type}')
        plt.xlabel('Frame Index')
        plt.ylabel('Angle (Degrees)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # train_df, val_df, test_df = custom_split_by_video_id(gait_data)
    # X_train, y_train = split_data(train_df)
    # X_val, y_val = split_data(val_df)
    # X_test, y_test = split_data(test_df)
    # # Hyper parameter settings for LSTM
    # input_size = X_train.shape[-1]
    # sequence_length = X_train.shape[2]
    # hidden_size = input_size
    # num_layers = 2
    # output_size = y_train.shape[-1]
    # # LSTM Declaration
    # lstm = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # print(lstm)
    # summary(lstm, input_size=(32, sequence_length, input_size))
    # dummy_input = torch.randn(32, sequence_length, input_size).to(device)
    # lstm = lstm.to(device)
    # output = lstm(dummy_input)
    # writer = SummaryWriter('runs/model_experiment')
    # writer.add_graph(lstm, dummy_input)
    # writer.close()
    # train_lstm(lstm, (X_train, y_train), (X_val, y_val),
    #            sequence_length, input_size, batch_size=32, num_epochs=100, learning_rate=0.001, device='cuda')
    # plot_confusion_matrix(lstm, (X_val, y_val), device='cuda')
    # plot_confusion_matrix(lstm, (X_test, y_test), device='cuda')