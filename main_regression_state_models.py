from random_forest import RandomForestModel
from xgboost_model import XGBoostModel
from ann import ANNModel
from ann_pso import ANN_PSO_Model
from transformer import TransformerModel
from transformer_pso import Transformer_PSO_Model
import numpy as np
import pandas as pd
import os
import shutil
import re
from sklearn.model_selection import train_test_split


def create_sequences(data, feature_cols, target_col, sequence_length):
    """Creates 3D sequences from time-series data."""
    X_sequences = []
    y_sequences = []

    data_features = data[feature_cols].values
    data_target = data[target_col].values

    for i in range(len(data) - sequence_length):
        X_sequences.append(data_features[i:i + sequence_length])
        y_sequences.append(data_target[i + sequence_length])

    return np.array(X_sequences), np.array(y_sequences)

def clean_col_names(df):
    """Cleans DataFrame column names to be valid Python identifiers."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', col) # Remove special chars
        new_col = new_col.replace('__', '_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

if __name__ == '__main__':
    STATE_FILES = [
        'Victoria_master_with_weather.csv',
        'New South Wales_master_with_weather.csv',
        'Queensland_master_with_weather.csv'
    ]

    MODELS_TO_RUN = [
        RandomForestModel,
        XGBoostModel,
        ANNModel,
        ANN_PSO_Model,
        TransformerModel,
        Transformer_PSO_Model
    ]

    FEATURE_COLS = [
        'WindMW',
        'SolarUtilityMW',
        'SolarRooftopMW',
        'TempC',
        'PressurehPa',
        'CloudCover',
        'WindSpeed10mkmh',
        'WindSpeed100mkmh',
        'SolarGHIWm',
        'SolarDNIWm'
    ]
    TARGET_COL = 'RRP'
    # The sequence length here will be a day, or 288 time frames. Each sequence will be made of 7 of these to make a full week
    SEQUENCE_LENGTH = 288

    if os.path.exists('model_results'):
        print("Cleaning up old results directory...")
        shutil.rmtree('model_results')

    for state_file in STATE_FILES:
        try:
            state_name = state_file.split('_')[0]
            print(f"\n{'=' * 50}")
            print(f"PROCESSING STATE: {state_name}")
            print(f"{'=' * 50}")

            print(f"Loading data from {state_file}...")
            df = pd.read_csv(state_file, parse_dates=['DateTime'], index_col='DateTime')
            df = clean_col_names(df)

            print(f"Cleaned Columns for {state_name}: {df.columns.tolist()}")

        except FileNotFoundError:
            print(f"--- WARNING: {state_file} not found. Skipping. ---")
            continue
        except Exception as e:
            print(f"An error occurred while loading {state_file}: {e}")
            continue

        print("Sample of loaded data with cleaned columns:")
        print(df.head())
        # 3D Sequence, includes time, for transformers
        X_seq, y_seq = create_sequences(
            data=df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            sequence_length=SEQUENCE_LENGTH
        )
        print(f"Created sequential data (3D). X_seq shape: {X_seq.shape}")
        # non-temporal
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
        print(f"Created non-temporal data {X.shape}, y shape: {y.shape}")

        # --- 70:15:15 TRAIN-VALIDATION-TEST SPLIT ---
        # First, split into 70% train and 30% temporary (for val/test)
        X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False, random_state=42)
        # Then, split the 30% temporary set in half to get 15% val and 15% test
        X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
            X_temp_seq, y_temp_seq, test_size=0.5, shuffle=False, random_state=42)

        # Do the same for the flattened data
        X_train_flat, X_temp_flat, y_train_flat, y_temp_flat = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42)
        X_val_flat, X_test_flat, y_val_flat, y_test_flat = train_test_split(
            X_temp_flat, y_temp_flat, test_size=0.5, shuffle=False, random_state=42)

        # Inner Loop: Run Models
        print(f"\n--- Training models for {state_name} ---")
        for model_class in MODELS_TO_RUN:

            model_class_name = model_class.__name__
            if "Transformer" in model_class_name:
                X_train, y_train = X_train_seq, y_train_seq
                X_val, y_val = X_val_seq, y_val_seq
                X_test, y_test = X_test_seq, y_test_seq
                model = model_class(region=state_name)
            elif "ANN" in model_class_name:
                X_train, y_train = X_train_flat.values, y_train_flat.values
                X_val, y_val = X_val_flat.values, y_val_flat.values
                X_test, y_test = X_test_flat.values, y_test_flat.values
                input_shape = X_train_flat.shape[1]
                model = model_class(region=state_name, input_shape=input_shape)
            else:
                X_train, y_train = X_train_flat, y_train_flat
                X_val, y_val = X_val_flat, y_val_flat
                X_test, y_test = X_test_flat, y_test_flat
                model = model_class(region=state_name)

            results_dir = os.path.join('model_results', state_name, model_class_name)
            if "ANN_PSO_Model" in model_class_name:
                print(f"Detected PSO model. Starting hyperparameter search...")
                MAX_LAYERS = 5
                NEURON_POWER_MIN, NEURON_POWER_MAX = 4, 9
                search_space = {
                    # 1. A parameter to control the number of layers (from 1 to MAX_LAYERS)
                    'num_layers': (1, MAX_LAYERS),

                    # 2. A parameter for each potential layer's neuron count
                    **{f'neurons_power_layer_{i}': (NEURON_POWER_MIN, NEURON_POWER_MAX) for i in range(MAX_LAYERS)},

                    # 3. Your other parameters
                    'learning_rate': (1e-4, 1e-2),
                    'epochs': (50, 150),
                    'batch_size_power': (5, 7)  # For batch sizes 32, 64, 128
                }
                model.run_training_loop(
                    X_train, y_train,
                    X_val, y_val,
                    search_space=search_space,
                    pso_n_particles=20,
                    pso_iters=10
                )
            if model_class_name == "Transformer_PSO_Model":
                print("Detected Transformer PSO model. Setting up search space...")

                # Search space for Transformer hyperparameters
                search_space = {
                    'd_model': (64, 256),  # Embedding dimension
                    'n_heads_power': (1, 3),  # For 2, 4, or 8 heads
                    'num_encoder_layers': (1, 6),  # Number of transformer blocks
                    'dim_feedforward': (128, 1024),  # Size of the feedforward network

                    # Common training hyperparameters
                    'learning_rate': (1e-5, 1e-3),
                    'epochs': (20, 100),
                    'batch_size_power': (4, 6)  # For batch sizes 16, 32, 64
                }

                # Call the training loop with this search space
                model.train_model(
                    X_train, y_train, X_val, y_val,
                    search_space=search_space,
                    pso_n_particles=20,
                    pso_iters=10
                )
            else:
                print(f"Detected standard model. Starting training...")
                # Standard models use train_model
                model.train_model(X_train, y_train, X_val, y_val)
            results = model.evaluate(X_test, y_test)
            model.save_results(results, results_dir)
            print("-" * 40)

    print("\n\nAll states and models processed successfully.")
