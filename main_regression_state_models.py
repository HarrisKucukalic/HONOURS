from models.ann import ANNModel
from models.ann_pso import ANN_PSO_Model
from models.random_forest import RandomForestModel
from models.transformer import TransformerModel
from models.transformer_pso import Transformer_PSO_Model
from models.xgboost_model import XGBoostModel
from models.LSTM_model import LSTMModel
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer


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


def fit_transformer(data, method='robust'):
    # Initialise and fit the chosen scaler on the clipped training data
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = PowerTransformer(method='yeo-johnson')
    scaler.fit(data)
    return scaler


def transform_data(data, scaler):
    return scaler.transform(data)


def inverse_transform_data(data, scaler):
    return scaler.inverse_transform(data)


if __name__ == '__main__':
    STATE_FILES = [
        'Victoria_master_with_weather.csv',
        'New South Wales_master_with_weather.csv',
        'Queensland_master_with_weather.csv'
    ]

    MODELS_TO_RUN = [
        # RandomForestModel,
        # XGBoostModel,
        # ANNModel,
        # LSTMModel,
        # ANN_PSO_Model,
        # TransformerModel,
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
        df['RRP_volatility_6hr'] = df['RRP'].rolling(window=72).std().shift(1)
        if 'RRP_volatility_6hr' not in FEATURE_COLS:
            FEATURE_COLS.append('RRP_volatility_6hr')
        df = df.bfill()
        print("With previous 6 hrs (6X12 = 72) RRP Volatility:")
        print(df.head())
        # 3D Sequence, includes time, for transformers & LSTM
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size: train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        # Transformation of data:
        print("Applying data transformation to RRP target...")
        scaler = fit_transformer(
            train_df[[TARGET_COL]].values,
            method='power'
        )
        # Store original test labels for final evaluation
        y_test_original_flat = test_df[TARGET_COL].values
        y_test_original_seq = test_df[TARGET_COL].iloc[SEQUENCE_LENGTH:].values
        train_df[TARGET_COL] = transform_data(train_df[[TARGET_COL]].values, scaler)
        val_df[TARGET_COL] = transform_data(val_df[[TARGET_COL]].values, scaler)
        test_df[TARGET_COL] = transform_data(test_df[[TARGET_COL]].values, scaler)

        X_train_seq, y_train_seq = create_sequences(train_df, FEATURE_COLS, TARGET_COL, SEQUENCE_LENGTH)
        X_val_seq, y_val_seq = create_sequences(val_df, FEATURE_COLS, TARGET_COL, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = create_sequences(test_df, FEATURE_COLS, TARGET_COL, SEQUENCE_LENGTH)
        print(f"Created sequential train data. X_train_seq shape: {X_train_seq.shape}")
        # 2D sequence, for other models
        X_train_flat = train_df.drop(columns=[TARGET_COL])
        y_train_flat = train_df[TARGET_COL]
        X_val_flat = val_df.drop(columns=[TARGET_COL])
        y_val_flat = val_df[TARGET_COL]
        X_test_flat = test_df.drop(columns=[TARGET_COL])
        y_test_flat = test_df[TARGET_COL]
        print(f"Created flat train data. X_train_flat shape: {X_train_flat.shape}")

        # Inner Loop: Run Models
        print(f"\n--- Training models for {state_name} ---")
        for model_class in MODELS_TO_RUN:

            model_class_name = model_class.__name__
            if "Transformer" in model_class_name or "LSTMModel" in model_class_name:
                X_train, y_train = X_train_seq, y_train_seq
                X_val, y_val = X_val_seq, y_val_seq
                X_test, y_test = X_test_seq, y_test_seq
                y_true_original_for_eval = y_test_original_seq
                input_dim = X_train_seq.shape[2]
                model = model_class(region=state_name, input_shape=input_dim)
            elif "ANN" in model_class_name:
                X_train, y_train = X_train_flat.values, y_train_flat.values
                X_val, y_val = X_val_flat.values, y_val_flat.values
                X_test, y_test = X_test_flat.values, y_test_flat.values
                y_true_original_for_eval = y_test_original_flat
                input_shape = X_train_flat.shape[1]
                model = model_class(region=state_name, input_shape=input_shape)
            else:
                X_train, y_train = X_train_flat, y_train_flat
                X_val, y_val = X_val_flat, y_val_flat
                X_test, y_test = X_test_flat, y_test_flat
                y_true_original_for_eval = y_test_original_flat
                model = model_class(region=state_name)

            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_dir = os.path.join('model_results', state_name, model_class_name)
            results_dir = os.path.join(base_dir, run_name)
            os.makedirs(results_dir, exist_ok=True)
            if "ANN_PSO_Model" in model_class_name:
                print(f"Detected ANN PSO model. Starting hyperparameter search...")
                MAX_LAYERS = 5
                NEURON_POWER_MIN, NEURON_POWER_MAX = 4, 9
                search_space = {
                    'num_layers': (1, MAX_LAYERS),
                    **{f'neurons_power_layer_{i}': (NEURON_POWER_MIN, NEURON_POWER_MAX) for i in range(MAX_LAYERS)},
                    'learning_rate': (1e-4, 1e-2),
                    'epochs': (50, 150),
                    'batch_size_power': (5, 7)
                }
                model.run_training_loop(
                    X_train, y_train, X_val, y_val,
                    search_space=search_space, pso_n_particles=10, pso_iters=5
                )

            elif model_class_name == "Transformer_PSO_Model":
                print("Detected Transformer PSO model. Starting hyperparameter search...")
                search_space = {
                    'model_dim': (32, 128),
                    'n_heads': (1, 3),
                    'n_encoder_layers': (1, 4),
                    'ff_dim': (32, 512),
                    'learning_rate': (1e-6, 1e-4),
                    'epochs': (20, 200),
                    'batch_size_power': (2, 4)
                }
                model.train_model(
                    X_train, y_train, X_val, y_val,
                    search_space=search_space, pso_n_particles=10, pso_iters=5
                )
            elif model_class_name in ["TransformerModel", "ANNModel", "LSTMModel"]:
                print(f"Detected standard model: {model_class_name}. Starting training...")
                model.train_model(X_train, y_train, X_val, y_val)
            else:
                print(f"Detected standard model: {model_class_name}. Starting training...")
                model.train_model(X_train, y_train)

            print("\n--- Evaluating and Saving Final Model ---")
            # Get predictions on the TRANSFORMED data
            _, y_true_transformed, y_pred_transformed = model.evaluate(X_test, y_test)
            # INVERSE TRANSFORM the predictions back to the original RRP scale
            print("Inverse transforming predictions to original RRP scale...")
            y_pred_original = inverse_transform_data(
                y_pred_transformed.reshape(-1, 1),
                scaler
            ).flatten()

            print("Recalculating metrics in true dollar values...")
            mae_dollars = mean_absolute_error(y_true_original_for_eval, y_pred_original)
            rmse_dollars = np.sqrt(mean_squared_error(y_true_original_for_eval, y_pred_original))
            results_in_dollars = {
                "MAE": mae_dollars,
                "RMSE": rmse_dollars
            }
            print(f"Final Results -> MAE (dollars): ${mae_dollars:,.2f}, RMSE (dollars): ${rmse_dollars:,.2f}")
            # Call save_results with the ORIGINAL true values and ORIGINAL-SCALE predictions
            model.save_results(
                results_in_dollars,
                y_true_original_for_eval,  # The original, untransformed test labels you saved earlier
                y_pred_original,  # The newly inverse-transformed predictions
                results_dir
            )

            # Save the actual model weights if it was a PSO model
            if "PSO" in model_class_name:
                model.save_model(results_dir)

            print("-" * 50)

    print("\n\nAll states and models processed successfully.")
