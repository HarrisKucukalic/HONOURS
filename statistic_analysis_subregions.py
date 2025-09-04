import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    brier_score_loss,
    roc_curve
)
import xgboost as xgb
import gc
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibrationDisplay
DATA_DIRECTORY = r'C:\projects\HONOURS\historical weather data'
EXCEL_FILES = [
    'NSW-SOLAR.xlsx',
    'NSW-WIND.xlsx',
    'QLD-SOLAR.xlsx',
    'QLD-WIND.xlsx',
    'VIC-SOLAR.xlsx',
    'VIC-WIND.xlsx'
]


def consolidate_generator_data(directory: str, files: list) -> pd.DataFrame:
    # Initialize an empty DataFrame to build upon.
    master_df = pd.DataFrame()
    print("Starting data consolidation...")

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        try:
            parts = file_name.replace('.xlsx', '').split('-')
            state = parts[0]
            gen_type = parts[1].capitalize()
            print(f"Processing {file_name} for State: {state}, Type: {gen_type}")

            sheets_dict = pd.read_excel(file_path, sheet_name=None)

            # Create a temporary list for sheets from THIS FILE ONLY
            current_file_sheets = []

            for sheet_name, sheet_df in sheets_dict.items():
                print(f"  - Reading sheet: {sheet_name}")
                sheet_df['Location'] = sheet_name
                sheet_df['State'] = state
                sheet_df['Type'] = gen_type

                # Clean up column names FIRST
                sheet_df.columns = sheet_df.columns.str.replace('.', '', regex=False).str.replace(' ', '_')
                sheet_df.columns = sheet_df.columns.str.replace(r'[\(\)°/]', '', regex=True)

                # Find the likely datetime column (often named 'time' or 'DateTime')
                datetime_col = None
                if 'time' in sheet_df.columns:
                    datetime_col = 'time'
                elif 'DateTime' in sheet_df.columns:
                    datetime_col = 'DateTime'

                if datetime_col:
                    sheet_df[datetime_col] = pd.to_datetime(sheet_df[datetime_col])

                # Efficiently convert data types to save memory
                for col in sheet_df.columns:
                    if sheet_df[col].dtype == 'float64':
                        sheet_df[col] = sheet_df[col].astype('float32')
                    if sheet_df[col].dtype == 'int64':
                        sheet_df[col] = pd.to_numeric(sheet_df[col], downcast='integer')

                sheet_df['Location'] = sheet_df['Location'].astype('category')
                sheet_df['State'] = sheet_df['State'].astype('category')
                sheet_df['Type'] = sheet_df['Type'].astype('category')

                current_file_sheets.append(sheet_df)

            # Concatenate the sheets from the current file
            if current_file_sheets:
                file_df = pd.concat(current_file_sheets, ignore_index=True)
                # Now, concatenate this file's data to the master DataFrame
                master_df = pd.concat([master_df, file_df], ignore_index=True)

        except FileNotFoundError:
            print(f"--- WARNING: {file_name} not found. Skipping. ---")
            continue
        except Exception as e:
            print(f"An error occurred with {file_name}: {e}")
            continue

    if master_df.empty:
        print("No data was loaded. Please check file paths and names.")
        return pd.DataFrame()

    print("\nConsolidation complete.")
    return master_df
def consolidate_state_rrp_data(directory: str, state_files: dict) -> pd.DataFrame:
    all_states_list = []
    print("--- Starting state-level RRP consolidation and downsampling ---")
    for file_name, state_code in state_files.items():
        file_path = os.path.join(directory, file_name)
        try:
            print(f"Processing {file_name} for State: {state_code}")
            # Specify dtypes on read for efficiency
            df = pd.read_csv(file_path, usecols=['DateTime', 'RRP'],
                           dtype={'RRP': 'float32'})
            df['State'] = state_code
            all_states_list.append(df)
        except (FileNotFoundError, KeyError) as e:
            print(f"--- WARNING: Could not process {file_name}. Error: {e}. Skipping. ---")
            continue

    if not all_states_list:
        print("No state data was loaded.")
        return pd.DataFrame()

    master_df = pd.concat(all_states_list, ignore_index=True)

    master_df['DateTime'] = pd.to_datetime(master_df['DateTime'])
    # Convert State to category type after concatenation
    master_df['State'] = master_df['State'].astype('category')
    master_df = master_df.set_index('DateTime')

    hourly_list = []
    for state, group_df in master_df.groupby('State'):
        hourly_df = group_df[['RRP']].resample('h').mean()
        hourly_df['State'] = state
        hourly_list.append(hourly_df)

    hourly_master_df = pd.concat(hourly_list)
    hourly_master_df.reset_index(inplace=True)

    print("\nState-level RRP consolidation and hourly aggregation complete.")
    return hourly_master_df


def create_final_dataset(hourly_state_df: pd.DataFrame, hourly_local_weather_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Starting final hourly data merging process ---")

    # 1. Ensure datetime types are consistent
    hourly_state_df['DateTime'] = pd.to_datetime(hourly_state_df['DateTime'])
    hourly_local_weather_df['time'] = pd.to_datetime(hourly_local_weather_df['time'])

    # --- OPTIMIZATION START ---
    # Sort both dataframes by the keys you are merging on.
    # This dramatically improves performance and reduces memory usage.
    print("Sorting dataframes before merge...")
    hourly_local_weather_df.sort_values(by=['State', 'time'], inplace=True)
    hourly_state_df.sort_values(by=['State', 'DateTime'], inplace=True)
    # --- OPTIMIZATION END ---

    # 2. Perform the merge on the sorted data
    print("Merging dataframes...")
    final_df = pd.merge(
        hourly_local_weather_df,
        hourly_state_df,
        left_on=['State', 'time'],      # Match the sort order
        right_on=['State', 'DateTime'], # Match the sort order
        how='inner'
    )

    # 3. Clean up the final DataFrame
    final_df.drop(columns=['DateTime'], inplace=True, errors='ignore')
    final_df.rename(columns={'time': 'DateTime'}, inplace=True)

    print("\n--- Data merging process complete ---")
    return final_df


def calculate_conditional_probability(df: pd.DataFrame, location: str, condition_query: str) -> float:
    print(f"\n--- Calculating Conditional Probability for '{location}' ---")
    print(f"Condition: {condition_query}")

    # Filter for the specific location
    location_df = df[df['Location'] == location].copy()

    # Apply the condition using the query method
    condition_met_df = location_df.query(condition_query)
    total_condition_events = len(condition_met_df)

    if total_condition_events == 0:
        print("Warning: The specified condition never occurred in the dataset for this location.")
        return np.nan

    # Count how many of those events also had a curtailment event
    curtailment_with_condition = condition_met_df['Curtailment_Event'].sum()

    # Calculate the probability
    probability = curtailment_with_condition / total_condition_events
    return probability


def plot_probability_distribution(df: pd.DataFrame, attribute: str, filter_by: str, filter_value: str,
                                  num_bins: int = 15, output_dir: str = "subregion_statistical_analysis"):
    print(f"\n--- Generating Probability Distribution for '{attribute}' filtered by {filter_by} = '{filter_value}' ---")

    filtered_df = df[df[filter_by] == filter_value].copy()

    if filtered_df.empty:
        print(f"Warning: No data found for {filter_by} = '{filter_value}'. Skipping plot.")
        return

    # Create bins for the attribute
    filtered_df['bins'] = pd.cut(filtered_df[attribute], bins=num_bins)

    # Calculate the probability of curtailment in each bin
    prob_df = filtered_df.groupby('bins', observed=True)['Curtailment_Event'].mean().reset_index()
    prob_df.rename(columns={'Curtailment_Event': 'Probability'}, inplace=True)

    if len(prob_df) < 2:
        print(
            f"Warning: Not enough data points to create a distribution plot for {attribute} at {filter_value}. Skipping.")
        return

    # Prepare for plotting
    plt.figure(figsize=(12, 7))
    prob_df['bins_mid'] = prob_df['bins'].apply(lambda x: x.mid)

    bin_width = (prob_df['bins_mid'].iloc[1] - prob_df['bins_mid'].iloc[0]) * 0.9 if len(prob_df) > 1 else 1

    plt.bar(prob_df['bins_mid'], prob_df['Probability'], width=bin_width)

    plt.title(f'Probability of Curtailment vs. {attribute} for {filter_by}: {filter_value}')
    plt.xlabel(f'{attribute} Bins')
    plt.ylabel('Probability of Curtailment Event')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{filter_value}_{attribute}_distribution.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    print(f"✅ Distribution plot saved to '{os.path.join(output_dir, plot_filename)}'")
    plt.close()


def train_logistic_regression_model(
        df: pd.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        output_dir: str = "subregion_statistical_analysis"
) -> Dict[str, Any]:
    print("\n--- Training Logistic Regression Model ---")

    # --- ADD THIS CHECK ---
    # Ensure there's enough data to train a model and perform a split
    if df.shape[0] < 100 or df['Curtailment_Event'].nunique() < 2:
        print(f"⚠️ WARNING: Skipping model training. Not enough data or only one class present.")
        print(f"Data points: {df.shape[0]}, Unique classes: {df['Curtailment_Event'].nunique()}")
        return None  # Return None to indicate that training was skipped

    df_clean = df.dropna(subset=feature_cols + categorical_cols).copy()
    print(f"Cleaned data for training. New shape: {df_clean.shape}")

    # Another check after dropping NAs
    if df_clean.shape[0] < 100 or df_clean['Curtailment_Event'].nunique() < 2:
        print(f"⚠️ WARNING: Skipping model training after dropping NAs. Not enough data or only one class present.")
        return None

    y = df_clean['Curtailment_Event']
    # Use the cleaned dataframe for X
    X = pd.get_dummies(df_clean[feature_cols + categorical_cols], columns=categorical_cols, drop_first=True)

    # ... (the rest of the function remains the same) ...
    model_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()

    # Create a copy to avoid SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    model = LogisticRegression(class_weight='balanced', random_state=42,
                               max_iter=1000)  # Increased max_iter for convergence
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    print("\n--- Model Performance ---")
    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(report_str)

    if output_dir:
        print(f"\n--- Saving results to '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(model_columns, os.path.join(output_dir, 'model_columns.joblib'))  # Save columns
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report_str)
        print("✅ Model, scaler, columns, and report saved.")

    return {'model': model, 'scaler': scaler, 'metrics': report_dict, 'model_columns': model_columns}


def train_xgboost_model(
        df: pd.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        output_dir: str = "xgboost_models"
) -> Dict[str, Any]:
    # ... (all the data prep, splitting, and scaling code remains exactly the same) ...

    y = df['Curtailment_Event']
    X = pd.get_dummies(df[feature_cols + categorical_cols], columns=categorical_cols, drop_first=True)
    model_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    print(f"Training XGBoost with scale_pos_weight: {scale_pos_weight:.2f}")
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- NEW: EVALUATION BASED ON PROBABILITY ---
    # Get the predicted probabilities for the 'positive' class (Curtailment = 1)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    # Get the binary predictions (this uses a default 0.5 threshold)
    y_pred_binary = model.predict(X_test_scaled)

    # Calculate probability-based metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    brier_loss = brier_score_loss(y_test, y_pred_proba)

    print("\n--- Model Performance ---")
    print(f"ROC AUC Score: {auc_score:.4f}")
    print(f"Brier Score Loss: {brier_loss:.4f} (lower is better)")
    print("\nClassification Report (based on 0.5 threshold):")
    report_str = classification_report(y_test, y_pred_binary)
    print(report_str)
    # --- END NEW SECTION ---

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Append new scores to the report
        full_report_str = (
            f"ROC AUC Score: {auc_score:.4f}\n"
            f"Brier Score Loss: {brier_loss:.4f}\n\n"
            f"{report_str}"
        )
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(full_report_str)

        # --- NEW: Save a Calibration Plot ---
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(y_test, y_pred_proba, n_bins=10, ax=ax, name="XGBoost")
        plt.title(f"Calibration Plot for {output_dir.split('/')[-1]}")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'calibration_plot.png'))
        plt.close(fig)
        # --- END NEW SECTION ---

        joblib.dump(model, os.path.join(output_dir, 'xgboost_model.joblib'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(model_columns, os.path.join(output_dir, 'model_columns.joblib'))
        print(f"\n✅ XGBoost Model, scaler, columns, report, and calibration plot saved to '{output_dir}'.")

    return {'model': model, 'scaler': scaler, 'metrics': {'roc_auc': auc_score, 'brier_loss': brier_loss}}

if __name__ == '__main__':
    STATE_DATA_DIRECTORY = '.'
    GROUP_FILE_MAP = {
        ('NSW', 'SOLAR'): {
            'generator_file': 'NSW-SOLAR.xlsx',
            'rrp_file': 'New South Wales_master_with_weather.csv'
        },
        ('NSW', 'WIND'): {
            'generator_file': 'NSW-WIND.xlsx',
            'rrp_file': 'New South Wales_master_with_weather.csv'
        },
        ('QLD', 'SOLAR'): {
            'generator_file': 'QLD-SOLAR.xlsx',
            'rrp_file': 'Queensland_master_with_weather.csv'
        },
        ('QLD', 'WIND'): {
            'generator_file': 'QLD-WIND.xlsx',
            'rrp_file': 'Queensland_master_with_weather.csv'
        },
        ('VIC', 'SOLAR'): {
            'generator_file': 'VIC-SOLAR.xlsx',
            'rrp_file': 'Victoria_master_with_weather.csv'
        },
        ('VIC', 'WIND'): {
            'generator_file': 'VIC-WIND.xlsx',
            'rrp_file': 'Victoria_master_with_weather.csv'
        }
    }
    all_models_artifacts = {}

    for (state, gen_type), files in GROUP_FILE_MAP.items():
        model_name = f"{state}_{gen_type}"
        print(f"\n{'=' * 30}")
        print(f"   PROCESSING GROUP: {model_name.upper()}")
        print(f"{'=' * 30}\n")

        # --- Step 1: Load ONLY the data for the current group ---
        print(f"--- Loading data for {model_name} ---")
        # The 'files' argument to consolidate_generator_data is now a list with just one file
        local_weather_df = consolidate_generator_data(DATA_DIRECTORY, [files['generator_file']])

        # We also load only the single, relevant state RRP file
        master_rrp_df = consolidate_state_rrp_data(STATE_DATA_DIRECTORY, {files['rrp_file']: state})

        # --- Step 2: Merge the (much smaller) dataframes ---
        final_combined_df = create_final_dataset(
            hourly_state_df=master_rrp_df,
            hourly_local_weather_df=local_weather_df
        )

        # --- Step 3: Feature Engineering ---
        final_combined_df['Curtailment_Event'] = (final_combined_df['RRP'] < -40).astype(int)
        final_combined_df['hour'] = final_combined_df['DateTime'].dt.hour

        # --- Step 4: Train the model for this group ---
        output_folder = os.path.join("subregion_models_xgboost", model_name)

        print(f"\n--- Training XGBoost model for {model_name.upper()} ---")
        model_artifacts = train_xgboost_model(
            df=final_combined_df,
            feature_cols=['RRP', 'Temp_C', 'Wind_Speed_100m_kmh', 'hour'],
            categorical_cols=[],
            output_dir=output_folder
        )

        if model_artifacts:
            all_models_artifacts[model_name] = model_artifacts

        # --- Step 5: Clean up memory before the next loop ---
        del local_weather_df, master_rrp_df, final_combined_df
        gc.collect()

        # --- Optional: Print a summary of all trained models ---
    print(f"\n\n{'=' * 30}")
    print("   MODEL TRAINING SUMMARY")
    print(f"{'=' * 30}\n")

    for model_name, artifacts in all_models_artifacts.items():
        f1_score = artifacts['metrics'].get('1', {}).get('f1-score', 'N/A')
        if f1_score != 'N/A':
            print(f"✅ Model: {model_name:<12} | F1-Score (for Curtailment): {f1_score:.4f}")
        else:
            print(f"ℹ️ Model: {model_name:<12} | Metrics not available.")






