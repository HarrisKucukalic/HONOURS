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
    all_generators_list = []
    print("Starting data consolidation...")
    for file_name in files:
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        try:
            # Extract State and Type from the filename
            # Example: "NSW-SOLAR.xlsx" -> ["NSW", "SOLAR"]
            parts = file_name.replace('.xlsx', '').split('-')
            state = parts[0]
            # Capitalise to match 'Solar' or 'Wind'
            gen_type = parts[1].capitalize()
            print(f"Processing {file_name} for State: {state}, Type: {gen_type}")
            # Read all sheets from the Excel file into a dictionary of DataFrames
            # The key is the sheet name, the value is the DataFrame
            sheets_dict = pd.read_excel(file_path, sheet_name=None)
            # Loop through each sheet's DataFrame in the dictionary
            for sheet_name, sheet_df in sheets_dict.items():
                print(f"  - Reading sheet: {sheet_name}")
                # Add the state and type columns
                sheet_df['State'] = state
                # In case the 'Type' column doesn't exist or is inconsistent
                sheet_df['Type'] = gen_type
                # Append the processed DataFrame to our master list
                all_generators_list.append(sheet_df)
        except FileNotFoundError:
            print(f"--- WARNING: {file_name} not found. Skipping. ---")
            continue
        except Exception as e:
            print(f"An error occurred with {file_name}: {e}")
            continue

    if not all_generators_list:
        print("No data was loaded. Please check file paths and names.")
        return pd.DataFrame()

    # Concatenate all the DataFrames in the list into a single DataFrame
    master_df = pd.concat(all_generators_list, ignore_index=True)

    # Standardize column names (e.g., 'Gen. Capa' -> 'Gen_Capa')
    master_df.columns = master_df.columns.str.replace('.', '', regex=False).str.replace(' ', '_')

    print("\nConsolidation complete.")
    return master_df



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

def train_logistic_regression_model(df: pd.DataFrame, feature_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
    print("\n--- Training Logistic Regression Model ---")

    # 1. Define Target and Features
    y = df['Curtailment_Event']
    X = pd.get_dummies(df[feature_cols + categorical_cols], columns=categorical_cols, drop_first=True)

    # Store column order for later predictions
    model_columns = X.columns.tolist()

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Scale Numerical Features
    scaler = StandardScaler()
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    # 4. Train Model
    # Using class_weight='balanced' is helpful if curtailment events are rare
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate Model
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    y_pred = model.predict(X_test)

    print("\n--- Model Performance ---")
    auc_score = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")
    print(f"Brier Score Loss: {brier:.4f} (lower is better)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'model': model,
        'scaler': scaler,
        'model_columns': model_columns,
        'metrics': {
            'roc_auc': auc_score,
            'brier_score': brier,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    }
if __name__ == '__main__':
    final_generators_df = consolidate_generator_data(DATA_DIRECTORY, EXCEL_FILES)
    print(final_generators_df.head())
    # # --- Example 1: Use the Conditional Probability function ---
    # prob = calculate_conditional_probability(
    #     df=final_generators_df,
    #     location='Gullen Range',
    #     condition_query='WindSpeed > 60 and hour < 6'  # high wind in early morning
    # )
    # if not np.isnan(prob):
    #     print(f"Result: The probability is {prob:.2%}")
    #
    # # --- Example 2: Use the Logistic Regression function ---
    # model_artifacts = train_logistic_regression_model(
    #     df=final_generators_df,
    #     feature_cols=['RRP', 'WindSpeed', 'SolarGHI', 'hour'],
    #     categorical_cols=['Location']
    # )
    #
    # # --- Example 3: Use the trained model to predict on new data ---
    # print("\n--- Predicting on new hypothetical data ---")
    # # New data must have the same structure as the training data
    # new_data = pd.DataFrame({
    #     'RRP': [-120],
    #     'WindSpeed': [75],
    #     'SolarGHI': [0],
    #     'hour': [3],
    #     'Location': ['Gullen Range']
    # })
    #
    # # Get the trained model components
    # trained_model = model_artifacts['model']
    # scaler = model_artifacts['scaler']
    # model_cols = model_artifacts['model_columns']
    #
    # # Prepare the new data exactly like the training data
    # new_data_encoded = pd.get_dummies(new_data, columns=['Location'], drop_first=True)
    # # Ensure all model columns are present, fill missing with 0
    # new_data_aligned = new_data_encoded.reindex(columns=model_cols, fill_value=0)
    # # Scale the numerical features
    # new_data_aligned[['RRP', 'WindSpeed', 'SolarGHI', 'hour']] = scaler.transform(
    #     new_data_aligned[['RRP', 'WindSpeed', 'SolarGHI', 'hour']])
    #
    # # Make the prediction
    # new_prediction_prob = trained_model.predict_proba(new_data_aligned)[:, 1]
    # print(f"Predicted curtailment likelihood for new data: {new_prediction_prob[0]:.2%}")