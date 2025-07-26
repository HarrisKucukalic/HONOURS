import pandas as pd
import os
import glob

# --- Configuration ---

# The states to process.
STATES = ['Victoria', 'New South Wales', 'Queensland']

# Suffixes for the files we need to find and the final output file.
PROCESSED_SUFFIX = ' processed.csv'  # This file contains the weather data
FINAL_SUFFIX = '_final.csv'

# Directory where the script will look for the input files.
# '.' means the current directory where the script is running.
SOURCE_DIRECTORY = r'C:\projects\HONOURS\day_to_5min'

# The factor to multiply the temperature data by to revert it to its original scale.
TEMP_REVERSION_FACTOR = 288.0


# --- Pre-processing Function ---

def preprocess_weather_data(file_path):
    """
    Loads a '_processed.csv' file, converts energy from GWh to MWh,
    reverts temperature calculations, and returns a processed DataFrame.
    """
    print(f"Pre-processing weather file: '{os.path.basename(file_path)}'")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # --- 1. Convert GWh columns to MWh ---
    gwh_cols = [col for col in df.columns if 'GWh' in col]
    if gwh_cols:
        print(f"Found GWh columns to convert: {', '.join(gwh_cols)}")
        df[gwh_cols] = df[gwh_cols] * 1000
        # Rename columns to reflect the new unit
        df.rename(columns={col: col.replace('GWh', 'MWh') for col in gwh_cols}, inplace=True)
        print("Converted GWh columns to MWh and updated column names.")
    else:
        print("No GWh columns found for conversion.")

    # --- 2. Revert Temperature columns ---
    temp_cols = [col for col in df.columns if 'Temp' in col]
    if temp_cols:
        print(f"Found temperature columns to revert: {', '.join(temp_cols)}")
        df[temp_cols] = df[temp_cols] * TEMP_REVERSION_FACTOR
        print(f"Multiplied temperature columns by {TEMP_REVERSION_FACTOR}.")
    else:
        print("No temperature columns found for reversion.")

    return df


# --- Main Processing Function ---

def process_weather_files_only():
    """
    Finds and pre-processes weather (_processed) files, then saves the result.
    """
    print("--- Starting Weather Data Processing ---")

    for state in STATES:
        print(f"\n--- Processing state: {state} ---")

        # --- 1. Find the processed file for the current state ---
        processed_search = os.path.join(SOURCE_DIRECTORY, f"*{state}{PROCESSED_SUFFIX}")
        processed_files = glob.glob(processed_search)

        # --- 2. Validate that we found exactly one file ---
        if not processed_files:
            print(f"⚠️  SKIPPING: Could not find the '{state}{PROCESSED_SUFFIX}' file.")
            continue

        weather_file = processed_files[0]
        print(f"Found weather file: '{os.path.basename(weather_file)}'")

        try:
            # --- 3. Use the function to load and clean the weather data ---
            processed_df = preprocess_weather_data(weather_file)
            print("Successfully pre-processed the file.")

            # --- 4. Save the final result ---
            output_filename = f"{state}{FINAL_SUFFIX}"
            processed_df.to_csv(output_filename)
            print(f"✅ Successfully saved processed data to '{output_filename}'")

        except Exception as e:
            print(f"❌ ERROR processing file for {state}: {e}")

    print("\n--- All states processed successfully. ---")


# --- Run the script ---
if __name__ == "__main__":
    process_weather_files_only()
