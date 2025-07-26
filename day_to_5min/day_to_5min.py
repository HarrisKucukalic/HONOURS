import pandas as pd
import os

# --- Configuration ---

# List of files to process. Add more filenames to this list as needed.
FILENAMES = [
    "20240703 New South Wales.csv",
    "20240703 Queensland.csv",
    "20240703 Victoria.csv"
]

# List of keywords to identify columns for dropping. This is more robust
# than matching exact, fragile column names.
KEYWORDS_TO_DROP = [
    'Battery', 'Pump', 'Export', 'Import', 'Coal', 'Bioenergy',
    'Distillate', 'Gas', 'Hydro', 'Emissions', 'Volume', 'Net'
]

# The factor to divide the data by (24 hours * 60 minutes / 5 minutes = 288 intervals)
DIVISION_FACTOR = 288.0


# --- Main Processing Function ---

def process_energy_files():
    """
    Reads energy data files, drops specified columns, resamples to 5-minute
    intervals, divides by a factor, and saves the processed files.
    """
    print("--- Starting Data Processing ---")

    for filename in FILENAMES:
        # Check if the file exists before trying to process it
        if not os.path.exists(filename):
            print(f"\n⚠️ WARNING: File '{filename}' not found. Skipping.")
            continue

        print(f"\n--- Processing file: {filename} ---")

        try:
            # 1. Read the CSV file
            # Assuming the first column is the date/time index
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Original data loaded. Shape: {df.shape}")

            # Clean column names to remove leading/trailing whitespace
            df.columns = df.columns.str.strip()

            # 2. Find and drop columns based on keywords
            # This is more flexible and handles variations in column names.
            cols_to_drop_in_file = [
                col for col in df.columns
                if any(keyword in col for keyword in KEYWORDS_TO_DROP)
            ]

            if cols_to_drop_in_file:
                df.drop(columns=cols_to_drop_in_file, inplace=True)
                print(f"Dropped {len(cols_to_drop_in_file)} columns: {', '.join(cols_to_drop_in_file)}")
            else:
                print("No columns matching the keywords were found to drop.")

            # 3. Resample to 5-minute intervals (Fixed the deprecated 'T' to 'min')
            # '5min' is the updated frequency string.
            # `ffill()` (forward-fill) carries the last valid observation forward.
            df_resampled = df.resample('5min').ffill()
            print(f"Resampled to 5-minute intervals. New shape: {df_resampled.shape}")

            # 4. Divide all numerical data by the division factor
            # We select only numeric columns to perform the division
            numeric_cols = df_resampled.select_dtypes(include='number').columns
            df_resampled[numeric_cols] = df_resampled[numeric_cols] / DIVISION_FACTOR
            print(f"Divided all numeric data by {DIVISION_FACTOR}.")

            # 5. Save the processed DataFrame to a new CSV file
            # Create the new filename
            base_name, extension = os.path.splitext(filename)
            new_filename = f"{base_name}_processed{extension}"

            df_resampled.to_csv(new_filename)
            print(f"✅ Successfully saved processed data to '{new_filename}'")

        except Exception as e:
            print(f"❌ ERROR processing file '{filename}': {e}")

    print("\n--- All files processed. ---")


# --- Run the script ---
if __name__ == "__main__":
    process_energy_files()
