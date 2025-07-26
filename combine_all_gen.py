import pandas as pd
import os
import glob

FINAL_COLUMNS = [
    'date',
    'Wind -  MW',
    'Solar (Utility) -  MW',
    'Solar (Rooftop) -  MW'
]

DATA_DIRECTORY = r'C:\projects\HONOURS'


def process_and_combine_state_data():
    """
    Loads, filters, and combines historical and current data for each state,
    then saves the result to a new CSV file.
    """
    # A dictionary to hold the file paths for each state
    state_files = {
        'New South Wales': {
            'hist': os.path.join(DATA_DIRECTORY, 'New South Wales_processed.csv'),
            'curr': os.path.join(DATA_DIRECTORY, 'New South Wales_combined.csv')
        },
        'Queensland': {
            'hist': os.path.join(DATA_DIRECTORY, 'Queensland_processed.csv'),
            'curr': os.path.join(DATA_DIRECTORY, 'Queensland_combined.csv')
        },
        'Victoria': {
            'hist': os.path.join(DATA_DIRECTORY, 'Victoria_processed.csv'),
            'curr': os.path.join(DATA_DIRECTORY, 'Victoria_combined.csv')
        }
    }

    print("--- Starting Data Merging and Saving Process ---")

    # Loop through each state defined in the dictionary
    for state, files in state_files.items():
        print(f"\n--- Processing: {state} ---")
        try:
            # --- 1. Load the historical and current data files ---
            df_hist = pd.read_csv(files['hist'])
            df_curr = pd.read_csv(files['curr'])
            print(f"Loaded '{os.path.basename(files['hist'])}' and '{os.path.basename(files['curr'])}'")

            # --- 2. Filter and prepare each DataFrame ---
            dataframes_to_combine = []
            for name, df in [('hist', df_hist), ('curr', df_curr)]:
                # Clean column names to remove leading/trailing whitespace
                df.columns = df.columns.str.strip()

                # Ensure the 'date' column exists before proceeding
                if 'date' not in df.columns:
                    print(f"⚠️  Warning: 'date' column not found in {name} file for {state}. Skipping this file.")
                    continue

                # Convert date column to datetime objects for proper merging
                # Using format='mixed' allows pandas to infer the format for each date string individually.
                # This is more robust for handling multiple date formats (e.g., DD/MM/YYYY and YYYY-MM-DD).
                df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

                # Find which of the FINAL_COLUMNS are present in this DataFrame
                cols_to_keep = [col for col in FINAL_COLUMNS if col in df.columns]

                if 'date' not in cols_to_keep:
                    cols_to_keep.insert(0, 'date')  # Make sure date is always included

                print(f"Found columns in {name} file: {cols_to_keep}")
                dataframes_to_combine.append(df[cols_to_keep])

            # --- 3. Combine the filtered DataFrames ---
            if not dataframes_to_combine:
                print(f"⚠️  SKIPPING {state}: No data to combine after filtering.")
                continue

            # Concatenate all prepared dataframes vertically
            combined_df = pd.concat(dataframes_to_combine, ignore_index=True)

            # Remove duplicate rows based on the 'date' column, keeping the last entry.
            # This ensures that data from the 'curr' file overwrites 'hist' data if dates overlap.
            combined_df.drop_duplicates(subset='date', keep='last', inplace=True)

            # Sort the final dataframe by date
            combined_df.sort_values(by='date', inplace=True)

            print(f"Combined data for {state}. Final shape: {combined_df.shape}")

            # --- 4. Save the final result ---
            output_path = os.path.join(DATA_DIRECTORY, f'{state}_final.csv')
            # index=False prevents pandas from writing row numbers into the file
            combined_df.to_csv(output_path, index=False)
            print(f"✅ Successfully saved merged data to '{output_path}'")

        except FileNotFoundError as e:
            print(f"❌ ERROR for {state}: File not found. Please check the path. Details: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {state}: {e}")

    print("\n--- All states processed. ---")


# --- Run the script ---
if __name__ == "__main__":
    process_and_combine_state_data()