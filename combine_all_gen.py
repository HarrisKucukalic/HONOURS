import pandas as pd
import os
import glob

FINAL_COLUMNS = [
    'DateTime', 'Wind -  MW', 'Solar (Utility) -  MW', 'Solar (Rooftop) -  MW'
]
DATA_DIRECTORY = r'C:\projects\HONOURS'


def process_and_combine_state_data():
    """
    Loads, filters, and combines historical and current data for each state,
    then saves the result to a new CSV file.
    """
    state_files = {
        'New South Wales': {
            'hist': r'C:\projects\HONOURS\day_to_5min\20240703 New South Wales_processed.csv',
            'curr': r'C:\projects\HONOURS\combined_5min\New South Wales_combined.csv'
        },
        'Queensland': {
            'hist': r'C:\projects\HONOURS\day_to_5min\20240703 Queensland_processed.csv',
            'curr': r'C:\projects\HONOURS\combined_5min\Queensland_combined.csv'
        },
        'Victoria': {
            'hist': r'C:\projects\HONOURS\day_to_5min\20240703 Victoria_processed.csv',
            'curr': r'C:\projects\HONOURS\combined_5min\Victoria_combined.csv'
        }
    }

    print("Starting Data Merging and Saving Process.")

    for state, files in state_files.items():
        print(f"\nProcessing: {state}")

        # load daily that was converted and true 5-min data
        df_hist = pd.read_csv(files['hist'])
        df_curr = pd.read_csv(files['curr'])

        print(df_hist.head())
        print(f"Historical columns: {df_hist.columns.tolist()}")
        print(df_curr.head())
        print(f"Current columns:    {df_curr.columns.tolist()}")
        if df_hist is None or df_curr is None:
            print(f"⚠️  SKIPPING {state}: Could not load one or both data files.")
            continue

        print(f"Historical data range: {df_hist['DateTime'].min()} to {df_hist['DateTime'].max()}")
        print(f"Current data range:    {df_curr['DateTime'].min()} to {df_curr['DateTime'].max()}")

        df_hist_filtered = df_hist[[col for col in FINAL_COLUMNS if col in df_hist.columns]]
        df_curr_filtered = df_curr[[col for col in FINAL_COLUMNS if col in df_curr.columns]]

        # Concatenate all prepared dataframes vertically
        combined_df = pd.concat([df_hist_filtered, df_curr_filtered], ignore_index=True)
        rows_before_dedupe = len(combined_df)
        print(f"Combined data before deduplication. Total rows: {rows_before_dedupe}")

        # Remove duplicate rows based on the 'DateTime' column, keeping the last entry.
        combined_df.drop_duplicates(subset='DateTime', keep='last', inplace=True)
        rows_after_dedupe = len(combined_df)

        print(
            f"  - Found and removed {rows_before_dedupe - rows_after_dedupe} duplicate rows (prioritizing current data).")

        # Sort the final dataframe by date
        combined_df.sort_values(by='DateTime', inplace=True)
        print(f"Combined data for {state}. Final shape: {combined_df.shape}")

        # Save Results
        output_path = os.path.join(DATA_DIRECTORY, f'{state}_final.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Successfully saved merged data to '{output_path}'")

    print("\n--- All states processed. ---")


if __name__ == "__main__":
    process_and_combine_state_data()
