import pandas as pd
import os
import glob

# Keywords to identify states and group files.
STATES = ['Victoria', 'New South Wales', 'Queensland']

# List of keywords to identify columns for dropping.
KEYWORDS_TO_DROP = [
    'Battery', 'Pump', 'Export', 'Import', 'Coal', 'Bioenergy',
    'Distillate', 'Gas', 'Hydro', 'Emissions', 'Volume', 'Net'
]
# Directory to search for files. Using the path you provided.
SOURCE_DIRECTORY = r'C:\projects\HONOURS\combined_5min'


def combine_and_clean_by_state():
    """
    Finds all CSV files, groups them by state, combines them,
    drops specified columns, and saves a new summary file for each state.
    """
    print("Starting State Data Combination and Cleaning")

    # Loop through each state to process
    for state in STATES:
        print(f"\nProcessing state: {state}")

        # Find all .csv files in the directory that contain the state's name
        search_pattern = os.path.join(SOURCE_DIRECTORY, f"*{state}*.csv")
        file_list = glob.glob(search_pattern)

        if not file_list:
            print(f"No CSV files found for state '{state}'. Skipping.")
            continue

        print(f"Found {len(file_list)} files to combine: {', '.join(file_list)}")

        # List to hold the DataFrame of each file for this state
        list_of_dfs = []

        # Read each file and append its DataFrame to the list
        for filename in file_list:
            try:
                # Read the file
                df = pd.read_csv(filename, index_col=0)
                df.index = pd.to_datetime(df.index, errors='coerce')

                # Drop any rows where the date could not be parsed.
                original_rows = len(df)
                df = df[df.index.notna()]

                if original_rows > len(df):
                    print(
                        f"Dropped {original_rows - len(df)} rows from '{os.path.basename(filename)}' due to unparseable dates.")

                if not df.empty:
                    list_of_dfs.append(df)
                else:
                    print(f"File '{os.path.basename(filename)}' was empty after cleaning.")

            except Exception as e:
                print(f"⚠️ Could not read or process file '{filename}'. Error: {e}")

        if not list_of_dfs:
            print(f"No valid data could be read for state '{state}'.")
            continue

        # Combine all DataFrames in the list into a single DataFrame
        combined_df = pd.concat(list_of_dfs, ignore_index=False)
        print(f"Combined data for {state}. Total rows: {len(combined_df)}")

        # Clean column names to remove leading/trailing whitespace
        combined_df.columns = combined_df.columns.str.strip()

        # Find and drop columns based on keywords
        cols_to_drop = [
            col for col in combined_df.columns
            if any(keyword in col for keyword in KEYWORDS_TO_DROP)
        ]

        if cols_to_drop:
            combined_df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped {len(cols_to_drop)} columns based on keywords.")

        # Sort the data by the index (date/time) to ensure it's in order
        combined_df.sort_index(inplace=True)

        # Save the final, combined DataFrame to a new CSV file
        # The output file will be saved in the same directory as the script.
        output_filename = f"{state}_combined.csv"
        combined_df.to_csv(output_filename, index_label='DateTime')
        print(f"✅ Successfully saved combined data to '{output_filename}'")

    print("\nAll states processed.")


if __name__ == "__main__":
    combine_and_clean_by_state()
