import pandas as pd
import os

def merge_csv_files_in_directory(directory_path):
    """
    Merges all AEMO Price/Demand files into a single CSV file,
    removing duplicate rows based on the 'SETTLEMENTDATE' column.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Get a list of all CSV files in the directory
    try:
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    except Exception as e:
        print(f"Error reading directory {directory_path}: {e}")
        return

    if not csv_files:
        print(f"No CSV files found in '{directory_path}'.")
        return

    # Create a list to hold the dataframes
    df_list = []
    print(f"\nProcessing directory: {directory_path}")

    # Loop through the list of CSV files and read them into dataframes
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        try:
            # Read each CSV file and append its content to the list
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"Reading {file}...")
        except Exception as e:
            print(f"Could not read file {file}. Error: {e}")

    # Concatenate all dataframes in the list into a single dataframe
    if not df_list:
        print(f"No data could be read from the CSV files in {directory_path}.")
        return

    print("Combining all files...")
    merged_df = pd.concat(df_list, ignore_index=True)

    # --- Deduplication Step ---
    # Check if the key column for deduplication exists
    if 'SETTLEMENTDATE' not in merged_df.columns:
        print(f"Error: 'SETTLEMENTDATE' column not found. Cannot deduplicate.")
        # Save the merged file without deduplication if an error occurs.
        output_filename = f"{os.path.basename(directory_path)}_combined_no_dedupe.csv"
    else:
        print(f"Original row count: {len(merged_df)}")
        # Remove duplicate rows based on the 'SETTLEMENTDATE' column
        # 'keep="first"' ensures that the first occurrence is kept
        merged_df.drop_duplicates(subset=['SETTLEMENTDATE'], keep='first', inplace=True)
        print(f"Deduplicated row count: {len(merged_df)}")
        output_filename = f"{os.path.basename(directory_path)}_combined.csv"


    # --- Save the Merged Dataframe to a new CSV file ---
    try:
        merged_df.to_csv(output_filename, index=False)
        print(f"Successfully created '{output_filename}' with {len(merged_df)} unique rows.")
    except Exception as e:
        print(f"Error saving the combined file: {e}")


if __name__ == "__main__":
    # List of directories to process
    directories_to_process = [
        r'C:\projects\HONOURS\NSW_PRICE_DEMAND_AEMO',
        r'C:\projects\HONOURS\QLD_PRICE_DEMAND_AEMO',
        r'C:\projects\HONOURS\VIC_PRICE_DEMAND_AEMO'
    ]

    # Run the merge function for each directory
    for directory in directories_to_process:
        merge_csv_files_in_directory(directory)

    print("\nAll directories processed.")