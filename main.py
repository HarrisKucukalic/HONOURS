import pandas as pd
import os
import random

FILENAMES = [
    r'C:\projects\HONOURS\New South Wales_final.csv',
    r'C:\projects\HONOURS\Queensland_final.csv',
    r'C:\projects\HONOURS\Victoria_final.csv'
]

STATE_FILES_TO_MERGE = {
    'New South Wales': {
        "master_file": r'C:\projects\HONOURS\New South Wales_master.csv',
        "weather_file": r'C:\projects\HONOURS\historical weather data\historical weather data\NSW_weather_data.csv'
    },
    'Queensland': {
        "master_file": r'C:\projects\HONOURS\Queensland_master.csv',
        "weather_file": r'C:\projects\HONOURS\historical weather data\historical weather data\QLD_weather_data.csv'
    },
    'Victoria': {
        "master_file": r'C:\projects\HONOURS\Victoria_master.csv',
        "weather_file": r'C:\projects\HONOURS\historical weather data\historical weather data\VIC_weather_data.csv'
    }
}


# Columns to check for low generation.
COLUMNS_TO_CHECK = [
    'Wind -  MW', 'Solar (Utility) -  MW', 'Solar (Rooftop) -  MW'
]

# The threshold for daily generation. If the sum of the checked columns for a
# day is below this value, it will be considered a "low-generation" day.

LOW_GENERATION_THRESHOLD = 100

DATA_DIRECTORY = r'C:\projects\HONOURS'


def correct_low_generation_days():
    """
    Reads final data files, identifies days with abnormally low generation or missing data,
    and replaces their data with a varied pattern based on the average of all
    previous valid days.
    """
    print("Starting Low-Generation Data Correction Process")

    for filepath in FILENAMES:
        if not os.path.exists(filepath):
            print(f"\n⚠️ WARNING: File not found at '{filepath}'. Skipping.")
            continue

        print(f"\nProcessing file: {os.path.basename(filepath)} ---")

        try:
            # Load data
            df = pd.read_csv(filepath)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)

            # Clean column names for robust matching
            df.columns = df.columns.str.strip()

            # Ensure all columns to check exist in the DataFrame
            actual_cols_to_check = [col for col in COLUMNS_TO_CHECK if col in df.columns]
            if not actual_cols_to_check:
                print(
                    f"⚠️ WARNING: None of the specified columns to check were found in {os.path.basename(filepath)}. Skipping.")
                continue

            print("Creating a complete 5-minute timeline to find gaps.")
            full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
            df_reindexed = df.reindex(full_index)

            # Identify days to be corrected (low generation OR missing data)
            daily_sums = df_reindexed[actual_cols_to_check].resample('D').sum()
            daily_counts = df_reindexed[actual_cols_to_check].resample('D').count().sum(axis=1)
            daily_total_gen = daily_sums.sum(axis=1)

            # A day is bad if its total generation is too low OR if it has no data points
            low_gen_dates = daily_total_gen[daily_total_gen < LOW_GENERATION_THRESHOLD].index
            missing_data_dates = daily_counts[daily_counts == 0].index
            dates_to_correct = low_gen_dates.union(missing_data_dates)

            if dates_to_correct.empty:
                print("No low-generation or missing days found. No changes needed.")
                continue

            print(f"Found {len(dates_to_correct)} days with low generation or missing data to correct.")

            # Correct the data using a "running average" logic
            df_corrected = df_reindexed.copy()

            # Create a DataFrame to hold the running sums, indexed by time of day
            time_index = pd.date_range("00:00", "23:55", freq="5min").time
            running_sums = pd.DataFrame(0.0, index=time_index, columns=actual_cols_to_check)
            good_day_count = 0
            corrected_days_count = 0

            # Iterate through each unique day in the complete timeline
            for day in df_corrected.index.normalize().unique():
                current_date = day.date()
                is_day_to_correct = day in dates_to_correct

                if is_day_to_correct:
                    # This is a "bad" day
                    if good_day_count > 0:
                        print(f"Correcting {current_date} using average of previous {good_day_count} good days.")

                        average_pattern = running_sums / good_day_count
                        scaling_factor = random.uniform(0.85, 1.15)
                        scaled_pattern = average_pattern * scaling_factor

                        target_index = df_corrected[df_corrected.index.date == current_date].index
                        replacement_values = scaled_pattern.values
                        replacement_values[replacement_values < 0] = 0

                        df_corrected.loc[target_index, actual_cols_to_check] = replacement_values
                        corrected_days_count += 1
                    else:
                        print(
                            f"Skipping correction for {current_date}: No previous good day found to create an average.")
                else:
                    # This is a "good" day, so we update the running sums
                    good_day_data = df_corrected.loc[df_corrected.index.date == current_date, actual_cols_to_check]
                    good_day_data.index = good_day_data.index.time
                    running_sums = running_sums.add(good_day_data, fill_value=0)
                    good_day_count += 1

            print(f"\nCorrected a total of {corrected_days_count} days.")

            print("Forward-filling all columns to ensure data continuity.")
            df_corrected.ffill(inplace=True)

            # Save the final result
            output_path = filepath.replace('_final.csv', '_corrected.csv')
            df_corrected.to_csv(output_path, index_label='DateTime')
            print(f"✅ Successfully saved corrected data to '{os.path.basename(output_path)}'")

        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {os.path.basename(filepath)}: {e}")

    print("\nAll files processed.")

def combine_with_price_demand():
    """
    Combines the corrected generation data with the RRP column from the
    price and demand data files.
    """
    print("\nStarting Final Combination with Price/Demand Data")

    # Define the files to be combined for each state
    state_files = {
        'New South Wales': {
            'corrected_gen': r'C:\projects\HONOURS\New South Wales_corrected.csv',
            'price_demand': r'C:\projects\HONOURS\AEMO_combined_price\NSW_PRICE_DEMAND_AEMO_combined.csv'
        },
        'Queensland': {
            'corrected_gen': r'C:\projects\HONOURS\Queensland_corrected.csv',
            'price_demand': r'C:\projects\HONOURS\AEMO_combined_price\QLD_PRICE_DEMAND_AEMO_combined.csv'
        },
        'Victoria': {
            'corrected_gen': r'C:\projects\HONOURS\Victoria_corrected.csv',
            'price_demand': r'C:\projects\HONOURS\AEMO_combined_price\VIC_PRICE_DEMAND_AEMO_combined.csv'
        }
    }

    for state, paths in state_files.items():
        print(f"\n--- Combining data for: {state} ---")

        try:
            # Load both datasets
            df_gen = pd.read_csv(paths['corrected_gen'], index_col='DateTime', parse_dates=True)
            df_price = pd.read_csv(paths['price_demand'])  # Load without setting index first

            print(f"  - Loaded '{os.path.basename(paths['corrected_gen'])}'")
            print(f"  - Loaded '{os.path.basename(paths['price_demand'])}'")

            # Prepare the price/demand dataframe
            # Rename 'SETTLEMENTDATE' to 'DateTime' to match the other file
            if 'SETTLEMENTDATE' in df_price.columns:
                df_price.rename(columns={'SETTLEMENTDATE': 'DateTime'}, inplace=True)

            # Check if required columns exist
            if 'DateTime' not in df_price.columns or 'RRP' not in df_price.columns:
                print(
                    f"⚠️ WARNING: 'DateTime' or 'RRP' column not found in {os.path.basename(paths['price_demand'])}. Skipping merge.")
                continue

            # Convert to datetime and set as index
            df_price['DateTime'] = pd.to_datetime(df_price['DateTime'])
            df_price.set_index('DateTime', inplace=True)

            # Keep only the 'RRP' column
            df_price_filtered = df_price[['RRP']]
            print(f"Isolated the 'RRP' column from the price/demand file.")

            # Merge the two dataframes on their DateTime index
            # A left merge ensures the complete timeline from the corrected file is used.
            df_master = pd.merge(df_gen, df_price_filtered, left_index=True, right_index=True, how='left')
            print(f"Merged generation data with RRP data. Final shape: {df_master.shape}")

            # Forward-fill any gaps that might have been created in the RRP column
            df_master['RRP'] = df_master['RRP'].ffill()
            print("Forward-filled any remaining gaps in the RRP column.")

            # Save the final master file
            output_path = os.path.join(DATA_DIRECTORY, f'{state}_master.csv')
            df_master.to_csv(output_path, index_label='DateTime')
            print(f"✅ Successfully saved master data to '{os.path.basename(output_path)}'")

        except FileNotFoundError as e:
            print(f"❌ ERROR for {state}: File not found. Please check the path. Details: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {state}: {e}")

def resample_and_merge_weather():
    """
    Loads master data and hourly weather data, resamples the weather data to
    5-minute intervals, and merges them into a final file.
    """
    print("Starting Weather Data Resampling and Merging Process")

    for state, paths in STATE_FILES_TO_MERGE.items():
        print(f"\nProcessing: {state}")

        master_filepath = paths['master_file']
        weather_filepath = paths['weather_file']

        # Validate that both required files exist before proceeding
        if not os.path.exists(master_filepath) or not os.path.exists(weather_filepath):
            print(f"⚠️ WARNING: Missing master or weather file for {state}. Skipping.")
            if not os.path.exists(master_filepath):
                print(f"  - File not found: {master_filepath}")
            if not os.path.exists(weather_filepath):
                print(f"  - File not found: {weather_filepath}")
            continue

        try:
            # Load the master and weather data files
            # Both files should have a 'DateTime' index after being parsed.
            df_master = pd.read_csv(master_filepath, index_col='DateTime', parse_dates=True)
            df_weather_hourly = pd.read_csv(weather_filepath, index_col='DateTime', parse_dates=True)

            print(f"Loaded '{os.path.basename(master_filepath)}' ({len(df_master)} rows)")
            print(f"Loaded '{os.path.basename(weather_filepath)}' ({len(df_weather_hourly)} rows)")

            # Resample the hourly weather data to 5-minute intervals
            # 'ffill()' (forward-fill) carries the hourly value (e.g., 12:00) forward
            # to fill the 5-minute intervals (12:05, 12:10, ..., 12:55).
            print("Resampling hourly weather data to 5-minute intervals.")
            df_weather_5min = df_weather_hourly.resample('5min').ffill()

            # Merge the 5-minute weather data with the master file
            print("Merging master data with resampled weather data...")
            df_final = pd.merge(df_master, df_weather_5min, left_index=True, right_index=True, how='left')

            # Forward-fill any remaining gaps in the newly merged weather columns
            # This handles cases where the master file might start slightly before the weather data.
            df_final[df_weather_5min.columns] = df_final[df_weather_5min.columns].ffill()
            print(f"Merged data. Final shape: {df_final.shape}")

            # Save the final, combined file
            output_filename = f"{state}_master_with_weather.csv"
            output_path = os.path.join(DATA_DIRECTORY, output_filename)
            df_final.to_csv(output_path, index_label='DateTime')
            print(f"✅ Successfully saved final combined data to '{output_filename}'")

        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {state}: {e}")

    print("\nAll states processed.")

if __name__ == "__main__":
    # correct_low_generation_days()
    # combine_with_price_demand()
    resample_and_merge_weather()