import pandas as pd
import os
import random
# --- Configuration ---

# List of the final combined files to process.
FILENAMES = [
    r'C:\projects\HONOURS\New South Wales_final.csv',
    r'C:\projects\HONOURS\Queensland_final.csv',
    r'C:\projects\HONOURS\Victoria_final.csv'
]

# Columns to check for low generation.
# Make sure these names exactly match the columns in your CSV files.
COLUMNS_TO_CHECK = [
    'Wind -  MW', 'Solar (Utility) -  MW', 'Solar (Rooftop) -  MW'
]

# The threshold for daily generation. If the sum of the checked columns for a
# day is below this value, it will be considered a "low-generation" day.
# The unit is the sum of average MW over the day. 100 is a reasonable starting point.
LOW_GENERATION_THRESHOLD = 100

DATA_DIRECTORY = r'C:\projects\HONOURS'

def correct_low_generation_days():
    """
    Reads final data files, identifies days with abnormally low generation or missing data,
    and replaces their data with a varied pattern based on the average of all
    previous valid days.
    """
    print("--- Starting Low-Generation Data Correction Process ---")

    for filepath in FILENAMES:
        if not os.path.exists(filepath):
            print(f"\n⚠️ WARNING: File not found at '{filepath}'. Skipping.")
            continue

        print(f"\n--- Processing file: {os.path.basename(filepath)} ---")

        try:
            # 1. Load the data
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

            # --- Create a complete index to handle missing periods ---
            print("  - Creating a complete 5-minute timeline to find gaps...")
            full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
            df_reindexed = df.reindex(full_index)

            # 2. Identify days to be corrected (low generation OR missing data)
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

            # 3. Correct the data using a "running average" logic
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
                        print(f"  - Correcting {current_date} using average of previous {good_day_count} good days.")

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
                            f"  - Skipping correction for {current_date}: No previous good day found to create an average.")
                else:
                    # This is a "good" day, so we update the running sums
                    good_day_data = df_corrected.loc[df_corrected.index.date == current_date, actual_cols_to_check]
                    good_day_data.index = good_day_data.index.time
                    running_sums = running_sums.add(good_day_data, fill_value=0)
                    good_day_count += 1

            print(f"\nCorrected a total of {corrected_days_count} days.")

            # --- NEW: Forward-fill all other columns to ensure no gaps remain ---
            print("  - Forward-filling all columns to ensure data continuity...")
            df_corrected.ffill(inplace=True)

            # 4. Save the final result
            output_path = filepath.replace('_final.csv', '_corrected.csv')
            df_corrected.to_csv(output_path, index_label='DateTime')
            print(f"✅ Successfully saved corrected data to '{os.path.basename(output_path)}'")

        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {os.path.basename(filepath)}: {e}")

    print("\n--- All files processed. ---")

if __name__ == "__main__":
    correct_low_generation_days()
