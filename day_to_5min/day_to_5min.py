import pandas as pd
import numpy as np
import os

# List of files to process.
FILENAMES = [
    r'C:\projects\HONOURS\day_to_5min\20240703 New South Wales.csv',
    r'C:\projects\HONOURS\day_to_5min\20240703 Queensland.csv',
    r'C:\projects\HONOURS\day_to_5min\20240703 Victoria.csv'
]

# List of keywords to identify columns for dropping.
KEYWORDS_TO_DROP = [
    'Battery', 'Pump', 'Export', 'Import', 'Coal', 'Bioenergy',
    'Distillate', 'Gas', 'Hydro', 'Emissions', 'Volume', 'Net'
]

def generate_solar_profile(intervals_per_day=288):
    """
    Generates a realistic "bell curve" daily profile for solar generation.
    The profile is normalised, so the sum of its values equals 1.
    """
    # Create a time axis from 0 to 2*pi for a full day cycle
    time_axis = np.linspace(0, 2 * np.pi, intervals_per_day)
    # Generate a sine wave profile (shifted to be positive during the day)
    # The sine wave naturally models the rise and fall of the sun.
    profile = np.sin(time_axis - np.pi/2) + 1
    # Solar generation is zero at night, so clip values outside of daylight hours.
    # This is approxmated by setting the first (early morning) and last (late evening) quarters of the day to 0.
    daylight_start = intervals_per_day // 4
    daylight_end = intervals_per_day * 3 // 4
    profile[:daylight_start] = 0
    profile[daylight_end:] = 0
    # Normalise the profile so that it sums to 1
    profile /= profile.sum()
    return profile


def generate_wind_profile(intervals_per_day=288):
    """
    Generates a fluctuating daily profile for wind generation.
    This simulates the natural variability of wind.
    """
    random_steps = np.random.randn(intervals_per_day)
    profile_raw = pd.Series(random_steps).cumsum()
    profile_smoothed = profile_raw.rolling(window=12, center=True, min_periods=1).mean()
    profile = profile_smoothed - profile_smoothed.min()

    # Add a check to prevent division by zero if the profile is flat
    if profile.sum() > 0:
        profile /= profile.sum()
    else:
        # Fallback to a flat profile if the sum is zero
        return np.full(intervals_per_day, 1.0 / intervals_per_day)

    # Return a NumPy array to prevent index alignment issues
    return profile.values


def process_energy_files():
    """
    Reads daily aggregated energy files, loops through each day,
    disaggregates the data into 5-minute intervals using realistic profiles (as above),
    converts units from daily GWh to average 5-minute MW, and saves the processed files.
    """
    print("Starting Advanced Data Disaggregation")

    solar_profile = generate_solar_profile()
    flat_profile = np.full(288, 1 / 288.0)

    for filename in FILENAMES:
        if not os.path.exists(filename):
            print(f"\n⚠️ WARNING: File '{filename}' not found. Skipping.")
            continue

        print(f"\nProcessing file: {filename}")

        try:
            df_daily = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Original daily data loaded. Contains {len(df_daily)} days.")
            df_daily.columns = df_daily.columns.str.strip()

            cols_to_drop = [
                col for col in df_daily.columns
                if any(keyword in col for keyword in KEYWORDS_TO_DROP)
            ]
            if cols_to_drop:
                df_daily.drop(columns=cols_to_drop, inplace=True)
                print(f"Dropped {len(cols_to_drop)} columns.")

            # This list will hold the processed 5-minute DataFrames for each day
            all_processed_days = []

            # Loop through each day (each row) in the input file
            for current_date, daily_data_row in df_daily.iterrows():
                print(f"Processing data for {current_date.date()}")

                # Generate a new wind profile for each day to ensure variability
                wind_profile = generate_wind_profile()

                # Create the 5-minute time index for the current day
                start_time = current_date
                end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
                five_min_index = pd.date_range(start=start_time, end=end_time, freq='5min')
                df_day_processed = pd.DataFrame(index=five_min_index)

                # Process each generation column for the current day
                for col_name in df_daily.columns:
                    # Get the total GWh for this specific day and column
                    daily_total_gwh = daily_data_row[col_name]

                    if pd.isna(daily_total_gwh):
                        continue  # Skip if data is missing for this day

                    # Choose the appropriate profile
                    if 'Solar' in col_name:
                        profile_to_use = solar_profile
                    elif 'Wind' in col_name:
                        profile_to_use = wind_profile
                    else:
                        profile_to_use = flat_profile

                    # Convert daily GWh (Energy) to average MW (Power) per 5-min interval.
                    # Power(MW) = Energy(MWh) / Time(h). A 5-min interval is 1/12th of an hour.
                    # So, Power(MW) = (Energy for interval in MWh) * 12
                    # Energy for interval in MWh = (Total GWh * 1000) * profile_fraction
                    power_mw_series = (daily_total_gwh * 1000 * 12) * profile_to_use
                    df_day_processed[col_name] = power_mw_series

                all_processed_days.append(df_day_processed)

            # Concatenate all the daily DataFrames into one continuous time series
            if not all_processed_days:
                print("No data was processed. Output file will not be created.")
                continue

            df_final_processed = pd.concat(all_processed_days)
            print(f"\nCombined all days into a single time series. Total 5-min intervals: {len(df_final_processed)}")

            # Rename columns to reflect the new units (MW)
            new_column_names = {col: col.replace('GWh', 'MW') for col in df_final_processed.columns}
            df_final_processed.rename(columns=new_column_names, inplace=True)
            print("Updated column units from GWh to MW.")

            # Save the final processed DataFrame to a new CSV file
            base_name, extension = os.path.splitext(filename)
            new_filename = f"{base_name}_processed{extension}"

            # Save the index and give it the name 'DateTime'
            df_final_processed.to_csv(new_filename, index_label='DateTime')
            print(f"✅ Successfully saved processed data to '{new_filename}'")

        except Exception as e:
            print(f"❌ ERROR processing file '{filename}': {e}")

    print("\n--- All files processed. ---")


# --- Run the script ---
if __name__ == "__main__":
    process_energy_files()