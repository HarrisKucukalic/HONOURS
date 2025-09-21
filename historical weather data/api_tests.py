import requests
import pandas as pd
from datetime import datetime, timedelta
import os

HOURLY_VARIABLES = (
    "temperature_2m,surface_pressure,cloud_cover,"
    "wind_speed_10m,wind_speed_100m,"
    "shortwave_radiation,direct_normal_irradiance"
)

INITIAL_START_DATE = "2024-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
TIMEZONE = "Australia/Sydney"
INPUT_EXCEL_FILE = "Wind & Solar Gen. Open Elec.xlsx"
OUTPUT_DIRECTORY = "historical weather data"

STATE_COORDINATES = {
    "NSW": {"lat": -33.87, "long": 151.21},
    "QLD": {"lat": -27.47, "long": 153.03},
    "VIC": {"lat": -37.81, "long": 144.96}
}


def pull_weather_data(state_name, start_date, end_date, lat, long):
    """
    Fetches historical weather data from the Open-Meteo API for a given location and date range.
    """
    print(f"Pulling data for {state_name} from {start_date} to {end_date}.")

    try:
        # Construct the API URL
        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={long}&start_date={start_date}&end_date={end_date}"
            f"&hourly={HOURLY_VARIABLES}"
            f"&timezone={TIMEZONE}"
        )

        # Make the API request
        response = requests.get(api_url)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        data = response.json()

        # Convert the 'hourly' data into a pandas DataFrame
        df = pd.DataFrame(data.get('hourly', {}))

        if df.empty:
            print("No new data found for this period.")
            return pd.DataFrame()

        # Convert the 'time' column to datetime objects
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # Rename columns to be more descriptive
        df.rename(columns={
            'temperature_2m': 'Temp (°C)',
            'surface_pressure': 'Pressure (hPa)',
            'cloud_cover': 'Cloud Cover (%)',
            'wind_speed_10m': 'Wind Speed 10m (km/h)',
            'wind_speed_100m': 'Wind Speed 100m (km/h)',
            'shortwave_radiation': 'Solar GHI (W/m²)',
            'direct_normal_irradiance': 'Solar DNI (W/m²)'
        }, inplace=True)

        print(f"✅ Successfully received {len(df)} new data points for {state_name}.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred fetching data for {state_name}: {e}")
        return pd.DataFrame()


def pull_hist_weather_data(start_date, end_date, lat, long):
    print(f"--> Pulling Historical Data for Lat: {lat} & Long: {long}")
    hist_df = pd.DataFrame()
    try:
        historical_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={long}&start_date={start_date}&end_date={end_date}"
            f"&hourly={HOURLY_VARIABLES}"  # <-- Using the expanded list of variables
            f"&timezone={TIMEZONE}"
        )

        response = requests.get(historical_url)
        response.raise_for_status()
        hist_data = response.json()

        # Convert to a pandas DataFrame
        hist_df = pd.DataFrame(hist_data.get('hourly', {}))
        print(hist_df.size)
        if not hist_df.empty:
            hist_df['time'] = pd.to_datetime(hist_df['time'])
            hist_df.set_index('time', inplace=True)
            # Rename columns for clarity
            hist_df.rename(columns={
                'temperature_2m': 'Temp (°C)',
                'surface_pressure': 'Pressure (hPa)',
                'cloud_cover': 'Cloud Cover (%)',
                'wind_speed_10m': 'Wind Speed 10m (km/h)',
                'wind_speed_100m': 'Wind Speed 100m (km/h)',
                'shortwave_radiation': 'Solar GHI (W/m²)',
                'direct_normal_irradiance': 'Solar DNI (W/m²)'
            }, inplace=True)
            print("✅ Historical data received.")
            print(hist_df.head(5))
        else:
            print("No historical data found.")

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred fetching historical data: {e}")

    return hist_df


def pull_forecast_weather_data(lat, long):
    # Retrieves Forecast Weather data
    print("\n Fetching hourly forecast data for the next 24 hours.")

    try:
        # Construct the Forecast API URL with the new variables
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={long}"
            f"&hourly={HOURLY_VARIABLES}"
            f"&forecast_days=1"  # Limit forecast to 1 day (24 hours)
        )

        response = requests.get(forecast_url)
        response.raise_for_status()
        forecast_data = response.json()

        # Convert to a pandas DataFrame
        forecast_df = pd.DataFrame(forecast_data.get('hourly', {}))
        if not forecast_df.empty:
            forecast_df['time'] = pd.to_datetime(forecast_df['time'])
            forecast_df.set_index('time', inplace=True)
            # Use the same renaming scheme for consistency
            forecast_df.rename(columns={
                'temperature_2m': 'Temp (°C)',
                'surface_pressure': 'Pressure (hPa)',
                'cloud_cover': 'Cloud Cover (%)',
                'wind_speed_10m': 'Wind Speed 10m (km/h)',
                'wind_speed_100m': 'Wind Speed 100m (km/h)',
                'shortwave_radiation': 'Solar GHI (W/m²)',
                'direct_normal_irradiance': 'Solar DNI (W/m²)'
            }, inplace=True)
            print("✅ Hourly forecast data received.")
            print(forecast_df.head(5))
        else:
            print("No hourly forecast data available.")

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred fetching forecast data: {e}")

    return forecast_df


def main():
    """
    Main function to create or update historical weather data files.
    If a file exists, it appends new data. Otherwise, it creates a new file.
    """
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Output will be saved to the '{OUTPUT_DIRECTORY}' directory.")

    try:
        xls_source = pd.ExcelFile(INPUT_EXCEL_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: Source file '{INPUT_EXCEL_FILE}' not found.")
        return

    # Processing loop
    for source_sheet_name in xls_source.sheet_names:
        print(f"\n--- Processing Main Sheet: {source_sheet_name} ---")
        df_locations = pd.read_excel(xls_source, sheet_name=source_sheet_name)
        output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{source_sheet_name}.xlsx")

        existing_data_sheets = {}
        # If the output file already exists, load all its sheets into memory.
        if os.path.exists(output_filepath):
            print(f"Found existing file: '{output_filepath}'. Checking for updates.")
            try:
                # index_col=0 ensures the first column (our 'time' column) is used as the index.
                existing_data_sheets = pd.read_excel(output_filepath, sheet_name=None, index_col=0)
            except Exception as e:
                print(
                    f"⚠️ Could not read existing file '{output_filepath}'. It might be corrupted. Will create a new one. Error: {e}")

        # Loop through each location from the source file.
        for index, row in df_locations.iterrows():
            location_name = row.get('Name', f'Row_{index + 2}')
            sheet_name_safe = location_name[:31]  # Excel sheet name limit
            latitude = row.get('Latitude')
            longitude = row.get('Longitude')

            if latitude is None or longitude is None:
                print(f"⚠️ Skipping '{location_name}' due to missing Latitude/Longitude.")
                continue

            print(f"\nProcessing Location: {location_name}")

            start_date = INITIAL_START_DATE

            # Check if we have existing data for this specific location.
            if sheet_name_safe in existing_data_sheets:
                df_existing = existing_data_sheets[sheet_name_safe]
                if not df_existing.empty and isinstance(df_existing.index, pd.DatetimeIndex):
                    # New start date is the day after the last recorded date.
                    last_date = df_existing.index.max()
                    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

            # Pull new data from the calculated start_date.
            df_new = pull_hist_weather_data(
                start_date=start_date,
                end_date=END_DATE,
                lat=latitude,
                long=longitude
            )

            # If new data was fetched, append it to the existing data.
            if not df_new.empty:
                if sheet_name_safe in existing_data_sheets:
                    # Combine old and new data.
                    combined_df = pd.concat([existing_data_sheets[sheet_name_safe], df_new])
                    # Update the dictionary with the combined data.
                    existing_data_sheets[sheet_name_safe] = combined_df
                else:
                    # This is a new location for an existing file.
                    existing_data_sheets[sheet_name_safe] = df_new
                print(f"Data for '{location_name}' updated.")

        # Save Files
        # Write all sheets (old and new) back to the Excel file, overwriting it.
        if not existing_data_sheets:
            print(f"No data processed for '{source_sheet_name}'. File will not be created/updated.")
            continue

        print(f"\nSaving all data to '{output_filepath}'...")
        with pd.ExcelWriter(output_filepath, engine='openpyxl') as writer:
            for sheet, df_to_save in existing_data_sheets.items():
                df_to_save.to_excel(writer, sheet_name=sheet)
        print(f"--- ✅ Finished updating file: {output_filepath} ---")


def main_states():
    """
    Main function to create or update historical weather data files for each QLD, NSW & VIC.
    """
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Output will be saved to the '{OUTPUT_DIRECTORY}' directory.")

    # Processing Loop
    for state, coords in STATE_COORDINATES.items():
        print(f"\n--- Processing State: {state} ---")

        output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{state}_weather_data.csv")
        start_date = INITIAL_START_DATE
        df_existing = None

        # Check if a data file for this state already exists.
        if os.path.exists(output_filepath):
            print(f"  - Found existing file: '{os.path.basename(output_filepath)}'.")
            try:
                df_existing = pd.read_csv(output_filepath, index_col=0, parse_dates=True)
                if not df_existing.empty:
                    # If the file exists and has data, set the new start date to the day
                    # after the last recorded date to avoid downloading duplicates.
                    last_date = df_existing.index.max()
                    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception as e:
                print(f"  - ⚠️ Could not read existing file. Will create a new one. Error: {e}")

        # Only proceed if the start date is not after the end date
        if start_date > END_DATE:
            print("  - Data is already up to date. No new data to pull.")
            continue

        # Fetch new data from the API
        df_new = pull_weather_data(
            state_name=state,
            start_date=start_date,
            end_date=END_DATE,
            lat=coords['lat'],
            long=coords['long']
        )

        # Combine and save the data
        if not df_new.empty:
            if df_existing is not None:
                # If there was existing data, concatenate the old and new dataframes.
                df_combined = pd.concat([df_existing, df_new])
            else:
                # Otherwise, the new dataframe is the combined dataframe.
                df_combined = df_new

            # Save the final result to a CSV file.
            df_combined.to_csv(output_filepath, index_label='DateTime')
            print(f"  - ✅ Successfully updated file: {os.path.basename(output_filepath)}")

    print("\n All states processed.")

if __name__ == "__main__":
    main_states()