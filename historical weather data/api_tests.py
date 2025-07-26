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
    # --- 2. GET HOURLY FORECAST DATA ---
    print("\n--> Fetching hourly forecast data for the next 24 hours...")

    try:
        # Construct the Forecast API URL with the new variables
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={long}"
            f"&hourly={HOURLY_VARIABLES}"  # <-- Using the expanded list of variables
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
    # --- 1. SETUP ---
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Output will be saved to the '{OUTPUT_DIRECTORY}' directory.")

    try:
        xls_source = pd.ExcelFile(INPUT_EXCEL_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: Source file '{INPUT_EXCEL_FILE}' not found.")
        return

    # --- 2. PROCESSING LOOP ---
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

        # --- 3. SAVE FILE ---
        # Write all sheets (old and new) back to the Excel file, overwriting it.
        if not existing_data_sheets:
            print(f"No data processed for '{source_sheet_name}'. File will not be created/updated.")
            continue

        print(f"\nSaving all data to '{output_filepath}'...")
        with pd.ExcelWriter(output_filepath, engine='openpyxl') as writer:
            for sheet, df_to_save in existing_data_sheets.items():
                df_to_save.to_excel(writer, sheet_name=sheet)
        print(f"--- ✅ Finished updating file: {output_filepath} ---")


if __name__ == "__main__":
    main()