import pandas as pd
import folium
import os

# --- Configuration ---

# The name of your Excel file containing the facility data.
INPUT_EXCEL_FILE = "Wind & Solar Gen. Open Elec.xlsx"

# The name of the output HTML file for the map.
OUTPUT_MAP_FILE = "facility_map.html"


# --- Main Map Creation Function ---

def create_facility_map():
    """
    Reads facility location data from an Excel file and creates an
    interactive geographic map using Folium.
    """
    print("--- Starting Interactive Map Generation ---")

    # --- 1. Load Data ---
    try:
        # Load the Excel file
        xls_source = pd.ExcelFile(INPUT_EXCEL_FILE)
        print(f"Successfully loaded '{INPUT_EXCEL_FILE}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: Source file '{INPUT_EXCEL_FILE}' not found.")
        print("Please make sure the Excel file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the Excel file: {e}")
        return

    # --- 2. Initialize the Map ---
    # Centered roughly on the middle of Australia.
    # You can adjust the location and zoom_start as needed.
    facility_map = folium.Map(location=[-25.27, 133.77], zoom_start=4)
    print("Map initialized.")

    # --- 3. Process Each Sheet and Add Markers ---
    # Loop through all sheets in the Excel file.
    for sheet_name in xls_source.sheet_names:
        print(f"\nProcessing sheet: '{sheet_name}'...")

        # --- Parse State and Technology from the sheet name ---
        try:
            # Assumes format "State-Technology" e.g., "Victoria-Wind"
            state, tech = sheet_name.split('-')
            # Clean up the parsed strings to remove any extra whitespace
            state = state.strip()
            tech = tech.strip()
            print(f"Parsed from sheet name -> State: {state}, Technology: {tech}")
        except ValueError:
            # Handle cases where the sheet name doesn't follow the format
            print(f"⚠️  Could not parse state and technology from sheet name '{sheet_name}'. Using defaults.")
            state = 'Unknown State'
            tech = 'Unknown Technology'

        df_locations = pd.read_excel(xls_source, sheet_name=sheet_name)

        # Loop through each row (each facility) in the current sheet.
        for index, row in df_locations.iterrows():
            try:
                # Extract data for the current facility
                name = row.get('Name', f'Unnamed Facility {index}')
                lat = row.get('Latitude')
                lon = row.get('Longitude')

                # Skip rows with missing coordinates
                if pd.isna(lat) or pd.isna(lon):
                    print(f"⚠️  Skipping '{name}' due to missing coordinates.")
                    continue

                # Create the HTML for the popup window, which is common for all markers
                popup_html = f"""
                <b>Name:</b> {name}<br>
                <b>Technology:</b> {tech}<br>
                <b>State:</b> {state}<br>
                <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})
                """
                iframe = folium.IFrame(popup_html, width=250, height=100)
                popup = folium.Popup(iframe, max_width=250)

                # --- 4. Customize and Add Marker ---
                # Assign a marker based on the technology type parsed from the sheet name
                if 'WIND' in tech:
                    # Use a blue cloud icon for Wind facilities
                    folium.Marker(
                        location=[lat, lon],
                        popup=popup,
                        tooltip=name,
                        icon=folium.Icon(color='blue', icon='cloud', prefix='glyphicon')
                    ).add_to(facility_map)
                elif 'SOLAR' in tech.upper():
                    # Use an orange sun-like icon (asterisk) for Solar facilities
                    folium.Marker(
                        location=[lat, lon],
                        popup=popup,
                        tooltip=name,
                        icon=folium.Icon(color='orange', icon='asterisk', prefix='glyphicon')
                    ).add_to(facility_map)
                else:
                    # Default marker for other technology types
                    folium.Marker(
                        location=[lat, lon],
                        popup=popup,
                        tooltip=name,
                        icon=folium.Icon(color='gray', icon='info-sign', prefix='glyphicon')
                    ).add_to(facility_map)

            except Exception as e:
                print(f"❌ Error processing row {index} in sheet '{sheet_name}': {e}")

    # --- 5. Save Map to HTML File ---
    try:
        facility_map.save(OUTPUT_MAP_FILE)
        print(f"\n--- ✅ Successfully created map! ---")
        print(f"Open the file '{os.path.abspath(OUTPUT_MAP_FILE)}' in your browser to view it.")
    except Exception as e:
        print(f"\n❌ An error occurred while saving the map: {e}")


# --- Run the script ---
if __name__ == "__main__":
    # Before running, make sure you have the required libraries installed:
    # pip install pandas openpyxl folium
    create_facility_map()
