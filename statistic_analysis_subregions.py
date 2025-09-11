import pandas as pd
import numpy as np
from pathlib import Path
import gc
import matplotlib.pyplot as plt
import os
SOURCE_GENERATOR_DIR = Path(r'C:\projects\HONOURS\historical weather data')
SOURCE_STATE_DIR = Path('.')

def calculate_conditional_probability(df: pd.DataFrame, location: str, condition_query: str) -> float:
    print(f"\n--- Calculating Conditional Probability for '{location}' ---")
    print(f"Condition: {condition_query}")

    # Filter for the specific location
    location_df = df[df['Location'] == location].copy()

    # Apply the condition using the query method
    condition_met_df = location_df.query(condition_query)
    total_condition_events = len(condition_met_df)

    if total_condition_events == 0:
        print("Warning: The specified condition never occurred in the dataset for this location.")
        return np.nan

    # Count how many of those events also had a curtailment event
    curtailment_with_condition = condition_met_df['Curtailment_Event'].sum()

    # Calculate the probability
    probability = curtailment_with_condition / total_condition_events
    return probability


def plot_probability_distribution(df: pd.DataFrame, attribute: str, filter_by: str, filter_value: str,
                                  num_bins: int = 15, output_dir: str = "subregion_statistical_analysis"):
    print(f"\n--- Generating Probability Distribution for '{attribute}' filtered by {filter_by} = '{filter_value}' ---")

    filtered_df = df[df[filter_by] == filter_value].copy()

    if filtered_df.empty:
        print(f"Warning: No data found for {filter_by} = '{filter_value}'. Skipping plot.")
        return

    # Create bins for the attribute
    filtered_df['bins'] = pd.cut(filtered_df[attribute], bins=num_bins)

    # Calculate the probability of curtailment in each bin
    prob_df = filtered_df.groupby('bins', observed=True)['Curtailment_Event'].mean().reset_index()
    prob_df.rename(columns={'Curtailment_Event': 'Probability'}, inplace=True)

    if len(prob_df) < 2:
        print(
            f"Warning: Not enough data points to create a distribution plot for {attribute} at {filter_value}. Skipping.")
        return

    # Prepare for plotting
    plt.figure(figsize=(12, 7))
    prob_df['bins_mid'] = prob_df['bins'].apply(lambda x: x.mid)

    bin_width = (prob_df['bins_mid'].iloc[1] - prob_df['bins_mid'].iloc[0]) * 0.9 if len(prob_df) > 1 else 1

    plt.bar(prob_df['bins_mid'], prob_df['Probability'], width=bin_width)

    plt.title(f'Probability of Curtailment vs. {attribute} for {filter_by}: {filter_value}')
    plt.xlabel(f'{attribute} Bins')
    plt.ylabel('Probability of Curtailment Event')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{filter_value}_{attribute}_distribution.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    print(f"✅ Distribution plot saved to '{os.path.join(output_dir, plot_filename)}'")
    plt.close()

def consolidate_generator_data(directory: Path, file_name: str, resample_freq: str = 'h',
                               limit_points: int = None) -> pd.DataFrame:
    file_path = directory / file_name
    print(f"Processing Excel file: {file_name}")
    try:
        # Load and clean data from all sheets
        parts = file_name.replace('.xlsx', '').split('-')
        state, gen_type = parts[0], parts[1].capitalize()
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        all_sheets = []
        for sheet_name, df in sheets_dict.items():
            df['Location'] = sheet_name
            df['State'] = state
            df['Type'] = gen_type
            df.columns = df.columns.str.replace('.', '', regex=False).str.replace(' ', '_')
            df.columns = df.columns.str.replace(r'[\(\)°/]', '', regex=True)
            if 'time' in df.columns:
                df.rename(columns={'time': 'DateTime'}, inplace=True)
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            all_sheets.append(df)

        if not all_sheets: return pd.DataFrame()
        consolidated_df = pd.concat(all_sheets, ignore_index=True)

        if limit_points and 'DateTime' in consolidated_df.columns:
            print(f"  -> Limiting to the most recent {limit_points:,} data points.")
            consolidated_df = consolidated_df.sort_values('DateTime').tail(limit_points)

        # Resample the weather data to the specified frequency
        print(f"  -> Resampling weather data to {resample_freq} intervals.")
        consolidated_df = consolidated_df.set_index('DateTime')
        agg_funcs = {col: 'mean' for col in consolidated_df.select_dtypes(include=np.number).columns}
        for col in consolidated_df.select_dtypes(exclude=np.number).columns:
            agg_funcs[col] = 'first'

        resampled_list = []
        for location, group_df in consolidated_df.groupby('Location'):
            resampled_list.append(group_df.resample(resample_freq).agg(agg_funcs))

        consolidated_df = pd.concat(resampled_list).reset_index()
        consolidated_df.dropna(subset=['Location'], inplace=True)

        # Final memory-saving optimisations
        for col in consolidated_df.columns:
            if consolidated_df[col].dtype == 'float64':
                consolidated_df[col] = consolidated_df[col].astype('float32')
            elif consolidated_df[col].dtype == 'int64':
                consolidated_df[col] = pd.to_numeric(consolidated_df[col], downcast='integer')
        for col in ['Location', 'State', 'Type']:
            if col in consolidated_df.columns:
                consolidated_df[col] = consolidated_df[col].astype('category')

        return consolidated_df
    except Exception as e:
        print(f"An error occurred with {file_name}: {e}")
        return pd.DataFrame()


def consolidate_state_rrp_data(directory: Path, file_name: str, state_code: str, resample_freq: str = 'h',
                               limit_points: int = None) -> pd.DataFrame:
    """
    Reads a state RRP CSV file, optionally limits it, and aggregates to a specified time frequency.
    """
    file_path = directory / file_name
    print(f"Processing RRP file: {file_name}")
    try:
        df = pd.read_csv(file_path, usecols=['DateTime', 'RRP'], dtype={'RRP': 'float32'})
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        if limit_points:
            print(f"  -> Limiting to the most recent {limit_points:,} RRP data points.")
            df = df.sort_values('DateTime').tail(limit_points)
        df['State'] = state_code
        df['State'] = df['State'].astype('category')
        df = df.set_index('DateTime')
        print(f"  -> Resampling RRP data to {resample_freq} intervals.")
        aggregated_df = df[['RRP']].resample(resample_freq).mean().reset_index()
        aggregated_df['State'] = state_code
        aggregated_df['State'] = aggregated_df['State'].astype('category')
        return aggregated_df
    except (FileNotFoundError, KeyError) as e:
        print(f"--- WARNING: Could not process {file_name}. Error: {e}. Skipping. ---")
        return pd.DataFrame()


def create_final_dataset(agg_state_df: pd.DataFrame, agg_local_weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merges aggregated state-level RRP data with aggregated local generator/weather data."""
    print("Merging aggregated RRP with aggregated weather data...")
    agg_local_weather_df.sort_values(by=['State', 'DateTime'], inplace=True)
    agg_state_df.sort_values(by=['State', 'DateTime'], inplace=True)
    final_df = pd.merge(
        agg_local_weather_df,
        agg_state_df,
        on=['State', 'DateTime'],
        how='inner'
    )
    print("Merge complete.")
    return final_df


def load_and_merge_capacity_data(df_main: pd.DataFrame, capacity_filepath: Path) -> pd.DataFrame:
    """Loads generator capacity data from a multi-sheet Excel file and merges it."""
    print(f"--- Loading generator capacity data from {capacity_filepath.name} ---")
    try:
        sheets_dict = pd.read_excel(capacity_filepath, sheet_name=None)
        all_sheets_list = []
        for sheet_name, sheet_df in sheets_dict.items():
            # --- ACTION REQUIRED: Verify these column names match your Excel file ---
            original_name_col = 'Name'
            original_capacity_col = 'Gen. Capacity (MW)'
            sheet_df.rename(columns={
                original_name_col: 'Location',
                original_capacity_col: 'Capacity_MW'
            }, inplace=True)
            if 'Location' in sheet_df.columns and 'Capacity_MW' in sheet_df.columns:
                all_sheets_list.append(sheet_df[['Location', 'Capacity_MW']])
            else:
                print(
                    f"⚠️ WARNING: Sheet '{sheet_name}' is missing '{original_name_col}' or '{original_capacity_col}'. Skipping.")
        if not all_sheets_list:
            raise ValueError("No valid generator capacity data found in any sheet with the specified column names.")
        df_capacity = pd.concat(all_sheets_list, ignore_index=True)
        df_capacity.drop_duplicates(subset=['Location'], inplace=True)
        print("Merging capacity data into the main dataset...")
        df_merged = pd.merge(df_main, df_capacity, on='Location', how='left')
        unmatched_count = df_merged['Capacity_MW'].isnull().sum()
        if unmatched_count > 0:
            unmatched_locations = df_merged[df_merged['Capacity_MW'].isnull()]['Location'].unique()
            print(f"⚠️ WARNING: {unmatched_count:,} records had no matching capacity information.")
            print(f"   (Example unmatched locations: {list(unmatched_locations[:5])})")
        return df_merged
    except FileNotFoundError:
        print(f"❌ ERROR: Capacity file not found at {capacity_filepath}")
        return df_main
    except Exception as e:
        print(f"❌ ERROR: Failed to process capacity file. Error: {e}")
        return df_main

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    SOURCE_FILE_MAP = {
        ('NSW', 'SOLAR'): {'generator_file': 'NSW-SOLAR.xlsx', 'rrp_file': 'New South Wales_master_with_weather.csv'},
        ('NSW', 'WIND'): {'generator_file': 'NSW-WIND.xlsx', 'rrp_file': 'New South Wales_master_with_weather.csv'},
        ('QLD', 'SOLAR'): {'generator_file': 'QLD-SOLAR.xlsx', 'rrp_file': 'Queensland_master_with_weather.csv'},
        ('QLD', 'WIND'): {'generator_file': 'QLD-WIND.xlsx', 'rrp_file': 'Queensland_master_with_weather.csv'},
        ('VIC', 'SOLAR'): {'generator_file': 'VIC-SOLAR.xlsx', 'rrp_file': 'Victoria_master_with_weather.csv'},
        ('VIC', 'WIND'): {'generator_file': 'VIC-WIND.xlsx', 'rrp_file': 'Victoria_master_with_weather.csv'}
    }

    DATA_LIMIT_PER_GROUP = 100_000
    CAPACITY_FILE_PATH = Path(r'C:\projects\HONOURS\historical weather data\Wind & Solar Gen. Open Elec.xlsx')

    for (state, gen_type), files in SOURCE_FILE_MAP.items():
        model_name = f"{state}_{gen_type}"
        print(f"\n{'=' * 30}")
        print(f"   PROCESSING GROUP: {model_name.upper()}")
        print(f"{'=' * 30}")

        # Initialise dataframes to prevent errors in the 'finally' block if loading fails
        local_weather_df, master_rrp_df, final_combined_df, df_with_capacity = [pd.DataFrame() for _ in range(4)]

        try:
            # Load and resample both datasets to a 3-hour frequency
            local_weather_df = consolidate_generator_data(
                SOURCE_GENERATOR_DIR,
                files['generator_file'],
                resample_freq='3h',
                limit_points=DATA_LIMIT_PER_GROUP
            )
            master_rrp_df = consolidate_state_rrp_data(
                SOURCE_STATE_DIR,
                files['rrp_file'],
                state,
                resample_freq='3h',
                limit_points=DATA_LIMIT_PER_GROUP
            )
            if local_weather_df.empty or master_rrp_df.empty:
                print(f"⚠️ WARNING: Data loading failed for {model_name}. Skipping.")
                continue

            final_combined_df = create_final_dataset(
                agg_state_df=master_rrp_df,
                agg_local_weather_df=local_weather_df
            )

            # Merge generator capacity
            df_with_capacity = load_and_merge_capacity_data(final_combined_df, CAPACITY_FILE_PATH)

            if df_with_capacity.empty:
                print(f"⚠️ WARNING: Dataframe is empty after merging for {model_name}. Skipping.")
                continue

            # Define the binary curtailment event for probability calculations
            # An event is defined as any period where the RRP is below -$50.
            print("--- Defining binary 'Curtailment_Event' based on RRP < -$50 ---")
            df_with_capacity['Curtailment_Event'] = (df_with_capacity['RRP'] < -50).astype(int)
            print("✅ 'Curtailment_Event' column created.")

            # Generate and plot probability distributions for each location
            attributes_to_plot = ['RRP', 'Temp_C', 'Wind_Speed_100m_kmh', 'GHI_Wm-2']
            unique_locations = df_with_capacity['Location'].unique()

            print(f"\n--- Found {len(unique_locations)} unique locations for {model_name}. Generating distributions... ---")

            for location in unique_locations:
                for attribute in attributes_to_plot:
                    # Check if the attribute exists in the dataframe before plotting
                    if attribute in df_with_capacity.columns:
                        output_folder = Path("probability_distributions") / model_name
                        plot_probability_distribution(
                            df=df_with_capacity,
                            attribute=attribute,
                            filter_by='Location',
                            filter_value=location,
                            output_dir=str(output_folder)
                        )
                    else:
                        # This is expected for e.g. 'Wind_Speed' in a solar dataset
                        pass

        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {model_name}: {e}")
        finally:
            # Clean up memory before the next loop
            del local_weather_df, master_rrp_df, final_combined_df, df_with_capacity
            gc.collect()

    print(f"\n\n{'=' * 30}")
    print("   PROCESSING COMPLETE")
    print("   Probability distribution plots have been saved to the 'probability_distributions' directory.")
    print(f"{'=' * 30}\n")