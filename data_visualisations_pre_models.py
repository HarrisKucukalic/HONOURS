import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---

# Directory where your final master files are located.
DATA_DIRECTORY = r'C:\projects\HONOURS'

# --- UPDATED: Point to the files that include weather data ---
MASTER_FILES = [
    os.path.join(DATA_DIRECTORY, 'New South Wales_master_with_weather.csv'),
    os.path.join(DATA_DIRECTORY, 'Queensland_master_with_weather.csv'),
    os.path.join(DATA_DIRECTORY, 'Victoria_master_with_weather.csv')
]

# Create an output directory for the plots if it doesn't exist
OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'plots')
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Created directory for plots at: {OUTPUT_DIRECTORY}")


def plot_all_time_series(filepath):
    """
    Loads a master data file and creates a separate time series plot for each
    numerical column against the DateTime index. This will now include weather data.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\n--- Generating Individual Time Series Plots for: {state_name} ---")

    try:
        # Load the data with DateTime as the index
        df = pd.read_csv(filepath, index_col='DateTime', parse_dates=True)

        # Loop through each column in the DataFrame
        for column in df.columns:
            # Skip non-numeric columns if any exist
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue

            print(f"  - Plotting {column}...")

            # --- Plotting ---
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(15, 7))

            ax.plot(df.index, df[column], label=column, linewidth=1)

            # --- Formatting ---
            ax.set_title(f'Time Series of {column} for {state_name}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date and Time', fontsize=12)
            ax.set_ylabel(f'{column}', fontsize=12)
            fig.autofmt_xdate()
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            # --- Saving ---
            # Sanitize column name for use in filename
            safe_col_name = column.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').replace('/', '')
            output_filename = f"{state_name}_timeseries_{safe_col_name}.png"
            output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

            plt.savefig(output_path, dpi=200)
            plt.close(fig)  # Close the figure to free up memory

        print(f"✅ All individual time series plots for {state_name} saved successfully.")

    except Exception as e:
        print(f"❌ An error occurred while generating time series plots for {state_name}: {e}")


def plot_combined_time_series(filepath):
    """
    Loads a master data file and plots generation and price columns on a single
    time series chart, using a secondary y-axis for RRP.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\n--- Generating Combined Generation vs. Price Plot for: {state_name} ---")

    try:
        # Load the data with DateTime as the index
        df = pd.read_csv(filepath, index_col='DateTime', parse_dates=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(18, 8))

        # Create a secondary y-axis for the RRP data
        ax2 = ax1.twinx()

        color_map = {
            'Wind': 'deepskyblue',
            'Solar (Utility)': 'green',
            'Solar (Rooftop)': 'gold'
        }

        lines, labels = [], []

        # Plot generation data on the primary axis (ax1)
        for column in df.columns:
            if 'RRP' not in column and 'MW' in column and pd.api.types.is_numeric_dtype(df[column]):
                plot_color = 'gray'
                if 'Wind' in column:
                    plot_color = color_map['Wind']
                elif 'Solar (Utility)' in column:
                    plot_color = color_map['Solar (Utility)']
                elif 'Solar (Rooftop)' in column:
                    plot_color = color_map['Solar (Rooftop)']

                line = ax1.plot(df.index, df[column], label=column, color=plot_color, linewidth=1.5)
                lines.extend(line)

        # Plot RRP data on the secondary axis (ax2)
        if 'RRP' in df.columns:
            line = ax2.plot(df.index, df['RRP'], label='RRP', color='red', linestyle='--', linewidth=2)
            lines.extend(line)

        # --- Formatting ---
        ax1.set_title(f'Combined Generation vs. Price for {state_name}', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Date and Time', fontsize=12)
        ax1.set_ylabel('Generation (MW)', fontsize=12, color='blue')
        ax2.set_ylabel('RRP ($/MWh)', fontsize=12, color='red')

        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')

        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        fig.autofmt_xdate()
        plt.tight_layout()

        # --- Saving ---
        output_filename = f"{state_name}_timeseries_gen_vs_price.png"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"✅ Combined generation vs. price plot for {state_name} saved successfully.")

    except Exception as e:
        print(f"❌ An error occurred while generating the combined plot for {state_name}: {e}")


def plot_correlation_heatmap(filepath):
    """
    Loads a master data file, calculates the correlation matrix for all numeric
    columns (including weather), and plots it as a Seaborn heatmap.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\n--- Generating Full Correlation Heatmap for: {state_name} ---")

    try:
        # Load the data
        df = pd.read_csv(filepath)

        # Select only numeric columns for correlation calculation
        numeric_df = df.select_dtypes(include='number')

        # Calculate the correlation matrix
        correlation_matrix = numeric_df.corr()

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        # Increase the figure size to accommodate more variables
        plt.figure(figsize=(16, 14))

        heatmap = sns.heatmap(
            correlation_matrix,
            annot=True,  # Show the correlation values on the map
            fmt='.2f',  # Format values to two decimal places
            cmap='coolwarm',  # Use a diverging color map
            linewidths=.5,
            annot_kws={"size": 8}  # Adjust font size for annotations
        )

        # --- Formatting ---
        heatmap.set_title(f'Full Correlation Matrix for {state_name}', fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # --- Saving ---
        output_filename = f"{state_name}_full_correlation_heatmap.png"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure

        print(f"✅ Full correlation heatmap for {state_name} saved successfully.")

    except Exception as e:
        print(f"❌ An error occurred while generating the correlation heatmap for {state_name}: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    for master_file in MASTER_FILES:
        if os.path.exists(master_file):
            plot_all_time_series(master_file)
            plot_combined_time_series(master_file)
            plot_correlation_heatmap(master_file)
        else:
            print(f"\n⚠️ WARNING: Master file not found at '{master_file}'. Skipping.")

    print("\n--- All analysis and plotting complete. ---")
