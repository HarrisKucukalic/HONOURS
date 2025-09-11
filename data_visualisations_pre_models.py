import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIRECTORY = r'C:\projects\HONOURS'

MASTER_FILES = [
    os.path.join(DATA_DIRECTORY, 'New South Wales_master_with_weather.csv'),
    os.path.join(DATA_DIRECTORY, 'Queensland_master_with_weather.csv'),
    os.path.join(DATA_DIRECTORY, 'Victoria_master_with_weather.csv')
]

OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'plots')
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Created directory for plots at: {OUTPUT_DIRECTORY}")


def plot_all_time_series(filepath):
    """
    Loads a master data file and creates a separate time series plot for each
    numerical column against the DateTime index.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\nGenerating Individual Time Series Plots for: {state_name}")

    try:
        df = pd.read_csv(filepath, index_col='DateTime', parse_dates=True)
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue

            print(f"  - Plotting {column}...")
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(18, 8))
            ax.plot(df.index, df[column], label=column, linewidth=1.5)

            ax.set_title(f'Time Series of {column} for {state_name}', fontsize=24, fontweight='bold')
            ax.set_xlabel('Date and Time', fontsize=20)
            ax.set_ylabel(f'{column}', fontsize=20)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=14)

            fig.autofmt_xdate()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            safe_col_name = column.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').replace('/', '')
            output_filename = f"{state_name}_timeseries_{safe_col_name}.png"
            output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
            plt.savefig(output_path, dpi=200)
            plt.close(fig)

        print(f"✅ All individual time series plots for {state_name} saved successfully.")

    except Exception as e:
        print(f"❌ An error occurred while generating time series plots for {state_name}: {e}")


def plot_rrp_time_series(filepath, y_max=300):
    """
    Creates a time series plot specifically for the RRP column, with a capped
    y-axis to show typical variations more clearly.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\nGenerating Capped RRP Time Series Plot for: {state_name}")

    try:
        df = pd.read_csv(filepath, index_col='DateTime', parse_dates=True)

        if 'RRP' not in df.columns:
            print(f"⚠️ WARNING: 'RRP' column not found in {state_name}'s file. Skipping RRP plot.")
            return

        print(f"Plotting RRP with y-axis capped at {y_max}.")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.plot(df.index, df['RRP'], label='RRP', color='darkviolet', linewidth=1.5)
        ax.set_title(f'RRP Time Series for {state_name} (Capped at ${y_max}/MWh)', fontsize=24, fontweight='bold')
        ax.set_xlabel('Date and Time', fontsize=20)
        ax.set_ylabel('RRP ($/MWh)', fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=14)

        ax.set_ylim(bottom=-50, top=y_max)

        fig.autofmt_xdate()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        output_filename = f"{state_name}_rrp_capped_timeseries.png"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"✅ Capped RRP plot for {state_name} saved successfully.")

    except Exception as e:
        print(f"❌ An error occurred while generating the capped RRP plot for {state_name}: {e}")


def plot_combined_time_series(filepath):
    """
    Loads a master data file and plots generation and price columns on a single
    time series chart, using a secondary y-axis for RRP.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\nGenerating Combined Generation vs. Price Plot for: {state_name}")

    try:
        df = pd.read_csv(filepath, index_col='DateTime', parse_dates=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(18, 8))
        ax2 = ax1.twinx()
        color_map = {'Wind': 'deepskyblue', 'Solar (Utility)': 'green', 'Solar (Rooftop)': 'gold'}
        lines, labels = [], []

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

        if 'RRP' in df.columns:
            line = ax2.plot(df.index, df['RRP'], label='RRP', color='red', linestyle='--', linewidth=2)
            lines.extend(line)

        ax1.set_title(f'Combined Generation vs. Price for {state_name}', fontsize=24, fontweight='bold')
        ax1.set_xlabel('Date and Time', fontsize=20)
        ax1.set_ylabel('Generation (MW)', fontsize=20, color='blue')
        ax2.set_ylabel('RRP ($/MWh)', fontsize=20, color='red')

        ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=14)

        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=14)

        fig.autofmt_xdate()
        plt.tight_layout()

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
        df = pd.read_csv(filepath)
        numeric_df = df.select_dtypes(include='number')
        correlation_matrix = numeric_df.corr()
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(16, 14))

        heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5,
                              annot_kws={"size": 14})

        heatmap.set_title(f'Full Correlation Matrix for {state_name}', fontsize=24, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()

        output_filename = f"{state_name}_full_correlation_heatmap.png"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Full correlation heatmap for {state_name} saved successfully.")
    except Exception as e:
        print(f"❌ An error occurred while generating the correlation heatmap for {state_name}: {e}")

def plot_negative_price_correlation_heatmap(filepath):
    """
    Filters data for periods of zero or negative RRP and plots a correlation
    heatmap for those specific periods.
    """
    state_name = os.path.basename(filepath).replace('_master_with_weather.csv', '')
    print(f"\nGenerating Correlation Heatmap for Zero/Negative Price Events in: {state_name}")

    try:
        df = pd.read_csv(filepath)

        # Filter the DataFrame for rows where RRP is <= 0
        df_negative_price = df[df['RRP'] <= 0].copy()

        # Check if there's enough data to create a meaningful plot
        if len(df_negative_price) < 10:
            print(f"  - ⚠️ Insufficient data ({len(df_negative_price)} rows) for negative price correlation. Skipping.")
            return

        print(f"  - Analyzing {len(df_negative_price)} data points with zero or negative prices.")

        numeric_df = df_negative_price.select_dtypes(include='number')
        correlation_matrix = numeric_df.corr()

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(16, 14))

        heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5,
                              annot_kws={"size": 14})

        heatmap.set_title(f'Correlation Matrix During Zero/Negative Price Events for {state_name}', fontsize=24,
                          fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()

        output_filename = f"{state_name}_negative_price_correlation_heatmap.png"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Negative price correlation heatmap for {state_name} saved successfully.")
    except Exception as e:
        print(
            f"❌ An unexpected error occurred while generating the negative price correlation heatmap for {state_name}: {e}")


def count_rrp_extremes(file_path):
    """
    Reads a CSV file, counts the number of days the RRP was $1000 and above
    and -$100 and below, as well as spikes ($10000 and above) and prints the results.

    Args:
        file_path (str): The path to the master CSV file.
    """
    try:
        print(f"\nAnalyzing file: {os.path.basename(file_path)}...")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        spikes = 10000
        max = 1000
        min = -100
        # Ensure the 'RRP' column exists
        if 'RRP' not in df.columns:
            print(f"  ❌ ERROR: 'RRP' column not found in {os.path.basename(file_path)}.")
            return

        # Count occurrences above 1000
        days_above_1000 = (df['RRP'] >= max).sum()
        max_price_spikes = (df['RRP'] >= spikes).sum()
        # Count occurrences below -500
        days_below_minus_500 = (df['RRP'] <= min).sum()

        print(f"  📈 Days with RRP >= {max}: {days_above_1000}")
        print(f"  📉 Days with RRP <= {min}: {days_below_minus_500}")
        print(f"  📉 Days with RRP >= {spikes}: {max_price_spikes}")

    except FileNotFoundError:
        print(f"  ❌ ERROR: File not found at '{file_path}'.")
    except Exception as e:
        print(f"  ❌ ERROR: An unexpected error occurred while processing {file_path}: {e}")


if __name__ == "__main__":
    # for master_file in MASTER_FILES:
    #     if os.path.exists(master_file):
    #         # plot_all_time_series(master_file)
    #         # plot_rrp_time_series(master_file)
    #         # plot_combined_time_series(master_file)
    #         # plot_correlation_heatmap(master_file)
    #         # plot_negative_price_correlation_heatmap(master_file)
    #     else:
    #         print(f"\n⚠️ WARNING: Master file not found at '{master_file}'. Skipping.")
    #
    # print("\nAll analysis and plotting complete.")

    MASTER_FILES = [
        "New South Wales_master_with_weather.csv",
        "Victoria_master_with_weather.csv",
        "Queensland_master_with_weather.csv",
    ]

    for master_file in MASTER_FILES:
        if os.path.exists(master_file):
            count_rrp_extremes(master_file)
            # You can still call your other plotting functions here if needed
            # plot_rrp_time_series(master_file)
        else:
            print(f"\n⚠️ WARNING: Master file not found at '{master_file}'. Skipping.")

    print("\nAnalysis complete.")
