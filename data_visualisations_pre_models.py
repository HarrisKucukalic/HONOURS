import pandas as pd
import matplotlib.pyplot as plt
import os


def create_time_series_plot(csv_filepath, date_column, value_column='RRP'):
    """
    Reads a CSV file and creates a time series plot of a specified column.

    Args:
        csv_filepath (str): The full path to the input CSV file.
        date_column (str): The name of the column containing datetime information.
        value_column (str): The name of the column containing the data to plot.
    """
    # --- File Validation ---
    if not os.path.exists(csv_filepath):
        print(f"Error: The file '{csv_filepath}' was not found.")
        return

    print(f"Processing file: {csv_filepath}")

    try:
        # --- Data Loading and Preparation ---
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_filepath)

        # Check if the required columns exist in the DataFrame
        if date_column not in df.columns:
            print(f"Error: Date column '{date_column}' not found in the CSV file.")
            return
        if value_column not in df.columns:
            print(f"Error: Value column '{value_column}' not found in the CSV file.")
            return

        # Convert the 'SETTLEMENTDATE' column to datetime objects
        # The errors='coerce' argument will turn any unparseable dates into NaT (Not a Time)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Drop rows where the date could not be parsed
        df.dropna(subset=[date_column], inplace=True)

        # Set the datetime column as the index of the DataFrame
        # This is a key step for creating a time series
        df.set_index(date_column, inplace=True)

        # Sort the data by date to ensure the plot is in chronological order
        df.sort_index(inplace=True)

        print(f"Successfully loaded and processed {len(df)} data points.")

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')  # Using a nice style for the plot
        fig, ax = plt.subplots(figsize=(15, 7))  # Create a figure and axes for plotting

        # Plot the specified value column against the index (which is our datetime)
        ax.plot(df.index, df[value_column], label=value_column, color='dodgerblue', linewidth=1.5)

        # --- Formatting the Plot ---
        # Set the title and labels for the axes
        region_name = os.path.basename(csv_filepath).split('_')[0]
        ax.set_title(f'Time Series of {value_column} for {region_name}', fontsize=16,
                     fontweight='bold')
        ax.set_xlabel('Date and Time', fontsize=12)
        ax.set_ylabel(f'{value_column} ($/MWh)', fontsize=12)

        # Improve the formatting of the date labels on the x-axis
        fig.autofmt_xdate()

        # Add a legend and grid for better readability
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add a tight layout
        plt.tight_layout()

        # --- Saving and Displaying the Plot ---
        # Create an output filename based on the input file
        output_filename = f"{os.path.splitext(os.path.basename(csv_filepath))[0]}_timeseries.png"

        # Save the plot to a PNG file
        plt.savefig(output_filename, dpi=300)
        print(f"Plot successfully saved as '{output_filename}'")

        # Display the plot
        plt.show()

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # Specify the CSV file you want to analyze.
    # This should be one of the files created by the previous script.
    # For example: 'NSW_PRICE_DEMAND_AEMO_combined.csv'

    input_csv = r'C:\projects\HONOURS\Queensland_corrected.csv'

    # Specify the columns you want to plot.
    # 'SETTLEMENTDATE' is the time column.
    # For the value, you could use 'RRP' (Regional Reference Price) or 'TOTALDEMAND'.
    date_col = 'DateTime'
    value_col_to_plot = 'Wind -  MW'  # You can change this to 'TOTALDEMAND' or another relevant column

    # --- Execution ---
    # Call the function with your specified configuration
    create_time_series_plot(input_csv, date_column=date_col, value_column=value_col_to_plot)
