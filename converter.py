import os
import glob
import pandas as pd

def convert_xlsx_to_csv(directory_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .xlsx files in the directory
    for file_path in glob.glob(os.path.join(directory_path, '*.xlsx')):
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Construct the CSV file name
        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)
        csv_file_path = os.path.join(output_folder, file_name + '.csv')

        # Save to CSV file
        df.to_csv(csv_file_path, index=False)
        print(f'Converted {file_path} to {csv_file_path}')

# Example usage
directory_path = 'dataset'  # Replace with the path to your directory containing .xlsx files
output_folder = 'dataset/csv'                      # The folder where the CSV files will be saved

convert_xlsx_to_csv(directory_path, output_folder)
