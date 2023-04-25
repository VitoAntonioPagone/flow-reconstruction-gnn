import os
import pandas as pd
import numpy as np

def combine_csv_files(input_folder, output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    combined_data = pd.concat([pd.read_csv(os.path.join(input_folder, file)) for file in all_files])
    combined_data.to_csv(os.path.join(output_folder, output_file), index=False)
    print(f"Combined {len(all_files)} CSV files from '{input_folder}' and saved in '{output_folder}' as '{output_file}'.")

if __name__ == "__main__":
    train_folder = os.path.abspath('./Dataset/train/csvs')
    test_folder = os.path.abspath('./Dataset/test/csvs')
    combined_train_folder = os.path.abspath('./MLP/Dataset/combined_train')
    combined_test_folder = os.path.abspath('./MLP/Dataset/combined_test')

    combine_csv_files(train_folder, combined_train_folder, 'combined_train.csv')
    combine_csv_files(test_folder, combined_test_folder, 'combined_test.csv')
