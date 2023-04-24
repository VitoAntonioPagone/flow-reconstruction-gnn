import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


def interpolate_data(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f'Analyzing file: {filename}')

        data = pd.read_csv(csv_file)

        grid_size = 256
        x_grid = np.linspace(data['x'].min(), data['x'].max(), grid_size)
        y_grid = np.linspace(data['y'].min(), data['y'].max(), grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        grid_data = {} 
        for feature in data.columns:
            if feature not in ['x', 'y']:
                grid_data[feature] = griddata(
                    data[['x', 'y']].values, data[feature].values,
                    (X_grid, Y_grid), method='nearest'
                )

        output_file_name = f'interpolated_{os.path.splitext(filename)[0]}.npz'
        output_file = os.path.join(output_folder, output_file_name)
        np.savez_compressed(output_file, **grid_data)


input_train_folder = 'Dataset/train/csvs'
output_train_folder = 'Dataset/train/ground_truth'
input_test_folder = 'Dataset/test/csvs'
output_test_folder = 'Dataset/test/ground_truth'

interpolate_data(input_train_folder, output_train_folder)
interpolate_data(input_test_folder, output_test_folder)
