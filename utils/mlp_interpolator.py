import os
import glob
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

GRID_SIZE = 256

def interpolate_data(npz_files, output_folder):
    
    for npz_file in npz_files:
        filename = os.path.basename(npz_file)
        print(f'Analyzing file: {filename}')
        
        data = np.load(npz_file)
        data_dict = {key: data[key] for key in data.files}
        
        x_grid = np.linspace(min(data_dict['x']), max(data_dict['x']), GRID_SIZE)
        y_grid = np.linspace(min(data_dict['y']), max(data_dict['y']), GRID_SIZE)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        grid_data = {}
        for feature in data_dict.keys():
            if feature not in ['x', 'y']:
                grid_data[feature] = griddata(
                    np.array([data_dict['x'], data_dict['y']]).transpose(), data_dict[feature],
                    (X_grid, Y_grid), method='nearest'
                )

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_name = f'interpolated_{os.path.splitext(filename)[0]}.npz'
        output_file = os.path.join(output_folder, output_file_name)
        np.savez_compressed(output_file, **grid_data)

if __name__ == "__main__":
    input_folder1 =  "/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset_mlp/reconstructed_mlp"
    output_folder1 = "/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset_mlp/interpolated/interpolated_reconstructed_mlp/"
    input_folder2 =  "/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset_mlp/npz_data/test"
    output_folder2 = "/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset_mlp/interpolated/interpolated_test/"

    # Get a list of all the npz files in the first input folder
    npz_files1 = glob.glob(os.path.join(input_folder1, '*.npz'))
    # Call the interpolate_data function with the list of npz files and the first output folder
    interpolate_data(npz_files1, output_folder1)

    # Get a list of all the npz files in the second input folder
    npz_files2 = glob.glob(os.path.join(input_folder2, '*.npz'))
    # Call the interpolate_data function with the list of npz files and the second output folder
    interpolate_data(npz_files2, output_folder2)

