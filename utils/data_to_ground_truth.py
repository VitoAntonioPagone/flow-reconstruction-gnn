import os
import glob
import numpy as np
import vtk as vtk
from scipy.interpolate import griddata
from pathlib import Path


GRID_SIZE = 256
INPUT_TRAIN_DIR  = '/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset/original_data/train'
OUTPUT_TRAIN_DIR = '/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset/npz_data/train/'
INPUT_TEST_DIR   = '/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset/original_data/test/'
OUTPUT_TEST_DIR  = '/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset/npz_data/test/'


def read_vtp_slice(file_name):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    data_in = reader.GetOutput()

    data_out = {
        'x': np.array(data_in.GetPoints().GetData())[:, 0],
        'y': np.array(data_in.GetPoints().GetData())[:, 1],
        'x_velocity': np.array(data_in.GetPointData().GetArray("x_velocity")),
        'y_velocity': np.array(data_in.GetPointData().GetArray("y_velocity")),
        'z_velocity': np.array(data_in.GetPointData().GetArray("z_velocity")),
        #'temperature': np.array(data_in.GetPointData().GetArray("temperature"))
        }

    return data_out


def interpolate_data(input_folder, *output_folders):
    vtp_files = glob.glob(os.path.join(input_folder, '*.vtp'))
    for vtp_file in vtp_files:
        filename = os.path.basename(vtp_file)
        print(f'Analyzing file: {filename}')

        data = read_vtp_slice(vtp_file)

        x_grid = np.linspace(min(data['x']), max(data['x']), GRID_SIZE)
        y_grid = np.linspace(min(data['y']), max(data['y']), GRID_SIZE)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # Grid data
        grid_data = {} 
        for feature in data.keys():
            if feature not in ['x', 'y']:
                grid_data[feature] = griddata(
                    np.array([data['x'], data['y']]).transpose(), data[feature],
                    (X_grid, Y_grid), method='nearest'
                )

        # Save output to all specified output folders
        for output_folder in output_folders:
            # Create output folder if it doesn't exist
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            output_file_name = f'interpolated_{os.path.splitext(filename)[0]}.npz'
            output_file = os.path.join(output_folder, output_file_name)
            np.savez_compressed(output_file, **grid_data)

if __name__ == "__main__":
    interpolate_data(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR)
    interpolate_data(INPUT_TEST_DIR, OUTPUT_TEST_DIR)