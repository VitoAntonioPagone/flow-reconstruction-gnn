import os
import glob
import numpy as np
import vtk
from scipy.interpolate import griddata
from pathlib import Path
import shutil
from natsort import natsorted

GRID_SIZE = 256
PERCENT_TO_REMOVE = 50
RANDOM_SEED = 1
TRAIN_SPLIT = 0.9

TRAIN_INPUT_FOLDER = "../dataset/original_data/train"
TRAIN_OUTPUT_FOLDER = "../dataset/original_data/npz_data/train"
TEST_INPUT_FOLDER = "../dataset/original_data/test"
TEST_OUTPUT_FOLDER = "../dataset/original_data/npz_data/test"
INTERPOLATED_TRAIN_OUTPUT_FOLDER = '../dataset/npz_data_interpolated/train/'
INTERPOLATED_TEST_OUTPUT_FOLDER = '../dataset/npz_data_interpolated/test/'
OUTPUT_TRAIN_INPUTS_DIR = f'../dataset/train_data_{PERCENT_TO_REMOVE}/train_inputs_{PERCENT_TO_REMOVE}/'
OUTPUT_TEST_INPUTS_DIR = f'../dataset/train_data_{PERCENT_TO_REMOVE}/test_inputs_{PERCENT_TO_REMOVE}/'
OUTPUT_TRAIN_LABELS_DIR = f'../dataset/train_data_{PERCENT_TO_REMOVE}/train_labels_{PERCENT_TO_REMOVE}/'
OUTPUT_TEST_LABELS_DIR = f'../dataset/train_data_{PERCENT_TO_REMOVE}/test_labels_{PERCENT_TO_REMOVE}/'
OUTPUT_VAL_INPUTS_DIR = f'../dataset/train_data_{PERCENT_TO_REMOVE}/val_inputs_{PERCENT_TO_REMOVE}/'
OUTPUT_VAL_LABELS_DIR = f'../dataset/train_data_{PERCENT_TO_REMOVE}/val_labels_{PERCENT_TO_REMOVE}/'

def read_vtp_slice(file_name):
    print(f"Reading vtp file: {file_name}")
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
    }
    print(f"Finished reading vtp file: {file_name}")
    return data_out

def vtp_to_npz(input_folder, output_folder):
    print(f"Processing files in folder: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    for vtp_file in glob.glob(os.path.join(input_folder, "*.vtp")):
        print(f"Processing file: {vtp_file}")
        data = read_vtp_slice(vtp_file)
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(vtp_file))[0] + ".npz")
        np.savez(output_file, **data)

    print(f"Finished processing files in folder: {input_folder}")

def interpolate_data(input_folder, output_folder):
    print(f"Interpolating files in folder: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    npz_files = glob.glob(os.path.join(input_folder, "*.npz"))
    for npz_file in npz_files:
        print(f"Interpolating file: {npz_file}")
        data = np.load(npz_file)
        x_grid = np.linspace(min(data['x']), max(data['x']), GRID_SIZE)
        y_grid = np.linspace(min(data['y']), max(data['y']), GRID_SIZE)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        grid_data = {} 
        for feature in data.keys():
            if feature not in ['x', 'y']:
                grid_data[feature] = griddata(
                    np.array([data['x'], data['y']]).transpose(), data[feature],
                    (X_grid, Y_grid), method='nearest'
                )

        output_file = os.path.join(output_folder, 'interpolated_' + os.path.splitext(os.path.basename(npz_file))[0] + '.npz')
        np.savez_compressed(output_file, **grid_data)

    print(f"Finished interpolating files in folder: {input_folder}")

def create_inputs_labels(input_dir, output_dir_input, output_dir_labels, percent_to_remove):
    print(f"Creating inputs and labels from folder: {input_dir}")
    Path(output_dir_input).mkdir(parents=True, exist_ok=True)
    Path(output_dir_labels).mkdir(parents=True, exist_ok=True)

    npz_files = glob.glob(input_dir + '*.npz')
    for npz_file in npz_files:
        data = np.load(npz_file)
        print(f'Analyzing file: {npz_file}')

        ny, nx = data[data.files[0]].shape
        n_points = ny * nx
        n_points_to_remove = int(n_points * percent_to_remove / 100)

        indices_to_remove = np.random.choice(n_points, n_points_to_remove, replace=False)

        features_reduced = []
        for feature_name in data.files:
            feature_data = data[feature_name].copy().flatten()
            feature_data[indices_to_remove] = 0
            feature_data = feature_data.reshape((ny, nx))
            features_reduced.append(feature_data)

        train_data = np.stack(features_reduced, axis=-1)

        file_name = os.path.splitext(os.path.basename(npz_file))[0]
        np.save(output_dir_input + file_name + '_input.npy', train_data)

        label_data = np.stack([data[feature_name] for feature_name in data.files], axis=-1)
        np.save(output_dir_labels + file_name + '_label.npy', label_data)

    print(f"Files saved in {output_dir_input} and {output_dir_labels}.")

def tran_val_split(train_dir, val_dir):
    print(f"Splitting training and validation data for folder: {train_dir}")
    files = glob.glob(train_dir + '*.npy')
    _, _, files = next(os.walk(train_dir))
    files = natsorted(files)
    num_files = len(files)

    range_files = np.arange(num_files)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(range_files)
    train_files = range_files[:int(num_files*TRAIN_SPLIT)]
    val_files = range_files[int(num_files*TRAIN_SPLIT):]

    Path(val_dir).mkdir(parents=True, exist_ok=True)

    for i in val_files:
        print(f'Moving file: {files[i]} from: {train_dir} to: {val_dir}')
        src = train_dir + files[i]
        dst = val_dir + files[i]
        shutil.move(src, dst)
    print(f"Finished splitting training and validation data for folder: {train_dir}")

if __name__ == "__main__":
    print("Starting to process TRAIN data...")
    vtp_to_npz(TRAIN_INPUT_FOLDER, TRAIN_OUTPUT_FOLDER)
    interpolate_data(TRAIN_OUTPUT_FOLDER, INTERPOLATED_TRAIN_OUTPUT_FOLDER)
    print("Finished processing TRAIN data.")

    print("Starting to process TEST data...")
    vtp_to_npz(TEST_INPUT_FOLDER, TEST_OUTPUT_FOLDER)
    interpolate_data(TEST_OUTPUT_FOLDER, INTERPOLATED_TEST_OUTPUT_FOLDER)
    print("Finished processing TEST data.")

    print("Starting to create inputs and labels for TRAIN data...")
    create_inputs_labels(INTERPOLATED_TRAIN_OUTPUT_FOLDER, OUTPUT_TRAIN_INPUTS_DIR, OUTPUT_TRAIN_LABELS_DIR, PERCENT_TO_REMOVE)
    print("Finished creating inputs and labels for TRAIN data.")

    print("Starting to create inputs and labels for TEST data...")
    create_inputs_labels(INTERPOLATED_TEST_OUTPUT_FOLDER, OUTPUT_TEST_INPUTS_DIR, OUTPUT_TEST_LABELS_DIR, PERCENT_TO_REMOVE)
    print("Finished creating inputs and labels for TEST data.")

    print("Starting to split TRAIN and VAL data...")
    tran_val_split(OUTPUT_TRAIN_INPUTS_DIR, OUTPUT_VAL_INPUTS_DIR)
    tran_val_split(OUTPUT_TRAIN_LABELS_DIR, OUTPUT_VAL_LABELS_DIR)
    print("Finished splitting TRAIN and VAL data.")