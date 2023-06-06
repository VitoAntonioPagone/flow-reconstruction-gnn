import numpy as np
import os
import random
import glob
from scipy.spatial import KDTree

def extract_uniform_points(data, percentage):
    num_points = int(data.shape[0] * percentage)
    x_min, y_min = np.min(data[:, :2], axis=0)
    x_max, y_max = np.max(data[:, :2], axis=0)

    x_values = np.linspace(x_min, x_max, int(np.sqrt(num_points)))
    y_values = np.linspace(y_min, y_max, int(np.sqrt(num_points)))

    grid_points = np.array([(x, y) for x in x_values for y in y_values])

    tree = KDTree(data[:, :2])
    _, indices = tree.query(grid_points, k=1)

    return np.unique(indices)

def process_npz_files(folder, output_folder, percentage):
    npz_files = glob.glob(os.path.join(folder, '*.npz'))
    
    for file_path in npz_files:
        print(f"Processing file: {file_path}")
        with np.load(file_path) as data:
            x = data['x']
            y = data['y']
            x_velocity = data['x_velocity']
            y_velocity = data['y_velocity']
            z_velocity = data['z_velocity']

        features = np.column_stack((x, y, x_velocity, y_velocity, z_velocity))
        indices_to_keep = extract_uniform_points(features, percentage)
        mask = np.ones(features.shape[0], dtype=bool)
        mask[indices_to_keep] = False

        features[mask, 2:] = 0
        output_file_path = os.path.join(output_folder, os.path.basename(file_path))

        np.savez(output_file_path, x=features[:, 0], y=features[:, 1],
                 x_velocity=features[:, 2], y_velocity=features[:, 3],
                 z_velocity=features[:, 4])

def main():
    train_folder = '../dataset_graph/original_data/npz_data/train'  # Training data folder
    test_folder = '../dataset_graph/original_data/npz_data/test'    # Testing data folder
    validation_folder = '../dataset_graph/original_data/npz_data/validation'
    train_output_folder = '../dataset_graph/original_data/npz_data/train_inputs'
    test_output_folder = '../dataset_graph/original_data/npz_data/test_inputs'
    validation_output_folder = '../dataset_graph/original_data/npz_data/validation_inputs'

    percentage = 0.5  

    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)
    os.makedirs(validation_output_folder, exist_ok=True)

    process_npz_files(train_folder, train_output_folder, percentage)
    process_npz_files(test_folder, test_output_folder, percentage)
    process_npz_files(validation_folder, validation_output_folder, percentage)

if __name__ == "__main__":
    main()
