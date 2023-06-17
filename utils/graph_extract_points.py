import numpy as np
import os
import random
import glob

def extract_random_points(data, percentage):
    num_points = int(data.shape[0] * percentage)
    indices = random.sample(range(data.shape[0]), num_points)
    return np.array(indices)

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

        indices_to_remove = extract_random_points(features, percentage)
        features[indices_to_remove, 2:] = 0

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

