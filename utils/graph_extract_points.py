import numpy as np
import os
import random
import glob

# Constants
TRAIN_FOLDER = '../dataset_graph/original_data/npz_data/train'
TEST_FOLDER = '../dataset_graph/original_data/npz_data/test'
VALIDATION_FOLDER = '../dataset_graph/original_data/npz_data/validation'
PERCENTAGE = 0.9

def extract_random_points(data, percentage):
    num_points = int(data.shape[0] * percentage)
    indices = random.sample(range(data.shape[0]), num_points)
    return np.array(indices)

def process_npz_files(folder, percentage):
    npz_files = glob.glob(os.path.join(folder, '*.npz'))
    output_folder = f'{folder}_inputs_{percentage*100:.0f}'  
    
    os.makedirs(output_folder, exist_ok=True)
    
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
    process_npz_files(TRAIN_FOLDER, PERCENTAGE)
    process_npz_files(TEST_FOLDER, PERCENTAGE)
    process_npz_files(VALIDATION_FOLDER, PERCENTAGE)

if __name__ == "__main__":
    main()
