import numpy as np
import glob
import os


def create_train_labels(input_dir, output_dir_train, output_dir_labels, percent_to_remove):
    if not os.path.exists(output_dir_train):
        os.makedirs(output_dir_train)

    if not os.path.exists(output_dir_labels):
        os.makedirs(output_dir_labels)

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
        np.save(output_dir_train + file_name + '_train.npy', train_data)

        label_data = np.stack([data[feature_name] for feature_name in data.files], axis=-1)
        np.save(output_dir_labels + file_name + '_label.npy', label_data)

    print(f"Files saved in {output_dir_train} and {output_dir_labels}.")


percent_to_remove = 50

input_train_dir = 'Dataset/train/ground_truth/'
input_test_dir = 'Dataset/test/ground_truth/'

output_train_dir = f'Dataset/train/train-{percent_to_remove}/'
output_test_dir = f'Dataset/test/test-{percent_to_remove}/'

output_train_labels_dir = f'Dataset/train/train-labels-{percent_to_remove}/'
output_test_labels_dir = f'Dataset/test/test-labels-{percent_to_remove}/'

create_train_labels(input_train_dir, output_train_dir, output_train_labels_dir, percent_to_remove)
create_train_labels(input_test_dir, output_test_dir, output_test_labels_dir, percent_to_remove)
