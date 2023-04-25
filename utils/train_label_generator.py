import numpy as np
import glob
import os


PERCENT_TO_REMOVE = 50
INPUT_TRAIN_DIR = 'Dataset/train_npz/'
INPUT_TEST_DIR = 'Dataset/test_npz/'
OUTPUT_TRAIN_DIR = f'Dataset/train_npz_{PERCENT_TO_REMOVE}/'
OUTPUT_TEST_DIR = f'Dataset/test_npz_{PERCENT_TO_REMOVE}/'
OUTPUT_TRAIN_LABELS_DIR = f'Dataset/train_labels_npz_{PERCENT_TO_REMOVE}/'
OUTPUT_TEST_LABELS_DIR = f'Dataset/test_labels_npz_{PERCENT_TO_REMOVE}/'


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


if __name__ == "__main__":
    create_train_labels(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR, OUTPUT_TRAIN_LABELS_DIR, PERCENT_TO_REMOVE)
    create_train_labels(INPUT_TEST_DIR, OUTPUT_TEST_DIR, OUTPUT_TEST_LABELS_DIR, PERCENT_TO_REMOVE)
