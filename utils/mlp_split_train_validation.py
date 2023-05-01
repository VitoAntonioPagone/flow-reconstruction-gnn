import os
import shutil
from natsort import natsorted
from pathlib import Path
import numpy as np

RANDOM_SEED = 1337
VALIDATION_SPLIT = 0.1
TRAIN_DIR = 'flow_reconstruction/dataset_mlp/npz_data/train/'
VALIDATION_DIR = 'flow_reconstruction/dataset_mlp/npz_data/validation/'


def train_validation_split(train_dir, validation_dir):
    _, _, files = next(os.walk(train_dir))
    files = natsorted(files)
    num_files = len(files)

    range_files = np.arange(num_files)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(range_files)
    
    validation_files = range_files[:int(num_files * VALIDATION_SPLIT)]

    # Create validation dir
    Path(validation_dir).mkdir(parents=True, exist_ok=True)

    # Move files to validation dir
    for i in validation_files:
        print(f'Moving file: {files[i]} from: {train_dir} to: {validation_dir}')
        src = train_dir + files[i]
        dst = validation_dir + files[i]
        shutil.move(src, dst)


if __name__ == "__main__":
    train_validation_split(TRAIN_DIR, VALIDATION_DIR)
