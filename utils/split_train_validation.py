import numpy as np
import os
import shutil
from natsort import natsorted
from pathlib import Path
import glob

RANDOM_SEED = 1337
TRAIN_SPLIT = 0.9
INPUT_TRAIN_INPUTS_DIR = 'dataset/train_data/train_inputs_50/'
OUTPUT_VAL_INPUTS_DIR = 'dataset/train_data/val_inputs_50/'
INPUT_TRAIN_LABELS_DIR = 'dataset/train_data/train_labels_50/'
OUTPUT_VAL_LABELS_DIR = 'dataset/train_data/val_labels_50/'


def tran_val_split(train_dir, val_dir):

    # files = glob.glob(train_dir + '*.npy')
    _, _, files = next(os.walk(train_dir))
    files = natsorted(files)
    num_files = len(files)

    range_files = np.arange(num_files)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(range_files)
    train_files = range_files[:int(num_files*TRAIN_SPLIT)]
    val_files = range_files[int(num_files*TRAIN_SPLIT):]

    # Create val dir
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    # Split images and masks into train and validation
    for i in val_files:
        print(f'Moving file: {files[i]} from: {train_dir} to: {val_dir}')
        src = train_dir + files[i]
        dst = val_dir + files[i]
        shutil.move(src, dst)


if __name__ == "__main__":
    tran_val_split(INPUT_TRAIN_INPUTS_DIR, OUTPUT_VAL_INPUTS_DIR)
    tran_val_split(INPUT_TRAIN_LABELS_DIR, OUTPUT_VAL_LABELS_DIR)

