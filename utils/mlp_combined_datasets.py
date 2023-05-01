import os
import numpy as np

# Set the paths to original train, test, and validation folders
train_input_dir = "flow_reconstruction/dataset_mlp/npz_data/train"
test_input_dir = "flow_reconstruction/dataset_mlp/npz_data/test"
val_input_dir = "flow_reconstruction/dataset_mlp/npz_data/validation"

# Set the paths to the output folders
train_output_dir = "flow_reconstruction/dataset_mlp/train_data/train_combined"
test_output_dir = "flow_reconstruction/dataset_mlp/train_data/test_combined"
val_output_dir = "flow_reconstruction/dataset_mlp/train_data/validation_combined"

# Create output directories if they do not exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Define a function to concatenate data from a list of .npz files
def concatenate_npz_files(npz_files, input_dir):
    combined_data = []
    for npz_file in npz_files:
        file_path = os.path.join(input_dir, npz_file)
        data = np.load(file_path)["data"]
        combined_data.append(data)

    return np.vstack(combined_data)

# Define a function to process data for a given input directory and output directory
def process_data(input_dir, output_dir):
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    combined_data = concatenate_npz_files(npz_files, input_dir)
    output_file = os.path.join(output_dir, "combined_data.npz")
    np.savez_compressed(output_file, data=combined_data)
    print(f"Combined file: {output_file}")

# Process data for train, test, and validation folders
process_data(train_input_dir, train_output_dir)
process_data(test_input_dir, test_output_dir)
process_data(val_input_dir, val_output_dir)
