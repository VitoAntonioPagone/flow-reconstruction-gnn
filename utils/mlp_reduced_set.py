import numpy as np
import os
file1_path = "flow_reconstruction/dataset_mlp/train_data/train_combined/combined_data.npz"
file2_path = "flow_reconstruction/dataset_mlp/train_data/validation_combined/combined_data.npz"
output_folder1 = "flow_reconstruction/dataset_mlp/train_data_reduced/train_combined/"
output_folder2 = "flow_reconstruction/dataset_mlp/train_data_reduced/validation_combined/"
def load_and_extract_npz(file_path, num_samples=10000):
    data = np.load(file_path)
    extracted_data = {}
    for key in data.keys():
        extracted_data[key] = data[key][:num_samples]

    return extracted_data

def save_extracted_data(extracted_data, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    np.savez(os.path.join(output_folder, filename), **extracted_data)

def main():

    extracted_data1 = load_and_extract_npz(file1_path)
    extracted_data2 = load_and_extract_npz(file2_path)
    save_extracted_data(extracted_data1, output_folder1, "extracted_data1.npz")
    save_extracted_data(extracted_data2, output_folder2, "extracted_data2.npz")

if __name__ == "__main__":
    main()
