import numpy as np

def load_npz_file(file_path):
    data = np.load(file_path)
    return data

def print_npz_info(npz_data):
    print("NPZ Information:")
    print("------------------")
    for key, value in npz_data.items():
        print(f"Name: {key}")
        print(f"Shape: {value.shape}")
        print(f"Data Type: {value.dtype}\n")

    # If the .npz file contains the expected keys, calculate the percentage of samples where all velocities are zero
    if set(['x_velocity', 'y_velocity', 'z_velocity']).issubset(npz_data.files):
        velocities = np.column_stack((npz_data['x_velocity'], npz_data['y_velocity'], npz_data['z_velocity']))
        zero_velocity_samples = np.all(velocities == 0, axis=1)
        zero_velocity_percentage = np.mean(zero_velocity_samples) * 100
        print(f"Percentage of samples with zero velocity: {zero_velocity_percentage:.2f}%")

npz_file_path = "../dataset/npz_data_interpolated/test/interpolated_cyc10_CAD615_Y0_Z0_X0.npz"
npz_data = load_npz_file(npz_file_path)
print_npz_info(npz_data)
