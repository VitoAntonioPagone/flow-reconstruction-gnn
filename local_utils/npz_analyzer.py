import numpy as np


def load_npz_file(file_path):
    data = np.load(file_path)
    return data


def print_npz_info(npz_data):
    print("NPZ Information:")
    print("------------------")
    for key in npz_data.files:
        value = npz_data[key]
        print(f"Name: {key}")
        print(f"Shape: {value.shape}")
        print(f"Data Type: {value.dtype}")

        if value.shape[0] >= 10:
            print("First 10 data points:")
            print(value[:10])
        else:
            print("Data points:")
            print(value)
        print()

    if set(["x_velocity", "y_velocity", "z_velocity"]).issubset(npz_data.files):
        velocities = np.column_stack(
            (npz_data["x_velocity"], npz_data["y_velocity"], npz_data["z_velocity"])
        )
        zero_velocity_samples = np.all(velocities == 0, axis=1)
        zero_velocity_percentage = np.mean(zero_velocity_samples) * 100
        print(
            f"Percentage of samples with zero velocity: {zero_velocity_percentage:.2f}%"
        )


npz_file_path = (
    "../dataset/npz_data_interpolated/train/interpolated_cyc05_CAD620_Y1_Z0_X2.npz"
)
npz_data = load_npz_file(npz_file_path)
print_npz_info(npz_data)
