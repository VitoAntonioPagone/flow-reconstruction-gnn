

import numpy as np

# Load the .npz file
data = np.load('/Users/vitoantonio/Desktop/feature_network_architectures/flow_reconstruction/dataset_graph/original_data/npz_data/validation_inputs/cyc11_CAD605_Y3_Z0_X1.npz')

# Print the dimensions of the arrays
print("Dimensions:")
for name, arr in data.items():
    print(f"{name}: {arr.shape}")

# Print the first 100 rows of the arrays
print("\nFirst 100 rows:")
for name, arr in data.items():
    print(f"{name}:")
    if len(arr.shape) == 1:
        print(arr[:100])
    else:
        print(arr[:100, :])
    print()
