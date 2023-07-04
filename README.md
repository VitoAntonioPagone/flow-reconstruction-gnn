# flow_reconstruction

## Project structure considerations
Branches:
- master branch: any code in the master branch should be working
- feature branch: create new descriptively-named branches off the main branch for new work, such as feature/add_variable_percentage

Good practice (not mandatory):
- Commit new work to your local branches and regularly push work to the remote
- To request feedback or help, or when you think your work is ready to merge into the master branch, open a pull request
- After your work or feature has been reviewed and approved, it can be merged into the master branch
- Follow PEP 8 style of coding whenever possible

## Data - Pipeline
Standard Convolution: standard_convolutions_data_pipeline.py

This script allows for the conversion of flow field simulation data from .vtp files to .npz format, interpolation of data onto a uniform grid, creation of input-label pairs with a certain percentage of the data removed from the inputs, and finally splitting the dataset into training and validation sets.

Steps

- Conversion from VTP to NPZ: Flow field simulation data stored in .vtp files is read and saved in .npz format for easier manipulation. This step is executed by the function vtp_to_npz.
- Interpolation: The data is interpolated onto a uniform grid of size GRID_SIZE x GRID_SIZE. Interpolation is performed separately for each feature in the dataset. This step is performed by the interpolate_data function.
- Input-label creation: For each interpolated .npz file, an input file and a corresponding label file are created. In the input file, a certain percentage (defined by PERCENT_TO_REMOVE) of the points have their velocity features set to zero, while the label file contains the complete data. The input and label files are saved in .npy format. This step is performed by the create_inputs_labels function.
- Training and validation split: The dataset is split into training and validation sets based on TRAIN_SPLIT (default is 90%). This means 90% of the data will be used for training and 10% for validation. This step is performed by the tran_val_split function.


Usage

This script is intended to be run from the command line with Python:

standard_convolutions_data_pipeline.py

The script uses a number of constants that can be modified at the beginning of the script to fit your specific use case, including grid size, percentage of data to remove, and directory paths for input and output data.

All output directories are created by the script if they do not already exist.

This script requires the following libraries: os, glob, numpy, vtk, scipy, pathlib, shutil, and natsort. Make sure to install these libraries before running the script.

You can install the necessary libraries using pip:

pip install vtk scipy pathlib numpy natsort

This script provides a streamlined way to process flow field simulation data for machine learning applications, particularly for tasks involving sparse data or data completion.
