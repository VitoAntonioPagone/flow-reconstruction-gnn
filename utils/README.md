## Flow Reconstruction


### AutoEncoder - Data Pipeline

1. **data_to_ground_truth**: The "read_vtp_slice(file_name)" function is defined to read and extract data from VTP (Visualization Toolkit PolyData) files. It returns a dictionary containing the extracted data.
The "interpolate_data(input_folder, *output_folders)" function is defined to interpolate the data using a griddata method from the Scipy library. It takes one input folder and one or more output folders as arguments, reads the VTP files in the input folder, extracts the data using the "read_vtp_slice()" function, interpolates the data, and saves the output to the specified output folders as compressed numpy files.
2. **input_label_generator.py**:  This code is an implementation of a data processing algorithm that reduces the amount of data by removing a specified percentage of data points using random selection and saves the processed data as numpy files. The algorithm is used to prepare the data for training and testing in a machine learning model.Several constant variables are defined to specify input and output directories, file names, and other constants. Specifically, the "PERCENT_TO_REMOVE" variable specifies the percentage of data points to remove from the original data.
The "create_inputs_labels(input_dir, output_dir_input, output_dir_labels, percent_to_remove)" function is defined to reduce the amount of data by removing a specified percentage of data points and saves the processed data as numpy files. It takes an input directory, output directories for inputs and labels, and a percentage of points to remove as arguments. The function reads the npz files in the input directory, removes the specified percentage of data points using random selection, saves the reduced data as numpy files in the specified output directories, and returns the file names of the saved files.
3. **split_train_validation**: this code is an implementation of a train-validation split algorithm that shuffles the input and label data files in a specified directory, splits them into training and validation sets using a specified fraction, and moves the validation files to the specified validation directories. The algorithm is used to prepare the data for training and testing in a machine learning model.

### MLP - Data Pipeline

1. **mlp_combined_datasets.py** : The code combines the data from multiple npz files in the specified input directories for the train, test, and validation sets, respectively, and saves the combined data in new npz files in the specified output directories.

First, the paths to the original input directories and output directories are set. Then, the output directories are created if they don't exist yet.

Next, there are two functions defined:

concatenate_npz_files(npz_files, input_dir) that takes a list of npz files and the input directory as arguments, loads the data from each file using numpy's load() function, and appends the data into a list. Finally, it stacks the data into a single numpy array using numpy's vstack() function and returns the result.
process_data(input_dir, output_dir) that takes the input and output directories as arguments, gets a list of npz files in the input directory, calls the concatenate_npz_files() function to get the combined data, and saves it as an npz file in the output directory using numpy's savez_compressed() function.
Finally, the process_data() function is called three times to process the train, test, and validation sets, respectively. The combined data for each set is saved in new npz files in the corresponding output directories.
2. **mlp_data_reader_from_vtp.py** : This code defines a function called "read_vtp_slice(file_name)" that reads and extracts data from a VTP file. It uses the vtk library to read the file and extract the x, y coordinates, x, y, and z velocity, and temperature data. The extracted data is returned in a dictionary format.

The code then sets the input and output directories and creates the output directory if it doesn't exist. It loops through all the VTP files in the input directory and uses the "read_vtp_slice()" function to extract the data from each file. It then saves the extracted data as an NPZ (compressed numpy) file with the same base name as the input VTP file in the output directory. The saved data can be used for further processing or analysis. The progress of the loop is printed to the console for each file being analyzed.


