## Flow Reconstruction


### AutoEncoder - Data Pipeline

1. data_to_ground_truth: The "read_vtp_slice(file_name)" function is defined to read and extract data from VTP (Visualization Toolkit PolyData) files. It returns a dictionary containing the extracted data.
The "interpolate_data(input_folder, *output_folders)" function is defined to interpolate the data using a griddata method from the Scipy library. It takes one input folder and one or more output folders as arguments, reads the VTP files in the input folder, extracts the data using the "read_vtp_slice()" function, interpolates the data, and saves the output to the specified output folders as compressed numpy files.
2. input_label_generator.py:  This code is an implementation of a data processing algorithm that reduces the amount of data by removing a specified percentage of data points using random selection and saves the processed data as numpy files. The algorithm is used to prepare the data for training and testing in a machine learning model.Several constant variables are defined to specify input and output directories, file names, and other constants. Specifically, the "PERCENT_TO_REMOVE" variable specifies the percentage of data points to remove from the original data.
The "create_inputs_labels(input_dir, output_dir_input, output_dir_labels, percent_to_remove)" function is defined to reduce the amount of data by removing a specified percentage of data points and saves the processed data as numpy files. It takes an input directory, output directories for inputs and labels, and a percentage of points to remove as arguments. The function reads the npz files in the input directory, removes the specified percentage of data points using random selection, saves the reduced data as numpy files in the specified output directories, and returns the file names of the saved files.
3. split_train_validation: this code is an implementation of a train-validation split algorithm that shuffles the input and label data files in a specified directory, splits them into training and validation sets using a specified fraction, and moves the validation files to the specified validation directories. The algorithm is used to prepare the data for training and testing in a machine learning model.

### MLP - Data Pipeline

To install the project, follow these steps:

1. Step 1: Do this.
2. Step 2: Do that.
3. Step 3: Do something else.

