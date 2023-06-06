import os
import glob
import numpy as np
import vtk

TRAIN_INPUT_FOLDER = "../dataset/original_data/train"
TRAIN_OUTPUT_FOLDER = "../dataset/original_data/npz_data/train"
TEST_INPUT_FOLDER = "../dataset/original_data/test"
TEST_OUTPUT_FOLDER = "../flow_reconstruction/dataset/original_data/npz_data/test"


def read_vtp_slice(file_name):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    data_in = reader.GetOutput()

    data_out = {
        'x': np.array(data_in.GetPoints().GetData())[:, 0],
        'y': np.array(data_in.GetPoints().GetData())[:, 1],
        'x_velocity': np.array(data_in.GetPointData().GetArray("x_velocity")),
        'y_velocity': np.array(data_in.GetPointData().GetArray("y_velocity")),
        'z_velocity': np.array(data_in.GetPointData().GetArray("z_velocity")),
        #'temperature': np.array(data_in.GetPointData().GetArray("temperature"))
        }

    return data_out

def vtp_to_npz(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for vtp_file in glob.glob(os.path.join(input_folder, "*.vtp")):
        print(f"Processing file: {vtp_file}")
        data = read_vtp_slice(vtp_file)
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(vtp_file))[0] + ".npz")
        np.savez(output_file, **data)

# Convert train and test vtp files to npz files
vtp_to_npz(TRAIN_INPUT_FOLDER, TRAIN_OUTPUT_FOLDER)
vtp_to_npz(TEST_INPUT_FOLDER, TEST_OUTPUT_FOLDER)

