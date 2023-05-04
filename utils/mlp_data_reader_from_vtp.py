
import os
import numpy as np
import vtk

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
        'temperature': np.array(data_in.GetPointData().GetArray("temperature"))}

    return data_out

input_dir = 'flow_reconstruction/dataset/original_data/test'
output_dir = 'flow_reconstruction/dataset_mlp/npz_data/test/'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all VTP files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.vtp'):
        vtp_file_path = os.path.join(input_dir, file_name)
        vtp_data = read_vtp_slice(vtp_file_path)
        print('Analysing file:', file_name)
        # Save the data as an NPZ file with the same base name as the input VTP file
        output_npz_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.npz')
        np.savez(output_npz_path, **vtp_data)
