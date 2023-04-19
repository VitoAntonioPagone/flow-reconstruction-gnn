import os
import glob
import time
import numpy as np
import vtk
from vtk.util import numpy_support

input_folder = '/Users/vitoantonio/Desktop/Thesis/Dataset/2D_slices/train/vtp_files'
output_folder = '/Users/vitoantonio/Desktop/Thesis/Dataset/2D_slices/train/csvs'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all VTP files in the input folder
vtp_files = glob.glob(os.path.join(input_folder, '*.vtp'))

for fnamevtk in vtp_files:
    t00 = time.time()

    #----------------- VTK file ---------------------
    print(f'\n>>>> Reading VTK file {fnamevtk}', flush=True)
    t0 = time.time()
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fnamevtk)
    reader.Update()
    data_in = reader.GetOutput()
    t1 = time.time()
    print(f'Time elapsed [sec]: {t1-t0:.1f}', flush=True)

    points = numpy_support.vtk_to_numpy(data_in.GetPoints().GetData())

    # Get the point data arrays excluding 'avtOriginalNodeNumbers'
    point_data_arrays = [data_in.GetPointData().GetArray(i) for i in range(data_in.GetPointData().GetNumberOfArrays()) if data_in.GetPointData().GetArray(i).GetName() != 'avtOriginalNodeNumbers']

    # Convert VTK arrays to NumPy arrays
    np_arrays = []
    for vtk_array in point_data_arrays:
        np_arrays.append(numpy_support.vtk_to_numpy(vtk_array))

    # Get the feature names
    feature_names = ['x', 'y'] + [arr.GetName() for arr in point_data_arrays]

    # Write the data to a CSV file
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(fnamevtk))[0] + '.csv')
    with open(output_file, 'w') as f:
        # Write the header
        f.write(','.join(feature_names) + '\n')

        # Write the data
        for i in range(points.shape[0]):
            values = [str(points[i, 0]), str(points[i, 1])] + [str(np_array[i]) for np_array in np_arrays]
            f.write(','.join(values) + '\n')

    print(f"Converted {fnamevtk} to {output_file}")
