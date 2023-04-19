import os
import glob
import time
import numpy as np
import vtk
from vtk.util import numpy_support


def vtk_to_csv(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vtp_files = glob.glob(os.path.join(input_folder, '*.vtp'))

    for fnamevtk in vtp_files:
        t00 = time.time()

        print(f'\n>>>> Reading VTK file {fnamevtk}', flush=True)
        t0 = time.time()
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fnamevtk)
        reader.Update()
        data_in = reader.GetOutput()
        t1 = time.time()
        print(f'Time elapsed [sec]: {t1-t0:.1f}', flush=True)

        points = numpy_support.vtk_to_numpy(data_in.GetPoints().GetData())

        point_data_arrays = [
            data_in.GetPointData().GetArray(i)
            for i in range(data_in.GetPointData().GetNumberOfArrays())
            if data_in.GetPointData().GetArray(i).GetName() != 'avtOriginalNodeNumbers'
        ]

        np_arrays = [
            numpy_support.vtk_to_numpy(vtk_array)
            for vtk_array in point_data_arrays
        ]

        feature_names = ['x', 'y'] + [
            arr.GetName() for arr in point_data_arrays
        ]

        output_file = os.path.join(
            output_folder, os.path.splitext(os.path.basename(fnamevtk))[0] + '.csv'
        )

        with open(output_file, 'w') as f:
            f.write(','.join(feature_names) + '\n')

            for i in range(points.shape[0]):
                values = (
                    [str(points[i, 0]), str(points[i, 1])]
                    + [str(np_array[i]) for np_array in np_arrays]
                )
                f.write(','.join(values) + '\n')

        print(f"Converted {fnamevtk} to {output_file}")


input_train_folder = (
    'Dataset/train/vtp_files'
)
output_train_folder = (
    'Dataset/train/csvs'
)
input_test_folder = (
    'Dataset/test/vtp_files'
)
output_test_folder = (
    'Dataset/test/csvs'
)

vtk_to_csv(input_train_folder, output_train_folder)
vtk_to_csv(input_test_folder, output_test_folder)
