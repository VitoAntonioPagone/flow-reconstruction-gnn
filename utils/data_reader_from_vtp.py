import os
import glob
import time
import numpy as np
import vtk
from vtk.util import numpy_support


INPUT_TRAIN_FODLER = 'flow_reconstruction/dataset/train/vtp_files'
OUTPUT_TRAIN_FOLDER = 'flow_reconstruction/dataset/train/csvs'
INPUT_TEST_FOLDER = 'flow_reconstruction/dataset/test/vtp_files'
OUTPUT_TEST_FOLDER = 'flow_reconstruction/dataset/test/csvs'


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


if __name__ == "__main__":
    vtk_to_csv(INPUT_TRAIN_FODLER, OUTPUT_TRAIN_FOLDER)
    vtk_to_csv(INPUT_TEST_FOLDER, OUTPUT_TEST_FOLDER)
