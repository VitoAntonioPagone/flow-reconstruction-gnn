import sys
import time
# Tries
import numpy as np
import vtk as vtk
sys.path.append('../src/')

t00 = time.time()

# Specify source file:
fnamevtk = 'slice01.vtp'

# Reference values for conversion:
xref = 0.086          # [m]
uref = 11.72571875    # [m/s] max piston speed, 2500 rpm
tref = 333.15         # [K]


#----------------- VTK file ---------------------
print(f'\n>>>> Reading VTK file {fnamevtk}', flush=True)
t0 = time.time()
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(fnamevtk)
reader.Update()
data_in = reader.GetOutput()
t1 = time.time()
print(f'Time elapsed [sec]: {t1-t0:.1f}', flush=True)

points = np.array(data_in.GetPoints().GetData())
x_vel = np.array(data_in.GetPointData().GetArray("x_velocity"))
y_vel = np.array(data_in.GetPointData().GetArray("y_velocity"))
z_vel = np.array(data_in.GetPointData().GetArray("z_velocity"))
temp = np.array(data_in.GetPointData().GetArray("temperature"))

