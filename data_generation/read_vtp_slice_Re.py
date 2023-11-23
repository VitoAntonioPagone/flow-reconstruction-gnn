import sys
import time
import numpy as np
import vtk as vtk
from math import sin, cos, sqrt, radians
sys.path.append('../src/')

t00 = time.time()

# Specify source file:
fnamevtk = 'cyc03_CAD605_test.vtp'

# Reference values for conversion:
x_ref   = 8.6             # [cm]
rpm     = 1500            # [rev/min]
u_ref   = 703.5423        # [cm/s] max piston speed at 1500 rpm
p_ref   = 1.154325e6      # [dyn/cm2]
T_ref   = 333.15          # [K]
R_air   = 2.9680291e+06   # [erg/cm/K/s]
vis_ref = 1.9479592e-04   # [g/cm/s]
rho_ref = 1.1674010e-03   # [g/cm3]
Re_ref  = 3.6260123e+04   # [-]

# Different rpm
# rpm  = 800               # [rev/min]
# uref = 375.223           # [cm/s] max piston speed, 800 rpm
# rpm  = 2500              # [rev/min]
# uref = 1172.571875       # [cm/s] max piston speed, 2500 rpm

# Piston geometry
l_rod  = 14.8           # rod length [cm]
bore   = 8.6            # [cm]
stroke = 8.6            # [cm]
cr     = 0.5 * stroke   # crank radius [cm]
Hcl    = 0.26           # clearance height [cm]


def get_piston_pos(CAD):
    # Returns piston position based on crank-angle degree
    CAD = radians(CAD)
    x_pis = cr * cos(CAD) + sqrt(l_rod ** 2 - cr ** 2 * sin(CAD) ** 2) - (l_rod+cr+Hcl)
    x_pis /= x_ref
    return x_pis


def get_piston_vel(CAD):
    # Returns piston velocity based on crank-angle degree
    CAD = radians(CAD)
    v = -cr * np.sin(CAD) + (-cr ** 2 * np.sin(CAD) * np.cos(CAD)) / (np.sqrt(l_rod ** 2 - cr ** 2 * np.sin(CAD)))  # cm/rad
    v_pis = v * rpm * 2 * np.pi / 60  # cm/s
    return v_pis


def get_thermo_pressure(CAD):
    # Returns thermodynamic pressure inside the cylinder based on piston position
    p_min = 1        # [-]
    p_max = 8.07165  # [-]
    z_min = get_piston_pos(600)
    z_max = get_piston_pos(720)
    z = get_piston_pos(CAD)
    pos_perc = (z-z_min)/(z_max-z_min)
    p0th = p_min + pos_perc * p_max
    return p0th


def get_density(pressure, temp):
    # Returns density based on temperature and pressure
    p = pressure * p_ref
    t = temp * T_ref
    density = p/(R_air*t)
    return density


def get_dynamic_viscosity(temp):
    # returns dynamic viscosity based on temperature
    t = temp * T_ref
    vis = (2.46317040e-05 +
           t*(6.10895392e-07 + t*(-3.5394496e-10 + t*(1.75040791e-13 + t*(-4.5734874e-17 + 4.7456719e-21*t)))))
    return vis


# ----------------- VTK file ---------------------
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
pres = np.array(data_in.GetPointData().GetArray("pressure"))

# Get piston velocity based on CAD
index = fnamevtk.find('CAD')
if index != -1:
    # Extract the first 3 characters after the substring
    extracted_cad = fnamevtk[index + len('CAD'):index + len('CAD') + 3]
else:
    exit("File name doesn't have CAD in it!")
cad = int(extracted_cad)
v_pis = get_piston_vel(cad)

# Get density
p_avg = get_thermo_pressure(cad)
t_avg = np.abs(np.average(temp))
rho = get_density(p_avg, t_avg)

# Get dynamic viscosity
mu = get_dynamic_viscosity(t_avg)

# Reynolds number
U = v_pis
L = x_ref
Re = rho*U*L/mu
Re_array = np.ones(np.shape(temp))*Re

t11 = time.time()
print(f">>>> Finished converting arrys. Total time elapsed [sec]: {t11-t00:.1f}")
