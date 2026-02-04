import sys
import time
import numpy as np
import vtk as vtk
from math import sin, cos, sqrt, radians

sys.path.append("../src/")

t00 = time.time()

fnamevtk = "cyc03_CAD605_test.vtp"

x_ref = 8.6
rpm = 1500
u_ref = 703.5423
p_ref = 1.154325e6
T_ref = 333.15
R_air = 2.9680291e06
vis_ref = 1.9479592e-04
rho_ref = 1.1674010e-03
Re_ref = 3.6260123e04

l_rod = 14.8
bore = 8.6
stroke = 8.6
cr = 0.5 * stroke
Hcl = 0.26


def get_piston_pos(CAD):
    CAD = radians(CAD)
    x_pis = cr * cos(CAD) + sqrt(l_rod**2 - cr**2 * sin(CAD) ** 2) - (l_rod + cr + Hcl)
    x_pis /= x_ref
    return x_pis


def get_piston_vel(CAD):
    CAD = radians(CAD)
    v = -cr * np.sin(CAD) + (-(cr**2) * np.sin(CAD) * np.cos(CAD)) / (
        np.sqrt(l_rod**2 - cr**2 * np.sin(CAD))
    )
    v_pis = v * rpm * 2 * np.pi / 60
    return v_pis


def get_thermo_pressure(CAD):
    p_min = 1
    p_max = 8.07165
    z_min = get_piston_pos(600)
    z_max = get_piston_pos(720)
    z = get_piston_pos(CAD)
    pos_perc = (z - z_min) / (z_max - z_min)
    p0th = p_min + pos_perc * p_max
    return p0th


def get_density(pressure, temp):
    p = pressure * p_ref
    t = temp * T_ref
    density = p / (R_air * t)
    return density


def get_dynamic_viscosity(temp):
    t = temp * T_ref
    vis = 2.46317040e-05 + t * (
        6.10895392e-07
        + t
        * (
            -3.5394496e-10
            + t * (1.75040791e-13 + t * (-4.5734874e-17 + 4.7456719e-21 * t))
        )
    )
    return vis


print(f"\n>>>> Reading VTK file {fnamevtk}", flush=True)
t0 = time.time()
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(fnamevtk)
reader.Update()
data_in = reader.GetOutput()
t1 = time.time()
print(f"Time elapsed [sec]: {t1 - t0:.1f}", flush=True)

points = np.array(data_in.GetPoints().GetData())
x_vel = np.array(data_in.GetPointData().GetArray("x_velocity"))
y_vel = np.array(data_in.GetPointData().GetArray("y_velocity"))
z_vel = np.array(data_in.GetPointData().GetArray("z_velocity"))
temp = np.array(data_in.GetPointData().GetArray("temperature"))
pres = np.array(data_in.GetPointData().GetArray("pressure"))

index = fnamevtk.find("CAD")
if index != -1:
    extracted_cad = fnamevtk[index + len("CAD") : index + len("CAD") + 3]
else:
    exit("File name doesn't have CAD in it!")
cad = int(extracted_cad)
v_pis = get_piston_vel(cad)

p_avg = get_thermo_pressure(cad)
t_avg = np.abs(np.average(temp))
rho = get_density(p_avg, t_avg)

mu = get_dynamic_viscosity(t_avg)

U = v_pis
L = x_ref
Re = rho * U * L / mu
Re_array = np.ones(np.shape(temp)) * Re

t11 = time.time()
print(f">>>> Finished converting arrys. Total time elapsed [sec]: {t11 - t00:.1f}")
