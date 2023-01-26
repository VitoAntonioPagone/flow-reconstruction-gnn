#import matplotlib.pyplot as plt
from pymech.neksuite import readnek
from pymech.dataset import open_dataset
from pymech.dataset import open_mfdataset
from pymech.neksuite import writenek
from pymech.neksuite import readnek
# import xarray as xr
import numpy as np

### Read data set from one file
# ds = open_dataset('tud0.f00003')
# print(ds, '\n')
# # Add velocity magnitude to the data set
# umag = np.linalg.norm(np.stack((ds.uz.values, ds.uy.values, ds.ux.values)), axis=0)
# ds['umag'] = (['z', 'y', 'x'], umag)
#
# # Calculate the biggest velocity magnitude value and it's mesh position
# max_idx = np.where(ds.umag.values == np.amax(ds.umag.values))
# max_x_pos = ds.xmesh[max_idx]
# max_y_pos = ds.xmesh[max_idx]
# max_z_pos = ds.zmesh[max_idx]
# print("Maximum velocity magnitude mesh position: x:{:.6f}, y:{:.6f}, z:{:.6f} \n".format(max_x_pos.values[0][0][0],
#                                                                                          max_y_pos.values[0][0][0],
#                                                                                          max_z_pos.values[0][0][0]))

### Plot slice
# ds.ux.mean('z').plot()
# # Show figure
# # plt.show()
# # Save figure
# plt.savefig('ux_mean.png')


### Read multiple files
# df = open_mfdataset('eddy_uv0.f00*')
# print(df, '\n')
# # Slice based on time
# print(df.ux.sel(time=slice(0.01)), '\n')

### Load data in parallel
# df = open_mfdataset('eddy_uv0.f00*', chunks='auto', parallel=True)
# print(df, '\n')

### Read nek file with pymech
field = readnek('tud0.f00006')
# Flags for activating max calculation
flag_max_velocity = False
flag_max_scalar = False
flag_max_temp = True
# Velocity field
if flag_max_velocity:
    vel_fields = []
    for i in range(len(field.elem)):
        vel_fields.append(field.elem[i].vel)
    velocity_fields = np.stack(vel_fields)
    # Calculate the velocity magnitude
    vel_mag = np.linalg.norm(velocity_fields, axis=1)
    max_ids = np.where(vel_mag == np.amax(vel_mag))
    if len(max_ids[0]) > 0:
        for i in range(len(max_ids[0])):
            print("Maximum velocity magnitude mesh position: x:{:.12f}, y:{:.12f}, z:{:.12f} \n".format(
                field.elem[int(max_ids[0][i])].pos[0][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])],
                field.elem[int(max_ids[0][i])].pos[1][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])],
                field.elem[int(max_ids[0][i])].pos[2][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])]))
    else:
        print("Maximum velocity magnitude mesh position: x:{:.12f}, y:{:.12f}, z:{:.12f} \n".format(
            field.elem[int(max_ids[0])].pos[0][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])],
            field.elem[int(max_ids[0])].pos[1][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])],
            field.elem[int(max_ids[0])].pos[2][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])]))
# Scalar field
if flag_max_scalar:
    # Which scalar to compute
    scalar_number = 1
    scalar_fields = []
    for i in range(len(field.elem)):
        scalar_fields.append(field.elem[i].scal)
    scalar_fields = np.stack(scalar_fields)
    max_ids = np.where(scalar_fields[:, scalar_number] == np.amax(scalar_fields[:, scalar_number]))
    if len(max_ids[0]) > 0:
        for i in range(len(max_ids[0])):
            print("Maximum scalar mesh position: x:{:.12f}, y:{:.12f}, z:{:.12f} \n".format(
                field.elem[int(max_ids[0][i])].pos[0][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])],
                field.elem[int(max_ids[0][i])].pos[1][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])],
                field.elem[int(max_ids[0][i])].pos[2][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])]))
    else:
        print("Maximum scalar mesh position: x:{:.12f}, y:{:.12f}, z:{:.12f} \n".format(
            field.elem[int(max_ids[0])].pos[0][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])],
            field.elem[int(max_ids[0])].pos[1][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])],
            field.elem[int(max_ids[0])].pos[2][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])]))
if flag_max_temp:
    # Which scalar to compute
    temp_fields = []
    for i in range(len(field.elem)):
        temp_fields.append(field.elem[i].temp)
    temp_fields = np.stack(temp_fields)
    print(np.mean(temp_fields[:, 0]))
#    max_ids = np.where(temp_fields[:, 0] == np.mean(temp_fields[:, 0]))
#    if len(max_ids[0]) > 0:
#        for i in range(len(max_ids[0])):
#            print("Maximum temperature mesh position: x:{:.12f}, y:{:.12f}, z:{:.12f} \n".format(
#                field.elem[int(max_ids[0][i])].pos[0][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])],
#                field.elem[int(max_ids[0][i])].pos[1][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])],
#                field.elem[int(max_ids[0][i])].pos[2][int(max_ids[1][i])][int(max_ids[2][i])][int(max_ids[3][i])]))
#    else:
#        print("Maximum temperature mesh position: x:{:.12f}, y:{:.12f}, z:{:.12f} \n".format(
#            field.elem[int(max_ids[0])].pos[0][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])],
#            field.elem[int(max_ids[0])].pos[1][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])],
#            field.elem[int(max_ids[0])].pos[2][int(max_ids[1])][int(max_ids[2])][int(max_ids[3])]))
# print(field)
# Work on elements in the field
# print(field.elem[0].vel)
# for i in range(len(field.elem)):
#     field.elem[i].vel = field.elem[i].vel + 10
# print(field.elem[0].vel)

### Wrtie new nek file
# writenek('eddy_uv0.f00013', field)
# field_new = readnek('eddy_uv0.f00013')
# print(field_new.elem[0].vel)
