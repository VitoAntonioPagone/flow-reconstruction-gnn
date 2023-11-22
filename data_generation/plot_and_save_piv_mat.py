import numpy as np
import matplotlib.pyplot as plt
import h5py
from math import sin, cos, sqrt, radians

# Mat file name
in_folder = 'files/'
in_mat_file = 'OP-C_181114A005.mat'
out_folder = 'out/'
out_npz_file = 'test.npy'
out_vtk_file = 'test.vtp'

cad_range = np.arange(50, 52, 1)
cycle_range = np.arange(10, 12, 1)

# Flags
flag_plot_slice = True
flag_save_slice = True
flag_extract_20x20 = True
flag_plot_20x20 = True
flag_save_20x20 = True


# ----- Reference quanitties -----

# Engine data
rpm  = 1500    # [rev/min]
x_ref = 86      # [mm]
u_ref   = 7.035423        # [m/s] max piston speed at 1500 rpm
# uref = 3.75223           # [m/s] max piston speed, 800 rpm
# uref = 11.72571875       # [m/s] max piston speed, 2500 rpm

# Piston geometry
l_rod  = 14.8           # rod length [cm]
bore   = 8.6            # [cm]
stroke = 8.6            # [cm]
cr     = 0.5 * stroke   # crank radius [cm]
Hcl    = 0.26           # clearance height [cm]


def get_piston_vel(CAD):
    # Returns piston velocity based on crank-angle degree
    CAD = radians(CAD)
    v = -cr * np.sin(CAD) + (-cr ** 2 * np.sin(CAD) * np.cos(CAD)) / (np.sqrt(l_rod ** 2 - cr ** 2 * np.sin(CAD)))  # cm/rad
    v_pis = v * rpm * 2 * np.pi / 60  # cm/s
    return v_pis
# --------------------------------

# Load the .mat file
# mat_data = scipy.io.loadmat(in_folder + in_mat_file)
mat_data = h5py.File(in_folder+in_mat_file, 'r')

# Structure of mat file
# print(f"Structure of the mat file: {mat_data['Vel'].dtype}")
# Vel
# Vx - 4D double array
# Vy - 4D double array
# Vz - 4D double array
# Vm - 4D double array
# x - 2D double array
# y - 2D double array
# z - 2D double array
# CAD - 1D double array
# mask - 3D logical array
# maskP - 3D logical array

# Extract the arrays and reshape them as necessary
vx = mat_data['Vel']['Vx']
vy = mat_data['Vel']['Vy']
vz = mat_data['Vel']['Vz']
vm = mat_data['Vel']['Vm']
x = mat_data['Vel']['x']
y = mat_data['Vel']['y']
z = mat_data['Vel']['z']
cad = mat_data['Vel']['Cad']
mask = mat_data['Vel']['mask']
maskP = mat_data['Vel']['maskP']

for m in cad_range:
    for n in cycle_range:
        i_cad = m
        i_cycle = n

        # Get CAD and piston velocity
        CAD = int(720 + cad[0, i_cad])
        # v_pis = get_piston_vel(CAD)
        # v_pis /= 100

        # Plot 2D velocity array
        if flag_plot_slice:
            # Create a pseudocolor plot for vx
            plt.figure(figsize=(8, 6))  # Set the figure size

            # Plot the pseudocolor plot
            plt.pcolormesh(x, y, vx[i_cad, i_cycle, :], cmap='jet')  # Adjust the colormap as needed

            # Add color bar for reference
            plt.colorbar()

            # Set axis labels and a title
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title(f'Pseudocolor plot of vx data at CAD: {CAD}')

            # Show the plot
            plt.show()

            # Create a pseudocolor plot for vy
            plt.figure(figsize=(8, 6))  # Set the figure size

            # Plot the pseudocolor plot
            plt.pcolormesh(x, y, vy[i_cad, i_cycle, :], cmap='jet')  # Adjust the colormap as needed

            # Add color bar for reference
            plt.colorbar()

            # Set axis labels and a title
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title(f'Pseudocolor plot of vy data at CAD: {CAD}')

            # Show the plot
            plt.show()

        if flag_save_slice:
            vx_cad = vx[i_cad, i_cycle, :]
            vy_cad = vy[i_cad, i_cycle, :]
            vz_cad = vz[i_cad, i_cycle, :]

            # Normalize and flatten arrays
            x_new = x[:] / x_ref
            x_new[np.isnan(x_new)] = 0
            x_new = x_new.flatten()
            y_new = y[:] / x_ref
            y_new[np.isnan(y_new)] = 0
            y_new = y_new.flatten()
            vx_new = vx_cad[:] / u_ref
            vx_new[np.isnan(vx_new)] = 0
            vx_new = vx_new.flatten()
            vy_new = vy_cad[:] / u_ref
            vy_new[np.isnan(vy_new)] = 0
            vy_new = vy_new.flatten()
            vz_new = vz_cad[:] / u_ref
            vz_new[np.isnan(vz_new)] = 0
            vz_new = vz_new.flatten()

            # Extra features
            temp_new = np.zeros(np.shape(vx_new))
            pres_new = np.zeros(np.shape(vx_new))
            visc_new = np.zeros(np.shape(vx_new))

            # Save slice as npz file
            # out_array = np.stack((x_slice, y_slice, vx_slice), axis=2)
            # np.save(out_folder+out_npz_file, out_array)
            data_out = {
                'x': x_new,
                'y': y_new,
                'x_velocity': vx_new,
                'y_velocity': vy_new,
                'z_velocity': vz_new,
                'pressure': pres_new,
                'viscosity': visc_new
            }
            file_name = f'{out_folder}PIV_cyc_{i_cycle}_CAD_{CAD}_full.npz'
            print(f'Saving file: {file_name}')
            np.savez(file_name, **data_out)

            # Extract 20x20 slice
            if flag_extract_20x20:
                limit_min = -10
                limit_max = 10
                if (np.abs(np.max(x) - np.min(x)) > 20 and
                        np.abs(np.max(y) - np.min(y)) > 20):
                    # Take a 20x20 slice in the middle of the data values
                    # x range
                    x_mid = (np.min(x) + np.max(x)) / 2
                    x_range_min = x_mid + limit_min
                    x_range_max = x_mid + limit_max
                    # y range
                    y_mid = (np.min(y) + np.max(y)) / 2
                    y_range_min = y_mid + limit_min
                    y_range_max = y_mid + limit_max

                    for i in range(len(x[:, 0])):
                        if x[i, 0] >= x_range_min:
                            start_x = i
                            break
                    for i in range(len(x[:, 0])):
                        if x[i, 0] >= x_range_max:
                            end_x = i
                            break
                    for i in range(len(y[0, :])):
                        if y[0, i] <= y_range_max:
                            start_y = i
                            break
                    for i in range(len(y[0, :])):
                        if y[0, i] <= y_range_min:
                            end_y = i
                            break

                    # velocity for 1 CAD
                    vx_cad = vx[i_cad, i_cycle, :]
                    vy_cad = vy[i_cad, i_cycle, :]
                    vz_cad = vz[i_cad, i_cycle, :]

                    x_slice = x[start_y:end_y, start_x:end_x]
                    y_slice = y[start_y:end_y, start_x:end_x]
                    vx_slice = vx_cad[start_y:end_y, start_x:end_x]
                    vy_slice = vy_cad[start_y:end_y, start_x:end_x]
                    vz_slice = vz_cad[start_y:end_y, start_x:end_x]

                    if flag_plot_20x20:
                        # Create a pseudocolor plot
                        plt.figure(figsize=(8, 6))  # Set the figure size

                        # Plot the pseudocolor plot
                        plt.pcolormesh(x_slice, y_slice, vx_slice, cmap='jet')  # Adjust the colormap as needed

                        # Add color bar for reference
                        plt.colorbar()

                        # Set axis labels and a title
                        plt.xlabel('X-axis')
                        plt.ylabel('Y-axis')
                        plt.title(f'Pseudocolor plot of 20x20 slice of vx data at CAD: {cad[0, i_cad]}')

                        # Show the plot
                        plt.show()

                        # Create a pseudocolor plot
                        plt.figure(figsize=(8, 6))  # Set the figure size

                        # Plot the pseudocolor plot
                        plt.pcolormesh(x_slice, y_slice, vy_slice, cmap='jet')  # Adjust the colormap as needed

                        # Add color bar for reference
                        plt.colorbar()

                        # Set axis labels and a title
                        plt.xlabel('X-axis')
                        plt.ylabel('Y-axis')
                        plt.title(f'Pseudocolor plot of 20x20 slice of vy data at CAD: {cad[0, i_cad]}')

                        # Show the plot
                        plt.show()

                    if flag_save_20x20:
                        # Normalize and flatten arrays
                        x_slice = x_slice/x_ref
                        x_slice = x_slice.flatten()
                        y_slice = y_slice/x_ref
                        y_slice = y_slice.flatten()
                        vx_slice = vx_slice/u_ref
                        vx_slice = vx_slice.flatten()
                        vy_slice = vy_slice/u_ref
                        vy_slice = vy_slice.flatten()
                        vz_slice = vz_slice/u_ref
                        vz_slice = vz_slice.flatten()

                        # Extra features
                        temp = np.zeros(np.shape(vx_slice))
                        pres = np.zeros(np.shape(vx_slice))
                        visc = np.zeros(np.shape(vx_slice))

                        # Save slice as npz file
                        # out_array = np.stack((x_slice, y_slice, vx_slice), axis=2)
                        # np.save(out_folder+out_npz_file, out_array)
                        data_out = {
                            'x': x_slice,
                            'y': y_slice,
                            'x_velocity': vx_slice,
                            'y_velocity': vy_slice,
                            'z_velocity': vz_slice,
                            'pressure': pres,
                            'viscosity': visc
                        }
                        file_name = f'{out_folder}PIV_cyc_{i_cycle}_CAD_{CAD}.npz'
                        print(f'Saving file: {file_name}')
                        np.savez(file_name, **data_out)

                        # # Write vtk file
                        # # points
                        # points = np.transpose(nekdata.pos)
                        # # point data
                        # point_data = {}
                        # point_data["x_velocity"] = vx_slice
                        # point_data["y_velocity"] = vy_slice
                        # point_data["z_velocity"] = vz_slice
                        #
                        # meshio.write_points_cells(
                        #     fnamevtk_out,
                        #     points,
                        #     cells,
                        #     point_data=point_data,
                        #     file_format=vtk_file_format
                        # )
