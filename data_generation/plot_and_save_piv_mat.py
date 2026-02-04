import numpy as np
import matplotlib.pyplot as plt
import h5py
from math import sin, cos, sqrt, radians

in_mat_file = "files/OP-C_181114A005.mat"
out_folder = "out/"
out_npz_file = "test.npy"
out_vtk_file = "test.vtp"

cad_range = np.arange(50, 52, 1)
cycle_range = np.arange(10, 12, 1)

flag_plot_slice = True
flag_save_slice = True
flag_extract_20x20 = True
flag_plot_20x20 = True
flag_save_20x20 = True


rpm = 1500
x_ref = 86
u_ref = 7.035423

l_rod = 14.8
bore = 8.6
stroke = 8.6
cr = 0.5 * stroke
Hcl = 0.26


def get_piston_vel(CAD):
    CAD = radians(CAD)
    v = -cr * np.sin(CAD) + (-(cr**2) * np.sin(CAD) * np.cos(CAD)) / (
        np.sqrt(l_rod**2 - cr**2 * np.sin(CAD))
    )
    v_pis = v * rpm * 2 * np.pi / 60
    return v_pis


mat_data = h5py.File(in_mat_file, "r")

vx = mat_data["Vel"]["Vx"]
vy = mat_data["Vel"]["Vy"]
vz = mat_data["Vel"]["Vz"]
vm = mat_data["Vel"]["Vm"]
x = mat_data["Vel"]["x"]
y = mat_data["Vel"]["y"]
z = mat_data["Vel"]["z"]
cad = mat_data["Vel"]["Cad"]
mask = mat_data["Vel"]["mask"]
maskP = mat_data["Vel"]["maskP"]

for m in cad_range:
    for n in cycle_range:
        i_cad = m
        i_cycle = n

        CAD = int(720 + cad[0, i_cad])

        if flag_plot_slice:
            plt.figure(figsize=(8, 6))

            plt.pcolormesh(x, y, vx[i_cad, i_cycle, :], cmap="jet")

            plt.colorbar()

            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title(f"Pseudocolor plot of vx data at CAD: {CAD}")

            plt.show()

            plt.figure(figsize=(8, 6))

            plt.pcolormesh(x, y, vy[i_cad, i_cycle, :], cmap="jet")

            plt.colorbar()

            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title(f"Pseudocolor plot of vy data at CAD: {CAD}")

            plt.show()

        if flag_save_slice:
            vx_cad = vx[i_cad, i_cycle, :]
            vy_cad = vy[i_cad, i_cycle, :]
            vz_cad = vz[i_cad, i_cycle, :]

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

            temp_new = np.zeros(np.shape(vx_new))
            pres_new = np.zeros(np.shape(vx_new))
            visc_new = np.zeros(np.shape(vx_new))

            data_out = {
                "x": x_new,
                "y": y_new,
                "x_velocity": vx_new,
                "y_velocity": vy_new,
                "z_velocity": vz_new,
                "pressure": pres_new,
                "viscosity": visc_new,
            }
            file_name = f"{out_folder}PIV_cyc_{i_cycle}_CAD_{CAD}_full.npz"
            print(f"Saving file: {file_name}")
            np.savez(file_name, **data_out)

            if flag_extract_20x20:
                limit_min = -10
                limit_max = 10
                if (
                    np.abs(np.max(x) - np.min(x)) > 20
                    and np.abs(np.max(y) - np.min(y)) > 20
                ):
                    x_mid = (np.min(x) + np.max(x)) / 2
                    x_range_min = x_mid + limit_min
                    x_range_max = x_mid + limit_max
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

                    vx_cad = vx[i_cad, i_cycle, :]
                    vy_cad = vy[i_cad, i_cycle, :]
                    vz_cad = vz[i_cad, i_cycle, :]

                    x_slice = x[start_y:end_y, start_x:end_x]
                    y_slice = y[start_y:end_y, start_x:end_x]
                    vx_slice = vx_cad[start_y:end_y, start_x:end_x]
                    vy_slice = vy_cad[start_y:end_y, start_x:end_x]
                    vz_slice = vz_cad[start_y:end_y, start_x:end_x]

                    if flag_plot_20x20:
                        plt.figure(figsize=(8, 6))

                        plt.pcolormesh(x_slice, y_slice, vx_slice, cmap="jet")

                        plt.colorbar()

                        plt.xlabel("X-axis")
                        plt.ylabel("Y-axis")
                        plt.title(
                            f"Pseudocolor plot of 20x20 slice of vx data at CAD: {cad[0, i_cad]}"
                        )

                        plt.show()

                        plt.figure(figsize=(8, 6))

                        plt.pcolormesh(x_slice, y_slice, vy_slice, cmap="jet")

                        plt.colorbar()

                        plt.xlabel("X-axis")
                        plt.ylabel("Y-axis")
                        plt.title(
                            f"Pseudocolor plot of 20x20 slice of vy data at CAD: {cad[0, i_cad]}"
                        )

                        plt.show()

                    if flag_save_20x20:
                        x_slice = x_slice / x_ref
                        x_slice = x_slice.flatten()
                        y_slice = y_slice / x_ref
                        y_slice = y_slice.flatten()
                        vx_slice = vx_slice / u_ref
                        vx_slice = vx_slice.flatten()
                        vy_slice = vy_slice / u_ref
                        vy_slice = vy_slice.flatten()
                        vz_slice = vz_slice / u_ref
                        vz_slice = vz_slice.flatten()

                        temp = np.zeros(np.shape(vx_slice))
                        pres = np.zeros(np.shape(vx_slice))
                        visc = np.zeros(np.shape(vx_slice))

                        data_out = {
                            "x": x_slice,
                            "y": y_slice,
                            "x_velocity": vx_slice,
                            "y_velocity": vy_slice,
                            "z_velocity": vz_slice,
                            "pressure": pres,
                            "viscosity": visc,
                        }
                        file_name = f"{out_folder}PIV_cyc_{i_cycle}_CAD_{CAD}.npz"
                        print(f"Saving file: {file_name}")
                        np.savez(file_name, **data_out)
