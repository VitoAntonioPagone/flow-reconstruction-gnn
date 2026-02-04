import numpy as np
import matplotlib.pyplot as plt
import os

data = np.load("../dataset_graph/original_data/npz_data/test/cyc10_CAD615_Y0_Z0_X0.npz")

x_velocity = data["x_velocity"]
y_velocity = data["y_velocity"]
z_velocity = data["z_velocity"]


def create_plot(velocity_data, title, save_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(velocity_data, origin="lower")
    plt.title(title)
    plt.colorbar(label="Velocity")
    plt.savefig(save_path, dpi=300)
    plt.close()


if not os.path.exists("plots"):
    os.makedirs("plots")

create_plot(x_velocity, "X Velocity", "../pics/x_velocity.png")
create_plot(y_velocity, "Y Velocity", "../pics/y_velocity.png")
create_plot(z_velocity, "Z Velocity", "../pics/z_velocity.png")
