import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use("science")
data = pd.read_csv("../network/velocity_errors_cnn.csv", delim_whitespace=True)
colors = ["blue", "green", "orange", "red"]
fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot(
    [data["x_rmse"], data["x_mae"], data["y_rmse"], data["y_mae"]],
    labels=["x_rmse", "x_mae", "y_rmse", "y_mae"],
    patch_artist=True,
)
ax.set_title("Box Plots for Each Metric", fontsize=16)
ax.set_ylabel("Prediction error [m/s]", fontsize=16)
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
plt.show()
