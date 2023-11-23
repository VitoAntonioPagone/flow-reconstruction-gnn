import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use('science')

# Read the data from the file into a pandas DataFrame
data = pd.read_csv('gnn_metrics_PIV_full.txt', delim_whitespace=True)

# Define colors for each box
colors = ['blue', 'green', 'orange', 'red']

# Create a box plot for all columns with different colors
fig, ax = plt.subplots(figsize=(10, 6))

# Box plot for all columns with different colors
box = ax.boxplot([data['x_rmse'], data['x_mae'], data['y_rmse'], data['y_mae']],
                 labels=['x_rmse', 'x_mae', 'y_rmse', 'y_mae'],
                 patch_artist=True)

# Set plot title and labels
ax.set_title('Box Plots for Each Metric', fontsize=16)
ax.set_ylabel('Prediction error [m/s]', fontsize=16)

# Set different colors for each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Show the plot
plt.show()
