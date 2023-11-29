import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use('science')

# Read the data from the file into a pandas DataFrame
data_cnn = pd.read_csv('cnn_metrics_PIV_full_mag.txt', delim_whitespace=True)
data_gnn = pd.read_csv('gnn_metrics_PIV_full_mag.txt', delim_whitespace=True)

# Define colors for each box
colors = ['blue', 'green']

# Create a box plot for all columns with different colors
fig, ax = plt.subplots(figsize=(6, 6))

# Define the positions of the boxes
positions = [0.9, 1.2]

# Box plot for all columns with different colors
box = ax.boxplot([data_gnn['v_mae'], data_cnn['v_mae']],
                 labels=['GACN', 'CNN'],
                 patch_artist=True,
                 positions=positions)

# Set plot title and labels
# ax.set_title('Box Plots for Each Metric', fontsize=16)
ax.set_ylabel(r'Mean absolute error $|\vec{V}|$ [m/s]', fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# Set different colors for each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Change the color of the median line to red
for median in box['medians']:
    median.set(color='red')

# Calculate and plot mean values with black-yellow star
for i, mean_val in enumerate([data_gnn['v_mae'].mean(), data_cnn['v_mae'].mean()]):
    ax.scatter(positions[i], mean_val, marker='*', color='yellow', edgecolors='black', s=200, zorder=5, label='Mean')

# Plot options
ax.grid(color='k', linestyle='--', linewidth=0.2)
ax.set_ylim(0.4, 1.8)
ax.set_xlim(0.7, 1.4)

# Show the plot
plt.show()

# Save figure
fig.savefig('mae_piv.jpg', dpi=800)
