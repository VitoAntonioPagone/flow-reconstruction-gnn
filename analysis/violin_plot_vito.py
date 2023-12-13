import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use('science')

data_cnn = pd.read_csv('L2velocity_errors_gnn.csv')

fig, ax = plt.subplots(figsize=(6, 6))

box = ax.boxplot(data_cnn['MAE Velocity Magnitude'],
                 labels=['GACN (L2)'],
                 patch_artist=True)

ax.set_ylabel(r'Mean absolute error $|\vec{V}|$ [m/s]', fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
box['boxes'][0].set_facecolor('blue')
box['medians'][0].set(color='red')

mean_val = data_cnn['MAE Velocity Magnitude'].mean()
ax.scatter(1, mean_val, marker='*', color='yellow', edgecolors='black', s=200, zorder=5, label='Mean')
ax.grid(color='k', linestyle='--', linewidth=0.2)
ax.set_ylim(0, 0.9)
ax.set_xlim(0.5, 1.5)

plt.show()

fig.savefig('L2ONly.jpg', dpi=800)
