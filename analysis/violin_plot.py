import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots  
plt.style.use('science')

file_path_gnn = 'velocity_errors_gnn.csv' 
file_path_cnn = 'velocity_errors_cnn.csv'  
data_gnn = pd.read_csv(file_path_gnn)
data_cnn = pd.read_csv(file_path_cnn)

data_gnn = data_gnn.add_prefix('GAT ')
data_cnn = data_cnn.add_prefix('CNN ')

combined_data = pd.concat([data_gnn, data_cnn], axis=1)

data_mae_melted = combined_data.filter(like='MAE').reset_index().melt(id_vars='index').drop(columns='index')
data_rmse_melted = combined_data.filter(like='RMSE').reset_index().melt(id_vars='index').drop(columns='index')

colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(10, 6))
sns.boxplot(x='variable', y='value', data=data_mae_melted, palette=colors)
plt.title('Comparison of MAE for Velocity Components Between GNN and CNN', fontsize=16)
plt.ylabel('Prediction Error [m/s]', fontsize=16)
plt.grid(True)
plt.savefig('mae_comparison.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()
plt.figure(figsize=(10, 6))
sns.boxplot(x='variable', y='value', data=data_rmse_melted, palette=colors)
plt.title('Comparison of RMSE for Velocity Components Between GNN and CNN', fontsize=16)
plt.ylabel('Prediction Error [m/s]', fontsize=16)
plt.grid(True)
plt.savefig('rmse_comparison.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()
