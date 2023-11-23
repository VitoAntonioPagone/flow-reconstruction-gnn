import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'velocity_errors_cnn.csv' 
data = pd.read_csv(file_path)

print("Statistics for each test set:")
print(data.describe())
plt.figure(figsize=(12, 6))
data_melted = data.melt(var_name='Test Set', value_name='Relative Prediction Error')
sns.boxplot(x='Test Set', y='Relative Prediction Error', data=data_melted)
plt.title('Box Plots of Relative Prediction Error per Test Set')
min_val = data_melted['Relative Prediction Error'].min()
max_val = data_melted['Relative Prediction Error'].max()
whitespace = (max_val - min_val) * 0.1 
plt.ylim(min_val - whitespace, max_val + whitespace)
plt.grid(True)

plt.show()
