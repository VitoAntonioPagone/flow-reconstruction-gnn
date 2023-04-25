import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Read the combined data
train_data = pd.read_csv('MLP/Dataset/combined_train/combined_train.csv')
test_data  = pd.read_csv('MLP/Dataset/combined_test/combined_test.csv')
print(train_data)
# Data preprocessing
def preprocess_data(data):
    X = data[['x', 'y', 'temperature']].values
    y = data[['x_velocity', 'y_velocity', 'z_velocity', 'velocity_magnitude']].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    #X = scaler_X.fit_transform(X)
    #y = scaler_y.fit_transform(y)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


X_train, y_train = preprocess_data(train_data)
X_test, y_test   = preprocess_data(test_data)

batch_size = 64

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def navier_stokes_loss(pred, x, y, temp, rho=1.0, mu=1.0, delta=1e-3):
    u, v, w, _ = torch.split(pred, 1, dim=1)

    du_dx = (model(torch.cat((x + delta, y, temp), dim=1))[:, 0] - u) / delta
    du_dy = (model(torch.cat((x, y + delta, temp), dim=1))[:, 0] - u) / delta
    dv_dx = (model(torch.cat((x + delta, y, temp), dim=1))[:, 1] - v) / delta
    dv_dy = (model(torch.cat((x, y + delta, temp), dim=1))[:, 1] - v) / delta
    dw_dx = (model(torch.cat((x + delta, y, temp), dim=1))[:, 2] - w) / delta
    dw_dy = (model(torch.cat((x, y + delta, temp), dim=1))[:, 2] - w) / delta

    div = du_dx + dv_dy + dw_dy

    d2u_dx2 = (model(torch.cat((x + 2 * delta, y, temp), dim=1))[:, 0] - 2 * model(torch.cat((x + delta, y, temp), dim=1))[:, 0] + u) / (delta ** 2)
    d2u_dy2 = (model(torch.cat((x, y + 2 * delta, temp), dim=1))[:, 0] - 2 * model(torch.cat((x, y + delta, temp), dim=1))[:, 0] + u) / (delta ** 2)
    d2v_dx2 = (model(torch.cat((x + 2 * delta, y, temp), dim=1))[:, 1] - 2 * model(torch.cat((x + delta, y, temp), dim=1))[:, 1] + v) / (delta ** 2)
    d2v_dy2 = (model(torch.cat((x, y + 2 * delta, temp), dim=1))[:, 1] - 2 * model(torch.cat((x, y + delta, temp), dim=1))[:, 1] + v) / (delta ** 2)
    d2w_dx2 = (model(torch.cat((x + 2 * delta, y, temp), dim=1))[:, 2] - 2 * model(torch.cat((x + delta, y, temp), dim=1))[:, 2] + w) / (delta ** 2)
    d2w_dy2 = (model(torch.cat((x, y + 2 * delta, temp), dim=1))[:, 2] - 2 * model(torch.cat((x, y + delta, temp), dim=1))[:, 2] + w) / (delta ** 2)


    # Convective terms
    u_du_dx = u * du_dx
    v_du_dy = v * du_dy
    u_dv_dx = u * dv_dx
    v_dv_dy = v * dv_dy
    u_dw_dx = u * dw_dx
    v_dw_dy = v * dw_dy

    NS_x = rho * (u_du_dx + v_du_dy) - mu * (d2u_dx2 + d2u_dy2)
    NS_y = rho * (u_dv_dx + v_dv_dy) - mu * (d2v_dx2 + d2v_dy2)
    NS_z = rho * (u_dw_dx + v_dw_dy) - mu * (d2w_dx2 + d2w_dy2)



    NS_loss = torch.mean(torch.abs(NS_x)) + torch.mean(torch.abs(NS_y)) + torch.mean(torch.abs(NS_z))

    div_loss = torch.mean(torch.abs(div))

    return NS_loss + div_loss

mse_loss = nn.MSELoss()

# Model, optimizer, and training loop
input_size = 3
hidden_size = 128
output_size = 4
learning_rate = 1e-3
num_epochs = 1000

model = MLP(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (input_tensor, target) in enumerate(train_loader):
        # Forward pass
        pred = model(input_tensor)

        # Calculate losses
        ns_loss = navier_stokes_loss(pred, input_tensor[:, 0].unsqueeze(1), input_tensor[:, 1].unsqueeze(1), input_tensor[:, 2].unsqueeze(1))
        l2_loss = mse_loss(pred, target)
        combined_loss = ns_loss + l2_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Navier-Stokes Loss: {ns_loss.item():.4f}, L2 Loss: {l2_loss.item():.4f}, Combined Loss: {combined_loss.item():.4f}')