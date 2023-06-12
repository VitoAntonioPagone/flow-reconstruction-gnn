import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import GCN
import torch.nn as nn
from collections import OrderedDict
import os

# Hyperparameters
FEAT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_DIM = 4
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {DEVICE}')

# Paths
TRAIN_INPUT_DIR = '../dataset_graph/training/train_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'
VALID_INPUT_DIR = '../dataset_graph/training/validation_input_graphs'
VALID_TARGET_DIR = '../dataset_graph/training/validation_graphs'

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' from the key
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

train_dataset = CustomDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)
valid_dataset = CustomDataset(VALID_INPUT_DIR, VALID_TARGET_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

model = GCN(feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
criterion = MSELoss()

# Load checkpoint if exists
if os.path.isfile("my_checkpoint.pth.tar"):
    checkpoint = torch.load("my_checkpoint.pth.tar")
    load_checkpoint(checkpoint, model)
    print("Checkpoint loaded successfully.")

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch+1}")
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        batch.x = batch.x.to(DEVICE)  
        batch.edge_index = batch.edge_index.to(DEVICE)
        batch.y = batch.y.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss}')

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch.x = batch.x.to(DEVICE)  
            batch.edge_index = batch.edge_index.to(DEVICE)
            batch.y = batch.y.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch.y)
            valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')
