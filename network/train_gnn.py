import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import GCN
import torch.nn as nn
# Hyperparameters
FEAT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_DIM = 4
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
TRAIN_INPUT_DIR = '../dataset_graph/training/train_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'
VALID_INPUT_DIR = '../dataset_graph/training/validation_input_graphs'
VALID_TARGET_DIR = '../dataset_graph/training/validation_graphs'

train_dataset = CustomDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)
valid_dataset = CustomDataset(VALID_INPUT_DIR, VALID_TARGET_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

model = GCN(feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)

    # Check for multiple GPUs and wrap model
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    model = nn.DataParallel(model)

model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
criterion = MSELoss()

for epoch in range(EPOCHS):
    print(f"Starting Epoch: {epoch+1}")
    model.train()
    for i, batch in enumerate(train_loader):
        batch.to(DEVICE)  # Move batch to GPU if available
        print(f"Training Batch: {i+1}")
        optimizer.zero_grad()
        out = model(batch)
        print(f"Model output size: {out.shape}")
        loss = criterion(out, batch.y)
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch.to(DEVICE)  # Move batch to GPU if available
            print(f"Validation Batch: {i+1}")
            out = model(batch)
            loss = criterion(out, batch.y)
            valid_loss += loss.item()
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss/len(valid_loader)}')
