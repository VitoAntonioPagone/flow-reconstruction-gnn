import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import GCN
from collections import OrderedDict
import os

# Hyperparameters
FEAT_DIM = 6
HIDDEN_DIM = 64
OUTPUT_DIM = 6
BATCH_SIZE = 16
LR = 0.01
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False  
MODEL_NAME = f"GCN"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'

print(f'Starting script with Device: {DEVICE}')

# Paths
TRAIN_INPUT_DIR = '../dataset_graph/training/train_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'
VALID_INPUT_DIR = '../dataset_graph/training/validation_input_graphs'
VALID_TARGET_DIR = '../dataset_graph/training/validation_graphs'

def save_checkpoint(state, filename=SAVE_CHECKPOINT_FILE):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

print('Loading train dataset...')
train_dataset = CustomDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)
print('Train dataset loaded.')

print('Loading validation dataset...')
valid_dataset = CustomDataset(VALID_INPUT_DIR, VALID_TARGET_DIR)
print('Validation dataset loaded.')

print('Creating data loaders...')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
print('Data loaders created.')

print('Building model...')
model = GCN(feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
model.to(DEVICE)
print('Model built.')

optimizer = Adam(model.parameters(), lr=LR)
criterion = MSELoss()

if LOAD_MODEL and os.path.isfile("my_checkpoint.pth.tar"):
    print('Loading checkpoint...')
    checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)
    load_checkpoint(checkpoint, model, optimizer)
    print("Checkpoint loaded successfully.")

for epoch in range(EPOCHS):
    print(f'Starting epoch {epoch + 1}...')
    model.train()
    train_loss = 0
    print('Processing training data...')
    for batch_idx, batch in enumerate(train_loader):
        batch.x = batch.x.to(DEVICE)  
        batch.edge_index = batch.edge_index.to(DEVICE)
        batch.y = batch.y.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)

        # Calculate loss only for the first 3 features
        loss = criterion(out[:,3], batch.y[:,3],)
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Print batch number and loss for each batch
        print(f"  Batch {batch_idx + 1}, Training Loss: {loss.item()}")

    train_loss /= len(train_loader)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss}')

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    model.eval()
    valid_loss = 0
    print('Processing validation data...')
    with torch.no_grad():
        for batch in valid_loader:
            batch.x = batch.x.to(DEVICE)  
            batch.edge_index = batch.edge_index.to(DEVICE)
            batch.y = batch.y.to(DEVICE)
            out = model(batch)
            
            # Calculate loss only for the first 3 features
            loss = criterion(out[:,3], batch.y[:,3])
            
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')
