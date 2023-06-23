import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import GCN, GraphSAGE, GAT, GraphSAGE_90
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 100
PERCENTAGE_OF_MISSING_POINTS = 90  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False  
MODEL_NAME = f"GraphSAGE_{PERCENTAGE_OF_MISSING_POINTS}"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'

# Directory for storing the loss plot
LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'

print(f'Starting script with Device: {DEVICE}')

# Paths
TRAIN_INPUT_DIR = f'../dataset_graph/training/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  
TRAIN_TARGET_DIR = f'../dataset_graph/training/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}' 
VALID_INPUT_DIR = f'../dataset_graph/training/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  
VALID_TARGET_DIR = f'../dataset_graph/training/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  

def save_checkpoint(state, filename=SAVE_CHECKPOINT_FILE):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 7))  # Set a larger figure size
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'../losses_plot/{MODEL_NAME}loss_plot_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg', format='jpg', dpi=350)
    plt.close()

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
model = GraphSAGE_90()
model.to(DEVICE)
print('Model built.')

optimizer = Adam(model.parameters(), lr=LR)
criterion = MSELoss()

train_losses, val_losses = [], []

if LOAD_MODEL and os.path.isfile(LOAD_CHECKPOINT_FILE):
    print('Loading checkpoint...')
    checkpoint = torch.load(LOAD_CHECKPOINT_FILE, map_location=DEVICE)
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
        loss = criterion(out[:,:3], batch.y[:,:3])  # Modified this line
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Print batch number and loss for each batch
        print(f"  Batch {batch_idx + 1}, Training Loss: {loss.item()}")

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
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
            loss = criterion(out[:,:3], batch.y[:,:3])  # Modified this line
            
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    val_losses.append(valid_loss)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')

# After the training loop ends, plot the losses
plot_losses(train_losses, val_losses)
