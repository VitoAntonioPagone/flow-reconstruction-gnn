import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader, DataListLoader
from datasets import CustomDataset
from models import (GraphSAGE_90, GraphSAGE_95, GAT_90, GAT_95, ChebNet_90, AGNN_90,
                    GIN_90, GraphSAGE_99, Hole_GraphSAGE_99, Hole_GAT_90, Hole_GCN_90, Hole_GAT_20Layers, Hole_GAT_25Layers, Hole_GAT_30Layers)
from torch_geometric.data import Batch
from losses import GraphNavierStokesLoss
from utils import graph_initialize_weights
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import DataParallel

HOLE = False

if not HOLE:
    ALPHA = 1e-4  
    BATCH_SIZE = 4
    LR = 0.0001
    EPOCHS = 100
    PERCENTAGE_OF_MISSING_POINTS = 90
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = False  
    MODEL_NAME = f"GAT_{PERCENTAGE_OF_MISSING_POINTS}"  
    LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'
    TRAIN_INPUT_DIR = f'../dataset_graph/training/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    TRAIN_TARGET_DIR = f'../dataset_graph/training/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}/' 
    VALID_INPUT_DIR = f'../dataset_graph/training/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    VALID_TARGET_DIR = f'../dataset_graph/training/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'
else:
    ALPHA = 1e-4  
    BATCH_SIZE = 4
    LR = 0.0001
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOAD_MODEL = False
    BOX_PERCENTAGE = 0.90
    TRAIN_INPUT_DIR = f'../dataset_graph/training_hole/train_input_graphs_box_{BOX_PERCENTAGE * 100}/'
    TRAIN_TARGET_DIR = f'../dataset_graph/training_hole/train_graphs_box_{BOX_PERCENTAGE * 100}/'
    VALID_INPUT_DIR = f'../dataset_graph/training_hole/validation_input_graphs_box_{BOX_PERCENTAGE * 100}/'
    VALID_TARGET_DIR = f'../dataset_graph/training_hole/validation_graphs_box_{BOX_PERCENTAGE * 100}/'
    MODEL_NAME = f"Hole_GAT_35Layers_box_{BOX_PERCENTAGE * 100}"
    LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'


def save_checkpoint(state, filename=SAVE_CHECKPOINT_FILE):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'../losses_plot/{MODEL_NAME}_losses_plot_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg', format='jpg', dpi=350)
    plt.close()

print('Loading train dataset...')
train_dataset = CustomDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)
print(f'Train dataset loaded with {len(train_dataset)} samples.')
print(f'First training sample: {train_dataset[0]}')

print('Loading validation dataset...')
valid_dataset = CustomDataset(VALID_INPUT_DIR, VALID_TARGET_DIR)
print(f'Validation dataset loaded with {len(valid_dataset)} samples.')
print(f'First validation sample: {valid_dataset[0]}')


print('Creating data loaders...')
train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataListLoader(valid_dataset, batch_size=BATCH_SIZE)
print('Data loaders created.')

print('Building model...')
model = GAT_90()

# Multi-GPU setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = DataParallel(model)

model.to(DEVICE)
graph_initialize_weights(model)  # Initialize weights of the model
print('Model built.')
print(f'Model Name: {type(model).__name__}')  # Print the name of the model
print(model)  # Print the model's structure


optimizer = Adam(model.parameters(), lr=LR)
criterion = MSELoss()
#navier_stokes_loss = GAT_95().to(DEVICE)

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
    for batch_idx, data_list in enumerate(train_loader):
        data_list = [data.to(DEVICE) for data in data_list] # Send each Data object to the correct device
        optimizer.zero_grad()
        out = model(data_list)  # Pass data_list to model
        batch = Batch.from_data_list(data_list) # Convert to Batch for loss calculation
        loss = criterion(out[:, :3], batch.y[:, :3])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f"  Batch {batch_idx + 1}, Training Loss: {loss.item()}")

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch: {epoch + 1}, Training Loss: {train_loss}')

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    model.eval()
    valid_loss = 0
    print('Processing validation data...')
    with torch.no_grad():
        for data_list in valid_loader:
            data_list = [data.to(DEVICE) for data in data_list] # Send each Data object to the correct device
            out = model(data_list)  # Pass data_list to model
            batch = Batch.from_data_list(data_list) # Convert to Batch for loss calculation
            loss = criterion(out[:, :3], batch.y[:, :3])
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    val_losses.append(valid_loss)
    print(f'Epoch: {epoch + 1}, Validation Loss: {valid_loss}')

plot_losses(train_losses, val_losses)

