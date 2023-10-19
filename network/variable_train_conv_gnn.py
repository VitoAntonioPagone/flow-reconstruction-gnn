import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import GAT_50, GCN_98_10,GraphSAGE_98_10,GCN_95_8, GraphSAGE_95_8, GCN_90_6_Double,GAT_98_8_Modified, GAT_98_8,GAT_98_10, GAT_95_12, GAT_95_10,GAT_95_8,GAT_90_6_Double,GATv2_90_8_2heads,GATv2_90_8,GraphSAGE_90_8, GAT_90_8_Increased, GAT_90_8,GCN_90_6, GraphSAGE_90,GraphSAGE_90_6_Double, GraphSAGE_95, GAT_90, GraphSAGE_99, GCN_90, GAT_90_3, GAT_90_3_2heads, GAT_90_6, GAT_90_6_2heads
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from losses import GraphNavierStokesLoss
from utils import graph_initialize_weights

HOLE = False
USE_LINF_LOSS = False  # set this to False to use L2 loss

if not HOLE:
    ALPHA = 1e-6  
    BATCH_SIZE = 2
    LR = 1e-6
    EPOCHS = 50
    #PERCENTAGE_OF_MISSING_POINTS = 90
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = False  
    MODEL_NAME = f"GCN_10_variable"  
    LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'
    TRAIN_INPUT_DIR = f'../dataset_graph/training_FP/train_input_graphs_variable/'  
    TRAIN_TARGET_DIR = f'../dataset_graph/training_FP/train_graphs_variable/' 
    VALID_INPUT_DIR = f'../dataset_graph/training_FP/validation_input_graphs_variable/'  
    VALID_TARGET_DIR = f'../dataset_graph/training_FP/validation_graphs_variable/'
else:
    ALPHA = 1e-4  
    BATCH_SIZE = 1
    PERCENTAGE_OF_MISSING_POINTS = 0

    LR = 0.00001
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOAD_MODEL = False
    BOX_PERCENTAGE = 0.90
    TRAIN_INPUT_DIR = f'../dataset_graph/training_hole/train_input_graphs_box_{BOX_PERCENTAGE * 100}'
    TRAIN_TARGET_DIR = f'../dataset_graph/training_hole/train_graphs_box_{BOX_PERCENTAGE * 100}'
    VALID_INPUT_DIR = f'../dataset_graph/training_hole/validation_input_graphs_box_{BOX_PERCENTAGE * 100}'
    VALID_TARGET_DIR = f'../dataset_graph/training_hole/validation_graphs_box_{BOX_PERCENTAGE * 100}'
    MODEL_NAME = f"Hole_GAT_8_box_{BOX_PERCENTAGE * 100}"
    LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'
'''
# Hyperparameters
ALPHA = 1e-7  
BATCH_SIZE = 1
LR = 0.00001
EPOCHS = 50
PERCENTAGE_OF_MISSING_POINTS = 98
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False  
MODEL_NAME = f"INCR_UNIV_NS_GAT_8'_{PERCENTAGE_OF_MISSING_POINTS}"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'

print(f'Starting script with Device: {DEVICE}')

TRAIN_INPUT_DIR = f'../dataset_graph/training/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  
TRAIN_TARGET_DIR = f'../dataset_graph/training/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}' 
VALID_INPUT_DIR = f'../dataset_graph/training/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  
VALID_TARGET_DIR = f'../dataset_graph/training/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}'
'''
print(f'Starting script with Device: {DEVICE}')

def Linfinity_loss(pred, target):
    return (pred - target).abs().max()

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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
print('Data loaders created.')

print('Building model...')
model = GCN_98_10()
model.to(DEVICE)
graph_initialize_weights(model)  # Initialize weights of the model

print('Model built.')
print("--------------------------")
print(f"  Model Name: {type(model).__name__}")  
print("--------------------------")
print(f"  Device: {DEVICE}")
print("--------------------------")
print(f"  Alpha: {ALPHA}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Number of Epochs: {EPOCHS}")
print(f"  Percentage of Missing Points: {PERCENTAGE_OF_MISSING_POINTS}")
print("--------------------------")
print(f"  Load Model: {'Yes' if LOAD_MODEL else 'No'}")
print(f"  Checkpoint File: {LOAD_CHECKPOINT_FILE}")
print("--------------------------")
print(f"  Training Input Directory: {TRAIN_INPUT_DIR}")
print(f"  Training Target Directory: {TRAIN_TARGET_DIR}")
print(f"  Validation Input Directory: {VALID_INPUT_DIR}")
print(f"  Validation Target Directory: {VALID_TARGET_DIR}")
print("--------------------------")
print(f"  Training Dataset Size: {len(train_dataset)} samples")
print(f"  Validation Dataset Size: {len(valid_dataset)} samples")
print("--------------------------")
print(f"  Model Structure: \n{model}")
print("--------------------------")



optimizer = Adam(model.parameters(), lr=LR)
criterion = MSELoss()
navier_stokes_loss = GraphNavierStokesLoss().to(DEVICE)

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
        l2_loss = criterion(out[:,:3], batch.y[:,:3])
        linf_loss = Linfinity_loss(out[:,:3], batch.y[:,:3])
        main_loss = linf_loss if USE_LINF_LOSS else l2_loss
        loss = main_loss + ALPHA * navier_stokes_loss(batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
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
            l2_loss = criterion(out[:,:3], batch.y[:,:3])
            linf_loss = Linfinity_loss(out[:,:3], batch.y[:,:3])
            main_loss = linf_loss if USE_LINF_LOSS else l2_loss
            loss = main_loss + ALPHA * navier_stokes_loss(batch)
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    val_losses.append(valid_loss)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')

plot_losses(train_losses, val_losses)
