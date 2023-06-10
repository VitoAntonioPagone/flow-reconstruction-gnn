import torch
from torch_geometric.nn import GraphSage
from torch_geometric.data import DataLoader
import torch.optim as optim
from models import  GraphSage

from utils import check_accuracy_graphs, load_checkpoint, save_checkpoint, get_loaders_g
import torch.nn as nn
import os 
from datasets import GraphDataset

TRAIN_INPUT_DIR = '../dataset_graph/training/test_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'
VAL_INPUT_DIR = '../dataset_graph/training/validation_input_graphs'
VAL_TARGET_DIR = '../dataset_graph/training/validation_graphs'
NUM_WORKERS = 6
PIN_MEMORY = True
HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_FEATURES = 4
LOAD_MODEL = False
PERCENTAGE_OF_MISSING_POINTS = 50
MODEL_NAME = f"GCN_{PERCENTAGE_OF_MISSING_POINTS}"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}_autoencoder_checkpoint_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}_autoencoder_checkpoint_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}.pth.tar'



def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    total_losses = []
    for incomplete_graph, complete_graph in train_loader:
        # Move data to the same device as the model
        incomplete_graph = incomplete_graph.to(device)
        complete_graph.x = complete_graph.x.to(device)
        complete_graph.edge_index = complete_graph.edge_index.to(device)
        
        optimizer.zero_grad()
        reconstructed_graph = model(incomplete_graph)
        loss = criterion(reconstructed_graph.x, complete_graph.x) # Use the target node features for calculating the loss
        loss.backward()
        optimizer.step()
        
        total_losses.append(loss.item())

    avg_loss = sum(total_losses) / len(total_losses)
    return avg_loss

def train():
    # Instantiate the model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Selected device: {DEVICE}")
    model = GraphSage(NUM_FEATURES, HIDDEN_CHANNELS).to(DEVICE)  
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Check for multiple GPUs and wrap model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    # Get the data loaders
    train_loader, val_loader = get_loaders_g(
        TRAIN_INPUT_DIR,
        TRAIN_TARGET_DIR,
        VAL_INPUT_DIR,
        VAL_TARGET_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL and os.path.isfile(LOAD_CHECKPOINT_FILE):
        checkpoint = torch.load(LOAD_CHECKPOINT_FILE)
        load_checkpoint(checkpoint, model)
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)

        val_loss = check_accuracy_graphs(val_loader, model, criterion, DEVICE) # Assume it returns validation loss
        val_losses.append(val_loss)

        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, SAVE_CHECKPOINT_FILE)

    print('Training complete.')


if __name__ == "__main__":
    train()
