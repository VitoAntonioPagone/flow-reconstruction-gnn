import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
from datasets import GraphDataset
from models import GCN
from utils import check_accuracy_graphs, load_checkpoint, save_checkpoint, get_loaders_g
import torch.nn as nn
import os 

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data_list in loader:
        for data in data_list:
            data = data.to(device)

            # Forward pass
            preds = model(data)
            loss = criterion(preds, data.x)  # Change 'data.y' to 'data.x' as the model is predicting node features

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs

    return running_loss / len(loader.dataset)

def train():
    # Instantiate the model
    model = GCN(num_features=NUM_FEATURES, hidden_channels=HIDDEN_CHANNELS)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = check_accuracy_graphs(val_loader, model, criterion, device) # Assume it returns validation loss
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