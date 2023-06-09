import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
from datasets import GraphDataset
from models import GCN
from utils import check_accuracy_graphs, load_checkpoint, save_checkpoint
import torch.nn as nn
import os 
HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 32

TRAIN_INPUT_DIR = '../dataset_graph/training/test_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'
VAL_INPUT_DIR = '../dataset_graph/training/validation_input_graphs'
VAL_TARGET_DIR = '../dataset_graph/training/validation_graphs'
NUM_FEATURES = 4
LOAD_CHECKPOINT_FILE = ''
SAVE_CHECKPOINT_FILE = ''
LOAD_MODEL = False
PERCENTAGE_OF_MISSING_POINTS = 50
MODEL_NAME = f"GCN_{PERCENTAGE_OF_MISSING_POINTS}"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}_autoencoder_checkpoint_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}_autoencoder_checkpoint_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}.pth.tar'



# Load datasets
train_dataset = GraphDataset(input_dir=TRAIN_INPUT_DIR, target_dir=TRAIN_TARGET_DIR)
val_dataset = GraphDataset(input_dir=VAL_INPUT_DIR, target_dir=VAL_TARGET_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Instantiate the model
model = GCN(num_features=NUM_FEATURES, hidden_channels=HIDDEN_CHANNELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Check if a load checkpoint is specified and file exists
if LOAD_MODEL and os.path.isfile(LOAD_CHECKPOINT_FILE):
    checkpoint = torch.load(LOAD_CHECKPOINT_FILE)
    print("=> Loading checkpoint")
    load_checkpoint(checkpoint, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for data_list in train_loader:
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

    avg_train_loss = running_loss / len(train_loader.dataset)

    # Validation
    val_rmse = check_accuracy_graphs(val_loader, model, criterion, device)

    print(f'Epoch: {epoch+1}, Loss: {avg_train_loss}, Validation RMSE: {val_rmse}')
    
    # Save model checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, SAVE_CHECKPOINT_FILE)

print('Training complete.')
