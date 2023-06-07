import torch
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
import torch.optim as optim
from models import GCN
from utils import check_accuracy, save_checkpoint, load_checkpoint, initialize_weights, print_autoencoder_dashboard, plot_losses
from datasets import GraphDataset
from torch_geometric.data import DataLoader  # <-- here
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
NUM_NODE_FEATURES = 4

TRAIN_INPUTS_DIR = "../dataset_graph/training/train_input_graphs/"
TRAIN_LABELS_DIR = "../dataset_graph/training/train_graphs/"
VAL_INPUTS_DIR = "../dataset_graph/training/validation_input_graphs/"
VAL_LABELS_DIR = "../dataset_graph/training/validation_graphs/"

def get_loaders_graphs(input_dir, labels_dir, val_input_dir, val_labels_dir, batch_size):
    train_ds = GraphDataset(input_dir, labels_dir)
    val_ds = GraphDataset(val_input_dir, val_labels_dir)
    
    # Updated to torch_geometric.data.DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def train(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        losses = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)

            model.train()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, data.y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        train_losses.append(train_loss)
        print(f"Train Epoch: {epoch}, Loss: {train_loss:.4f}")
        
        val_loss = check_accuracy(val_loader, model, device)
        val_losses.append(val_loss)

    return train_losses, val_losses


def check_accuracy(loader, model, device):
    model.eval()
    losses = []
    criterion = MSELoss()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, data.y)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss


def main():
    train_loader, val_loader = get_loaders_graphs(
            input_dir=TRAIN_INPUTS_DIR,
            labels_dir=TRAIN_LABELS_DIR,
            val_input_dir=VAL_INPUTS_DIR,
            val_labels_dir=VAL_LABELS_DIR,
            batch_size=BATCH_SIZE,
        )


    model = GCN(NUM_NODE_FEATURES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MSELoss()

    initialize_weights(model)
    print_autoencoder_dashboard(model)

    train_losses, val_losses = train(train_loader, val_loader, model, criterion, optimizer, NUM_EPOCHS, DEVICE)

    plot_losses(train_losses, val_losses, alpha=1, beta=0.1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()

