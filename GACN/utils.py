import torch
from datasets import FlowDataset, CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pickle
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GINConv, SAGEConv

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' from the key
        new_state_dict[name] = v

    # Load the modified state_dict to the model
    model.load_state_dict(new_state_dict)


def check_accuracy_graphs(loader, model, criterion, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = criterion(preds, batch.x)
            losses.append(loss.item())
    avg_rmse = torch.sqrt(torch.tensor(losses).mean()).item()
    
    model.train()
    
    return avg_rmse

def get_loaders_graphs(input_dir, target_dir, batch_size, num_workers=4, pin_memory=True):
    # Create a dataset
    ds = CustomDataset(input_dir, target_dir)
    
    # Create a DataLoader
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    return loader

def get_loaders(
    train_inputs_dir,
    train_labels_dir,
    val_inputs_dir,
    val_labels_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_ds = FlowDataset(
        input_dir=train_inputs_dir,
        label_dir=train_labels_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = FlowDataset(
        input_dir=val_inputs_dir,
        label_dir=val_labels_dir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def plot_losses(train_losses, val_losses, alpha, beta, learning_rate, batch_size, model_name, percentage, epochs):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))  # Set a larger figure size
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Generate the plot filename based on the hyperparameters and the model name
    plot_filename = f'../losses_plot/{model_name}_losses_plot_{epochs}_alpha_{alpha}_beta_{beta}_lr_{learning_rate}_batch_{batch_size}.jpg'

    plt.savefig(plot_filename, format='jpg', dpi=500)



def graph_initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    return print('Weights initialized with Glorot (Xavier) initializer')


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return print('Weights initialized with Glorot (Xavier) initializer, bias initialized to zero')

def print_autoencoder_dashboard(model):
    print("ConvAutoEncoder Architecture:\n")

    for name, module in model.named_children():
        print(f"{name}: {module}\n")



