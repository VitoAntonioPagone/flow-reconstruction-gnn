import torch
from datasets import FlowDataset
from torch.utils.data import DataLoader
from losses import MaskedMSELoss
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pickle

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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



def check_accuracy(loader, model, device="cuda"):
    model.eval()

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            x_with_mask = torch.cat((x, mask), dim=1)
            preds = model(x_with_mask)
            loss_fn = MaskedMSELoss()
            loss = loss_fn(preds, y, mask)
        print(f"Validation Loss: {loss.item():.4f}")

    model.train()


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

import torch

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return print('Weights initialized with Glorot (Xavier) initializer, bias initialized to zero')

def print_mlp_characteristics(model):
    print("Multi-Layer Perceptron Summary:")
    print("---------------------------------")
    print(model)
    print("---------------------------------")
    print("Number of parameters: {}".format(sum(p.numel() for p in model.parameters())))
    print("Number of layers: {}".format(len(model.layers)))
    print("Layer-wise architecture:")
    for i, layer in enumerate(model.layers):
        print("Layer {} - {}".format(i+1, layer))

def print_autoencoder_dashboard(model):
    print("ConvAutoEncoder Architecture:\n")

    for name, module in model.named_children():
        print(f"{name}: {module}\n")


def load_torch_geometric_graphs(input_path):
    tg_graphs = []

    for filename in os.listdir(input_path):
        if filename.endswith("_tg.pkl"):
            file_path = os.path.join(input_path, filename)
            with open(file_path, 'rb') as f:
                tg_graph = pickle.load(f)
                tg_graphs.append(tg_graph)

    return tg_graphs


