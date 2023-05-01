import torch
from datasets import FlowDataset, MLPDataset
from torch.utils.data import DataLoader
from losses import MaskedMSELoss
import matplotlib.pyplot as plt


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

def get_loaders_mlp(
    train_data_file,
    val_data_file,
    batch_size,
    num_workers=8,
    pin_memory=True,
):
    train_ds = MLPDataset(train_data_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MLPDataset(val_data_file)

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

def check_accuracy_mlp(loader, model, mse_loss, navier_stokes_loss, alpha, device="cuda"):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            loss_mse = mse_loss(preds, labels)
            loss_ns = navier_stokes_loss.forward(preds, *torch.split(inputs, 1, dim=1))
            loss = loss_mse + alpha * loss_ns
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    print(f"Validation Loss: {avg_loss:.4f}")

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
