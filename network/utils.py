import torch
from network.datasets import FlowDataset
from torch.utils.data import DataLoader
from network.losses import MaskedMSELoss


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

