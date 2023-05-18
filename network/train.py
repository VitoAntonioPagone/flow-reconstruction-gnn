import torch
import torch.nn as nn
from torch.optim import Adam
from models import (
    UNet,
    ConvAutoEncoder,
    DilatedConvAutoEncoder,
    MLP,
    SEConvAutoEncoder)
from losses import (
    MaskedMSELoss, 
    NavierStokesLoss)
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    plot_losses,
    initialize_weights,
    print_mlp_characteristics,
    print_autoencoder_dashboard
)

# Hyper-parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 6
PIN_MEMORY = True
LEARNING_RATE = 0.001
SHUFFLE = True
NUM_EPOCHS = 1
LOAD_MODEL = False
CHECKPOINT_FILE = 'flow_reconstruction/network/trained_models/autoencoder_checkpoint.pth.tar'
TRAIN_INPUTS_DIR = 'flow_reconstruction/dataset/train_data/train_inputs_50'
TRAIN_LABELS_DIR = 'flow_reconstruction/dataset/train_data/train_labels_50'
VAL_INPUTS_DIR = 'flow_reconstruction/dataset/train_data/val_inputs_50'
VAL_LABELS_DIR = 'flow_reconstruction/dataset/train_data/val_labels_50'
ALPHA = 1  # set this to the desired value

def train_fn(loader, model, optimizer, loss_fn, ns_loss, alpha, scaler):
    loop = tqdm(loader)
    losses = []

    for batch_idx, (inputs, labels, mask) in enumerate(loop):
        inputs = inputs.to(device=DEVICE)
        labels = labels.to(device=DEVICE)
        mask = mask.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            inputs_with_mask = torch.cat((inputs, mask), dim=1)
            outputs = model(inputs_with_mask)
            masked_loss = loss_fn(outputs, labels, mask)
            ns_loss_value = ns_loss(outputs)
            loss = masked_loss + alpha * ns_loss_value

        # print the Navier-Stokes and L2 losses separately
        print(f"Navier-Stokes Loss: {ns_loss_value.item()}, Masked L2 Loss: {masked_loss.item()}")

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())

    return sum(losses) / len(losses)



def train_unet_conv_autoencoder():

    print(f"Selected device: {DEVICE}")

    model = ConvAutoEncoder().to(DEVICE)
    print_autoencoder_dashboard(model)
    initialize_weights(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedMSELoss()
    ns_loss = NavierStokesLoss().to(DEVICE)

    train_loader, val_loader = get_loaders(
        TRAIN_INPUTS_DIR,
        TRAIN_LABELS_DIR,
        VAL_INPUTS_DIR,
        VAL_LABELS_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_FILE), model)
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, ns_loss, ALPHA, scaler)
        train_losses.append(train_loss)

        # save model
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, CHECKPOINT_FILE)

        # check accuracy
        val_loss = check_accuracy(val_loader, model, device=DEVICE)
        val_losses.append(val_loss)

    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    train_unet_conv_autoencoder()
