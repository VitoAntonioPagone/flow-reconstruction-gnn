import torch
import torch.nn as nn
from torch.optim import Adam
from models import (
    UNet,
    ConvAutoEncoder_seven,
    DilatedConvAutoEncoder,
    SEConvAutoEncoder,
    ConvAutoEncoder_simplified)
from losses import (
    MaskedMSELoss, 
    NavierStokesLoss,
    TVLoss)
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    plot_losses,
    initialize_weights,
    print_autoencoder_dashboard
)

# Global parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 6
PIN_MEMORY = True
SHUFFLE = True
NUM_EPOCHS = 100
LOAD_MODEL = False
MODEL_NAME = "ConvAutoEncoder_simplified"
TRAIN_INPUTS_DIR = '../dataset/train_data/train_inputs_50'
TRAIN_LABELS_DIR = '../dataset/train_data/train_labels_50'
VAL_INPUTS_DIR   = '../dataset/train_data/val_inputs_50'
VAL_LABELS_DIR   = '../dataset/train_data/val_labels_50'


def train_fn(loader, model, optimizer, loss_fn, ns_loss, tv_loss, alpha, beta, scaler):
    loop = tqdm(loader, leave=True)

    total_loss = 0
    total_batches = 0

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
            tv_loss_value = tv_loss(outputs) # calculate TV loss
            loss = masked_loss + alpha * ns_loss_value + beta * tv_loss_value  # added TV loss to total loss

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_description(f"Iter [{batch_idx}/{len(loader)}]")
        loop.set_postfix(loss=loss.item(), NSLoss=ns_loss_value.item(), MaskedL2Loss=masked_loss.item(), TVLoss=tv_loss_value.item())  # added TV loss to logging
        
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    return avg_loss

def train_unet_conv_autoencoder(learning_rate, batch_size, alpha, beta):

    print(f"Selected device: {DEVICE}")

    model = ConvAutoEncoder_simplified().to(DEVICE)
    # Check for multiple GPUs and wrap model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    initialize_weights(model)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = MaskedMSELoss(device=DEVICE) 
    ns_loss = NavierStokesLoss(DEVICE)
    tv_loss = TVLoss().to(DEVICE)  # instantiate TVLoss

    train_loader, val_loader = get_loaders(
        TRAIN_INPUTS_DIR,
        TRAIN_LABELS_DIR,
        VAL_INPUTS_DIR,
        VAL_LABELS_DIR,
        batch_size,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_FILE), model)

    check_accuracy(val_loader, model, alpha, beta, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, ns_loss, tv_loss, alpha, beta, scaler) # added tv_loss
        train_losses.append(train_loss)

        # save model
        CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}_autoencoder_checkpoint_alpha_{alpha}_beta_{beta}_lr_{learning_rate}_batch_{batch_size}.pth.tar'
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, CHECKPOINT_FILE)

        # check accuracy
        val_loss = check_accuracy(val_loader, model, alpha, beta, device=DEVICE)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss}")

    plot_losses(train_losses, val_losses, alpha, beta, learning_rate, batch_size)
    return val_losses[-1] # Assuming the last validation loss is the one you want


if __name__ == "__main__":

    learning_rates = [0.001, 0.0001, 0.00001]
    batch_sizes = [64, 128]
    alphas = [0.01,0.05,0.1]
    betas = [0.01,0.05,0.1]

    best_parameters = None
    best_val_loss = float('inf')

    for lr in learning_rates:
        for bs in batch_sizes:
            for alpha in alphas:
                for beta in betas:
                    print(f'Training with lr={lr}, bs={bs}, alpha={alpha}, beta={beta}')
                    val_loss = train_unet_conv_autoencoder(lr, bs, alpha, beta)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_parameters = (lr, bs, alpha, beta)

    print(f'Best parameters are: lr={best_parameters[0]}, bs={best_parameters[1]}, alpha={best_parameters[2]}, beta={best_parameters[3]}')
