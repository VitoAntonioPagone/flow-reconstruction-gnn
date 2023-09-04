import torch
import optuna
import torch.nn as nn
from torch.optim import Adam
from models import (
    ConvNet_50,
    ConvNet_90,
    ConvNet_95,
    ConvNet_98)
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
from torchsummary import summary

# Hyper-parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 6
PIN_MEMORY = True
LEARNING_RATE = 0.0001
SHUFFLE = True
NUM_EPOCHS = 500
ALPHA = 0.1
BETA = 0.1
LOAD_MODEL = False  
PERCENTAGE_OF_MISSING_POINTS = 98
MODEL_NAME = f"ConvNet_{PERCENTAGE_OF_MISSING_POINTS}"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}__alpha_{ALPHA}_beta_{BETA}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{NUM_EPOCHS}__alpha_{ALPHA}_beta_{BETA}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}.pth.tar'

TRAIN_INPUTS_DIR = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/train_inputs_{PERCENTAGE_OF_MISSING_POINTS}'
TRAIN_LABELS_DIR = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/train_labels_{PERCENTAGE_OF_MISSING_POINTS}'
VAL_INPUTS_DIR   = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/val_inputs_{PERCENTAGE_OF_MISSING_POINTS}'
VAL_LABELS_DIR   = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/val_labels_{PERCENTAGE_OF_MISSING_POINTS}'


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

def train_unet_conv_autoencoder():
    print(f"Selected device: {DEVICE}")

    model = ConvNet_98().to(DEVICE)

    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(4, 256, 256))

    # Check for multiple GPUs and wrap model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    print(f"\nHyperparameters:")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Alpha: {ALPHA}")
    print(f"Beta: {BETA}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    initialize_weights(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedMSELoss(device=DEVICE) 
    ns_loss = NavierStokesLoss(DEVICE)
    tv_loss = TVLoss().to(DEVICE)  # instantiate TVLoss

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
        load_checkpoint(torch.load(LOAD_CHECKPOINT_FILE), model)
    check_accuracy(val_loader, model, ALPHA, BETA, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, ns_loss, tv_loss, ALPHA, BETA, scaler) # added tv_loss
        train_losses.append(train_loss)

        # save model
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, SAVE_CHECKPOINT_FILE)

        # check accuracy
        val_loss = check_accuracy(val_loader, model, ALPHA, BETA, device=DEVICE)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss}")

    plot_losses(train_losses, val_losses, ALPHA, BETA, LEARNING_RATE, BATCH_SIZE, MODEL_NAME, PERCENTAGE_OF_MISSING_POINTS, NUM_EPOCHS)

if __name__ == "__main__":
    train_unet_conv_autoencoder()
