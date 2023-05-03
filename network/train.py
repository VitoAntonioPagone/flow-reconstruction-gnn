import torch
import torch.nn as nn
from torch.optim import Adam
from models import (
    UNet,
    ConvAutoEncoder,
    DilatedConvAutoEncoder,
    MLP)
from losses import MaskedMSELoss, NavierStokesLoss
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    plot_losses,
    check_accuracy_mlp,
    get_loaders_mlp,
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

# MLP parameters
MLP_INPUT_SIZE = 3
MLP_HIDDEN_SIZE1 = 512
MLP_HIDDEN_SIZE2 = 1024
MLP_HIDDEN_SIZE3 = 2048
MLP_HIDDEN_SIZE4 = 1024
MLP_HIDDEN_SIZE5 = 512
MLP_OUTPUT_SIZE = 3
MLP_LEARNING_RATE = 0.001
MLP_NUM_EPOCHS = 100
MLP_BATCH_SIZE = 64
MLP_TRAIN_DATA_FILE = "flow_reconstruction/dataset_mlp/train_data/train_combined/combined_data_10000.npz"
MLP_VAL_DATA_FILE = "flow_reconstruction/dataset_mlp/train_data/validation_combined/combined_data_10000.npz"
MLP_CHECKPOINT_FILE = 'flow_reconstruction/network/trained_models/mlp_checkpoint.pth.tar'
MLP_ALPHA = 1e-4
LOAD_MLP_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
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
            loss = loss_fn(outputs, labels, mask)

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

    # model = UNet(in_channels=5, out_channels=4).to(DEVICE)
    model = ConvAutoEncoder().to(DEVICE)
    print_autoencoder_dashboard(model)
    initialize_weights(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedMSELoss()

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
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
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


def main_mlp():
    # Create the MLP model
    model = MLP(MLP_INPUT_SIZE, MLP_HIDDEN_SIZE1, MLP_HIDDEN_SIZE2, MLP_HIDDEN_SIZE3, MLP_HIDDEN_SIZE4,MLP_HIDDEN_SIZE5, MLP_OUTPUT_SIZE)
    print_mlp_characteristics(model)
    initialize_weights(model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available, using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, using CPU.")
    
    model.to(device)
    # Load the data
    train_data_file = "/Users/vitoantonio/Desktop/feature_AutoEncoder/flow_reconstruction/dataset_mlp/train_data/train_combined/combined_data_10000.npz"
    val_data_file   = "/Users/vitoantonio/Desktop/feature_AutoEncoder/flow_reconstruction/dataset_mlp/train_data/validation_combined/combined_data_10000.npz"
    train_loader, val_loader = get_loaders_mlp(
        MLP_TRAIN_DATA_FILE,
        MLP_VAL_DATA_FILE, 
        MLP_BATCH_SIZE)
    
    if LOAD_MLP_MODEL:
        load_checkpoint(torch.load(MLP_CHECKPOINT_FILE), model)

    # Define the loss function and the optimizer
    mse_loss = nn.MSELoss()
    navier_stokes_loss = NavierStokesLoss(model)
    optimizer = Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(MLP_NUM_EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss_mse = mse_loss(outputs, labels)
            loss_ns = navier_stokes_loss.forward(outputs, *torch.split(inputs, 1, dim=1))
            loss = loss_mse + MLP_ALPHA * loss_ns

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{MLP_NUM_EPOCHS}], Loss: {loss.item():.4f}")

        # Save the checkpoint at the end of each epoch
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, MLP_CHECKPOINT_FILE)

        train_losses.append(loss.item())
        check_accuracy_mlp(val_loader, model, mse_loss, navier_stokes_loss, MLP_ALPHA, device)
        val_losses.append(loss.item())



def main():
    model_to_train = input("Enter the model to train ('MLP', 'UNet', 'ConvAutoEncoder', or 'DilatedConvAutoEncoder'): ")

    if model_to_train == "MLP":
        main_mlp()
    elif model_to_train == "UNet" or model_to_train == "ConvAutoEncoder" or model_to_train == "DilatedConvAutoEncoder":
        train_unet_conv_autoencoder()
    else:
        print("Invalid model_to_train value. Please choose 'MLP', 'UNet', 'ConvAutoEncoder', or 'DilatedConvAutoEncoder'.")

if __name__ == "__main__":
    main()
