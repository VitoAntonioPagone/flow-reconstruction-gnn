import torch
from torch.optim import Adam
from models import (
    UNet,
    ConvAutoEncoder)
from losses import MaskedMSELoss
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)


# Hyper-parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 6
PIN_MEMORY = True
LEARNING_RATE = 0.001
SHUFFLE = True
NUM_EPOCHS = 50
LOAD_MODEL = False
CHECKPOINT_FILE = 'network/trained_models/my_checkpoint.pth.tar'
TRAIN_INPUTS_DIR = 'dataset/train_data/train_inputs_50/'
TRAIN_LABELS_DIR = 'dataset/train_data/train_labels_50'
VAL_INPUTS_DIR = 'dataset/train_data/val_inputs_50/'
VAL_LABELS_DIR = 'dataset/train_data/val_labels_50'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

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


def train():
    print(f"Selected device: {DEVICE}")

    # model = UNet(in_channels=5, out_channels=4).to(DEVICE)
    model = ConvAutoEncoder().to(DEVICE)
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

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, CHECKPOINT_FILE)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    train()
