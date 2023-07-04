import torch
import optuna
import torch.nn as nn
from torch.optim import Adam
from models import (
    ConvNet_50,
    ConvNet_90,
    ConvNet_95,
    ConvNet_99)
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

# Hyper-parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False  
PERCENTAGE_OF_MISSING_POINTS = 90
MODEL_NAME = f"Optuna_ConvNet_{PERCENTAGE_OF_MISSING_POINTS}"
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}.pth.tar'
TRAIN_INPUTS_DIR = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/train_inputs_{PERCENTAGE_OF_MISSING_POINTS}'
TRAIN_LABELS_DIR = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/train_labels_{PERCENTAGE_OF_MISSING_POINTS}'
VAL_INPUTS_DIR   = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/val_inputs_{PERCENTAGE_OF_MISSING_POINTS}'
VAL_LABELS_DIR   = f'../dataset/train_data_{PERCENTAGE_OF_MISSING_POINTS}/val_labels_{PERCENTAGE_OF_MISSING_POINTS}'

def train_fn(trial, loader, model, optimizer, loss_fn, ns_loss, tv_loss, alpha, beta, scaler):
    loop = tqdm(loader, leave=True)

    total_loss = 0
    total_batches = 0

    for batch_idx, (inputs, labels, mask) in enumerate(loop):
        inputs = inputs.to(device=DEVICE)
        labels = labels.to(device=DEVICE)
        mask = mask.to(device=DEVICE)

        with torch.cuda.amp.autocast():
            inputs_with_mask = torch.cat((inputs, mask), dim=1)
            outputs = model(inputs_with_mask)
            masked_loss = loss_fn(outputs, labels, mask)
            ns_loss_value = ns_loss(outputs)
            tv_loss_value = tv_loss(outputs)
            loss = masked_loss + alpha * ns_loss_value + beta * tv_loss_value

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description(f"Iter [{batch_idx}/{len(loader)}]")
        loop.set_postfix(loss=loss.item(), NSLoss=ns_loss_value.item(), MaskedL2Loss=masked_loss.item(), TVLoss=tv_loss_value.item())
        
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    return avg_loss

def objective(trial):
    BATCH_SIZE = trial.suggest_categorical('batch_size', [64, 128])
    LEARNING_RATE = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    NUM_EPOCHS = trial.suggest_int('num_epochs', 100, 200)
    ALPHA = trial.suggest_float('alpha', 0, 2)
    BETA = trial.suggest_float('beta', 0, 2)

    print(f"\nTrial {trial.number} parameters: \n"
          f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, "
          f"Num_epochs: {NUM_EPOCHS}, Alpha: {ALPHA}, Beta: {BETA}\n")

    model = ConvNet_90().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    initialize_weights(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MaskedMSELoss(device=DEVICE)
    ns_loss = NavierStokesLoss(DEVICE)
    tv_loss = TVLoss().to(DEVICE)

    train_loader, val_loader = get_loaders(
        TRAIN_INPUTS_DIR,
        TRAIN_LABELS_DIR,
        VAL_INPUTS_DIR,
        VAL_LABELS_DIR,
        BATCH_SIZE,
        trial.suggest_int('num_workers', 1, 8),
        trial.suggest_categorical('pin_memory', [True, False]),
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_CHECKPOINT_FILE), model)

    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_fn(trial, train_loader, model, optimizer, loss_fn, ns_loss, tv_loss, ALPHA, BETA, scaler)
        train_losses.append(train_loss)

        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, SAVE_CHECKPOINT_FILE)

        val_loss = check_accuracy(val_loader, model, ALPHA, BETA, device=DEVICE)
        val_losses.append(val_loss)
        print(f"\nEpoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}\n")

    plot_losses(train_losses, val_losses, ALPHA, BETA, LEARNING_RATE, BATCH_SIZE, MODEL_NAME, PERCENTAGE_OF_MISSING_POINTS, NUM_EPOCHS)

    return val_losses[-1]

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25, show_progress_bar=True)

    print('\nBest trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
