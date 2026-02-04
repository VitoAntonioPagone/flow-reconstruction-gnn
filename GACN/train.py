import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from graph_models import GAT_98_8_SkipConnections
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from losses import GraphNavierStokesLoss
from utils import graph_initialize_weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import get_laplacian

HOLE = False
USE_LINF_LOSS = False
use_nmse = False

if not HOLE:
    GAMMA = 10
    ALPHA = 0
    LAPLACIAN_REG_WEIGHT = 0     
    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 50
    PERCENTAGE_OF_MISSING_POINTS = 98
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = False  
    MODEL_NAME = f"ONLYL2_{PERCENTAGE_OF_MISSING_POINTS}"  
    LOAD_CHECKPOINT_FILE = f'../gacn_trained_models/NOISY_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../gacn_trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot_full/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'
    TRAIN_INPUT_DIR = f'../gacn_dataset_graph/training_FP/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    TRAIN_TARGET_DIR = f'../gacn_dataset_graph/training_FP/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}/' 
    VALID_INPUT_DIR = f'../gacn_dataset_graph/training_FP/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    VALID_TARGET_DIR = f'../gacn_dataset_graph/training_FP/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'
else:
    ALPHA = 1e-4  
    BATCH_SIZE = 1
    PERCENTAGE_OF_MISSING_POINTS = 0
    LR = 0.00001
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOAD_MODEL = False
    BOX_PERCENTAGE = 0.90
    TRAIN_INPUT_DIR = f'../dataset_graph/training_hole/train_input_graphs_box_{BOX_PERCENTAGE * 100}'
    TRAIN_TARGET_DIR = f'../dataset_graph/training_hole/train_graphs_box_{BOX_PERCENTAGE * 100}'
    VALID_INPUT_DIR = f'../dataset_graph/training_hole/validation_input_graphs_box_{BOX_PERCENTAGE * 100}'
    VALID_TARGET_DIR = f'../dataset_graph/training_hole/validation_graphs_box_{BOX_PERCENTAGE * 100}'
    MODEL_NAME = f"Hole_GAT_8_box_{BOX_PERCENTAGE * 100}"
    LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'

print(f'Starting script with Device: {DEVICE}')

def relobralo(model, f_loss, b_losses, args:dict):
    """Compute weighted loss using dynamic weighting."""
    losses = [f_loss] + b_losses
    lambs_hat = (F.softmax(torch.tensor([losses[i] / (args['T'] + 1e-12) for i in range(len(losses))]), dim=0) * len(losses)).detach()
    for i in range(len(lambs_hat)):
        lambs_hat[i] = max(lambs_hat[i], args['lam'+str(i)])
    total_loss = sum([lambs_hat[i] * losses[i] for i in range(len(losses))])
    args = args.copy()
    for i in range(len(b_losses) + 1):
        args['lam'+str(i)] = lambs_hat[i]
        args['l'+str(i)] = losses[i]
    return total_loss, f_loss, b_losses, args

args = {
    'T': 10,     
    'rho': 0.5,    
    'alpha': 0.5,  
    'lam0': 1,    
    'lam1': 1,  
    'lam2': 1, 
    'l0': 1,    
    'l1': 1,  
    'l2': 1,  
}

def laplacian_regularization(graph):
    """Compute Laplacian regularization term for graph."""
    features = graph.x[:, :3]
    laplacian_indices, laplacian_values = get_laplacian(graph.edge_index, edge_weight=graph.edge_attr, normalization=None)
    num_nodes = graph.num_nodes
    laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))
    LX = torch.sparse.mm(laplacian, features)
    regularization_matrix = torch.mm(features.transpose(0, 1), LX)
    regularization = torch.trace(regularization_matrix)
    return regularization

def save_checkpoint(state, filename, epoch):
    """Save model checkpoint."""
    state['epoch'] = epoch
    print(f"=> Saving checkpoint at epoch {epoch}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load model checkpoint."""
    print("=> Loading checkpoint")
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file does not exist at {checkpoint_path}")
        return None, None
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint['state_dict'].keys())
    if model_keys != checkpoint_keys:
        print("Mismatch in model's state_dict keys and checkpoint keys")
        print("Missing in checkpoint:", model_keys - checkpoint_keys)
        print("Extra in checkpoint:", checkpoint_keys - model_keys)
        return None, None
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model's state_dict loaded successfully.")
    except Exception as e:
        print(f"Exception occurred while loading model: {e}")
        return None, None
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer's state_dict loaded successfully.")
    except Exception as e:
        print(f"Exception occurred while loading optimizer: {e}")
        return None, None
    start_epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    print(f"Checkpoint loaded successfully with all the available information. Starting from epoch {start_epoch} with best loss {best_loss}.")
    return start_epoch, best_loss

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'../losses_plot/{MODEL_NAME}_losses_plot_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg', format='jpg', dpi=350)
    plt.close()

print('Loading train dataset...')
train_dataset = CustomDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)
print(f'Train dataset loaded with {len(train_dataset)} samples.')
print(f'First training sample: {train_dataset[0]}')

print('Loading validation dataset...')
valid_dataset = CustomDataset(VALID_INPUT_DIR, VALID_TARGET_DIR)
print(f'Validation dataset loaded with {len(valid_dataset)} samples.')
print(f'First validation sample: {valid_dataset[0]}')

print('Creating data loaders...')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
print('Data loaders created.')

print('Building model...')
model = GAT_98_8_SkipConnections()
model.to(DEVICE)
graph_initialize_weights(model)
print(f'Total number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

print('Model built.')
print("--------------------------")
print(f"  Model Name: {type(model).__name__}")  
print("--------------------------")
print(f"  Device: {DEVICE}")
print("--------------------------")
print(f"  Alpha: {ALPHA}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Number of Epochs: {EPOCHS}")
print(f"  Percentage of Missing Points: {PERCENTAGE_OF_MISSING_POINTS}")
print("--------------------------")
print(f"  Load Model: {'Yes' if LOAD_MODEL else 'No'}")
print(f"  Checkpoint File: {LOAD_CHECKPOINT_FILE}")
print("--------------------------")
print(f"  Training Input Directory: {TRAIN_INPUT_DIR}")
print(f"  Training Target Directory: {TRAIN_TARGET_DIR}")
print(f"  Validation Input Directory: {VALID_INPUT_DIR}")
print(f"  Validation Target Directory: {VALID_TARGET_DIR}")
print("--------------------------")
print(f"  Training Dataset Size: {len(train_dataset)} samples")
print(f"  Validation Dataset Size: {len(valid_dataset)} samples")
print("--------------------------")
print(f"  Model Structure: \n{model}")
print("--------------------------")

optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
criterion = MSELoss()
navier_stokes_loss = GraphNavierStokesLoss().to(DEVICE)

train_losses, val_losses = [], []
PRINT_INTERVAL = 500

start_epoch, best_loss = 0, float('inf')
if LOAD_MODEL and os.path.isfile(LOAD_CHECKPOINT_FILE):
    start_epoch, best_loss = load_checkpoint(LOAD_CHECKPOINT_FILE, model, optimizer)
    if start_epoch is None:
        start_epoch = 0
        best_loss = float('inf')
    else:
        print(f"Resuming training from epoch {start_epoch} as saved in checkpoint.")

for epoch in range(start_epoch,EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch: {epoch+1}, Learning Rate: {current_lr}')
    model.train()
    train_loss = 0
    print('Processing training data...')
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        l2_loss = criterion(out[:,:3], batch.y[:,:3])
        main_loss = GAMMA*l2_loss
        ns_loss = ALPHA*navier_stokes_loss(batch)
        laplacian_loss = LAPLACIAN_REG_WEIGHT*laplacian_regularization(batch)
        _, _, _, updated_args = relobralo(model, main_loss, [ns_loss, laplacian_loss], args)
        args.update(updated_args)
        lam_main = args['lam0']
        lam_ns = args['lam1']
        lam_laplacian = args['lam2']
        total_loss = lam_main *  main_loss + lam_ns *  ns_loss + lam_laplacian * laplacian_loss
        train_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()
        if batch_idx % PRINT_INTERVAL == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)},,AlphaDIFFUSION: {model.alpha.item():.4f},Training Loss: {total_loss.item()} --> MAIN Loss: {(lam_main * main_loss).item()} (Weight: {lam_main}), Navier-Stokes Loss: {(lam_ns * ns_loss).item()} (Weight: {lam_ns}), Laplacian Regularization Loss: {(lam_laplacian*laplacian_loss).item()} (Weight: {lam_laplacian})")
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss}')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    model.eval()
    valid_loss = 0
    print('Processing validation data...')
    with torch.no_grad():
        for val_batch_idx, batch in enumerate(valid_loader):
            batch = batch.to(DEVICE)
            out = model(batch)
            l2_loss = criterion(out[:,:3], batch.y[:,:3])
            main_loss = GAMMA*l2_loss
            ns_loss = ALPHA*navier_stokes_loss(batch)
            laplacian_loss = LAPLACIAN_REG_WEIGHT*laplacian_regularization(batch)
            lam_main = args['lam0']
            lam_ns = args['lam1']
            lam_laplacian = args['lam2']
            total_loss = lam_main * main_loss + lam_ns * ns_loss + lam_laplacian *laplacian_loss
            valid_loss += total_loss.item()
            if val_batch_idx % PRINT_INTERVAL == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(train_loader)},AlphaDIFFUSION: {model.alpha.item():.4f},Training Loss: {total_loss.item()} --> MAIN Loss: {(lam_main * main_loss).item()} (Weight: {lam_main}), Navier-Stokes Loss: {(lam_ns * ns_loss).item()} (Weight: {lam_ns}), Laplacian Regularization Loss: {(lam_laplacian*laplacian_loss).item()} (Weight: {lam_laplacian})")
    valid_loss /= len(valid_loader)
    val_losses.append(valid_loss)
    scheduler.step(valid_loss)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')
    save_checkpoint(checkpoint, SAVE_CHECKPOINT_FILE, epoch)

plot_losses(train_losses, val_losses)
