import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import GAT_98_8_SkipConnections, GAT_98_10, red_GAT_98_5,red_GAT_98_6,GAT_98_2, GAT_98_3, GAT_98_6,GAT_98_4, GAT_50, GCN_98_10,GraphSAGE_98_10,GCN_95_8, GraphSAGE_95_8, GCN_90_6_Double,GAT_98_8_Modified, GAT_98_8,GAT_98_10, GAT_95_12, GAT_95_10,GAT_95_8,GAT_90_6_Double,GATv2_90_8_2heads,GATv2_90_8,GraphSAGE_90_8, GAT_90_8_Increased, GAT_90_8,GCN_90_6, GraphSAGE_90,GraphSAGE_90_6_Double, GraphSAGE_95, GAT_90, GraphSAGE_99, GCN_90, GAT_90_3, GAT_90_3_2heads, GAT_90_6, GAT_90_6_2heads
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from losses import GraphNavierStokesLoss
from utils import graph_initialize_weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import get_laplacian

HOLE = False
USE_LINF_LOSS = False  # set this to False to use L2 loss
use_nmse = False


if not HOLE:
    GAMMA = 1
    ALPHA = 1e-7
    LAPLACIAN_REG_WEIGHT = 1e-4     
    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 50
    PERCENTAGE_OF_MISSING_POINTS = 98
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = False  
    MODEL_NAME = f"skip_GAT_8_{PERCENTAGE_OF_MISSING_POINTS}"  
    LOAD_CHECKPOINT_FILE = f'../trained_models_FP/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models_FP/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'
    TRAIN_INPUT_DIR = f'../dataset_graph/training_FP/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    TRAIN_TARGET_DIR = f'../dataset_graph/training_FP/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}/' 
    VALID_INPUT_DIR = f'../dataset_graph/training_FP/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    VALID_TARGET_DIR = f'../dataset_graph/training_FP/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'
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
'''
# Hyperparameters
ALPHA = 1e-7  
BATCH_SIZE = 1
LR = 0.00001
EPOCHS = 50
PERCENTAGE_OF_MISSING_POINTS = 98
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False  
MODEL_NAME = f"INCR_UNIV_NS_GAT_8'_{PERCENTAGE_OF_MISSING_POINTS}"  
LOAD_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
SAVE_CHECKPOINT_FILE = f'../trained_models/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'

print(f'Starting script with Device: {DEVICE}')

TRAIN_INPUT_DIR = f'../dataset_graph/training/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  
TRAIN_TARGET_DIR = f'../dataset_graph/training/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}' 
VALID_INPUT_DIR = f'../dataset_graph/training/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}'  
VALID_TARGET_DIR = f'../dataset_graph/training/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}'
'''
print(f'Starting script with Device: {DEVICE}')

def relobralo(model, f_loss, b_losses, args:dict):
    T = args['T']
    losses = [f_loss] + b_losses

    # Compute dynamic weights
    lambs_hat = (F.softmax(torch.tensor([losses[i] / (T + 1e-12) for i in range(len(losses))]), dim=0) * len(losses)).detach()

    # Ensure the dynamic weights are at least the initial values
    for i in range(len(lambs_hat)):
        lambs_hat[i] = max(lambs_hat[i], args['lam'+str(i)])

    # Compute the weighted total loss using the dynamic weights
    total_loss = sum([lambs_hat[i] * losses[i] for i in range(len(losses))])
    
    # Update args to store the computed dynamic weights and losses
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
    features = graph.x[:, :3]  # Select only the first three features
    laplacian_indices, laplacian_values = get_laplacian(graph.edge_index, normalization=None)

    # Create Laplacian matrix
    num_nodes = graph.num_nodes
    laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))

    # Compute L * X
    LX = torch.sparse.mm(laplacian, features)
    
    # Compute X^T * (L * X)
    regularization_matrix = torch.mm(features.transpose(0, 1), LX)
    
    regularization = torch.trace(regularization_matrix)
    
    return regularization

def loss_nmse(pred, target, epsilon=1e-5):
    diff_up = torch.norm(pred - target, p=2)    # ||\hat{p} - p||_2
    norm_target = torch.norm(target, p=2) + epsilon    # ||p||_2 + \epsilon
    return diff_up / norm_target 

def Linfinity_loss(pred, target):
    return (pred - target).abs().max()

def save_checkpoint(state, filename=SAVE_CHECKPOINT_FILE):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def plot_losses(train_losses, val_losses):
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
graph_initialize_weights(model)  # Initialize weights of the model
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
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = MSELoss()
navier_stokes_loss = GraphNavierStokesLoss().to(DEVICE)

train_losses, val_losses = [], []
PRINT_INTERVAL = 1  # adjust this value to print every n batches

if LOAD_MODEL and os.path.isfile(LOAD_CHECKPOINT_FILE):
    print('Loading checkpoint...')
    checkpoint = torch.load(LOAD_CHECKPOINT_FILE, map_location=DEVICE)
    load_checkpoint(checkpoint, model, optimizer)
    print("Checkpoint loaded successfully.")

for epoch in range(EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch: {epoch+1}, Learning Rate: {current_lr}')
    model.train()
    train_loss = 0
    print('Processing training data...')
    for batch_idx, batch in enumerate(train_loader):
        batch.x = batch.x.to(DEVICE)  
        batch.edge_index = batch.edge_index.to(DEVICE)
        batch.y = batch.y.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        # Compute the three losses
        l2_loss = criterion(out[:,:3], batch.y[:,:3])

        main_loss = GAMMA*l2_loss
        ns_loss = ALPHA*navier_stokes_loss(batch)
        laplacian_loss = LAPLACIAN_REG_WEIGHT*laplacian_regularization(batch)

        _, _, _, updated_args = relobralo(model, main_loss, [ns_loss, laplacian_loss], args)
        args.update(updated_args)  # Update the args dictionary with the new values

        # Compute the weighted total loss using the updated weights
        lam_main = args['lam0']
        lam_ns = args['lam1']
        lam_laplacian = args['lam2']
        total_loss = lam_main *  main_loss + lam_ns *  ns_loss + lam_laplacian * laplacian_loss

        # Backpropagate and update the model parameters
        total_loss.backward()
        optimizer.step()
       
        # Printing individual loss components
        if batch_idx % PRINT_INTERVAL == 0:   # Only print every PRINT_INTERVAL batches
            print(f"  Batch {batch_idx + 1}/{len(train_loader)},Training Loss: {total_loss.item()} --> MAIN Loss: {(lam_main * main_loss).item()} (Weight: {lam_main}), Navier-Stokes Loss: {(lam_ns * ns_loss).item()} (Weight: {lam_ns}), Laplacian Regularization Loss: {(lam_laplacian*laplacian_loss).item()} (Weight: {lam_laplacian})")
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss}')

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    model.eval()
    valid_loss = 0
    print('Processing validation data...')
    with torch.no_grad():
        for batch in valid_loader:
            batch.x = batch.x.to(DEVICE)  
            batch.edge_index = batch.edge_index.to(DEVICE)
            batch.y = batch.y.to(DEVICE)
            out = model(batch)
            l2_loss = criterion(out[:,:3], batch.y[:,:3])
            main_loss = GAMMA*l2_loss
            ns_loss = ALPHA*navier_stokes_loss(batch)
            laplacian_loss = LAPLACIAN_REG_WEIGHT*laplacian_regularization(batch)

            # Compute the weighted total loss using the updated weights from training
            lam_main = args['lam0']
            lam_ns = args['lam1']
            lam_laplacian = args['lam2']
            total_loss = lam_main * GAMMA* main_loss + lam_ns * ALPHA* ns_loss + lam_laplacian * LAPLACIAN_REG_WEIGHT*laplacian_loss

            valid_loss += total_loss.item()
            if batch_idx % PRINT_INTERVAL == 0:   # Only print every PRINT_INTERVAL batches
                print(f"Validation Batch {batch_idx + 1}/{len(train_loader)},Training Loss: {total_loss.item()} --> MAIN Loss: {(lam_main * main_loss).item()} (Weight: {lam_main}), Navier-Stokes Loss: {(lam_ns * ns_loss).item()} (Weight: {lam_ns}), Laplacian Regularization Loss: {(lam_laplacian*laplacian_loss).item()} (Weight: {lam_laplacian})")

    valid_loss /= len(valid_loader)
    val_losses.append(valid_loss)
    scheduler.step(valid_loss)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')


plot_losses(train_losses, val_losses)
