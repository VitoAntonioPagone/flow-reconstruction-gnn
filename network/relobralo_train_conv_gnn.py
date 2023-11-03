import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from models import fluid_GAT_98_8_SkipConnections,GAT_98_8_SkipConnections, GAT_98_10, red_GAT_98_5,red_GAT_98_6,GAT_98_2, GAT_98_3, GAT_98_6,GAT_98_4, GAT_50, GCN_98_10,GraphSAGE_98_10,GCN_95_8, GraphSAGE_95_8, GCN_90_6_Double, GAT_98_8,GAT_98_10, GAT_95_12, GAT_95_10,GAT_95_8,GAT_90_6_Double,GATv2_90_8_2heads,GATv2_90_8,GraphSAGE_90_8, GAT_90_8_Increased, GAT_90_8,GCN_90_6, GraphSAGE_90,GraphSAGE_90_6_Double, GraphSAGE_95, GAT_90, GraphSAGE_99, GCN_90, GAT_90_3, GAT_90_3_2heads, GAT_90_6, GAT_90_6_2heads
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
    """
    Configuration Parameters for Graph-Based Model Training

    This section defines various configuration parameters required for training and evaluation of a graph-based model. 
    These parameters may vary based on the dataset, model architecture, and training requirements.

    Parameters:
        HOLE (externally defined): Indicates whether the training is being done with a hole in the data.
        GAMMA: Weighting factor for the main L2 loss component.
        ALPHA: Weighting factor for the Navier-Stokes loss component.
        LAPLACIAN_REG_WEIGHT: Weighting factor for the Laplacian regularization loss component.
        BATCH_SIZE: The number of samples that will be propagated through the network simultaneously.
        LR: Learning rate for the optimizer.
        EPOCHS: Number of times the entire dataset is passed through the network.
        PERCENTAGE_OF_MISSING_POINTS: Percentage of data points that are missing from the training dataset.
        DEVICE: The device on which the model will run, e.g., 'cuda' for GPU or 'cpu' for CPU.
        LOAD_MODEL: Boolean flag to determine whether to load a pre-trained model for further training or evaluation.
        MODEL_NAME: A name assigned to the model, used mainly for saving and retrieving checkpoints.
        LOAD_CHECKPOINT_FILE: Path to the file from which a pre-trained model checkpoint should be loaded.
        SAVE_CHECKPOINT_FILE: Path to the file where the model checkpoint will be saved.
        LOSS_PLOT_DIR: Directory path for saving the plot that displays the training and validation loss.
        TRAIN_INPUT_DIR: Directory path for the input graphs used during training.
        TRAIN_TARGET_DIR: Directory path for the target graphs used during training.
        VALID_INPUT_DIR: Directory path for the input graphs used during validation.
        VALID_TARGET_DIR: Directory path for the target graphs used during validation.
    """
    GAMMA = 10
    ALPHA = 1e-7
    LAPLACIAN_REG_WEIGHT = 1e-5     
    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 50
    PERCENTAGE_OF_MISSING_POINTS = 98
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = False  
    MODEL_NAME = f"FLUID_skip_GAT_8_{PERCENTAGE_OF_MISSING_POINTS}"  
    LOAD_CHECKPOINT_FILE = f'../trained_models_FP/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    SAVE_CHECKPOINT_FILE = f'../trained_models_FP/{MODEL_NAME}_epochs_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.pth.tar'
    LOSS_PLOT_DIR = f'../losses_plot/{MODEL_NAME}_losses_plot_{EPOCHS}_lr_{LR}_batch_{BATCH_SIZE}.jpg'
    TRAIN_INPUT_DIR = f'../dataset_graph/training_FP_10/train_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    TRAIN_TARGET_DIR = f'../dataset_graph/training_FP_10/train_graphs_{PERCENTAGE_OF_MISSING_POINTS}/' 
    VALID_INPUT_DIR = f'../dataset_graph/training_FP_10/validation_input_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'  
    VALID_TARGET_DIR = f'../dataset_graph/training_FP_10/validation_graphs_{PERCENTAGE_OF_MISSING_POINTS}/'
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
    """
    Computes the weighted total loss using dynamic weights based on the provided forward and backward losses.

    Parameters:
    - model: The model on which the loss is being computed (though it's not used in this snippet).
    - f_loss: The forward loss.
    - b_losses: A list of backward losses.
    - args: A dictionary containing arguments. Expected keys:
        - 'T': A temperature parameter for the softmax computation.
        - 'lamX': The initial weight values, where X is an index (e.g., 'lam0', 'lam1', etc.).
    
    Returns:
    - total_loss: The computed weighted total loss.
    - f_loss: The forward loss (unchanged).
    - b_losses: A list of backward losses (unchanged).
    - args: An updated dictionary with computed dynamic weights and losses.

    Notes:
    The function uses the provided losses to compute dynamic weights via a softmax mechanism. The resulting dynamic weights
    are then used to calculate the weighted total loss. The weights are also stored in the returned args dictionary.
    """
    
    # Combine forward loss and backward losses into a single list for processing
    losses = [f_loss] + b_losses

    # Compute dynamic weights using softmax, with a temperature parameter to control sharpness
    lambs_hat = (F.softmax(torch.tensor([losses[i] / (args['T'] + 1e-12) for i in range(len(losses))]), dim=0) * len(losses)).detach()

    # Ensure the computed dynamic weights are at least as large as the provided initial values
    for i in range(len(lambs_hat)):
        lambs_hat[i] = max(lambs_hat[i], args['lam'+str(i)])

    # Calculate the weighted total loss using the dynamic weights
    total_loss = sum([lambs_hat[i] * losses[i] for i in range(len(losses))])

    # Update args dictionary to include the computed dynamic weights and the individual losses
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
    """
    Computes the Laplacian regularization term for the given graph.

    Parameters:
    - graph: An object representing the graph. Expected to have attributes:
        - x: A tensor representing node features.
        - edge_index: A tensor representing edge indices.
        - num_nodes: An integer indicating the number of nodes in the graph.
    
    Returns:
    - regularization: A scalar tensor representing the Laplacian regularization term.

    Notes:
    The function extracts the first three features of each node from the graph, then computes the Laplacian matrix.
    The regularization term is derived from the product of the node features and the Laplacian matrix. This term is
    useful in various graph-related learning tasks to promote smoothness of node features.
    """

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
PRINT_INTERVAL = 100  # adjust this value to print every n batches

if LOAD_MODEL and os.path.isfile(LOAD_CHECKPOINT_FILE):
    print('Loading checkpoint...')
    checkpoint = torch.load(LOAD_CHECKPOINT_FILE, map_location=DEVICE)
    load_checkpoint(checkpoint, model, optimizer)
    print("Checkpoint loaded successfully.")

"""
This script performs training and validation for a given number of epochs on a graph-based model.
For each epoch, the model is trained using a combination of three loss functions: 
- A main L2 loss
- A Navier-Stokes loss
- A Laplacian regularization loss

The weights for these losses are dynamically computed using the `relobralo` function. 
During training, the loss values and their corresponding weights are printed periodically for insight.

At the end of each epoch:
- The model's state is saved as a checkpoint.
- The model is evaluated on a validation set, using the same loss components.

Required external variables (not defined in this code snippet):
- EPOCHS: The number of epochs for training.
- optimizer: The optimization algorithm used for training.
- DEVICE: The device on which tensors are processed, e.g., 'cuda' for GPU or 'cpu' for CPU.
- train_loader: Data loader providing batches for training.
- valid_loader: Data loader providing batches for validation.
- model: The graph-based model being trained.
- GAMMA, ALPHA, LAPLACIAN_REG_WEIGHT: Constants specifying the initial weights for the main, Navier-Stokes, and Laplacian losses, respectively.
- criterion: The loss function for computing the L2 loss.
- navier_stokes_loss: A function that computes the Navier-Stokes loss for a given batch.
- laplacian_regularization: A function that computes the Laplacian regularization for a given batch.
- PRINT_INTERVAL: Specifies how often loss values should be printed during training and validation.
- save_checkpoint: A function to save the model and optimizer states.
- scheduler: Learning rate scheduler which adjusts the learning rate based on validation loss.
- args: A dictionary containing arguments for the `relobralo` function.
"""

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
        train_loss += total_loss.item()
        # Backpropagate and update the model parameters
        total_loss.backward()
        optimizer.step()
       
        # Printing individual loss components
        if batch_idx % PRINT_INTERVAL == 0:   # Only print every PRINT_INTERVAL batches
            print(f"  Batch {batch_idx + 1}/{len(train_loader)},,AlphaDIFFUSION: {model.alpha.item():.4f},Training Loss: {total_loss.item()} --> MAIN Loss: {(lam_main * main_loss).item()} (Weight: {lam_main}), Navier-Stokes Loss: {(lam_ns * ns_loss).item()} (Weight: {lam_ns}), Laplacian Regularization Loss: {(lam_laplacian*laplacian_loss).item()} (Weight: {lam_laplacian})")
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
                print(f"Validation Batch {batch_idx + 1}/{len(train_loader)},AlphaDIFFUSION: {model.alpha.item():.4f},Training Loss: {total_loss.item()} --> MAIN Loss: {(lam_main * main_loss).item()} (Weight: {lam_main}), Navier-Stokes Loss: {(lam_ns * ns_loss).item()} (Weight: {lam_ns}), Laplacian Regularization Loss: {(lam_laplacian*laplacian_loss).item()} (Weight: {lam_laplacian})")

    valid_loss /= len(valid_loader)
    val_losses.append(valid_loss)
    scheduler.step(valid_loss)
    print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss}')


plot_losses(train_losses, val_losses)
