import torch
from datasets import CustomDataset
from torch_geometric.data import DataListLoader
from models import GCN
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
FEAT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_DIM = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path
TEST_INPUT_DIR = '../dataset_graph/training/test_input_graphs/graph_48_input.pt'
TEST_TARGET_DIR = '../dataset_graph/training/test_graphs/graph_48_label.pt'
CHECKPOINT_PATH = '../trained_models/GCN_epochs_10_lr_0.01_batch_32.pth.tar' 

test_dataset = CustomDataset(TEST_INPUT_DIR, TEST_TARGET_DIR)

model = GCN(feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
model.to(DEVICE)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()

    # Filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Overwrite entries in the existing state dict
    model_dict.update(state_dict)

    # Load the new state dict
    model.load_state_dict(model_dict)

# Load trained weights
load_checkpoint(model, CHECKPOINT_PATH)

model.eval()

# Select one test graph
single_graph_input = test_dataset[0]

single_graph_input.x = single_graph_input.x.to(DEVICE)
single_graph_input.edge_index = single_graph_input.edge_index.to(DEVICE)
single_graph_input.y = single_graph_input.y.to(DEVICE)

# Perform prediction
with torch.no_grad():
    out = model(single_graph_input)

error = (out - single_graph_input.y).abs().mean()
print(f"Mean absolute error in predictions for test graph: {error.item()}")

fig, axs = plt.subplots(3, 3, figsize=(18, 18))  # 3 rows for 3 channels, 3 columns for input/output/target

# Assuming that the positions are the last 2 features in the feature vector
positions = single_graph_input.x.cpu()[:, -2:].numpy()

for i in range(3):  # iterate over channels
    input_values = single_graph_input.x.cpu()[:, i].numpy()
    output_values = out.cpu()[:, i].numpy()
    target_values = single_graph_input.y.cpu()[:, i].numpy()
    
    # Input grid
    sc = axs[i, 0].scatter(positions[:, 0], positions[:, 1], c=input_values, cmap='jet')
    plt.colorbar(sc, ax=axs[i, 0])
    axs[i, 0].set_title(f'Input Channel {i+1}')

    # Output grid
    sc = axs[i, 1].scatter(positions[:, 0], positions[:, 1], c=output_values, cmap='jet')
    plt.colorbar(sc, ax=axs[i, 1])
    axs[i, 1].set_title(f'Output Channel {i+1}')

    # Target grid
    sc = axs[i, 2].scatter(positions[:, 0], positions[:, 1], c=target_values, cmap='jet')
    plt.colorbar(sc, ax=axs[i, 2])
    axs[i, 2].set_title(f'Target Channel {i+1}')

plt.tight_layout()
plt.show()
