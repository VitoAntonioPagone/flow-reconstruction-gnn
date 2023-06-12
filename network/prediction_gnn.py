import torch
from datasets import CustomDataset
from torch_geometric.data import DataListLoader
from models import GCN
from collections import OrderedDict

# Hyperparameters
FEAT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_DIM = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path
TEST_INPUT_DIR = '../dataset_graph/trainint/test_input_graphs'
TEST_TARGET_DIR = '../dataset_graph/training/test_graphs'
CHECKPOINT_PATH = 'my_checkpoint.pth.tar' 

test_dataset = CustomDataset(TEST_INPUT_DIR, TEST_TARGET_DIR)

model = GCN(feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
model.to(DEVICE)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' from the key
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

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
