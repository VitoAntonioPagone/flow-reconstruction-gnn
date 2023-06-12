from torch_geometric.loader import DataLoader
from datasets import CustomDataset
from utils import get_loaders_graphs 
TRAIN_INPUT_DIR = '../dataset_graph/training/train_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'

VALID_INPUT_DIR = '../dataset_graph/training/validation_input_graphs'  # Replace with your validation input directory
VALID_TARGET_DIR = '../dataset_graph/training/validation_graphs'  # Replace with your validation target directory


# Create DataLoaders for training and validation data
train_loader = DataLoader(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, batch_size=32)
valid_loader = DataLoader(VALID_INPUT_DIR, VALID_TARGET_DIR, batch_size=32)

# Print the type of the data loaders
print(f'Type of train_loader: {type(train_loader)}')
print(f'Type of valid_loader: {type(valid_loader)}')

