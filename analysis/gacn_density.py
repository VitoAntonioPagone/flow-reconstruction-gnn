import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset as TorchDataset
import random
from network.models import GAT_98_8_SkipConnections

def calculate_velocity_magnitude(velocities):
    print("Calculating velocity magnitude...")
    return torch.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)

def plot_predictions_with_density(x, y, title, ax, sample_fraction=0.1):
    print("Plotting predictions with density...")

    num_points = int(len(x) * sample_fraction)

    sampled_indices = random.sample(range(len(x)), num_points)

    x_sampled = x[sampled_indices]
    y_sampled = y[sampled_indices]

    xy_sampled = np.vstack([x_sampled, y_sampled])
    z = gaussian_kde(xy_sampled)(xy_sampled)

    scatter = ax.scatter(x_sampled, y_sampled, c=z, s=50)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], '--', lw=2, color='blue', label="Perfect Prediction")
    ax.set_xlabel("DNS")
    ax.set_ylabel("GACN Predictions")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.figure.colorbar(scatter, ax=ax, label='Density')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
CHECKPOINT_PATH = '../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar'

class CustomDataset(TorchDataset):
    def __init__(self, input_files, label_files):
        print("Initializing dataset...")
        self.input_files = input_files
        self.label_files = label_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        print(f"Loading data for index {idx}...")
        input_data = torch.load(self.input_files[idx])
        label_data = torch.load(self.label_files[idx])
        input_data.y = label_data.x
        input_data.x_complete = label_data.x
        return input_data

def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

def run_GCN(input_file, label_file):
    print("Running GCN...")
    test_dataset = CustomDataset([input_file], [label_file])
    single_graph = test_dataset[0]

    model = GAT_98_8_SkipConnections()
    model.to(DEVICE)
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    print("Model loaded and set to evaluation mode.")

    single_graph.x = single_graph.x.to(DEVICE)
    single_graph.edge_index = single_graph.edge_index.to(DEVICE)
    single_graph.y = single_graph.y.to(DEVICE)

    print("Data moved to device.")

    with torch.no_grad():
        print("Performing prediction...")
        out = model(single_graph)

    predicted_velocity_magnitude = calculate_velocity_magnitude(out.cpu())* 7.035423
    target_velocity_magnitude = calculate_velocity_magnitude(single_graph.y.cpu())* 7.035423

    print("Prediction complete. Plotting results...")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_predictions_with_density(target_velocity_magnitude.numpy(), predicted_velocity_magnitude.numpy(), "GAT Predictions", ax, sample_fraction=1)
    plt.tight_layout()

    plt.savefig('gat_predictions_density_plot.png', dpi=1200)
    print("Plot saved as gat_predictions_density_plot.png")

    plt.show()

if __name__ == "__main__":
    label_file = '../dataset_graph_full/training_FP/test_graphs_98/cyc11_CAD660_Y7_Z0_X0_label.pt'
    input_file = '../dataset_graph_full/training_FP/test_input_graphs_98/cyc11_CAD660_Y7_Z0_X0_input.pt'
    print("Starting process...")
    run_GCN(input_file, label_file)
