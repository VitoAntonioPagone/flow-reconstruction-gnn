import torch
import matplotlib.pyplot as plt
from datasets import FlowDataset
from models import ConvAutoEncoder
from utils import load_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = 'flow_reconstruction/network/trained_models/my_checkpoint.pth.tar'
TEST_INPUTS_DIR = 'flow_reconstruction/dataset/train_data/test_inputs_50/'
TEST_LABELS_DIR = 'flow_reconstruction/dataset/train_data/test_labels_50/'
TEST_FILE_NUM = 5


def reconstruct_flow(model, input_tensor, missing_mask_tensor):
    with torch.no_grad():
        # Concatenate the input tensor and the single-channel missing mask tensor
        input_with_mask = torch.cat((input_tensor, missing_mask_tensor), dim=1)
        output_tensor = model(input_with_mask)
        reconstructed_flow = output_tensor * missing_mask_tensor + input_tensor * (1 - missing_mask_tensor)
    return reconstructed_flow


def predict():
    # Get test data set
    test_dataset = FlowDataset(TEST_INPUTS_DIR, TEST_LABELS_DIR)
    # Get a sample input tensor, ground truth label tensor, and its single-channel missing mask tensor
    test_input_tensor, test_label_tensor, test_missing_mask_tensor = test_dataset[TEST_FILE_NUM]
    test_input_tensor = test_input_tensor.to(DEVICE)
    test_label_tensor = test_label_tensor.to(DEVICE)
    test_missing_mask_tensor = test_missing_mask_tensor.to(DEVICE)
    # Add the batch dimension
    test_input_tensor = test_input_tensor.unsqueeze(0)
    test_label_tensor = test_label_tensor.unsqueeze(0)
    test_missing_mask_tensor = test_missing_mask_tensor.unsqueeze(0)
    # Load pre-trained model
    model = ConvAutoEncoder().to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=torch.device('cpu')), model)  
    reconstructed_flow_tensor = reconstruct_flow(model, test_input_tensor, test_missing_mask_tensor)
    reconstructed_flow_tensor = reconstructed_flow_tensor.cpu()  # Move the tensor back to CPU for visualization

    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=100)  
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  

    for i in range(4):
        # Plot the input tensor
        im1 = axes[0, i].imshow(test_input_tensor[0, i], cmap='jet', aspect='auto')
        axes[0, i].set_title(f"Input Tensor - Channel {i + 1}", fontsize=10)
        cbar1 = fig.colorbar(im1, ax=axes[0, i], shrink=0.6)
        cbar1.ax.tick_params(labelsize=8)

        # Plot the ground truth label tensor
        im2 = axes[1, i].imshow(test_label_tensor[0, i], cmap='jet', aspect='auto')
        axes[1, i].set_title(f"Ground Truth Label Tensor - Channel {i + 1}", fontsize=10)
        cbar2 = fig.colorbar(im2, ax=axes[1, i], shrink=0.6)
        cbar2.ax.tick_params(labelsize=8)

        # Plot the reconstructed flow tensor
        im3 = axes[2, i].imshow(reconstructed_flow_tensor[0, i], cmap='jet', aspect='auto')
        axes[2, i].set_title(f"Reconstructed Flow Tensor - Channel {i + 1}", fontsize=10)
        cbar3 = fig.colorbar(im3, ax=axes[2, i], shrink=0.6)
        cbar3.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=2)  
    plt.show()

if __name__ == "__main__":
    predict()