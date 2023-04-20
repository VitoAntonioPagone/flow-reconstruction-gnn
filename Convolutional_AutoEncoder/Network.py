import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import re
from FlowDataset import FlowDataset
from ConvAutoEncoder import ConvAutoencoder
from tqdm import tqdm

# Define the custom loss function
def masked_mse_loss(input, target, mask):
    diff = input - target
    masked_diff = diff * mask
    loss = torch.mean(masked_diff ** 2)
    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")

# Load the data
train_folder = 'Dataset/train/train-50'
label_folder = 'Dataset/train/train-labels-50'

test_folder = 'Dataset/test/test-50'
test_label_folder = 'Dataset/test/test-labels-50'


label_files = glob.glob(label_folder + '/*_label.npy')
train_files = [re.sub('_label.npy$', '_train.npy', f).replace(label_folder, train_folder) for f in label_files]


train_files.sort()
label_files.sort()

# Create dataset and data loader
dataset = FlowDataset(train_files, label_files)
batch_size = 8
shuffle = True
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Initialize the model, optimizer, and training parameters
model = ConvAutoencoder().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 2

# Training loop
for epoch in range(num_epochs):
    for train, labels, missing_mask in data_loader:
        train, labels, missing_mask = train.to(device), labels.to(device), missing_mask.to(device)
        train_with_mask = torch.cat((train, missing_mask), dim=1)  # Concatenate the train tensor and the single-channel missing mask tensor
        outputs = model(train_with_mask)
        loss = masked_mse_loss(outputs, labels, missing_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Function to reconstruct the flow field
def reconstruct_flow(model, input_tensor, missing_mask_tensor):
    with torch.no_grad():
        input_with_mask = torch.cat((input_tensor, missing_mask_tensor), dim=1)  # Concatenate the input tensor and the single-channel missing mask tensor
        output_tensor = model(input_with_mask)
        reconstructed_flow = output_tensor * missing_mask_tensor + input_tensor * (1 - missing_mask_tensor)
    return reconstructed_flow

# Example usage TYUIO
test_input_tensor, test_label_tensor, test_missing_mask_tensor = dataset[0]  # Get a sample input tensor, ground truth label tensor, and its single-channel missing mask tensor
test_input_tensor, test_label_tensor, test_missing_mask_tensor = test_input_tensor.to(device), test_label_tensor.to(device), test_missing_mask_tensor.to(device)
test_input_tensor = test_input_tensor.unsqueeze(0)  # Add the batch dimension
test_label_tensor = test_label_tensor.unsqueeze(0)  # Add the batch dimension
test_missing_mask_tensor = test_missing_mask_tensor.unsqueeze(0)  # Add the batch dimension
reconstructed_flow_tensor = reconstruct_flow(model, test_input_tensor, test_missing_mask_tensor)
reconstructed_flow_tensor = reconstructed_flow_tensor.cpu()  # Move the tensor back to CPU for visualization

# Plot the original input tensor, ground truth label tensor, and the reconstructed flow tensor
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for i in range(5):
    # Plot the input tensor
    im1 = axes[0, i].imshow(test_input_tensor[0, i], cmap='jet', aspect='auto')
    axes[0, i].set_title(f"Input Tensor - Channel {i + 1}")
    fig.colorbar(im1, ax=axes[0, i])

    # Plot the ground truth label tensor
    im2 = axes[1, i].imshow(test_label_tensor[0, i], cmap='jet', aspect='auto')
    axes[1, i].set_title(f"Ground Truth Label Tensor - Channel {i + 1}")
    fig.colorbar(im2, ax=axes[1, i])

    # Plot the reconstructed flow tensor
    im3 = axes[2, i].imshow(reconstructed_flow_tensor[0, i], cmap='jet', aspect='auto')
    axes[2, i].set_title(f"Reconstructed Flow Tensor - Channel {i + 1}")
    fig.colorbar(im3, ax=axes[2, i])

plt.show()