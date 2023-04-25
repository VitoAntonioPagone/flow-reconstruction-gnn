import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import glob
import re
from network.dataset import FlowDataset
from network.models import ConvAutoEncoder
from network.losses import masked_mse_loss
from tqdm import tqdm


BATCH_SIZE = 8
LEARNING_RATE = 0.001
SHUFFLE = True
NUM_EPOCHS = 2
TRAIN_FOLDER = 'dataset/train_npz_50'
LABEL_FOLDER = 'dataset/train_labels_npz_50'
TEST_FOLDER = 'dataset/test_npz_50'
TEST_LABEL_FOLDER = 'dataset/test_labels_npz_-50'


def train_fn(loader, model, optimizer, loss_fn, scaler):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    label_files = glob.glob(LABEL_FOLDER + '/*_label.npy')
    train_files = [re.sub('_label.npy$', '_train.npy', f).replace(LABEL_FOLDER, TRAIN_FOLDER) for f in label_files]

    train_files.sort()
    label_files.sort()

    # Create dataset and data loader
    dataset = FlowDataset(train_files, label_files)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # Initialize the model, optimizer, and training parameters
    model = ConvAutoEncoder().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()