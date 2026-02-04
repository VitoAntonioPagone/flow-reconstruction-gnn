import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset as TorchDataset
import json
import csv

MISSING_PERCENTAGE = 98

def calculate_additional_stats(values):
    """Calculate additional statistics for error values."""
    values = np.array(values)
    stats = {
        'min': np.min(values),
        'max': np.max(values),
        '1st_quantile': np.quantile(values, 0.25),
        'median': np.median(values),
        '3rd_quantile': np.quantile(values, 0.75)
    }
    return stats

def save_dict_to_file(d, file_name):
    """Save dictionary to JSON file."""
    with open(file_name, 'w') as f:
        json.dump(d, f, indent=4)

class FlowDataset(TorchDataset):
    """Dataset for flow field evaluation."""
    
    def __init__(self, input_files, label_files, add_mask=False):
        self.input_files = input_files
        self.label_files = label_files
        self.add_mask = add_mask

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_path = self.input_files[index]
        label_path = self.label_files[index]
        input_data = np.load(input_path)
        label_data = np.load(label_path)
        input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label_data).permute(2, 0, 1).float()
        missing_mask_tensor = (input_tensor[0] == 0).unsqueeze(0).float()
        if self.add_mask:
            input_tensor = torch.cat((input_tensor, missing_mask_tensor), dim=0)
        return input_tensor, label_tensor, missing_mask_tensor

def calculate_rmse(pred, target):
    """Calculate RMSE."""
    return torch.sqrt(((pred - target) ** 2).mean())

def calculate_mae(pred, target):
    """Calculate MAE."""
    return (torch.abs(pred - target)).mean()

def run_autoencoder(input_file, label_file):
    """Run autoencoder evaluation on single sample."""
    from models import ConvNet_98
    from utils import load_checkpoint
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_FILE = '../trained_models/ConvNet_98_epochs_100__alpha_0.1_beta_0.1_lr_0.0001_batch_32.pth.tar'

    def reconstruct_flow(model, input_tensor, missing_mask_tensor):
        with torch.no_grad():
            input_tensor = input_tensor[:, :3, :, :]
            input_with_mask = torch.cat((input_tensor, missing_mask_tensor), dim=1)
            output_tensor = model(input_with_mask)
            reconstructed_flow = output_tensor * missing_mask_tensor + input_tensor * (1 - missing_mask_tensor)
        return reconstructed_flow
    
    def velocity_magnitude(velocities):
        """Compute velocity magnitude from components."""
        return torch.sqrt(torch.sum(velocities ** 2, dim=0))
    
    def predict():
        test_dataset = FlowDataset([input_file], [label_file])
        test_input_tensor, test_label_tensor, test_missing_mask_tensor = test_dataset[0]
        test_input_tensor = test_input_tensor.to(DEVICE)
        test_label_tensor = test_label_tensor.to(DEVICE)
        test_missing_mask_tensor = test_missing_mask_tensor.to(DEVICE)
        test_input_tensor = test_input_tensor.unsqueeze(0)
        test_label_tensor = test_label_tensor.unsqueeze(0)
        test_missing_mask_tensor = test_missing_mask_tensor.unsqueeze(0)
        model = ConvNet_98().to(DEVICE)
        load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=torch.device(DEVICE)), model)
        reconstructed_flow_tensor = reconstruct_flow(model, test_input_tensor, test_missing_mask_tensor)
        rmses = []
        maes = []
        for i in range(2):  
            reconstructed_flow_tensor_denorm = reconstructed_flow_tensor[0, i] * 7.035423
            test_label_tensor_denorm = test_label_tensor[0, i] * 7.035423
            rmse = calculate_rmse(reconstructed_flow_tensor_denorm, test_label_tensor_denorm)
            mae = calculate_mae(reconstructed_flow_tensor_denorm, test_label_tensor_denorm)
            rmses.append(rmse.item())
            maes.append(mae.item())
        predicted_velocity_magnitude = velocity_magnitude(reconstructed_flow_tensor[0, :2] * 7.035423)
        ground_truth_velocity_magnitude = velocity_magnitude(test_label_tensor[0, :2] * 7.035423)
        rmse_velocity_magnitude = calculate_rmse(predicted_velocity_magnitude, ground_truth_velocity_magnitude).item()
        mae_velocity_magnitude = calculate_mae(predicted_velocity_magnitude, ground_truth_velocity_magnitude).item()
        rmses.append(rmse_velocity_magnitude)
        maes.append(mae_velocity_magnitude)
        return rmses, maes
    return predict()

if __name__ == "__main__":
    input_folder = f'../dataset/train_data_{MISSING_PERCENTAGE}/test_inputs_{MISSING_PERCENTAGE}/'
    label_folder = f'../dataset/train_data_{MISSING_PERCENTAGE}/test_labels_{MISSING_PERCENTAGE}/'
    all_rmses = []
    all_maes = []
    input_files = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
    label_files = sorted(glob.glob(os.path.join(label_folder, "*.npy")))
    channel_names = ['x-velocity', 'y-velocity']
    results = []
    for inp, lbl in zip(input_files, label_files):
        rmses, maes = run_autoencoder(inp, lbl)
        results.append(rmses + maes)  
        print(f"\nAnalyzing Files: {os.path.basename(inp)} and {os.path.basename(lbl)}")
        for i, channel in enumerate(channel_names + ['Velocity Magnitude']):
            print(f"{channel} RMSE: {rmses[i]}")
            print(f"{channel} MAE: {maes[i]}")
        all_rmses.append(rmses)
        all_maes.append(maes)
    print("\nFinal results to be written to CSV:")
    for result in results:
        print(result)
    with open('velocity_errors_cnn.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RMSE X Velocity', 'RMSE Y Velocity', 'RMSE Velocity Magnitude', 'MAE X Velocity', 'MAE Y Velocity', 'MAE Velocity Magnitude'])
        writer.writerows(results)
    print("RMSE and MAE values for each input slice saved to 'velocity_errors_cnn.csv'")
