import torch


# Define the custom loss function
def masked_mse_loss(input, target, mask):
    """"Custom masked mse loss function"""
    diff = input - target
    masked_diff = diff * mask
    loss = torch.mean(masked_diff ** 2)
    return loss

