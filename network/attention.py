import torch.nn as nn
import torch
import torch.nn.functional as F
# Squeeze-and-Excitation Layer (SELayer): adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels


# Spatial Attention (SA): focuses on learning spatial dependencies and computes a spatial attention map by pooling channel information

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)




