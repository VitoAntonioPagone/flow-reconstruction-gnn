import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Module, Linear, ReLU, Dropout
import torchvision.transforms.functional as TF
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import conv
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import MessagePassing

class ConvNet_90(nn.Module):
    """Convolutional network for 90% missing data."""
    
    def __init__(self):
        super(ConvNet_90, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32*2, 64*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64*2, 128*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128*2, 256*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384*2, 128*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192*2, 64*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96*2, 32*2, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 3, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        dec4 = self.decoder4(torch.cat((enc4, enc3), dim=1))
        dec3 = self.decoder3(torch.cat((dec4, enc2), dim=1))
        dec2 = self.decoder2(torch.cat((dec3, enc1), dim=1))
        dec1 = self.decoder1(dec2)
        return dec1
    
class ConvNet_98(nn.Module):
    """Convolutional network for 98% missing data."""
    
    def __init__(self):
        super(ConvNet_98, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32*4, 64*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64*4, 128*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128*4, 256*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384*4, 128*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192*4, 64*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96*4, 32*4, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32*4, 3, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        dec4 = self.decoder4(torch.cat((enc4, enc3), dim=1))
        dec3 = self.decoder3(torch.cat((dec4, enc2), dim=1))
        dec2 = self.decoder2(torch.cat((dec3, enc1), dim=1))
        dec1 = self.decoder1(dec2)
        return dec1


class GAT_98_8_SkipConnections(torch.nn.Module):
    """Graph Attention Network with skip connections."""
    
    def __init__(self):
        super(GAT_98_8_SkipConnections, self).__init__()
        self.feat_dim = 8
        self.output_dim = 8
        self.num_heads = 1
        self.alpha = torch.nn.Parameter(torch.tensor(0.25))
        self.conv1 = GATConv(self.feat_dim, 8, heads=self.num_heads)
        self.conv2 = GATConv(8, 16, heads=self.num_heads)
        self.conv3 = GATConv(16, 32, heads=self.num_heads)
        self.conv4 = GATConv(32, 64, heads=self.num_heads)
        self.conv5 = GATConv(64, 128, heads=self.num_heads)
        self.conv6 = GATConv(128, 256, heads=self.num_heads)
        self.conv7 = GATConv(256, 256, heads=self.num_heads)
        self.conv8 = GATConv(256, self.output_dim, heads=self.num_heads, concat=False)
        self.proj1_to_3 = torch.nn.Linear(8, 16)
        self.proj3_to_5 = torch.nn.Linear(32, 64)
        self.proj5_to_7 = torch.nn.Linear(128, 256)
        
    def diffuse_with_laplacian(self, x, edge_index):
        """Apply Laplacian diffusion to features."""
        laplacian_indices, laplacian_values = get_laplacian(edge_index, normalization=None)
        num_nodes = x.size(0)
        laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))
        return x + self.alpha * torch.sparse.mm(laplacian, x - x.mean(dim=0))
    
    def forward(self, data):
        """Forward pass through GAT network."""
        x, edge_index = data.x, data.edge_index
        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.conv2(x1, edge_index))
        x3 = torch.relu(self.conv3(x2 + self.proj1_to_3(x1), edge_index))
        x4 = torch.relu(self.conv4(x3, edge_index))
        x5 = torch.relu(self.conv5(x4 + self.proj3_to_5(x3), edge_index))
        x6 = torch.relu(self.conv6(x5, edge_index))
        x7 = torch.relu(self.conv7(x6 + self.proj5_to_7(x5), edge_index))
        x8 = self.conv8(x7, edge_index)
        x8_diffused = self.diffuse_with_laplacian(x8, edge_index)
        return x8_diffused
