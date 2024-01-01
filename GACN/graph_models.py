import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import get_laplacian

class GAT_98_8_SkipConnections(torch.nn.Module):
    def __init__(self):
        super(GAT_98_8_SkipConnections, self).__init__()
        self.feat_dim = 8
        self.output_dim = 8
        self.num_heads = 1  # Use single head
        self.alpha = torch.nn.Parameter(torch.tensor(0.25))

        # Layers with direct numbers instead of division
        self.conv1 = GATConv(self.feat_dim, 8, heads=self.num_heads)  # 16/2 = 8
        self.conv2 = GATConv(8, 16, heads=self.num_heads)             # 32/2 = 16
        self.conv3 = GATConv(16, 32, heads=self.num_heads)            # 64/2 = 32
        self.conv4 = GATConv(32, 64, heads=self.num_heads)            # 128/2 = 64
        self.conv5 = GATConv(64, 128, heads=self.num_heads)           # 256/2 = 128
        self.conv6 = GATConv(128, 256, heads=self.num_heads)          # 512/2 = 256
        self.conv7 = GATConv(256, 256, heads=self.num_heads)          # 512/2 = 256
        self.conv8 = GATConv(256, self.output_dim, heads=self.num_heads, concat=False)
        
        # Projection layers for skip connections with direct numbers
        self.proj1_to_3 = torch.nn.Linear(8, 16)   # 16/2 = 8, 32/2 = 16
        self.proj3_to_5 = torch.nn.Linear(32, 64)  # 64/2 = 32, 128/2 = 64
        self.proj5_to_7 = torch.nn.Linear(128, 256) # 256/2 = 128, 512/2 = 256
    def diffuse_with_laplacian(self, x, edge_index):
        laplacian_indices, laplacian_values = get_laplacian(edge_index, normalization=None)
        num_nodes = x.size(0)
        laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))
        
        return x + self.alpha * torch.sparse.mm(laplacian, x - x.mean(dim=0))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.conv2(x1, edge_index))
        
        # Project x1 to match x2's dimensions and then add
        x3 = torch.relu(self.conv3(x2 + self.proj1_to_3(x1), edge_index))
        
        x4 = torch.relu(self.conv4(x3, edge_index))
        
        # Project x3 to match x4's dimensions and then add
        x5 = torch.relu(self.conv5(x4 + self.proj3_to_5(x3), edge_index))
        
        x6 = torch.relu(self.conv6(x5, edge_index))
        
        # Project x5 to match x6's dimensions and then add
        x7 = torch.relu(self.conv7(x6 + self.proj5_to_7(x5), edge_index))
        
        x8 = self.conv8(x7, edge_index)  # Last layer without ReLU

        # Diffuse features with Laplacian
        x8_diffused = self.diffuse_with_laplacian(x8, edge_index)

        return x8_diffused
 