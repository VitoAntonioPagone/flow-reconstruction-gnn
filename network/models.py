import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Module, Linear, ReLU, Dropout
import torchvision.transforms.functional as TF
from torch_geometric.nn import (GCNConv, SAGEConv, GATConv, GravNetConv, 
                                GINConv, PNAConv, ChebConv, AGNNConv, ARMAConv)
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import conv

#### 50 % MISSING POINTS #####

#### CONVOLUTIONAL NETWORKS ####

class ConvNet_50(nn.Module):
    def __init__(self):
        super(ConvNet_50, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(torch.cat((enc4, enc3), dim=1))  # Skip connection from encoder3
        dec3 = self.decoder3(torch.cat((dec4, enc2), dim=1))  # Skip connection from encoder2
        dec2 = self.decoder2(torch.cat((dec3, enc1), dim=1))  # Skip connection from encoder1
        dec1 = self.decoder1(dec2)

        return dec1
        
#### GRAPHS NETWORKS ####

class ChebNet_50(torch.nn.Module):
    def __init__(self):
        super(ChebNet_50, self).__init__()
        self.feat_dim = 6
        self.conv1 = ChebConv(self.feat_dim, 128, K=2)
        self.conv2 = ChebConv(128, 256, K=2)
        self.conv3 = ChebConv(256, 128, K=2)
        self.conv4 = ChebConv(128, self.feat_dim, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        return x


class GAT_50(torch.nn.Module):
    def __init__(self):
        super(GAT_50, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = GATConv(self.feat_dim, 128)
        self.conv2 = GATConv(128, 256)
        self.conv3 = GATConv(256, 128)
        self.conv4 = GATConv(128, self.feat_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        return x

class GCN_50(torch.nn.Module):
    def __init__(self):
        super(GCN_50, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = GCNConv(self.feat_dim, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 128)
        self.conv4 = GCNConv(128, self.feat_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        return x

class GCN_50_deep(torch.nn.Module):
    def __init__(self):
        super(GCN_50_deep, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = GCNConv(self.feat_dim, 256)  # increased to 256
        self.conv2 = GCNConv(256, 512)  # increased to 512
        self.conv3 = GCNConv(512, 1024)  # new layer, dimension 1024
        self.conv4 = GCNConv(1024, 512)  # new layer, dimension 512
        self.conv5 = GCNConv(512, 256)  # increased to 256
        self.conv6 = GCNConv(256, self.feat_dim)  # output layer, back to original dimensionality

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = torch.relu(x)
        x = self.conv6(x, edge_index)
        return x


class GraphSAGE_50(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_50, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = SAGEConv(self.feat_dim, 128)
        self.conv2 = SAGEConv(128, 256)
        self.conv3 = SAGEConv(256, 128)
        self.conv4 = SAGEConv(128, self.feat_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        return x

 #### 90% MISSING POINTS ####

 #### CONVOLUTIONAL NETWORKS ####

class ConvNet_90(nn.Module):
    def __init__(self):
        super(ConvNet_90, self).__init__()

        # Encoder
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

        # Decoder
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
        
        dec4 = self.decoder4(torch.cat((enc4, enc3), dim=1))  # Skip connection from encoder3


        dec3 = self.decoder3(torch.cat((dec4, enc2), dim=1))  # Skip connection from encoder2
        dec2 = self.decoder2(torch.cat((dec3, enc1), dim=1))  # Skip connection from encoder1
        dec1 = self.decoder1(dec2)

        return dec1



#### GRAPH NETWORKS ####


class GCN_90(torch.nn.Module):
    def __init__(self):
        super(GCN_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        # Replace GATConv with GCNConv for GCN
        self.conv1 = GCNConv(self.feat_dim, 16)  # Layer 1
        self.conv2 = GCNConv(16, 32)  # Layer 2
        self.conv3 = GCNConv(32, 64)  # Layer 3
        self.conv4 = GCNConv(64, 128)  # Layer 4 (peak dimensionality)
        self.conv5 = GCNConv(128, 64)  # Layer 5
        self.conv6 = GCNConv(64, 32)  # Layer 6
        self.conv7 = GCNConv(32, 16)  # Layer 7
        self.conv8 = GCNConv(16, 8)  # Layer 8
        self.conv9 = GCNConv(8, 4)  # Layer 9
        self.conv10 = GCNConv(4, self.output_dim)  # Layer 10 (Output)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x

class GCN_90_6(torch.nn.Module):
    def __init__(self):
        super(GCN_90_6, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GCNConv(self.feat_dim, 32)   # Layer 1
        self.conv2 = GCNConv(32, 64)              # Layer 2
        self.conv3 = GCNConv(64, 256)             # Layer 3 (peak dimensionality)
        self.conv4 = GCNConv(256, 64)             # Layer 4
        self.conv5 = GCNConv(64, 32)              # Layer 5
        self.conv6 = GCNConv(32, self.output_dim) # Layer 6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        return x
    
class GraphSAGE_90(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = SAGEConv(self.feat_dim, 16)  # Layer 1
        self.conv2 = SAGEConv(16, 32)  # Layer 2
        self.conv3 = SAGEConv(32, 64)  # Layer 3
        self.conv4 = SAGEConv(64, 128)  # Layer 4 (peak dimensionality)
        self.conv5 = SAGEConv(128, 64)  # Layer 5
        self.conv6 = SAGEConv(64, 32)  # Layer 6
        self.conv7 = SAGEConv(32, 16)  # Layer 7
        self.conv8 = SAGEConv(16, 8)  # Layer 8
        self.conv9 = SAGEConv(8, 4)  # Layer 9
        self.conv10 = SAGEConv(4, self.output_dim)  # Layer 10 (Output)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x
    
class GraphSAGE_90_6_Double(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_90_6_Double, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = SAGEConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = SAGEConv(64, 128)               # Layer 2
        self.conv3 = SAGEConv(128, 512)              # Layer 3 (peak dimensionality)
        self.conv4 = SAGEConv(512, 128)              # Layer 4
        self.conv5 = SAGEConv(128, 64)               # Layer 5
        self.conv6 = SAGEConv(64, self.output_dim)   # Layer 6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        return x

class GCN_90_6_Double(torch.nn.Module):
    def __init__(self):
        super(GCN_90_6_Double, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GCNConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = GCNConv(64, 128)               # Layer 2
        self.conv3 = GCNConv(128, 512)              # Layer 3 (peak dimensionality)
        self.conv4 = GCNConv(512, 128)              # Layer 4
        self.conv5 = GCNConv(128, 64)               # Layer 5
        self.conv6 = GCNConv(64, self.output_dim)   # Layer 6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        return x
class GraphSAGE_90_8(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_90_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = SAGEConv(self.feat_dim, 32)     # Layer 1
        self.conv2 = SAGEConv(32, 64)                # Layer 2
        self.conv3 = SAGEConv(64, 128)               # Layer 3
        self.conv4 = SAGEConv(128, 256)              # Layer 4 (peak dimensionality)
        self.conv5 = SAGEConv(256, 128)              # Layer 5
        self.conv6 = SAGEConv(128, 64)               # Layer 6
        self.conv7 = SAGEConv(64, 32)                # Layer 7
        self.conv8 = SAGEConv(32, self.output_dim)   # Layer 8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)
        return x    
    
class GAT_90(torch.nn.Module):
    def __init__(self):
        super(GAT_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 16)  # Layer 1
        self.conv2 = GATConv(16, 32)             # Layer 2
        self.conv3 = GATConv(32, 64)             # Layer 3
        self.conv4 = GATConv(64, 128)            # Layer 4 (peak dimensionality)
        self.conv5 = GATConv(128, 64)            # Layer 5
        self.conv6 = GATConv(64, 32)             # Layer 6
        self.conv7 = GATConv(32, 16)             # Layer 7
        self.conv8 = GATConv(16, 8)              # Layer 8
        self.conv9 = GATConv(8, 4)               # Layer 9
        self.conv10 = GATConv(4, self.output_dim)# Layer 10 (Output)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x
    
class GAT_90_3(torch.nn.Module):
    def __init__(self):
        super(GAT_90_3, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 32)       # Layer 1
        self.conv2 = GATConv(32, 64)                  # Layer 2 (hidden layer)
        self.conv3 = GATConv(64, self.output_dim)     # Layer 3 (output layer)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class GAT_90_3_2heads(torch.nn.Module):
    def __init__(self):
        super(GAT_90_3_2heads, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.intermediate_dim = 32 * 2   # Multiply by 2 since we're using 2 heads and concatenating
        self.hidden_dim = 64 * 2         # Same reason as above

        # Define GAT layers with 2 heads and using concat for multi-head aggregation
        self.conv1 = GATConv(self.feat_dim, 32, heads=2, concat=True)  # Layer 1
        self.conv2 = GATConv(self.intermediate_dim, 64, heads=2, concat=True)  # Layer 2 (hidden layer)
        self.conv3 = GATConv(self.hidden_dim, self.output_dim, heads=1, concat=False)  # Layer 3 (output layer) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

        
class GAT_90_6(torch.nn.Module):
    def __init__(self):
        super(GAT_90_6, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 32)   # Layer 1
        self.conv2 = GATConv(32, 64)              # Layer 2
        self.conv3 = GATConv(64, 256)             # Layer 3 (peak dimensionality)
        self.conv4 = GATConv(256, 64)             # Layer 4
        self.conv5 = GATConv(64, 32)              # Layer 5
        self.conv6 = GATConv(32, self.output_dim) # Layer 6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        return x
    
class GAT_90_6_Double(torch.nn.Module):
    def __init__(self):
        super(GAT_90_6_Double, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = GATConv(64, 128)               # Layer 2
        self.conv3 = GATConv(128, 512)              # Layer 3 (peak dimensionality)
        self.conv4 = GATConv(512, 128)              # Layer 4
        self.conv5 = GATConv(128, 64)               # Layer 5
        self.conv6 = GATConv(64, self.output_dim)   # Layer 6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        return x

class GAT_90_6_2heads(torch.nn.Module):
    def __init__(self):
        super(GAT_90_6_2heads, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        # Using 2 heads for multi-head attention, doubling the dimensions where concat is used
        self.conv1 = GATConv(self.feat_dim, 32, heads=2, concat=True)   # Layer 1: Output is 64
        self.conv2 = GATConv(64, 32, heads=2, concat=True)              # Layer 2: Output is 64
        self.conv3 = GATConv(64, 128, heads=2, concat=True)             # Layer 3: Output is 256
        self.conv4 = GATConv(256, 32, heads=2, concat=True)             # Layer 4: Output is 64
        self.conv5 = GATConv(64, 16, heads=2, concat=True)              # Layer 5: Output is 32
        self.conv6 = GATConv(32, self.output_dim, heads=1, concat=False) # Layer 6: Output is 6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        return x 
    
class GAT_90_8(torch.nn.Module):
    def __init__(self):
        super(GAT_90_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 32)     # Layer 1
        self.conv2 = GATConv(32, 64)                # Layer 2
        self.conv3 = GATConv(64, 128)               # Layer 3
        self.conv4 = GATConv(128, 256)              # Layer 4 (peak dimensionality)
        self.conv5 = GATConv(256, 128)              # Layer 5
        self.conv6 = GATConv(128, 64)               # Layer 6
        self.conv7 = GATConv(64, 32)                # Layer 7
        self.conv8 = GATConv(32, self.output_dim)   # Layer 8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)
        return x

class GAT_90_8_Increased(torch.nn.Module):
    def __init__(self):
        super(GAT_90_8_Increased, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 64)    # Layer 1
        self.conv2 = GATConv(64, 128)              # Layer 2
        self.conv3 = GATConv(128, 256)             # Layer 3
        self.conv4 = GATConv(256, 512)             # Layer 4 (peak dimensionality)
        self.conv5 = GATConv(512, 256)             # Layer 5
        self.conv6 = GATConv(256, 128)             # Layer 6
        self.conv7 = GATConv(128, 64)              # Layer 7
        self.conv8 = GATConv(64, self.output_dim)  # Layer 8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)
        return x 
  
class GATv2_90_8(torch.nn.Module):
    def __init__(self, heads=1):
        super(GATv2_90_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = conv.GATv2Conv(self.feat_dim, 32, heads=heads)
        self.conv2 = conv.GATv2Conv(32 * heads, 64, heads=heads)  # Taking concat into account
        self.conv3 = conv.GATv2Conv(64 * heads, 128, heads=heads)
        self.conv4 = conv.GATv2Conv(128 * heads, 256, heads=heads)
        self.conv5 = conv.GATv2Conv(256 * heads, 128, heads=heads)
        self.conv6 = conv.GATv2Conv(128 * heads, 64, heads=heads)
        self.conv7 = conv.GATv2Conv(64 * heads, 32, heads=heads)
        self.conv8 = conv.GATv2Conv(32 * heads, self.output_dim, heads=heads)

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)

        if return_attention_weights:
            x, attn_weights = self.conv8(x, edge_index, return_attention_weights=True)
            return x, attn_weights

        return x
    
class GATv2_90_8_2heads(torch.nn.Module):
    def __init__(self):
        super(GATv2_90_8_2heads, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.heads = 2

        # Adjusting for concatenated output from 2 heads
        self.conv1 = conv.GATv2Conv(self.feat_dim, 32, heads=self.heads)
        self.conv2 = conv.GATv2Conv(32 * self.heads, 64, heads=self.heads)
        self.conv3 = conv.GATv2Conv(64 * self.heads, 128, heads=self.heads)
        self.conv4 = conv.GATv2Conv(128 * self.heads, 256, heads=self.heads)
        self.conv5 = conv.GATv2Conv(256 * self.heads, 128, heads=self.heads)
        self.conv6 = conv.GATv2Conv(128 * self.heads, 64, heads=self.heads)
        self.conv7 = conv.GATv2Conv(64 * self.heads, 32, heads=self.heads)
        self.conv8 = conv.GATv2Conv(32 * self.heads, self.output_dim, heads=self.heads, concat=False)  # Assuming you want a single output dimension

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)

        if return_attention_weights:
            x, attn_weights = self.conv8(x, edge_index, return_attention_weights=True)
            return x, attn_weights

        return x
#### 95 % MISSING POINTS ####

#### CONVOLUTIONAL NETWORKS ####

class ConvNet_95(nn.Module):
    def __init__(self):
        super(ConvNet_95, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32*3, 64*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64*3, 128*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128*3, 256*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384*3, 128*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192*3, 64*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96*3, 32*3, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32*3, 3, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(torch.cat((enc4, enc3), dim=1))  # Skip connection from encoder3
        dec3 = self.decoder3(torch.cat((dec4, enc2), dim=1))  # Skip connection from encoder2
        dec2 = self.decoder2(torch.cat((dec3, enc1), dim=1))  # Skip connection from encoder1
        dec1 = self.decoder1(dec2)

        return dec1


#### GRAPH NETWORKS ####

class GAT_95_8(torch.nn.Module):
    def __init__(self):
        super(GAT_95_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1  # Use single head

        # Layers
        self.conv1 = GATConv(self.feat_dim, 64, heads=self.num_heads)
        self.conv2 = GATConv(64, 128, heads=self.num_heads)
        self.conv3 = GATConv(128, 256, heads=self.num_heads)
        self.conv4 = GATConv(256, 512, heads=self.num_heads)  # Bottleneck layer
        self.conv5 = GATConv(512, 256, heads=self.num_heads)
        self.conv6 = GATConv(256, 128, heads=self.num_heads)
        self.conv7 = GATConv(128, 64, heads=self.num_heads)
        self.conv8 = GATConv(64, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)  # Last layer without ReLU
        return x

class GraphSAGE_95_8(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_95_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        # Layers
        self.conv1 = SAGEConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = SAGEConv(64, 128)               # Layer 2
        self.conv3 = SAGEConv(128, 256)              # Layer 3
        self.conv4 = SAGEConv(256, 512)              # Layer 4 (Bottleneck layer)
        self.conv5 = SAGEConv(512, 256)              # Layer 5
        self.conv6 = SAGEConv(256, 128)              # Layer 6
        self.conv7 = SAGEConv(128, 64)               # Layer 7
        self.conv8 = SAGEConv(64, self.output_dim)   # Layer 8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)  # Last layer without ReLU
        return x

class GCN_95_8(torch.nn.Module):
    def __init__(self):
        super(GCN_95_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        # Layers
        self.conv1 = GCNConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = GCNConv(64, 128)               # Layer 2
        self.conv3 = GCNConv(128, 256)              # Layer 3
        self.conv4 = GCNConv(256, 512)              # Layer 4 (Bottleneck layer)
        self.conv5 = GCNConv(512, 256)              # Layer 5
        self.conv6 = GCNConv(256, 128)              # Layer 6
        self.conv7 = GCNConv(128, 64)               # Layer 7
        self.conv8 = GCNConv(64, self.output_dim)   # Layer 8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)  # Last layer without ReLU
        return x
    
class GAT_95_10(torch.nn.Module):
    def __init__(self):
        super(GAT_95_10, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1  # Use single head

        # Layers
        self.conv1 = GATConv(self.feat_dim, 32, heads=self.num_heads)
        self.conv2 = GATConv(32, 64, heads=self.num_heads)
        self.conv3 = GATConv(64, 128, heads=self.num_heads)
        self.conv4 = GATConv(128, 256, heads=self.num_heads)
        self.conv5 = GATConv(256, 512, heads=self.num_heads)  # Bottleneck layer
        self.conv6 = GATConv(512, 256, heads=self.num_heads)
        self.conv7 = GATConv(256, 128, heads=self.num_heads)
        self.conv8 = GATConv(128, 64, heads=self.num_heads)
        self.conv9 = GATConv(64, 32, heads=self.num_heads)
        self.conv10 = GATConv(32, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)  # Last layer without ReLU
        return x

class GAT_95_12(torch.nn.Module):
    def __init__(self):
        super(GAT_95_12, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1  # Use single head

        # Layers
        self.conv1 = GATConv(self.feat_dim, 24, heads=self.num_heads)
        self.conv2 = GATConv(24, 48, heads=self.num_heads)
        self.conv3 = GATConv(48, 96, heads=self.num_heads)
        self.conv4 = GATConv(96, 192, heads=self.num_heads)
        self.conv5 = GATConv(192, 384, heads=self.num_heads)
        self.conv6 = GATConv(384, 512, heads=self.num_heads)  # Bottleneck layer
        self.conv7 = GATConv(512, 384, heads=self.num_heads)
        self.conv8 = GATConv(384, 192, heads=self.num_heads)
        self.conv9 = GATConv(192, 96, heads=self.num_heads)
        self.conv10 = GATConv(96, 48, heads=self.num_heads)
        self.conv11 = GATConv(48, 24, heads=self.num_heads)
        self.conv12 = GATConv(24, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = torch.relu(self.conv10(x, edge_index))
        x = torch.relu(self.conv11(x, edge_index))
        x = self.conv12(x, edge_index)  # Last layer without ReLU
        return x


class GraphSAGE_95(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_95, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = SAGEConv(self.feat_dim, 128*2)
        self.conv2 = SAGEConv(128*2, 256*2)
        self.conv3 = SAGEConv(256*2, 512*2)
        self.conv4 = SAGEConv(512*2, 256*2)
        self.conv5 = SAGEConv(256*2, 128*2)
        self.conv6 = SAGEConv(128*2, 64*2)
        self.conv7 = SAGEConv(64*2, 32*2) # New conv layer
        self.conv8 = SAGEConv(32*2, self.feat_dim) # New conv layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = torch.relu(x)
        x = self.conv6(x, edge_index)
        x = torch.relu(x)
        x = self.conv7(x, edge_index) 
        x = torch.relu(x)
        x = self.conv8(x, edge_index) # New conv layer
        return x
#### 99% ####
class ConvNet_98(nn.Module):
    def __init__(self):
        super(ConvNet_98, self).__init__()

        # Encoder
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

        # Decoder
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

        dec4 = self.decoder4(torch.cat((enc4, enc3), dim=1))  # Skip connection from encoder3
        dec3 = self.decoder3(torch.cat((dec4, enc2), dim=1))  # Skip connection from encoder2
        dec2 = self.decoder2(torch.cat((dec3, enc1), dim=1))  # Skip connection from encoder1
        dec1 = self.decoder1(dec2)

        return dec1
    
class GAT_98_10(torch.nn.Module):
    def __init__(self):
        super(GAT_98_10, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        # Layers
        self.conv1 = GATConv(self.feat_dim, 64, heads=self.num_heads)
        self.conv2 = GATConv(64, 128, heads=self.num_heads)
        self.conv3 = GATConv(128, 256, heads=self.num_heads)
        self.conv4 = GATConv(256, 384, heads=self.num_heads)
        self.conv5 = GATConv(384, 512, heads=self.num_heads)  # Bottleneck layer
        self.conv6 = GATConv(512, 384, heads=self.num_heads)
        self.conv7 = GATConv(384, 256, heads=self.num_heads)
        self.conv8 = GATConv(256, 128, heads=self.num_heads)
        self.conv9 = GATConv(128, 64, heads=self.num_heads)
        self.conv10 = GATConv(64, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x

class GAT_98_4(torch.nn.Module):
    def __init__(self):
        super(GAT_98_4, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        # Layers
        self.conv1 = GATConv(self.feat_dim, 64, heads=self.num_heads, concat=True)  # Output: 64*2 = 128
        self.conv2 = GATConv(64*self.num_heads, 128, heads=self.num_heads, concat=True)  # Output: 128*2 = 256
        self.conv3 = GATConv(128*self.num_heads, 256, heads=self.num_heads, concat=True)  # Output: 256*2 = 512
        self.conv4 = GATConv(256*self.num_heads, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

class red_GAT_98_5(torch.nn.Module):
    def __init__(self):
        super(red_GAT_98_5, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        self.conv1 = GATConv(self.feat_dim, 32, heads=self.num_heads, concat=True)
        self.conv2 = GATConv(32, 64, heads=self.num_heads, concat=True)
        self.conv3 = GATConv(64, 128, heads=self.num_heads, concat=True) 
        self.conv4 = GATConv(128, 256, heads=self.num_heads, concat=True)
        self.conv5 = GATConv(256, self.output_dim, heads=self.num_heads, concat=False)  # Adjusted to output dimensions

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.conv2(x1, edge_index))
        x3 = torch.relu(self.conv3(x2, edge_index)) 
        x4 = torch.relu(self.conv4(x3, edge_index)) 
        x5 = self.conv5(x4, edge_index)
        return x5


class GAT_98_6(torch.nn.Module):
    def __init__(self):
        super(GAT_98_6, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        self.conv1 = GATConv(self.feat_dim, 32, heads=self.num_heads, concat=True)
        self.conv2 = GATConv(32, 64, heads=self.num_heads, concat=True)
        self.conv3 = GATConv(64, 128, heads=self.num_heads, concat=True) 
        self.conv4 = GATConv(128, 256, heads=self.num_heads, concat=True)
        self.conv5 = GATConv(256, 512, heads=self.num_heads, concat=True)
        self.conv6 = GATConv(512, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.conv2(x1, edge_index))
        x3 = torch.relu(self.conv3(x2, edge_index)) 
        x4 = torch.relu(self.conv4(x3, edge_index)) 
        x5 = torch.relu(self.conv5(x4, edge_index))
        x6 = self.conv6(x5, edge_index)
        return x6

class red_GAT_98_6(torch.nn.Module):
    def __init__(self):
        super(red_GAT_98_6, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        self.conv1 = GATConv(self.feat_dim, 16, heads=self.num_heads, concat=True)
        self.conv2 = GATConv(16, 32, heads=self.num_heads, concat=True)
        self.conv3 = GATConv(32, 64, heads=self.num_heads, concat=True) 
        self.conv4 = GATConv(64, 128, heads=self.num_heads, concat=True)
        self.conv5 = GATConv(128, 256, heads=self.num_heads, concat=True)
        self.conv6 = GATConv(256, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.conv2(x1, edge_index))
        x3 = torch.relu(self.conv3(x2, edge_index)) 
        x4 = torch.relu(self.conv4(x3, edge_index)) 
        x5 = torch.relu(self.conv5(x4, edge_index))
        x6 = self.conv6(x5, edge_index)
        return x6


    
class GAT_98_3(torch.nn.Module):
    def __init__(self):
        super(GAT_98_3, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        # Layers
        self.conv1 = GATConv(self.feat_dim, 128, heads=self.num_heads, concat=True)
        # Bottleneck layer
        self.conv2 = GATConv(128, 64, heads=self.num_heads, concat=True)
        # Expand back
        self.conv3 = GATConv(64, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x   

class GAT_98_2(torch.nn.Module):
    def __init__(self):
        super(GAT_98_2, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1

        # Layers
        self.conv1 = GATConv(self.feat_dim, 128, heads=self.num_heads, concat=True)
        # Last layer
        self.conv2 = GATConv(128 * self.num_heads, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
class GCN_98_10(torch.nn.Module):
    def __init__(self):
        super(GCN_98_10, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        # Layers
        self.conv1 = GCNConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = GCNConv(64, 128)               # Layer 2
        self.conv3 = GCNConv(128, 256)              # Layer 3
        self.conv4 = GCNConv(256, 384)              # Layer 4
        self.conv5 = GCNConv(384, 512)              # Layer 5 (Bottleneck layer)
        self.conv6 = GCNConv(512, 384)              # Layer 6
        self.conv7 = GCNConv(384, 256)              # Layer 7
        self.conv8 = GCNConv(256, 128)              # Layer 8
        self.conv9 = GCNConv(128, 64)               # Layer 9
        self.conv10 = GCNConv(64, self.output_dim)  # Layer 10

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x    

class GraphSAGE_98_10(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_98_10, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        # Layers
        self.conv1 = SAGEConv(self.feat_dim, 64)     # Layer 1
        self.conv2 = SAGEConv(64, 128)               # Layer 2
        self.conv3 = SAGEConv(128, 256)              # Layer 3
        self.conv4 = SAGEConv(256, 384)              # Layer 4
        self.conv5 = SAGEConv(384, 512)              # Layer 5 (Bottleneck layer)
        self.conv6 = SAGEConv(512, 384)              # Layer 6
        self.conv7 = SAGEConv(384, 256)              # Layer 7
        self.conv8 = SAGEConv(256, 128)              # Layer 8
        self.conv9 = SAGEConv(128, 64)               # Layer 9
        self.conv10 = SAGEConv(64, self.output_dim)  # Layer 10

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x
class GraphSAGE_99(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_99, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = SAGEConv(self.feat_dim, 128)
        self.conv2 = SAGEConv(128, 256)
        self.conv3 = SAGEConv(256, 512)
        self.conv4 = SAGEConv(512, 512) # added
        self.conv5 = SAGEConv(512, 512) # added
        self.conv6 = SAGEConv(512, 256)
        self.conv7 = SAGEConv(256, 128)
        self.conv8 = SAGEConv(128, 64)
        self.conv9 = SAGEConv(64, 32) 
        self.conv10 = SAGEConv(32, self.feat_dim) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = torch.relu(x)
        x = self.conv6(x, edge_index)
        x = torch.relu(x)
        x = self.conv7(x, edge_index)
        x = torch.relu(x)
        x = self.conv8(x, edge_index)
        x = torch.relu(x)
        x = self.conv9(x, edge_index)
        x = torch.relu(x)
        x = self.conv10(x, edge_index)
        return x

    

#########################
#         HOLES         #
#                       # 
#########################

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, kernel_size=3, stride=1, padding=1), 
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)   
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(torch.cat((self.up(enc4), enc3), dim=1))
        dec3 = self.decoder3(torch.cat((self.up(dec4), enc2), dim=1))
        dec2 = self.decoder2(torch.cat((self.up(dec3), enc1), dim=1))
        dec1 = self.decoder1(self.up(dec2))

        return dec1

class Hole_GAT_10(torch.nn.Module):
    def __init__(self):
        super(Hole_GAT_10, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        
        self.conv1 = GATConv(self.feat_dim, 16)
        self.conv2 = GATConv(16, 32)
        self.conv3 = GATConv(32, 64)
        self.conv4 = GATConv(64, 128)
        self.conv5 = GATConv(128, 256) # Bottleneck
        self.conv6 = GATConv(256, 128)
        self.conv7 = GATConv(128, 64)
        self.conv8 = GATConv(64, 32)
        self.conv9 = GATConv(32, 16)
        self.conv10 = GATConv(16, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = torch.relu(self.conv8(x, edge_index))
        x = torch.relu(self.conv9(x, edge_index))
        x = self.conv10(x, edge_index)
        return x
    
class GAT_98_8(torch.nn.Module):
    def __init__(self):
        super(GAT_98_8, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1  # Use single head

        # Layers
        self.conv1 = GATConv(self.feat_dim, 64, heads=self.num_heads)
        self.conv2 = GATConv(64, 128, heads=self.num_heads)
        self.conv3 = GATConv(128, 256, heads=self.num_heads)
        self.conv4 = GATConv(256, 512, heads=self.num_heads)  # Bottleneck layer
        self.conv5 = GATConv(512, 256, heads=self.num_heads)
        self.conv6 = GATConv(256, 128, heads=self.num_heads)
        self.conv7 = GATConv(128, 64, heads=self.num_heads)
        self.conv8 = GATConv(64, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)  # Last layer without ReLU
        return x

class GAT_98_8_Modified(torch.nn.Module):
    def __init__(self):
        super(GAT_98_8_Modified, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.num_heads = 1  # Use single head

        # Layers
        self.conv1 = GATConv(self.feat_dim, 80, heads=self.num_heads)
        self.conv2 = GATConv(80, 160, heads=self.num_heads)
        self.conv3 = GATConv(160, 320, heads=self.num_heads)
        self.conv4 = GATConv(320, 640, heads=self.num_heads)  # Bottleneck layer
        self.conv5 = GATConv(640, 320, heads=self.num_heads)
        self.conv6 = GATConv(320, 160, heads=self.num_heads)
        self.conv7 = GATConv(160, 80, heads=self.num_heads)
        self.conv8 = GATConv(80, self.output_dim, heads=self.num_heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))
        x = torch.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)  # Last layer without ReLU
        return x