import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Module, Linear, ReLU, Dropout
import torchvision.transforms.functional as TF
from torch_geometric.nn import (GCNConv, SAGEConv, GATConv, GravNetConv, 
                                GINConv, PNAConv, ChebConv, AGNNConv, ARMAConv)
from torch_geometric.data import Data
from torch_geometric.data import Batch

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
        print(f'dec4 shape: {dec4.shape}')
        print(f'enc3 shape: {enc3.shape}')

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

class GAT_90_14(torch.nn.Module):
    def __init__(self):
        super(GAT_90_14, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 8)  # Layer 1
        self.conv2 = GATConv(8, 16)             # Layer 2
        self.conv3 = GATConv(16, 32)            # Layer 3
        self.conv4 = GATConv(32, 64)            # Layer 4
        self.conv5 = GATConv(64, 128)           # Layer 5
        self.conv6 = GATConv(128, 256)          # Layer 6
        self.conv7 = GATConv(256, 128)          # Layer 7 (peak dimensionality)
        self.conv8 = GATConv(128, 64)           # Layer 8
        self.conv9 = GATConv(64, 32)            # Layer 9
        self.conv10 = GATConv(32, 16)           # Layer 10
        self.conv11 = GATConv(16, 8)            # Layer 11
        self.conv12 = GATConv(8, 4)             # Layer 12
        self.conv13 = GATConv(4, 2)             # Layer 13
        self.conv14 = GATConv(2, self.output_dim)# Layer 14 (Output)

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
        x = torch.relu(self.conv12(x, edge_index))
        x = torch.relu(self.conv13(x, edge_index))
        x = self.conv14(x, edge_index)
        return x

class GAT_90_14_2(torch.nn.Module):
    def __init__(self):
        super(GAT_90_14_2, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6

        self.conv1 = GATConv(self.feat_dim, 8, heads=2)  # Layer 1
        self.conv2 = GATConv(8*2, 16, heads=2)           # Layer 2
        self.conv3 = GATConv(16*2, 32, heads=2)          # Layer 3
        self.conv4 = GATConv(32*2, 64, heads=2)          # Layer 4
        self.conv5 = GATConv(64*2, 128, heads=2)         # Layer 5
        self.conv6 = GATConv(128*2, 256, heads=2)        # Layer 6
        self.conv7 = GATConv(256*2, 128, heads=2)        # Layer 7 (peak dimensionality)
        self.conv8 = GATConv(128*2, 64, heads=2)         # Layer 8
        self.conv9 = GATConv(64*2, 32, heads=2)          # Layer 9
        self.conv10 = GATConv(32*2, 16, heads=2)         # Layer 10
        self.conv11 = GATConv(16*2, 8, heads=2)          # Layer 11
        self.conv12 = GATConv(8*2, 4, heads=2)           # Layer 12
        self.conv13 = GATConv(4*2, 2, heads=2)           # Layer 13
        self.conv14 = GATConv(2*2, self.output_dim)      # Layer 14 (Output)

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
        x = torch.relu(self.conv12(x, edge_index))
        x = torch.relu(self.conv13(x, edge_index))
        x = self.conv14(x, edge_index)
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

class GAT_95(torch.nn.Module):
    def __init__(self):
        super(GAT_95, self).__init__()
        self.feat_dim = 6
        self.hidden_dim = 64  # Increased hidden dimension
        self.output_dim = 6
        self.num_heads = 2  

        self.conv1 = GATConv(self.feat_dim, self.hidden_dim, heads=self.num_heads, dropout=0.5) 
        self.conv2 = GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads, dropout=0.5)
        self.conv3 = GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads, dropout=0.5)
        self.conv4 = GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads, dropout=0.5)
        self.conv5 = GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads, dropout=0.5)
        self.conv6 = GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads, dropout=0.5)  
        self.conv7 = GATConv(self.hidden_dim * self.num_heads, self.output_dim, concat=False)  
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        x = torch.relu(self.conv6(x, edge_index))  
        x = self.conv7(x, edge_index)  
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

