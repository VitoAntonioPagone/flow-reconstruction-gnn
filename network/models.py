import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Module, Linear, ReLU, Dropout
import torchvision.transforms.functional as TF
from torch_geometric.nn import (GCNConv, SAGEConv, GATConv, GravNetConv, 
                                GINConv, PNAConv, ChebConv, AGNNConv, ARMAConv)
from torch_geometric.data import Data
from attention import ChannelAttention

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
    
class ConvNet_50(nn.Module):
    def __init__(self):
        super(ConvNet_50, self).__init__()

        # Encoder with Channel Attention
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(32)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(64)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(128)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(256)
        )

        # Decoder with Channel Attention
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(128)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(64)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            ChannelAttention(32)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=7, stride=1, padding=3),
            ChannelAttention(3)
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

class AGNN_90(torch.nn.Module):
    def __init__(self):
        super(AGNN_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.lin1 = torch.nn.Linear(self.feat_dim, 64)
        self.conv1 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(64, 128)
        self.conv2 = AGNNConv(requires_grad=True)
        self.lin3 = torch.nn.Linear(128, 64)
        self.conv3 = AGNNConv(requires_grad=True)
        self.lin4 = torch.nn.Linear(64, 32)
        self.conv4 = AGNNConv(requires_grad=True)
        self.lin5 = torch.nn.Linear(32, self.output_dim)
        self.conv5 = AGNNConv(requires_grad=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin1(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.lin2(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.lin3(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.lin4(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.lin5(x)
        x = self.conv5(x, edge_index)
        return x

class GIN_90(torch.nn.Module):
    def __init__(self):
        super(GIN_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = GINConv(self.feat_dim, 128)
        self.conv2 = GINConv(128, 256)
        self.conv3 = GINConv(256, 512)
        self.conv4 = GINConv(512, 256)
        self.conv5 = GINConv(256, 128)
        self.conv6 = GINConv(128, 64) # Updated this line
        self.conv7 = GINConv(64, self.feat_dim) # New conv layer

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
        x = self.conv7(x, edge_index) # New conv layer
        return x


class GraphSAGE_90(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = SAGEConv(self.feat_dim, 128*2)
        self.conv2 = SAGEConv(128*2, 256*2)
        self.conv3 = SAGEConv(256*2, 512*2)
        self.conv4 = SAGEConv(512*2, 256*2)
        self.conv5 = SAGEConv(256*2, 128*2)
        self.conv6 = SAGEConv(128*2, 64*2) # Updated this line
        self.conv7 = SAGEConv(64*2, self.feat_dim) # New conv layer

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
        x = self.conv7(x, edge_index) # New conv layer
        return x

class GAT_90(torch.nn.Module):
    def __init__(self):
        super(GAT_90, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = GATConv(self.feat_dim, 64)
        self.conv2 = GATConv(64, 128)
        self.conv3 = GATConv(128, 256)
        self.conv4 = GATConv(256, 128)
        self.conv5 = GATConv(128, 64)
        self.conv6 = GATConv(64, self.feat_dim)

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



class ChebNet_90(torch.nn.Module):
    def __init__(self):
        super(ChebNet_90, self).__init__()
        self.feat_dim = 6
        self.conv1 = ChebConv(self.feat_dim, 128*2, K=2)
        self.conv2 = ChebConv(128*2, 256*2, K=2)
        self.conv3 = ChebConv(256*2, 512*2, K=2)
        self.conv4 = ChebConv(512*2, 256*2, K=2)
        self.conv5 = ChebConv(256*2, 128*2, K=2)
        self.conv6 = ChebConv(128*2, 64*2, K=2) # New conv layer
        self.conv7 = ChebConv(64*2, self.feat_dim, K=2) # New conv layer

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
        x = self.conv7(x, edge_index) # New conv layer
        return x

'''class DilatedConvAutoEncoder(nn.Module):
    def __init__(self):
        super(DilatedConvAutoEncoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32*2, 64*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64*2, 128*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128*2, 256*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384*2, 128*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192*2, 64*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96*2, 32*2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 3, kernel_size=7, stride=1, padding=6, dilation=2),
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
'''

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
    
class ConvNet_95_deeper(nn.Module):
    def __init__(self):
        super(ConvNet_95_deeper, self).__init__()

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

#### GRAPH NETWORKS ####

class GAT_95(torch.nn.Module):
    def __init__(self):
        super(GAT_95, self).__init__()
        self.feat_dim = 6
        self.output_dim = 6
        self.conv1 = GATConv(self.feat_dim, 64)
        self.conv2 = GATConv(64, 128)
        self.conv3 = GATConv(128, 256)
        self.conv4 = GATConv(256, 128)
        self.conv5 = GATConv(128, 64)
        self.conv6 = GATConv(64, self.feat_dim)

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
class ConvNet_99(nn.Module):
    def __init__(self):
        super(ConvNet_99, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32*5, 64*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64*5, 128*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128*5, 256*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(384*5, 128*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(192*5, 64*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96*5, 32*5, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32*5, 3, kernel_size=7, stride=1, padding=3),
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
'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
            self, in_channels=5, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
'''

