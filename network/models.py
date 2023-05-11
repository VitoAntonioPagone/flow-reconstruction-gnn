import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from attention import (
    SELayer,
    ChannelAttention,
    SpatialAttention,
    NonLocalBlock)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConvAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim):
        super(GraphConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, embedding_dim)

        # Decoder layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_features)

    def encode(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

    def decode(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z)
        return x_hat


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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size3, hidden_size4),
            nn.BatchNorm1d(hidden_size4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size4, hidden_size5),
            nn.BatchNorm1d(hidden_size5),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size5, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.dense(x)

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder7 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )

        # Dense block
        self.dense_block = DenseBlock(2048)

        # Decoder
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)

        # Apply dense block
        bottleneck = self.dense_block(enc7)

        dec7 = self.decoder7(bottleneck)
        dec7 = torch.cat((dec7, enc6), dim=1)  # Skip connection
        dec6 = self.decoder6(dec7)
        dec6 = torch.cat((dec6, enc5), dim=1)  # Skip connection
        dec5 = self.decoder5(dec6)
        dec5 = torch.cat((dec5, enc4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)  # Skip connection
        x = self.decoder1(dec2)

        return x


class DilatedConvAutoEncoder(nn.Module):
    def __init__(self):
        super(DilatedConvAutoEncoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=32, dilation=32),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder7 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=64, dilation=64),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        # Decoder
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)

        dec6 = self.decoder6(enc7)
        dec6 = torch.cat((dec6, enc6), dim=1)  # Skip connection
        dec5 = self.decoder5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)  # Skip connection
        dec4 = self.decoder4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        x = self.decoder1(dec2)

        return x
    
class SEConvAutoEncoder(nn.Module):
    def __init__(self):
        super(SEConvAutoEncoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(32),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(64),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(128),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(256),
            nn.MaxPool2d(2, 2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(512),
            nn.MaxPool2d(2, 2)
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(1024),
            nn.MaxPool2d(2, 2)
        )
        self.encoder7 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(2048),
            nn.MaxPool2d(2, 2)
        )

        # Dense block
        self.dense_block = DenseBlock(2048)

        # Decoder
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            SELayer(1024),
            nn.Dropout(0.2)
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SELayer(512),
            nn.Dropout(0.2)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SELayer(256),
            nn.Dropout(0.2)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SELayer(128),
            nn.Dropout(0.2)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SELayer(64),
            nn.Dropout(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SELayer(32),
            nn.Dropout(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)

        # Apply dense block
        bottleneck = self.dense_block(enc7)

        dec7 = self.decoder7(bottleneck)
        dec7 = torch.cat((dec7, enc6), dim=1)  # Skip connection
        dec6 = self.decoder6(dec7)
        dec6 = torch.cat((dec6, enc5), dim=1)  # Skip connection
        dec5 = self.decoder5(dec6)
        dec5 = torch.cat((dec5, enc4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)  # Skip connection
        x = self.decoder1(dec2)

        return x
    


class SpatialAttentionConvAutoEncoder(nn.Module):
    def __init__(self):
        super(SpatialAttentionConvAutoEncoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder7 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )

        # Dense block
        self.dense_block = DenseBlock(2048)

        # Decoder
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.spatial_attention_enc = SpatialAttention(kernel_size=7)
        self.spatial_attention_dec = SpatialAttention(kernel_size=7)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_att = self.spatial_attention_enc(enc1)
        enc1 = enc1 * enc1_att  # Apply attention weights

        enc2 = self.encoder2(enc1)
        enc2_att = self.spatial_attention_enc(enc2)
        enc2 = enc2 * enc2_att  # Apply attention weights

        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)

        # Apply dense block
        bottleneck = self.dense_block(enc7)

        dec7 = self.decoder7(bottleneck)
        dec7 = torch.cat((dec7, enc6), dim=1)  # Skip connection
        dec7_att = self.spatial_attention_dec(dec7)
        dec7 = dec7 * dec7_att  # Apply attention weights

        dec6 = self.decoder6(dec7)
        dec6 = torch.cat((dec6, enc5), dim=1)  # Skip connection
        dec6_att = self.spatial_attention_dec(dec6)
        dec6 = dec6 * dec6_att  # Apply attention weights

        dec5 = self.decoder5(dec6)
        dec5 = torch.cat((dec5, enc4), dim=1)  # Skip connection
        dec5_att = self.spatial_attention_dec(dec5)
        dec5 = dec5 * dec5_att  # Apply attention weights

        dec4 = self.decoder4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)  # Skip connection
        dec4_att = self.spatial_attention_dec(dec4)
        dec4 = dec4 * dec4_att  # Apply attention weights

        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)  # Skip connection
        dec3_att = self.spatial_attention_dec(dec3)
        dec3 = dec3 * dec3_att  # Apply attention weights

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)  # Skip connection
        dec2_att = self.spatial_attention_dec(dec2)
        dec2 = dec2 * dec2_att  # Apply attention weights

        x = self.decoder1(dec2)

        return x
    
class NonLocalConvAutoEncoder(nn.Module):
    def __init__(self):
        super(NonLocalConvAutoEncoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder7 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )

        # Dense block
        self.dense_block = DenseBlock(2048)

        # Decoder
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.non_local_block1 = NonLocalBlock(32)
        self.non_local_block2 = NonLocalBlock(64)
        self.non_local_block3 = NonLocalBlock(128)
        self.non_local_block4 = NonLocalBlock(256)
        self.non_local_block5 = NonLocalBlock(512)
        self.non_local_block6 = NonLocalBlock(1024)
        self.non_local_block7 = NonLocalBlock(2048)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = self.non_local_block1(enc1)

        enc2 = self.encoder2(enc1)
        enc2 = self.non_local_block2(enc2)

        enc3 = self.encoder3(enc2)
        enc3 = self.non_local_block3(enc3)

        enc4 = self.encoder4(enc3)
        enc4 = self.non_local_block4(enc4)

        enc5 = self.encoder5(enc4)
        enc5 = self.non_local_block5(enc5)

        enc6 = self.encoder6(enc5)
        enc6 = self.non_local_block6(enc6)

        enc7 = self.encoder7(enc6)
        enc7 = self.non_local_block7(enc7)

        # Apply dense block
        bottleneck = self.dense_block(enc7)

        bottleneck = self.dense_block(enc7)

        dec7 = self.decoder7(bottleneck)
        dec7 = torch.cat((dec7, enc6), dim=1)  # Skip connection
        dec7 = self.non_local_block_dec6(dec7)

        dec6 = self.decoder6(dec7)
        dec6 = torch.cat((dec6, enc5), dim=1)  # Skip connection
        dec6 = self.non_local_block_dec5(dec6)

        dec5 = self.decoder5(dec6)
        dec5 = torch.cat((dec5, enc4), dim=1)  # Skip connection
        dec5 = self.non_local_block_dec4(dec5)

        dec4 = self.decoder4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)  # Skip connection
        dec4 = self.non_local_block_dec3(dec4)

        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)  # Skip connection
        dec3 = self.non_local_block_dec2(dec3)

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)  # Skip connection
        dec2 = self.non_local_block_dec1(dec2)

        x = self.decoder1(dec2)
        return x