import torch
import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 5, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.final_conv = nn.ConvTranspose2d(5, 5, kernel_size=3, stride=1, padding=1) # Added this layer

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        dec3 = self.decoder3(enc3)
        dec3 = torch.cat((dec3, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)  # Skip connection
        x = self.decoder1(dec2)
        x = self.final_conv(x)  # Apply the final ConvTranspose2d layer

        return x
