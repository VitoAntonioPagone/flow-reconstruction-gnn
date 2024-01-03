import torch
import torch.nn as nn

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

 