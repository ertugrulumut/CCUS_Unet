import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, features):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.features = features
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding='same')
        self.ac1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding='same')
        self.ac2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.ac2(x)
        x = self.bn2(x)
        return x


class Unet2d(nn.Module):
    def __init__(self, in_channels, out_channels=3, init_features=32):
        super(Unet2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = init_features
        self.encoder1 = ConvBlock(in_channels, self.features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(self.features, self.features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(self.features * 2, self.features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(self.features * 4, self.features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(self.features * 8, self.features * 16)

        self.upscaling4 = nn.ConvTranspose2d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock((self.features * 8) * 2, self.features * 8)
        self.upscaling3 = nn.ConvTranspose2d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ConvBlock((self.features * 4) * 2, self.features * 4)
        self.upscaling2 = nn.ConvTranspose2d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ConvBlock((self.features * 2) * 2, self.features * 2)
        self.upscaling1 = nn.ConvTranspose2d(
            self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock((self.features * 2), self.features)
        self.conv = nn.Conv2d(self.features, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, 14, 0, 14])
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upscaling4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upscaling3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upscaling2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upscaling1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = self.conv(dec1)
        output = output[..., :-14, :-14]
        output = output.permute(0, 2, 3, 1)
        output = self.activation(output)
        return output




        


