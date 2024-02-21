import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.instance_norm = norm_layer(out_channels)
        
    def forward(self, x):
        feat = self.instance_norm(self.conv(x))
        return F.relu(feat, inplace=True)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        
        self.blocks = nn.Sequential(
            ConvNormReLU(in_channels, in_channels, 3, 1, 1, norm_layer),
            ConvNormReLU(in_channels, in_channels, 3, 1, 1, norm_layer)
        )
        self.down = ConvNormReLU(in_channels, out_channels, 3, 2, 1, norm_layer)
        
    def forward(self, x):
        feat = self.blocks(x)
        return self.down(feat)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        
        self.convs = nn.Sequential(
            ConvNormReLU(in_channels, in_channels, 3, 1, 1, norm_layer),
            ConvNormReLU(in_channels, out_channels, 3, 1, 1, norm_layer)
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.convs(x)


class AE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            EncoderBlock(3, 16),
            EncoderBlock(16, 32),
            EncoderBlock(32, 64),
            nn.Conv2d(64, 64, 8),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 8),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            DecoderBlock(64, 32),
            DecoderBlock(32, 32),
            DecoderBlock(32, 16),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        recon_img = self.decoder(latent)
        return F.sigmoid(recon_img), latent