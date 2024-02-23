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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.InstanceNorm2d) -> None:
        super().__init__()

        self.convs = nn.Sequential(
            ConvNormReLU(in_channels, in_channels, 3, 1, 1, norm_layer),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            norm_layer(in_channels)
        )

    def forward(self, x):
        return F.relu(x + self.convs(x))

class StyleMapping(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )
    
    def forward(self, x):
        return self.model(x)
    
class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta

        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super().__init__()

        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels, norm_layer),
        )
        self.down = ConvNormReLU(in_channels, out_channels, 3, 2, 1, norm_layer)

    def forward(self, x):
        feat = self.res_blocks(x)
        return self.down(feat)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d, style_dim=64):
        super().__init__()

        self.convs = nn.Sequential(
            ConvNormReLU(in_channels, in_channels, 3, 1, 1, norm_layer),
            ConvNormReLU(in_channels, out_channels, 3, 1, 1, norm_layer)
        )
        self.style_mapping = StyleMapping(style_dim)
        self.adain = AdaIN(out_channels, style_dim)

    def forward(self, x, latent):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        feat = self.convs(x)
        style = self.style_mapping(latent)
        return self.adain(feat, style)


class AE(nn.Module):
    def __init__(self, latent_dim: int=256) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            EncoderBlock(3, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            nn.Conv2d(128, latent_dim, 8),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 8),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = DecoderBlock(128, 64, style_dim=latent_dim)
        self.decoder3 = DecoderBlock(64, 32, style_dim=latent_dim)
        self.decoder4 = DecoderBlock(32, 16, style_dim=latent_dim)
        self.final = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, x):
        latent = self.encoder(x)
        _latent = latent.view(latent.size(0), self.latent_dim)
        feat = self.decoder1(latent)
        feat = self.decoder2(feat, _latent)
        feat = self.decoder3(feat, _latent)
        feat = self.decoder4(feat, _latent)
        return F.sigmoid(self.final(feat)), latent
    
    def encode(self, x):
        latent = self.encoder(x)
        return latent

    def decode(self, latent):
        _latent = latent.view(latent.size(0), self.latent_dim)
        feat = self.decoder1(latent)
        feat = self.decoder2(feat, _latent)
        feat = self.decoder3(feat, _latent)
        feat = self.decoder4(feat, _latent)
        return F.sigmoid(self.final(feat))
    
    def interpolate_with_style_space(self, l1, l2, alphas=[0.5, 0.5, 0.5, 0.5]):
        _l1 = l1.view(l1.size(0), self.latent_dim)
        _l2 = l2.view(l2.size(0), self.latent_dim)
        
        feat = self.decoder1(l1*alphas[0] + l2*(1-alphas[0]))
        
        for decoder, a in zip([self.decoder2, self.decoder3, self.decoder4], alphas[1:]):
            feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
            feat = decoder.convs(feat)
            style1 = decoder.style_mapping(_l1)
            style2 = decoder.style_mapping(_l2)
            feat = decoder.adain(feat, style1*a + style2*(1-a))
        return F.sigmoid(self.final(feat))