import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_scales=3, block_expansion=64, num_blocks=4, max_features=512, sn=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            SingleScaleDiscriminator(in_channels, block_expansion, num_blocks, max_features, sn)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, img_A, img_B):
        results = []
        for discriminator in self.discriminators:
            logging.debug(f"Shape of img_A before concat: {img_A.shape}")
            logging.debug(f"Shape of img_B before concat: {img_B.shape}")
            assert img_A.shape == img_B.shape, f"Shapes of img_A and img_B must match. Got {img_A.shape} and {img_B.shape}."
            results.append(discriminator(img_A, img_B))
            img_A = self.downsample(img_A)
            img_B = self.downsample(img_B)
        return results

class SingleScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, block_expansion=64, num_blocks=4, max_features=512, sn=False):
        super(SingleScaleDiscriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_channels * 2 if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kernel_size=4,
                    pool=(i != num_blocks - 1),
                    sn=sn,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(
            self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1
        )
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        out = img_input

        for down_block in self.down_blocks:
            out = down_block(out)

        return self.conv(out)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):

            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class DownBlock2d(nn.Module):

    def __init__(
        self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False
    ):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size
        )

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out
