import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.cust_resblock import ResBlock2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class G2d(nn.Module):
    def __init__(self, in_channels):
        super(G2d, self).__init__()
        self.reshape = nn.Conv2d(96, 1536, kernel_size=1).to(device)  # Reshape C96xD16 → C1536
        self.conv1x1 = nn.Conv2d(1536, 512, kernel_size=1).to(device)  # 1x1 convolution to reduce channels to 512

        self.res_blocks = nn.Sequential(
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
        ).to(device)

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(512, 256)
        ).to(device)

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128)
        ).to(device)

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64)
        ).to(device)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        logging.debug(f"G2d > x:{x.shape}")
        x = self.reshape(x)
        x = self.conv1x1(x)  # Added 1x1 convolution to reduce channels to 512
        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.final_conv(x)
        return x
