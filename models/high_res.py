import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cust_resblock import ResBlock2D
from models.meta_loss import Vgg19
from models.gbase import Gbase

class Genh(nn.Module):
    def __init__(self):
        super(Genh, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            ResBlock2D(64, 64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64, 64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64, 64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64, 64),
        )
        self.res_blocks = nn.Sequential(
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
            ResBlock2D(64, 64),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64, 64),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

    def unsupervised_loss(self, x, x_hat):
        x_cycle = self.forward(x_hat)
        cycle_loss = F.l1_loss(x_cycle, x)
        return cycle_loss

    def supervised_loss(self, x_hat, y):
        l1_loss = F.l1_loss(x_hat, y)
        perceptual_loss = self.perceptual_loss(x_hat, y)

        return l1_loss + perceptual_loss

    def perceptual_loss(self, x, y):
        vgg = Vgg19(pretrained=True).features.eval().to(x.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        perceptual_layers = [1, 6, 11, 20, 29]
        perceptual_loss = 0.0
        for i, layer in enumerate(vgg):
            x = layer(x)
            y = layer(y)
            if i in perceptual_layers:
                perceptual_loss += nn.functional.l1_loss(x, y)
        return perceptual_loss

class GHR(nn.Module):
    def __init__(self):
        super(GHR, self).__init__()
        self.Gbase = Gbase()
        self.Genh = Genh()

    def forward(self, xs, xd):
        xhat_base, _ = self.Gbase(xs, xd)  
        xhat_hr = self.Genh(xhat_base)
        return xhat_hr

