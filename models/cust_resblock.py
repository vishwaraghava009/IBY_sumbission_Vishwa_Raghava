import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from models.apply_warping import apply_warping_field
from models.conv2d_ws import Conv2d_WS
from models.conv3d_ws import Conv3d_WS
from models.adgnorm import AdaptiveGroupNorm


class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, in_channels, out_channels):
        super().__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
            self.conv_ws = Conv2d_WS(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.in_channels, self.out_channels, 3, padding=1)
            self.conv_ws = Conv3d_WS(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
    
#    @profile
    def forward(self, x):
        logging.debug(f"ResBlock_Custom > x.shape:  %s",x.shape)
        
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        assert output.shape[1] == self.out_channels, f"Expected {self.out_channels} channels, got {output.shape[1]}"
        assert output.shape[2] == x.shape[2] and output.shape[3] == x.shape[3], \
            f"Expected spatial dimensions {(x.shape[2], x.shape[3])}, got {(output.shape[2], output.shape[3])}"

        return output

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1, 1)):
        super(ResBlock3D, self).__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='trilinear', align_corners=False)
        
        return out

    
    
class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1, 1)):
        super().__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_channels)
        self.norm2 = AdaptiveGroupNorm(out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

#    @profile
    def forward(self, x):
        residual = x
        logging.debug(f"   ResBlock3D x.shape:{x.shape}")
        out = self.conv1(x)
        logging.debug(f"   conv1 > out.shape:{out.shape}")
        out = self.norm1(out)
        logging.debug(f"   norm1 > out.shape:{out.shape}")
        out = F.relu(out)
        logging.debug(f"   F.relu(out) > out.shape:{out.shape}")
        out = self.conv2(out)
        logging.debug(f"   conv2 > out.shape:{out.shape}")
        out = self.norm2(out)
        logging.debug(f"   norm2 > out.shape:{out.shape}")
        
        residual = self.residual_conv(residual)
        logging.debug(f"   residual > residual.shape:{residual.shape}",)
        
        out += residual
        out = F.relu(out)

        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='trilinear', align_corners=False)
        
        return out

            
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock2D, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)
        
        identity = self.shortcut(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out 
            
class ResBlock2D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1)):
        super().__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_channels)
        self.norm2 = AdaptiveGroupNorm(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = F.relu(out)

        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='bilinear', align_corners=False)
        
        return out


class SPADEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_avatars):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        middle_channels = min(in_channels, out_channels)

        self.conv_0 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.norm_0 = SPADE(in_channels, num_avatars)
        self.norm_1 = SPADE(middle_channels, num_avatars)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, num_avatars)

    def forward(self, x, avatar_index):
        x_s = self.shortcut(x, avatar_index)

        dx = self.conv_0(self.actvn(self.norm_0(x, avatar_index)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, avatar_index)))

        out = x_s + dx

        return out

    def shortcut(self, x, avatar_index):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, avatar_index))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, num_avatars):
        super().__init__()
        self.num_avatars = num_avatars
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.conv_shared = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

        self.avatar_shared_emb = nn.Embedding(num_avatars, 128)
        self.avatar_gamma_emb = nn.Embedding(num_avatars, norm_nc)
        self.avatar_beta_emb = nn.Embedding(num_avatars, norm_nc)

    def forward(self, x, avatar_index):
        avatar_shared = self.avatar_shared_emb(avatar_index)
        avatar_gamma = self.avatar_gamma_emb(avatar_index)
        avatar_beta = self.avatar_beta_emb(avatar_index)

        x = self.norm(x)
        shared_emb = self.conv_shared(x)
        gamma = self.conv_gamma(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        beta = self.conv_beta(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        gamma = gamma + avatar_gamma.view(-1, self.norm_nc, 1, 1)
        beta = beta + avatar_beta.view(-1, self.norm_nc, 1, 1)

        out = x * (1 + gamma) + beta
        return out
