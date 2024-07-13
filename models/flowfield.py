import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.cust_resblock import ResBlock3D_Adaptive


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowField(nn.Module):
    def __init__(self):
        super(FlowField, self).__init__()
        
        self.conv1x1 = nn.Conv2d(512, 2048, kernel_size=1).to(device)


        
        # reshape the tensor from [batch_size, 2048, height, width] to [batch_size, 512, 4, height, width], effectively splitting the channels into a channels dimension of size 512 and a depth dimension of size 4.
        self.reshape_layer = lambda x: x.view(-1, 512, 4, *x.shape[2:]).to(device)

        self.resblock1 = ResBlock3D_Adaptive(in_channels=512, out_channels=256).to(device)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.resblock2 = ResBlock3D_Adaptive( in_channels=256, out_channels=128).to(device)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.resblock3 =  ResBlock3D_Adaptive( in_channels=128, out_channels=64).to(device)
        self.upsample3 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.resblock4 = ResBlock3D_Adaptive( in_channels=64, out_channels=32).to(device)
        self.upsample4 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.conv3x3x3 = nn.Conv3d(32, 3, kernel_size=3, padding=1).to(device)
        self.gn = nn.GroupNorm(1, 3).to(device)
        self.tanh = nn.Tanh().to(device)
    
#    @profile
    def forward(self, zs,adaptive_gamma, adaptive_beta): # 
       # zs = zs * adaptive_gamma.unsqueeze(-1).unsqueeze(-1) + adaptive_beta.unsqueeze(-1).unsqueeze(-1)
        



        logging.debug(f"      FlowField > zs sum.shape:{zs.shape}") #torch.Size([1, 512, 1, 1])
        x = self.conv1x1(zs)
        logging.debug(f"      conv1x1 > x.shape:{x.shape}") #  -> [1, 2048, 1, 1]
        x = self.reshape_layer(x)
        logging.debug(f"      reshape_layer > x.shape:{x.shape}") # -> [1, 512, 4, 1, 1]
        x = self.upsample1(self.resblock1(x))
        logging.debug(f"      upsample1 > x.shape:{x.shape}") # [1, 512, 4, 1, 1]
        x = self.upsample2(self.resblock2(x))
        logging.debug(f"      upsample2 > x.shape:{x.shape}") #[512, 256, 8, 16, 16]
        x = self.upsample3(self.resblock3(x))
        logging.debug(f"      upsample3 > x.shape:{x.shape}")# [512, 128, 16, 32, 32]
        x = self.upsample4(self.resblock4(x))
        logging.debug(f"      upsample4 > x.shape:{x.shape}")
        x = self.conv3x3x3(x)
        logging.debug(f"      conv3x3x3 > x.shape:{x.shape}")
        x = self.gn(x)
        logging.debug(f"      gn > x.shape:{x.shape}")
        x = F.relu(x)
        logging.debug(f"      F.relu > x.shape:{x.shape}")

        x = self.tanh(x)
        logging.debug(f"      tanh > x.shape:{x.shape}")

        assert x.shape[1] == 3, f"Expected 3 channels after conv3x3x3, got {x.shape[1]}"

        return x