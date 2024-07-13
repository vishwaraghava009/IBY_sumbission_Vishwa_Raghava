import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_profiler import profile
import logging
import numpy as np

from models.apply_warping import apply_warping_field
from models.cust_resblock import ResBlock_Custom, ResBlock3D_Adaptive
from models.cust_resnet50 import CustomResNet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPRESS_DIM = 512


class Eapp(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3).to(device)
        self.resblock_128 = ResBlock_Custom(dimension=2, in_channels=64, out_channels=128).to(device)
        self.resblock_256 = ResBlock_Custom(dimension=2, in_channels=128, out_channels=256).to(device)
        self.resblock_512 = ResBlock_Custom(dimension=2, in_channels=256, out_channels=512).to(device)

        # round 0
        self.resblock3D_96 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        # round 1
        self.resblock3D_96_1 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_1_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        # round 2
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_2_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0).to(device)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0).to(device)

        # Second part: producing global descriptor es
        self.custom_resnet50 = CustomResNet50().to(device)
        '''
        ### TODO 2: Change vs/es here for vector size
        According to the description of the paper (Page11: predict the head pose and expression vector), 
        zs should be a global descriptor, which is a vector. Otherwise, the existence of Emtn and Eapp is of little significance. 
        The output feature is a matrix, which means it is basically not compressed. This encoder can be completely replaced by a VAE.
        '''        
        filters = [64, 256, 512, 1024, 2048]
        outputs=COMPRESS_DIM
        self.fc = torch.nn.Linear(filters[4], outputs)       
       
    def forward(self, x):
        # First part
        logging.debug(f"image x: {x.shape}") # [1, 3, 256, 256]
        out = self.conv(x)
        logging.debug(f"After conv: {out.shape}")  # [1, 3, 256, 256]
        out = self.resblock_128(out)
        logging.debug(f"After resblock_128: {out.shape}") # [1, 128, 256, 256]
        out = self.avgpool(out)
        logging.debug(f"After avgpool: {out.shape}")
        
        out = self.resblock_256(out)
        logging.debug(f"After resblock_256: {out.shape}")
        out = self.avgpool(out)
        logging.debug(f"After avgpool: {out.shape}")
        
        out = self.resblock_512(out)
        logging.debug(f"After resblock_512: {out.shape}") # [1, 512, 64, 64]
        out = self.avgpool(out) 
        # logging.debug(f"After avgpool: {out.shape}") # [1, 256, 64, 64]
   
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.conv_1(out)
        logging.debug(f"After conv_1: {out.shape}") # [1, 1536, 32, 32]
        
        vs = out.view(out.size(0), 96, 16, *out.shape[2:]) 
        logging.debug(f"reshape 1546 -> C96 x D16 : {vs.shape}") 
        
        
        # 1
        vs = self.resblock3D_96(vs)
        logging.debug(f"After resblock3D_96: {vs.shape}") 
        vs = self.resblock3D_96_2(vs)
        logging.debug(f"After resblock3D_96_2: {vs.shape}") # [1, 96, 16, 32, 32]

        # 2
        vs = self.resblock3D_96_1(vs)
        logging.debug(f"After resblock3D_96_1: {vs.shape}") # [1, 96, 16, 32, 32]
        vs = self.resblock3D_96_1_2(vs)
        logging.debug(f"After resblock3D_96_1_2: {vs.shape}")

        # 3
        vs = self.resblock3D_96_2(vs)
        logging.debug(f"After resblock3D_96_2: {vs.shape}") # [1, 96, 16, 32, 32]
        vs = self.resblock3D_96_2_2(vs)
        logging.debug(f"After resblock3D_96_2_2: {vs.shape}")

        # Second part
        es_resnet = self.custom_resnet50(x)
        es_flatten = torch.flatten(es_resnet, start_dim=1)
        es = self.fc(es_flatten) # torch.Size([bs, 2048]) -> torch.Size([bs, 2])        
        return vs, es