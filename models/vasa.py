# Sanity check is not done. May need a few changes to work as expected



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from math import cos, sin, pi
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from app_feat import AppearnceFeatures
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
# from insightface.app import FaceAnalysis
from torchvision.models import resnet50
from models.cust_resnet50 import CustomResNet50
from models.app_encoder import Eapp
from models.warping_generator import WarpGeneratorS2C
from models.warping_generator import WarpGeneratorC2D
from models.g2d import G2d
from models.g3d import G3d
from models.apply_warping import apply_warping_field
from models.resnet import resnet18


FEATURE_SIZE_AVG_POOL = 2 
FEATURE_SIZE = (2, 2) 
COMPRESS_DIM = 512 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        self.appearanceEncoder = Eapp()
        self.identityEncoder = CustomResNet50()
        self.headPoseEstimator = resnet18(pretrained=True)
        self.headPoseEstimator.fc = nn.Linear(self.headPoseEstimator.fc.in_features, 6)
        self.facialDynamicsEncoder = nn.Sequential(*list(resnet18(pretrained=False, num_classes=512).children())[:-1])
        self.facialDynamicsEncoder.adaptive_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE)
        self.facialDynamicsEncoder.fc = nn.Linear(2048, COMPRESS_DIM)

    def forward(self, x):
        appearance_volume = self.appearanceEncoder(x)[0]  # Get only the appearance volume
        identity_code = self.identityEncoder(x)
        head_pose = self.headPoseEstimator(x)
        rotation = head_pose[:, :3]
        translation = head_pose[:, 3:]
        facial_dynamics_features = self.facialDynamicsEncoder(x) # es
        facial_dynamics = self.facialDynamicsEncoder.fc(torch.flatten(facial_dynamics_features, start_dim=1))
        return appearance_volume, identity_code, rotation, translation, facial_dynamics


class FaceDecoder(nn.Module):
    def __init__(self):
        super(FaceDecoder, self).__init__()
        self.warp_generator_s2c = WarpGeneratorS2C(num_channels=512)
        self.warp_generator_c2d = WarpGeneratorC2D(num_channels=512)
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)

    def forward(self, appearance_volume, identity_code, rotation, translation, facial_dynamics):
        w_s2c = self.warp_generator_s2c(rotation, translation, facial_dynamics, identity_code)
        canonical_volume = apply_warping_field(appearance_volume, w_s2c)
        assert canonical_volume.shape[1:] == (96, 16, 64, 64)

        vc2d = self.G3d(canonical_volume)
        w_c2d = self.warp_generator_c2d(rotation, translation, facial_dynamics, identity_code)
        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        assert vc2d_warped.shape[1:] == (96, 16, 64, 64)

        vc2d_projected = torch.sum(vc2d_warped, dim=2)
        xhat = self.G2d(vc2d_projected)
        return xhat



class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.identity_extractor = resnet50(pretrained=True)
        self.identity_extractor.fc = nn.Identity()

    def forward(self, x, y):
        x_feats = self.identity_extractor(x)
        y_feats = self.identity_extractor(y)
        return 1 - F.cosine_similarity(x_feats, y_feats, dim=1).mean()

class DPELoss(nn.Module):
    def __init__(self):
        super(DPELoss, self).__init__()
        self.identity_loss = IdentityLoss()
        self.recon_loss = nn.L1Loss()

    def forward(self, I_i, I_j, I_i_pose_j, I_j_pose_i, I_s, I_d, I_s_pose_d_dyn_d):

        pose_dyn_loss = self.recon_loss(I_i_pose_j, I_j_pose_i)

        identity_loss = self.identity_loss(I_s, I_s_pose_d_dyn_d)

        return pose_dyn_loss + identity_loss


class DiffusionTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout=0.1):
        super(DiffusionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, audio_features, gaze_direction, head_distance, emotion_offset, guidance_scale=1.0):

        input_features = torch.cat([x, audio_features, gaze_direction, head_distance, emotion_offset], dim=-1)
        
        for layer in self.layers:
            x = layer(input_features)
        
        x = self.norm(x)
        
        if guidance_scale != 1.0:
            uncond_input_features = torch.cat([x, audio_features, torch.zeros_like(gaze_direction), 
                                               torch.zeros_like(head_distance), torch.zeros_like(emotion_offset)], dim=-1)
            uncond_output = self.forward(uncond_input_features, audio_features, gaze_direction, head_distance, emotion_offset, guidance_scale=1.0)
            x = uncond_output + guidance_scale * (x - uncond_output)
        
        return x
    


class ClassifierFreeGuidance(nn.Module):
    def __init__(self, model, guidance_scales):
        super().__init__()
        self.model = model
        self.guidance_scales = guidance_scales

    def forward(self, x, t, cond):
        unconditional_output = self.model(x, t, None)

        conditional_output = self.model(x, t, cond)

        guidance_output = torch.zeros_like(unconditional_output)
        for scale in self.guidance_scales:
            guidance_output = guidance_output + scale * (conditional_output - unconditional_output)

        return guidance_output + unconditional_output

