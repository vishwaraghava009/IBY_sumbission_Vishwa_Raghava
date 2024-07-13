import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from models.resnet import ResNet,Bottleneck, resnet18
from models.mysixdrepnet import SixDRepNet_Detector

FEATURE_SIZE = (2, 2) 
COMPRESS_DIM = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Emtn(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_pose_net = resnet18(pretrained=True).to(device)
        self.head_pose_net.fc = nn.Linear(self.head_pose_net.fc.in_features, 6).to(device)  # 6 corresponds to rotation and translation parameters
        self.rotation_net =  SixDRepNet_Detector()

        model = resnet18(pretrained=False,num_classes=512).to(device)  
        self.expression_net = nn.Sequential(*list(model.children())[:-1])
        self.expression_net.adaptive_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE) 
        # self.expression_net.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) 

        outputs=COMPRESS_DIM ## 512 -> 2048 channel
        self.fc = torch.nn.Linear(2048, outputs)

    def forward(self, x):
        rotations,_ = self.rotation_net.predict(x)
        logging.debug(f"rotation :{rotations}")
       

        head_pose = self.head_pose_net(x)

        # rotation = head_pose[:, :3]  - this is shit
        translation = head_pose[:, 3:]


        expression_resnet = self.expression_net(x)
        expression_flatten = torch.flatten(expression_resnet, start_dim=1)
        expression = self.fc(expression_flatten)  # (bs, 2048) ->>> (bs, COMPRESS_DIM)

        return rotations, translation, expression
