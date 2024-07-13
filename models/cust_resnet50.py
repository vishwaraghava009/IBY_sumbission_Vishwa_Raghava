import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


FEATURE_SIZE_AVG_POOL = 2 

class CustomResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        resnet = models.resnet50(*args, **kwargs)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
      #  self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # self.layer4 = resnet.layer4
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE_AVG_POOL)
        
        self.conv_reduce = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        
        x = self.adaptive_avg_pool(x)
        
        x = self.conv_reduce(x)
        
        return x

